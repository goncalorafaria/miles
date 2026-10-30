"""Search-agent episode loop for ``agentic_tool_call.generate``: Prime-RL's ``search_agent.environment``.

Miles calls ``run`` once per episode (``--custom-agent-function-path search_agent_function.run``); the policy is
reached through the session server at ``base_url/v1``, which records every turn's tokens. The loop is a port of
Prime-RL's ``WebTerminalJudgeSearchAgentEnv`` (environments/search_agent/search_agent/{agent,environment}.py and the
verifiers ``MultiTurnEnv`` / ``ToolEnv`` it runs on). Everything of that environment that does not import
``verifiers`` runs unchanged from the pinned Prime-RL source (on ``PYTHONPATH``): the LiteRegistry tool clients
(search, URL fetch, the terminal, ``browse``), the stderr-penalty rules (``search_agent.tools``) and the tool-failure
categories (``search_agent.tool_metrics``).

  messages = system prompt + question
  repeat:
      chat completion offering search and webterminal
      stop: tool budget spent and the turn calls a tool | total completion tokens >= budget | max turns |
            no tool call (the final answer) | context full or length-truncated turn
      run the turn's tool calls in order; past the budget a call is answered without running; the last allowed call
      gets the final warning, and the token and turn budget warnings are appended once each

This function computes no reward: the judge reads the raw tokens of every turn, which only the session server's
sample has, so ``search_reward.reward_func`` scores the sample. ``run`` leaves what it needs under
``search_episode`` and the episode's metrics under ``agent_metrics``.

Settings come from ``SEARCH_ENV_ARGS`` (``search_config``); ``SEARCH_TRACE_DIR``, if set, gets one JSONL line per
episode (messages and metrics), and ``AGENT_MODEL_NAME`` names the policy model (default "model").
"""

import contextlib
import json
import logging
import math
import os
import socket
import time
from collections.abc import Mapping
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from literegistry_tool_client import (
    FetchClient,
    SearchClient,
    TerminalExecutionClient,
    WebTerminalExecutionClient,
    terminal_output_to_tool_content,
)
from literegistry_tool_client.asset_store import WebAssetStore
from openai import AsyncOpenAI, BadRequestError
from search_agent.tool_metrics import TOOL_FAILURE_CATEGORIES, tool_failure_category
from search_agent.tools import _error_output, _exception_is_model_caused, _tool_stderr_is_model_error
from search_config import env_args

from miles.rollout.agentic.rollout_limits import rollout_lock
from miles.rollout.agentic.session import openai_session_url

logger = logging.getLogger(__name__)

# Prime-RL never sends a Jina request with a local search model path; FetchClient still takes the URL.
_JINA_READER_URL = "https://r.jina.ai"


def tool_definitions(env: Mapping[str, Any]) -> list[dict]:
    """The two tools as Prime-RL offers them: verifiers derives the parameters from the environment's Python
    signatures (strict mode) and the descriptions come from the configured files. Checked against a Prime-RL
    trajectory's ``tools``."""

    def tool(name: str, description: str, parameter: str, parameter_description: str) -> dict:
        parameters = {
            "properties": {
                parameter: {"description": parameter_description, "title": parameter.title(), "type": "string"}
            },
            "required": [parameter],
            "title": f"{name}_args",
            "type": "object",
            "additionalProperties": False,
        }
        function = {"name": name, "description": description, "parameters": parameters, "strict": True}
        return {"type": "function", "function": function}

    return [
        tool("search", env["search_description"], "query", "Focused web search query."),
        tool(
            "webterminal",
            env["webterminal_description"],
            "command",
            "A `browse <url>` command, optionally followed by a pipeline.",
        ),
    ]


@dataclass
class _Episode:
    messages: list[dict]
    turns: int = 0
    # Executed tool calls, the budget's count (search_agent_tool_call_count); a call whose arguments do not parse
    # is not one.
    tool_calls: int = 0
    completion_tokens: int = 0
    last_total_tokens: int = 0
    last_finish_reason: str | None = None
    stop_condition: str = ""
    error: str | None = None
    truncated: bool = False
    stderr_count: int = 0
    token_warning_sent: bool = False
    turn_warning_sent: bool = False
    final_turn_tokens: int = 0
    # search_agent.agent._observed_call's counters, per tool.
    outcomes: dict[str, dict[str, int]] = field(
        default_factory=lambda: {tool: {"calls": 0, "failures": 0} for tool in ("search", "webterminal")}
    )
    gen_times: list[float] = field(default_factory=list)
    tool_times: list[float] = field(default_factory=list)


async def run(
    base_url: str,
    prompt: Any,
    request_kwargs: dict[str, Any] | None = None,
    metadata: dict[str, Any] | None = None,
    **kwargs: Any,
) -> dict[str, Any]:
    env = env_args()
    metadata = metadata or {}
    episode = _Episode(messages=list(prompt))
    policy = AsyncOpenAI(base_url=openai_session_url(base_url), api_key="EMPTY", timeout=3600)
    tools = _Tools(env, episode.outcomes)
    t_start = time.monotonic()
    await _loop(episode, env, policy, tools, dict(request_kwargs or {}), metadata.get("max_seq_len"))
    metrics = _metrics(episode, time.monotonic() - t_start)
    last = episode.messages[-1]
    result = {
        "agent_metrics": metrics,
        "search_episode": {
            "question": metadata["question"],
            "answer": metadata["answer"],
            "turns": episode.turns,
            "stop_condition": episode.stop_condition,
            "error": episode.error,
            # search_agent.judging: an error, a truncated step or a final turn that did not stop is not judged.
            "incomplete": bool(episode.error or episode.truncated or episode.last_finish_reason != "stop"),
            "final_role": last.get("role"),
            "final_has_tool_calls": bool(last.get("tool_calls")),
            "final_content": str(last.get("content") or ""),
            "stderr_count": episode.stderr_count,
        },
    }
    logger.info(
        f"[{metadata.get('id')}] episode done: stop={episode.stop_condition} turns={episode.turns} "
        f"tool_calls={episode.tool_calls} tokens={episode.last_total_tokens} time={metrics['total_time']:.0f}s"
    )
    _write_trace(episode, metadata, metrics)
    return result


async def _loop(
    episode: _Episode,
    env: Mapping[str, Any],
    policy: AsyncOpenAI,
    tools: "_Tools",
    request_kwargs: dict,
    max_seq_len: int | None,
) -> None:
    model = os.getenv("AGENT_MODEL_NAME", "model")
    per_turn_cap = request_kwargs.pop("max_tokens", None)
    tool_defs = tool_definitions(env)
    while True:
        # Prime-RL asks for the per-turn cap and vLLM shortens it to the context left; so does this.
        max_tokens = per_turn_cap
        if max_seq_len is not None:
            remaining = max_seq_len - episode.last_total_tokens
            max_tokens = remaining if per_turn_cap is None else min(per_turn_cap, remaining)
            if max_tokens <= 0:
                episode.stop_condition, episode.truncated = "prompt_too_long", True
                return
        t0 = time.monotonic()
        try:
            completion = await policy.chat.completions.create(
                model=model,
                messages=episode.messages,
                tools=tool_defs,
                **({"max_tokens": max_tokens} if max_tokens is not None else {}),
                extra_body=request_kwargs,
            )
        except BadRequestError as e:
            if any(word in str(e).lower() for word in ("context", "length", "too long")):
                # The prompt outgrew the context: Prime-RL's OverlongPromptError, a truncated rollout.
                episode.stop_condition, episode.truncated = "prompt_too_long", True
            else:
                episode.stop_condition, episode.error = "has_error", f"{type(e).__name__}: {e}"
            logger.warning(f"policy request rejected ({episode.stop_condition}): {e}")
            return
        except Exception as e:  # noqa: BLE001 - verifiers' has_error: the rollout carries no valid reward
            episode.stop_condition, episode.error = "has_error", f"{type(e).__name__}: {e}"
            return
        finally:
            episode.gen_times.append(time.monotonic() - t0)
        episode.turns += 1
        if completion.usage is not None:
            episode.last_total_tokens = completion.usage.total_tokens
            episode.completion_tokens += completion.usage.completion_tokens
            episode.final_turn_tokens = completion.usage.completion_tokens
        choice = completion.choices[0]
        episode.last_finish_reason = choice.finish_reason
        # Echo the assistant turn verbatim: the session server prefix-matches the next request against it.
        message = choice.message.model_dump(exclude_none=True)
        episode.messages.append(message)
        tool_calls = choice.message.tool_calls or []

        # Prime-RL's stop conditions, in its order (priority, then name: verifiers/decorators.py).
        if tool_calls and episode.tool_calls >= env["max_tool_calls"]:
            episode.stop_condition = "stop_after_tool_budget"
            return
        if 0 < env["max_total_completion_tokens"] <= episode.completion_tokens:
            episode.stop_condition = "max_total_completion_tokens_reached"
            return
        if episode.turns >= env["max_turns"]:
            episode.stop_condition = "max_turns_reached"
            return
        if not tool_calls:
            episode.stop_condition = "no_tools_called"
            return
        if max_seq_len is not None and episode.last_total_tokens > max_seq_len:
            episode.stop_condition, episode.truncated = "prompt_too_long", True
            return
        if choice.finish_reason == "length":
            # A departure: Prime-RL would run the complete tool calls of a length-truncated turn and go on. The
            # session server cannot extend a turn that has no <|im_end|>, so the episode ends here, unjudged.
            episode.stop_condition = "length"
            return

        for tool_call in tool_calls:
            content = await _run_tool_call(episode, env, tools, tool_call)
            episode.messages.append({"role": "tool", "tool_call_id": tool_call.id, "content": content})


async def _run_tool_call(episode: _Episode, env: Mapping[str, Any], tools: "_Tools", tool_call: Any) -> str:
    """One tool call, as search_agent.agent.SearchAgentEnv._env_response runs it: its observation."""
    if episode.tool_calls >= env["max_tool_calls"]:
        return f"Tool not executed: tool-call budget exhausted. {env['final_warning']}"
    try:
        name = tool_call.function.name
        arguments = json.loads(tool_call.function.arguments)
        if not isinstance(arguments, dict):
            raise ValueError("Tool arguments must be a JSON object")
        arguments.pop("documents", None)  # a hidden argument the environment owns, never the model
    except Exception as e:  # noqa: BLE001 - verifiers' error_formatter: the exception text is the observation
        return f"{e}"
    episode.tool_calls += 1
    t0 = time.monotonic()
    try:
        content = await tools.call(episode, name, arguments)
    finally:
        episode.tool_times.append(time.monotonic() - t0)
    warning = _budget_warning(episode, env)
    if warning:
        content = f"{content}\n\n{warning}"
    if episode.tool_calls == env["max_tool_calls"]:
        content = f"{content}\n\n{env['final_warning']}"
    return content


def _budget_warning(episode: _Episode, env: Mapping[str, Any]) -> str:
    """search_agent.agent.SearchAgentEnv._budget_warning: one-time notices near the end of the token and turn
    budgets (the "10%" is in the text whatever the fraction)."""
    warnings = []
    token_fraction, token_limit = env["token_budget_warning_fraction"], env["max_total_completion_tokens"]
    if token_fraction is not None and token_limit > 0 and not episode.token_warning_sent:
        remaining = max(0, token_limit - math.ceil(episode.completion_tokens))
        if remaining <= math.floor(token_limit * token_fraction):
            episode.token_warning_sent = True
            warnings.append(
                "[TOKEN BUDGET EXCEEDED] You are in the final 10% of the total "
                f"completion-token budget ({remaining:,} of {token_limit:,} "
                "tokens remain). You should now do your best to compose the final "
                "answer."
            )
    turn_fraction, max_turns = env["turn_budget_warning_fraction"], env["max_turns"]
    if turn_fraction is not None and max_turns > 0 and not episode.turn_warning_sent:
        remaining = max(0, max_turns - episode.turns)
        if remaining <= max(1, math.floor(max_turns * turn_fraction)):
            episode.turn_warning_sent = True
            warnings.append(
                "IMPORTANT: You are in the final 10% of the turn budget "
                f"({remaining} of {max_turns} assistant turns remain). "
                "Finish gathering essential evidence and prepare your final answer."
            )
    return "\n\n".join(warnings)


class _Tools:
    """The episode's tool clients, built as search_agent.environment.load_environment builds them."""

    def __init__(self, env: Mapping[str, Any], outcomes: dict[str, dict[str, int]]):
        self.outcomes = outcomes
        self.search_client = SearchClient(
            env["search_server_url"],
            model_path=env["local_search_model_path"],
            timeout=env["timeout"],
            max_retries=env["max_retries"],
        )
        terminal = TerminalExecutionClient(
            env["terminal_server_url"],
            timeout=env["timeout"],
            max_retries=env["max_retries"],
            truncation=env["terminal_truncation"],
        )
        fetch = FetchClient(
            _JINA_READER_URL,
            timeout=env["timeout"],
            max_retries=env["max_retries"],
            local_search_server_url=env["search_server_url"],
            local_search_model_path=env["local_search_model_path"],
        )
        self.webterminal_client = WebTerminalExecutionClient(terminal, fetch)
        self.asset_store = WebAssetStore()  # the rollout's page cache, keyed by URL

        num_results = env["search_num_results"]

        async def search(query: str, documents: Any = None) -> str:
            return await self._search(query, num_results)

        # Prime-RL's search tool is a closure of SearchAgentEnv.__init__; a call with bad arguments raises a
        # TypeError naming it, and that message is the observation.
        search.__qualname__ = "SearchAgentEnv.__init__.<locals>.search"
        self._tool_map = {"search": search}

    async def call(self, episode: _Episode, name: str, arguments: dict) -> str:
        if name == "webterminal":
            return await self._webterminal(episode, arguments)
        try:
            return str(await self._tool_map[name](**arguments))
        except Exception as e:  # noqa: BLE001 - an unknown tool's KeyError, a TypeError, a service failure
            return f"{e}"

    async def _search(self, query: str, num_results: int) -> str:
        output = await self._observed("search", self.search_client.execute, query=query, num_results=num_results)
        if not isinstance(output, Mapping):
            raise TypeError("search client returned a non-object response")
        return _format_search_output(output)

    async def _webterminal(self, episode: _Episode, arguments: dict) -> str:
        """search_agent.agent: the webterminal branch of _env_response, with environment.py's stderr count."""
        try:
            output = await self._observed(
                "webterminal",
                self.webterminal_client.execute,
                asset_store=self.asset_store,
                command=arguments["command"],
            )
            if not isinstance(output, Mapping):
                raise TypeError("webterminal client returned a non-object response")
            raw = dict(output)
            content = terminal_output_to_tool_content(raw)
        except Exception as e:  # noqa: BLE001
            self._count_stderr(episode, arguments, None, e)
            return f"{e}"
        self._count_stderr(episode, arguments, raw, None)
        return content

    @staticmethod
    def _count_stderr(episode: _Episode, arguments: dict, output: dict | None, error: Exception | None) -> None:
        """WebTerminalJudgeSearchAgentEnv.record_webterminal_result: count model-caused stderr only."""
        if output is None:
            raw, model_caused = _error_output("terminal", error), _exception_is_model_caused(error)
        else:
            raw, model_caused = output, True
        if _tool_stderr_is_model_error("terminal", arguments, raw, request_model_caused=model_caused):
            episode.stderr_count += 1

    async def _observed(self, name: str, execute, **kwargs) -> Any:
        """search_agent.agent._observed_call: count calls and failures by category."""
        counters = self.outcomes[name]
        counters["calls"] += 1
        try:
            result = await execute(**kwargs)
        except Exception as e:
            counters["failures"] += 1
            key = f"{tool_failure_category(error=e)}_failures"
            counters[key] = counters.get(key, 0) + 1
            raise
        if not isinstance(result, Mapping) or result.get("success") is False:
            counters["failures"] += 1
            key = f"{tool_failure_category(result=result)}_failures"
            counters[key] = counters.get(key, 0) + 1
        return result


def _format_search_output(output: Mapping[str, Any]) -> str:
    """search_agent.agent._format_search_output with URL references."""
    data = output.get("data") or {}
    results = (data.get("organic") or data.get("results") or []) if isinstance(data, Mapping) else []
    results = [item for item in results if isinstance(item, Mapping)] if isinstance(results, list) else []
    if not results:
        return json.dumps(dict(output), ensure_ascii=False)
    rendered = []
    for result in results:
        title = str(result.get("title") or "Untitled result").strip()
        snippet = str(result.get("snippet") or result.get("description") or "").strip()
        url = str(result.get("link") or result.get("url") or "").strip()
        lines = [f"**{title}**"]
        if snippet:
            lines.append(snippet)
        if url:
            lines.append(f"Source: {url}")
        rendered.append("\n".join(lines))
    return "\n\n".join(rendered)


def _metrics(episode: _Episode, total_time: float) -> dict[str, Any]:
    """Prime-RL's environment metrics (num_turns, the per-tool call and outcome counters, terminal_turn_tokens) plus
    the stop condition and time split."""
    completion = episode.messages
    called = [tc for m in completion if m.get("role") == "assistant" for tc in m.get("tool_calls") or []]
    metrics: dict[str, Any] = {
        "turns": episode.turns,
        "tool_calls": episode.tool_calls,
        "total_tool_calls": len(called),
        "search_calls": sum(tc["function"]["name"] == "search" for tc in called),
        "webterminal_calls": sum(tc["function"]["name"] == "webterminal" for tc in called),
        "stderr_count": episode.stderr_count,
        "stop_condition": episode.stop_condition,
        "completion_tokens": episode.completion_tokens,
        "total_tokens": episode.last_total_tokens,
        "terminal_turn_tokens": episode.final_turn_tokens if episode.stop_condition == "no_tools_called" else 0,
        "model_query_time_sum": sum(episode.gen_times),
        "env_execution_time_sum": sum(episode.tool_times),
        "total_tool_time": sum(episode.tool_times),
        "total_time": total_time,
    }
    for tool, counters in episode.outcomes.items():
        for key in ("calls", "failures", *(f"{category}_failures" for category in TOOL_FAILURE_CATEGORIES)):
            metrics[f"{tool}_retrieval_{key}"] = counters.get(key, 0)
    return metrics


def _write_trace(episode: _Episode, metadata: dict, metrics: dict) -> None:
    """Append the episode to SEARCH_TRACE_DIR/traces-<host>.jsonl; tracing never fails the episode."""
    trace_dir = os.getenv("SEARCH_TRACE_DIR")
    if not trace_dir:
        return
    record = {"time": time.time(), "id": metadata.get("id"), "metrics": metrics, "messages": episode.messages}
    with contextlib.suppress(Exception):
        path = Path(trace_dir) / f"traces-{socket.gethostname()}.jsonl"
        path.parent.mkdir(parents=True, exist_ok=True)
        line = json.dumps(record, default=str) + "\n"
        with rollout_lock(f"search-trace-{path.name}"), open(path, "a") as f:
            f.write(line)
