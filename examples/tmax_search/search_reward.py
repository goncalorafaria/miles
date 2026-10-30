"""Search-agent reward (``--custom-rm-path search_reward.reward_func``): Prime-RL's judge and penalties.

reward = judge (0 or 1) - tool_stderr_penalty x model-caused webterminal stderr + missing_final_answer_penalty

as ``WebTerminalJudgeSearchAgentEnv`` sums its three reward functions (environments/search_agent/search_agent/
environment.py). The judge is ``BrowseCompJudgeRubric`` in mode ``extract_then_judge`` (search_agent/judging.py),
ported here because that module imports ``verifiers``; its format gates run unchanged from the pinned Prime-RL source
(``search_agent.protocol``). They read each turn's raw generated tokens, which only the session server's sample has:
``search_agent_function.run`` leaves no reward, so Miles calls this function with the finished sample, whose loss mask
marks exactly the generated tokens of each turn.

A judge that never gives a parseable answer (Prime-RL's ``pipeline_failed``, which excludes the rollout from its group)
returns None here, and the missing-reward filter drops the group. Each judgment is kept in ``metadata["judge"]`` and,
with ``SEARCH_TRACE_DIR`` set, appended to ``judge-<host>.jsonl`` there.
"""

import asyncio
import contextlib
import json
import logging
import os
import re
import socket
import time
from functools import cache
from pathlib import Path
from typing import Any

from openai import APIConnectionError, APIResponseValidationError, APIStatusError, AsyncOpenAI
from search_agent.protocol import (
    apply_browsecomp_format_gates,
    format_reference_answer,
    intermediate_qwen_reasoning_format_error,
)
from search_config import env_args

from miles.rollout.agentic.rollout_limits import rollout_lock
from miles.utils.types import Sample

logger = logging.getLogger(__name__)

# The judge's transport errors worth retrying, and how often (search_agent/judging.py _completion_when_available).
_RETRYABLE_STATUSES = frozenset({408, 429, 500, 502, 503, 504})
_TRANSPORT_FAILURES = 3


async def reward_func(args, samples: Sample | list[Sample], **kwargs) -> float | None | list[float | None]:
    if isinstance(samples, list):
        return list(await asyncio.gather(*(_reward(args, s) for s in samples)))
    return await _reward(args, samples)


async def _reward(args, sample: Sample) -> float | None:
    env = env_args()
    episode = (sample.metadata or {}).get("search_episode")
    if episode is None:  # the agent function failed before returning
        return None
    trace = await judge(args, sample, episode, env)
    sample.metadata["judge"] = trace
    _write_trace(sample, trace)
    if not trace["valid"]:
        return None
    missing = episode["final_has_tool_calls"] or not episode["final_content"].strip()
    components = {
        "judge_answer": trace["score"],
        "tool_stderr_penalty_reward": -env["tool_stderr_penalty"] * max(episode["stderr_count"], 0),
        "missing_final_answer_penalty_reward": env["missing_final_answer_penalty"] if missing else 0.0,
    }
    sample.metadata["reward_components"] = components
    return sum(components.values())


async def judge(args, sample: Sample, episode: dict, env: dict) -> dict[str, Any]:
    """BrowseCompJudgeRubric.judge_answer: the judgment as a trace (score, valid, skipped or the stages' attempts)."""
    trace: dict[str, Any] = {"mode": "extract_then_judge", "model": env["judge_model_path"], "invoked": False}
    try:
        score = await _judge_answer(args, sample, episode, env, trace)
    except Exception as e:  # noqa: BLE001 - search_agent.judging: an unexpected failure invalidates the reward
        logger.exception("search judge failed; dropping the reward")
        return trace | {
            "score": None,
            "valid": False,
            "failure_kind": "unexpected",
            "error": f"{type(e).__name__}: {e}",
        }
    return trace | {"score": score, "valid": score is not None}


async def _judge_answer(args, sample: Sample, episode: dict, env: dict, trace: dict) -> float | None:
    if episode["final_role"] != "assistant" or episode["final_has_tool_calls"] or not episode["turns"]:
        trace["skipped"] = "no_terminal_model_text"
        return 0.0
    if episode["incomplete"]:
        trace["skipped"] = "incomplete_rollout"
        return 0.0
    turns = turn_texts(args, sample)
    if len(turns) != episode["turns"]:
        # One generated stretch per turn, unless the session server forked the episode or cut its last turn.
        trace["skipped"] = f"turn_count_mismatch:{len(turns)}!={episode['turns']}"
        return None
    terminal = turns[-1]
    if not terminal.strip():
        trace["skipped"] = "no_terminal_model_text"
        return 0.0
    reason, _ = apply_browsecomp_format_gates(terminal)
    if reason:
        trace["skipped"] = f"format:{reason}"
        return 0.0
    for turn in turns[:-1]:
        reason = intermediate_qwen_reasoning_format_error(turn)
        if reason:
            trace["skipped"] = f"format:trajectory:{reason}"
            return 0.0
    final_content = episode["final_content"]
    if not final_content.strip():
        trace["skipped"] = "no_terminal_model_text"
        return 0.0
    prompts = env["judge_prompts"]
    question = episode["question"].strip()
    extracted = await _stage(
        env,
        prompts["BROWSECOMP_ANSWER_EXTRACTION_PROMPT"].format(question=question, response=final_content),
        "extraction",
        trace,
    )
    if not extracted:
        trace["skipped"] = "extraction:failed" if extracted is None else "extraction:no_final_answer"
        return None if extracted is None else 0.0
    judge_prompt = prompts["BROWSECOMP_EXTRACTED_ANSWER_CORRECTNESS_JUDGE_PROMPT"].format(
        question=question, correct_answer=format_reference_answer(episode["answer"]), response=extracted
    )
    score = await _stage(env, judge_prompt, "verdict", trace)
    return None if score is None else float(score)


def turn_texts(args, sample: Sample) -> list[str]:
    """Each turn's generated text, special tokens included: the sample's stretches of loss-masked response tokens
    (search_agent.judging decodes each turn's completion_ids the same way)."""
    response = sample.tokens[len(sample.tokens) - sample.response_length :]
    turns, current = [], []
    for token, mask in zip(response, sample.loss_mask, strict=True):
        if mask:
            current.append(token)
        elif current:
            turns.append(current)
            current = []
    if current:
        turns.append(current)
    tokenizer = _tokenizer(args.hf_checkpoint)
    return [tokenizer.decode(ids, skip_special_tokens=False) for ids in turns]


@cache
def _tokenizer(path: str):
    from transformers import AutoTokenizer

    return AutoTokenizer.from_pretrained(path, trust_remote_code=True)


def _parse_extraction(content: str) -> str:
    if "</think>" in content:
        content = content.rsplit("</think>", 1)[1]
    matches = re.findall(r"<extracted_answer>(.*?)</extracted_answer>", content, re.I | re.S)
    if not matches:
        raise ValueError("Extractor returned no complete extracted_answer tag")
    answer = matches[-1].strip()
    return "" if answer.casefold() == "none" else answer


def _parse_verdict(content: str) -> float:
    if "</think>" in content:
        content = content.rsplit("</think>", 1)[1]
    match = re.fullmatch(
        r"<explanation>(.*?)</explanation>\s*<judgement>(.*?)</judgement>", content.strip(), re.I | re.S
    )
    verdict = re.search(r"\b(yes|no)\b", match.group(2), re.I) if match else None
    if verdict is None:
        raise ValueError("Judge returned an invalid structured judgement")
    return float(verdict.group(1).casefold() == "yes")


async def _stage(env: dict, prompt: str, stage: str, trace: dict) -> Any:
    """One stage (extraction or verdict): up to judge_max_retries attempts with growing token budgets and seeds; a
    transport failure ends the stage. None when no attempt parses."""
    attempts: list[dict] = []
    trace[stage] = {"prompt": prompt, "attempts": attempts}
    parse = _parse_extraction if stage == "extraction" else _parse_verdict
    for attempt in range(env["judge_max_retries"]):
        budget = 1024 * (1 if attempt < 2 else 2 ** (attempt - 1))
        request = {
            "model": env["judge_model_path"],
            "messages": [{"role": "user", "content": prompt}],
            "temperature": 0.7,
            "top_p": 0.8,
            "presence_penalty": 1.5,
            "max_completion_tokens": budget,
            "seed": env["judge_seed"] + attempt,
            "extra_body": {
                "top_k": 20,
                "min_p": 0.0,
                "repetition_penalty": 1.0,
                "chat_template_kwargs": {"enable_thinking": False},
            },
        }
        entry: dict[str, Any] = {"attempt": attempt, "max_completion_tokens": budget, "seed": request["seed"]}
        attempts.append(entry)
        try:
            trace["invoked"] = True
            response = await _completion_when_available(env, request, trace[stage])
            if not response.choices:
                raise ValueError("Judge returned no choices")
            content = response.choices[0].message.content
            entry["content"], entry["finish_reason"] = content, response.choices[0].finish_reason
            if not isinstance(content, str):
                raise ValueError("Judge returned no text content")
            entry["result"] = parse(content)
            return entry["result"]
        except (ValueError, APIResponseValidationError, APIConnectionError, APIStatusError) as e:
            transport = isinstance(e, (APIConnectionError, APIStatusError))
            entry["error"], entry["failure_kind"] = f"{type(e).__name__}: {e}", "transport" if transport else "parsing"
            if transport:
                break
            if attempt < env["judge_max_retries"] - 1:
                await asyncio.sleep(2**attempt)
    trace.update(failure_stage=stage, failure_kind=attempts[-1]["failure_kind"], error=attempts[-1]["error"])
    return None


async def _completion_when_available(env: dict, request: dict, stage_trace: dict) -> Any:
    """Retry the judge's transient transport failures with the same request; the third raises."""
    failures = 0
    while True:
        try:
            stage_trace["request_count"] = stage_trace.get("request_count", 0) + 1
            return await _client(env["judge_server_url"], env["judge_timeout"]).chat.completions.create(**request)
        except (APIConnectionError, APIStatusError) as e:
            if isinstance(e, APIStatusError) and e.status_code not in _RETRYABLE_STATUSES:
                raise
            failures += 1
            stage_trace["transport_failures"] = failures
            if failures == _TRANSPORT_FAILURES:
                raise
            await asyncio.sleep(min(30, 2 ** min(failures, 5)))


_clients: dict[tuple, AsyncOpenAI] = {}


def _client(base_url: str, timeout: float) -> AsyncOpenAI:
    """One judge client per event loop (the rollout process may run more than one)."""
    key = (id(asyncio.get_running_loop()), base_url)
    if key not in _clients:
        _clients[key] = AsyncOpenAI(base_url=base_url, api_key="EMPTY", max_retries=0, timeout=timeout)
    return _clients[key]


def _write_trace(sample: Sample, trace: dict) -> None:
    trace_dir = os.getenv("SEARCH_TRACE_DIR")
    if not trace_dir:
        return
    record = {"time": time.time(), "sample_index": sample.index, "group_index": sample.group_index, "judge": trace}
    with contextlib.suppress(Exception):
        path = Path(trace_dir) / f"judge-{socket.gethostname()}.jsonl"
        path.parent.mkdir(parents=True, exist_ok=True)
        with rollout_lock(f"search-judge-{path.name}"), open(path, "a") as f:
            f.write(json.dumps(record, default=str) + "\n")
