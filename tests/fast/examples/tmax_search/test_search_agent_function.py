"""The search episode loop (search_agent_function.py): Prime-RL's stop conditions, tool budget and warnings."""

import asyncio
import json
from types import SimpleNamespace

import pytest
from tests.fast.examples.tmax_search.conftest_paths import PRIME_RL_CONFIG, PRIME_RL_SOURCE, needs_prime_rl_source

pytestmark = needs_prime_rl_source


@pytest.fixture
def saf(monkeypatch):
    import search_agent_function
    import search_config

    env = search_config.resolve_env_args(
        prime_rl_source=str(PRIME_RL_SOURCE),
        prime_rl_config=PRIME_RL_CONFIG,
        prime_rl_checkout="/weka/stevenc/prime-rl",
        weka_mount="/weka/oe-adapt-default",
        search_url="http://search/search",
        terminal_url="http://terminal/terminal",
        judge_url="http://judge/v1",
    )
    monkeypatch.setenv(search_config.ENV_ARGS_VAR, json.dumps(env))
    search_config.env_args.cache_clear()
    return search_agent_function


def _call(name: str, arguments) -> SimpleNamespace:
    text = arguments if isinstance(arguments, str) else json.dumps(arguments)
    return SimpleNamespace(id=f"call_{name}", function=SimpleNamespace(name=name, arguments=text))


class _Policy:
    """Answers each turn from a script: a list of tool calls, or a final text."""

    def __init__(self, script: list, completion_tokens: int = 100):
        self.script = list(script)
        self.completion_tokens = completion_tokens
        self.requests = []
        self.chat = self.completions = self

    async def create(self, **kwargs):
        self.requests.append(kwargs)
        turn = self.script.pop(0)
        calls = turn if isinstance(turn, list) else []
        dumped = [{"id": c.id, "type": "function", "function": vars(c.function)} for c in calls]
        content = "" if calls else turn
        message = SimpleNamespace(
            tool_calls=calls or None,
            model_dump=lambda exclude_none: {
                "role": "assistant",
                "content": content,
                **({"tool_calls": dumped} if dumped else {}),
            },
        )
        return SimpleNamespace(
            usage=SimpleNamespace(
                total_tokens=1000 + 100 * len(self.requests), completion_tokens=self.completion_tokens
            ),
            choices=[SimpleNamespace(message=message, finish_reason="tool_calls" if calls else "stop")],
        )


class _Tools:
    """Echoes each call; counts what ran."""

    def __init__(self):
        self.ran = []

    async def call(self, episode, name, arguments):
        self.ran.append((name, arguments))
        return f"ran {name}"


def _run(saf, script, *, env_overrides=None, completion_tokens=100):
    env = saf.env_args() | (env_overrides or {})
    episode = saf._Episode(messages=[{"role": "system", "content": "s"}, {"role": "user", "content": "q"}])
    policy, tools = _Policy(script, completion_tokens), _Tools()
    asyncio.run(saf._loop(episode, env, policy, tools, {"max_tokens": 32768}, 131072))
    return episode, tools, policy


def test_final_answer_ends_the_episode(saf):
    episode, tools, _ = _run(saf, [[_call("search", {"query": "x"})], "The answer is 42."])
    assert episode.stop_condition == "no_tools_called" and episode.turns == 2 and episode.tool_calls == 1
    assert episode.messages[-1]["content"] == "The answer is 42."
    assert tools.ran == [("search", {"query": "x"})]


def test_tool_budget_final_warning_and_unexecuted_calls(saf):
    final = saf.env_args()["final_warning"]
    # Budget 3: the third call runs and gets the final warning, the fourth is answered without running.
    script = [
        [_call("search", {"query": "a"}), _call("search", {"query": "b"})],
        [_call("search", {"query": "c"}), _call("webterminal", {"command": "browse u"})],
        [_call("search", {"query": "d"})],
    ]
    episode, tools, _ = _run(saf, script, env_overrides={"max_tool_calls": 3, "max_turns": 4})
    tool_messages = [m["content"] for m in episode.messages if m["role"] == "tool"]
    assert len(tools.ran) == 3 and episode.tool_calls == 3
    assert tool_messages[2] == f"ran search\n\n{final}"
    assert tool_messages[3] == f"Tool not executed: tool-call budget exhausted. {final}"
    # The next turn calls a tool again: the budget is spent, so the episode stops without running it.
    assert episode.stop_condition == "stop_after_tool_budget" and episode.messages[-1]["tool_calls"]


def test_unparseable_arguments_do_not_count(saf):
    episode, tools, _ = _run(saf, [[_call("search", "[1, 2]"), _call("search", "{oops")], "done"])
    tool_messages = [m["content"] for m in episode.messages if m["role"] == "tool"]
    assert tool_messages[0] == "Tool arguments must be a JSON object"
    assert tool_messages[1].startswith("Expecting property name enclosed in double quotes")
    assert episode.tool_calls == 0 and not tools.ran


def test_turn_warning_fires_once(saf):
    # 15 of 151 turns left is the turn warning's threshold.
    script = [[_call("search", {"query": str(i)})] for i in range(140)] + ["done"]
    episode, _, _ = _run(saf, script, completion_tokens=100)
    tool_messages = [m["content"] for m in episode.messages if m["role"] == "tool"]
    assert [i for i, m in enumerate(tool_messages) if "final 10% of the turn budget" in m] == [135]
    assert "(15 of 151 assistant turns remain)" in tool_messages[135]
    assert not any("TOKEN BUDGET" in m for m in tool_messages)


def test_token_warning_fires_once_then_the_budget_stops(saf):
    # 13,107 of 131,072 tokens left is the token warning's threshold; at 131,072 the episode stops.
    script = [[_call("search", {"query": str(i)})] for i in range(140)]
    episode, _, _ = _run(saf, script, completion_tokens=1000)
    tool_messages = [m["content"] for m in episode.messages if m["role"] == "tool"]
    assert [i for i, m in enumerate(tool_messages) if "TOKEN BUDGET EXCEEDED" in m] == [117]
    assert "(13,072 of 131,072 tokens remain)" in tool_messages[117]
    assert episode.stop_condition == "max_total_completion_tokens_reached" and episode.turns == 132


def test_turn_limit(saf):
    script = [[_call("search", {"query": str(i)})] for i in range(5)]
    episode, _, _ = _run(saf, script, env_overrides={"max_turns": 3, "max_tool_calls": 2})
    assert episode.stop_condition == "stop_after_tool_budget" and episode.turns == 3


def test_tool_errors_are_observations(saf):
    tools = saf._Tools(saf.env_args(), {t: {"calls": 0, "failures": 0} for t in ("search", "webterminal")})
    episode = saf._Episode(messages=[])
    unknown = asyncio.run(tools.call(episode, "python", {"code": "1"}))
    extra = asyncio.run(tools.call(episode, "search", {"query": "x", "k": 3}))
    assert unknown == "'python'"
    assert extra == "SearchAgentEnv.__init__.<locals>.search() got an unexpected keyword argument 'k'"


def test_model_caused_terminal_stderr_is_counted(saf):
    tools = saf._Tools(saf.env_args(), {t: {"calls": 0, "failures": 0} for t in ("search", "webterminal")})
    episode = saf._Episode(messages=[])
    outputs = iter(
        [
            {"success": False, "stdout": "", "stderr": "grep: bad option", "exit_code": 2},  # a pipeline: counted
            {
                "success": False,
                "stdout": "",
                "stderr": "cat: x: failed to fetch",
                "error_type": "fetch",
            },  # browse: not
        ]
    )

    async def execute(**kwargs):
        return next(outputs)

    tools.webterminal_client = SimpleNamespace(execute=execute)
    asyncio.run(tools.call(episode, "webterminal", {"command": "grep -Z x"}))
    asyncio.run(tools.call(episode, "webterminal", {"command": "browse https://x.org"}))
    assert episode.stderr_count == 1
    assert tools.outcomes["webterminal"] == {"calls": 2, "failures": 2, "unknown_failures": 1, "fetch_failures": 1}
