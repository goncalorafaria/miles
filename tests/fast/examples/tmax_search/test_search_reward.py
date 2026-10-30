"""The search reward (search_reward.py): Prime-RL's format gates, two-stage judge and penalties on a sample."""

import asyncio
import json
from types import SimpleNamespace

import pytest
from tests.fast.examples.tmax_search.conftest_paths import PRIME_RL_CONFIG, PRIME_RL_SOURCE, needs_prime_rl_source

pytestmark = needs_prime_rl_source

_TOOL_TURN = "Let me search.\n</think>\n\n<tool_call>\n<function=search>\n<parameter=query>\nx\n</parameter>\n</function>\n</tool_call><|im_end|>"
_FINAL_TURN = "I found it.\n</think>\n\nThe answer is 1955.<|im_end|>"


@pytest.fixture
def reward(monkeypatch):
    import search_config
    import search_reward

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
    # Turn texts are token lists here: each "token" is a whole turn's text.
    monkeypatch.setattr(
        search_reward, "_tokenizer", lambda path: SimpleNamespace(decode=lambda ids, **kw: "".join(ids))
    )
    monkeypatch.setattr(search_reward.asyncio, "sleep", _no_sleep)
    return search_reward


async def _no_sleep(_):
    return None


class _Judge:
    def __init__(self, replies: list[str]):
        self.replies, self.requests = list(replies), []
        self.chat = self.completions = self

    async def create(self, **kwargs):
        self.requests.append(kwargs)
        reply = self.replies.pop(0)
        return SimpleNamespace(choices=[SimpleNamespace(message=SimpleNamespace(content=reply), finish_reason="stop")])


def _sample(turns: list[str], **episode) -> SimpleNamespace:
    tokens, mask = ["<prompt>"], []
    for turn in turns:
        tokens += [turn, "<tool response>"]
        mask += [1, 0]
    episode = {
        "question": "In what year?",
        "answer": "1955",
        "turns": len(turns),
        "stop_condition": "no_tools_called",
        "error": None,
        "incomplete": False,
        "final_role": "assistant",
        "final_has_tool_calls": False,
        "final_content": "The answer is 1955.",
        "stderr_count": 0,
    } | episode
    return SimpleNamespace(
        tokens=tokens,
        response_length=len(tokens) - 1,
        loss_mask=mask,
        index=0,
        group_index=0,
        metadata={"search_episode": episode},
    )


def _score(reward, monkeypatch, sample, replies):
    judge = _Judge(replies)
    monkeypatch.setattr(reward, "_client", lambda url, timeout: judge)
    value = asyncio.run(reward.reward_func(SimpleNamespace(hf_checkpoint="x"), sample))
    return value, sample.metadata["judge"], judge


def test_correct_answer_with_a_penalty(reward, monkeypatch):
    sample = _sample([_TOOL_TURN, _FINAL_TURN], stderr_count=2)
    value, trace, judge = _score(
        reward,
        monkeypatch,
        sample,
        [
            "<explanation>final</explanation>\n<extracted_answer>1955</extracted_answer>",
            "<explanation>same</explanation>\n<judgement>yes</judgement>",
        ],
    )
    assert value == pytest.approx(1.0 - 2 * 0.05)
    assert trace["valid"] and trace["score"] == 1.0 and "skipped" not in trace
    assert "[response]: The answer is 1955." in judge.requests[0]["messages"][0]["content"]
    assert judge.requests[0]["seed"] == 42 and judge.requests[0]["extra_body"]["chat_template_kwargs"] == {
        "enable_thinking": False
    }


def test_intermediate_format_failure_scores_zero_unjudged(reward, monkeypatch):
    sample = _sample(["Chatty text.\n</think>\n\nI will search.\n<tool_call>x</tool_call><|im_end|>", _FINAL_TURN])
    value, trace, judge = _score(reward, monkeypatch, sample, [])
    assert value == 0.0 and trace["skipped"] == "format:trajectory:content_before_tool_call" and not judge.requests


def test_missing_final_answer(reward, monkeypatch):
    sample = _sample(
        [_TOOL_TURN], final_has_tool_calls=True, final_content="", stop_condition="stop_after_tool_budget"
    )
    value, trace, _ = _score(reward, monkeypatch, sample, [])
    assert value == -1.0 and trace["skipped"] == "no_terminal_model_text"


def test_judge_that_never_parses_drops_the_reward(reward, monkeypatch):
    sample = _sample([_TOOL_TURN, _FINAL_TURN])
    value, trace, judge = _score(reward, monkeypatch, sample, ["no tags"] * 5)
    assert value is None and not trace["valid"] and trace["failure_stage"] == "extraction"
    assert [r["max_completion_tokens"] for r in judge.requests] == [1024, 1024, 2048, 4096, 8192]


def test_extracted_none_is_no_answer(reward, monkeypatch):
    sample = _sample([_FINAL_TURN])
    value, trace, _ = _score(reward, monkeypatch, sample, ["<extracted_answer>None</extracted_answer>"])
    assert value == 0.0 and trace["skipped"] == "extraction:no_final_answer"
