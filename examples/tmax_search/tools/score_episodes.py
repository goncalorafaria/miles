"""Score run_episodes.py output with the port's reward function (search_reward) against a judge.

Usage: score_episodes.py <episodes.jsonl> <judge_url> <tokenizer>
"""

import asyncio
import collections
import json
import os
import sys
from statistics import mean
from types import SimpleNamespace

import search_reward
from search_config import resolve_env_args

SRC = os.environ.get(
    "PRIME_RL_SOURCE",
    "/weka/oe-adapt-default/shashankg/miles_search/prime-rl-sources/"
    "a5fcb924645d9bd8f326efd5087c8de6447c83fe17c6d07c36e99ee736d30a37/source",
)


def as_sample(result: dict) -> SimpleNamespace:
    tokens, mask = [], []
    for ids in result["turn_token_ids"]:
        tokens += list(ids) + [0]  # a masked separator between turns, like the tool tokens of a real sample
        mask += [1] * len(ids) + [0]
    return SimpleNamespace(
        tokens=tokens,
        response_length=len(tokens),
        loss_mask=mask,
        index=0,
        group_index=0,
        metadata={"search_episode": result["search_episode"]},
    )


async def main() -> None:
    path, judge_url, tokenizer = sys.argv[1:4]
    env = resolve_env_args(
        prime_rl_source=SRC,
        prime_rl_config="configs/experiments/search-agent-bc-rl-v1-1-fp32-lr5e-7/rl.toml",
        prime_rl_checkout="/weka/stevenc/prime-rl",
        weka_mount="/weka/oe-adapt-default",
        search_url="http://unused/search",
        terminal_url="http://unused/terminal",
        judge_url=judge_url,
    )
    os.environ["SEARCH_ENV_ARGS"] = json.dumps(env)
    args = SimpleNamespace(hf_checkpoint=tokenizer)
    results = [json.loads(line) for line in open(path)]
    samples = [as_sample(r) for r in results]
    rewards = await asyncio.gather(*(search_reward.reward_func(args, s) for s in samples))
    skipped = collections.Counter(s.metadata["judge"].get("skipped") or "judged" for s in samples)
    correct = [s.metadata["reward_components"]["judge_answer"] for s in samples if "reward_components" in s.metadata]
    print("rewards", [None if r is None else round(r, 3) for r in rewards])
    print("judge", dict(skipped))
    print(
        f"mean reward {mean(r for r in rewards if r is not None):.3f}, correct {mean(correct):.3f}, n={len(rewards)}"
    )


asyncio.run(main())
