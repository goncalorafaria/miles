"""Score Prime-RL trajectories with the Miles port's reward function and compare with Prime-RL's own judgment.

Usage: judge_parity.py <step.jsonl> <max_trajectories> <judge_url> <tokenizer>
"""

import asyncio
import collections
import json
import os
import sys
from types import SimpleNamespace

import search_reward
from search_config import resolve_env_args

SRC = os.environ.get(
    "PRIME_RL_SOURCE",
    "/weka/oe-adapt-default/shashankg/miles_search/prime-rl-sources/"
    "a5fcb924645d9bd8f326efd5087c8de6447c83fe17c6d07c36e99ee736d30a37/source",
)


def as_sample_and_episode(trajectory: dict) -> tuple[SimpleNamespace, dict]:
    payload = trajectory["payload"]
    nodes = [n for n in payload["nodes"]]
    tokens, loss_mask, turns = [], [], 0
    for n in nodes:
        if n["message"]["role"] != "assistant":
            continue
        turns += 1
        ids, mask = n["token_ids"], n["mask"]
        tokens += ids
        loss_mask += [int(m) for m in mask]
    first = loss_mask.index(1)
    sample = SimpleNamespace(
        tokens=tokens,
        response_length=len(tokens) - first,
        loss_mask=loss_mask[first:],
        index=0,
        group_index=0,
        metadata={},
    )
    last = nodes[-1]["message"]
    data = payload["task"]["data"]
    metrics = payload["metrics"]
    calls = payload["calls"]
    episode = {
        "question": data.get("question") or data["prompt"],
        "answer": data["answer"],
        "turns": turns,
        "stop_condition": payload["stop_condition"],
        "error": None,
        "incomplete": bool(payload.get("errors")) or calls[-1]["finish_reason"] != "stop",
        "final_role": last["role"],
        "final_has_tool_calls": bool(last.get("tool_calls")),
        "final_content": str(last.get("content") or ""),
        "stderr_count": round(-metrics["tool_stderr_penalty_reward"] / 0.05),
    }
    return sample, episode


async def main() -> None:
    path, limit, judge_url, tokenizer = sys.argv[1], int(sys.argv[2]), sys.argv[3], sys.argv[4]
    env = resolve_env_args(
        prime_rl_source=SRC,
        prime_rl_config="configs/experiments/search-agent-001-3-quote-v3/rl.toml",
        prime_rl_checkout="/weka/stevenc/prime-rl",
        weka_mount="/weka/oe-adapt-default",
        search_url="http://unused/search",
        terminal_url="http://unused/terminal",
        judge_url=judge_url,
    )
    os.environ["SEARCH_ENV_ARGS"] = json.dumps(env)
    args = SimpleNamespace(hf_checkpoint=tokenizer)
    jobs = []
    with open(path) as f:
        for line in f:
            jobs.append(json.loads(line))
            if len(jobs) >= limit:
                break
    sem = asyncio.Semaphore(32)

    async def one(trajectory):
        sample, episode = as_sample_and_episode(trajectory)
        sample.metadata["search_episode"] = episode
        async with sem:
            reward = await search_reward.reward_func(args, sample)
        theirs = trajectory["payload"]["info"]["judge"]
        m = trajectory["payload"]["metrics"]
        their_reward = m["judge_answer"] + m["tool_stderr_penalty_reward"] + m["missing_final_answer_penalty_reward"]
        return sample.metadata["judge"], theirs, reward, their_reward

    results = await asyncio.gather(*(one(t) for t in jobs))
    counts = collections.Counter()
    for ours, theirs, reward, their_reward in results:
        ours_skip, theirs_skip = ours.get("skipped"), theirs.get("skipped")
        counts["n"] += 1
        counts["skip_agree"] += ours_skip == theirs_skip
        if ours_skip != theirs_skip:
            print("SKIP DIFF", ours_skip, theirs_skip, flush=True)
        if not ours_skip and not theirs_skip:
            counts["judged_both"] += 1
            counts["score_agree"] += ours["score"] == theirs["score"]
            counts["ours_correct"] += ours["score"] == 1.0
            counts["theirs_correct"] += theirs["score"] == 1.0
        counts["reward_agree"] += reward is not None and abs(reward - their_reward) < 1e-9
        counts["ours_invalid"] += reward is None
    print("SUMMARY", json.dumps(counts))


asyncio.run(main())
