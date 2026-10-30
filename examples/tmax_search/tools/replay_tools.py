"""Replay Prime-RL trajectories' tool calls through the Miles port's tool layer and compare the observations.

Usage: replay_tools.py <step.jsonl> <max_trajectories> <search_url> <terminal_url>
"""

import asyncio
import json
import os
import re
import sys
from types import SimpleNamespace

import search_agent_function as saf
from search_config import resolve_env_args

SRC = os.environ.get(
    "PRIME_RL_SOURCE",
    "/weka/oe-adapt-default/shashankg/miles_search/prime-rl-sources/"
    "a5fcb924645d9bd8f326efd5087c8de6447c83fe17c6d07c36e99ee736d30a37/source",
)
ASSET = re.compile(r"asset_[0-9a-f]{32}")
HOST = re.compile(r"at http://[^/\s]+/")


def normalize(content: str) -> str:
    content = HOST.sub("at http://HOST/", ASSET.sub("asset_X", content))
    head, sep, tail = content.partition("\n\n")
    try:
        payload = json.loads(head)
        payload.pop("execution_time", None)
        return json.dumps(payload, sort_keys=True) + sep + tail
    except (json.JSONDecodeError, AttributeError):
        return content


async def replay(trajectory: dict, env: dict) -> dict:
    payload = trajectory["payload"]
    nodes = payload["nodes"]
    calls = payload["calls"]
    episode = saf._Episode(messages=[])
    tools = saf._Tools(env, episode.outcomes)
    by_node_usage = {c["node"]: c.get("usage") or {} for c in calls}
    mismatches, total = [], 0
    for i, n in enumerate(nodes):
        m = n["message"]
        if m["role"] != "assistant":
            continue
        episode.turns += 1
        episode.completion_tokens += int(by_node_usage.get(i, {}).get("completion_tokens") or 0)
        # A turn's tool messages follow it in order (each parallel call's message hangs off the previous one).
        tool_msgs = []
        for later in nodes[i + 1 :]:
            if later["message"]["role"] != "tool":
                break
            tool_msgs.append(later["message"])
        for tc, recorded in zip(m.get("tool_calls") or [], tool_msgs, strict=False):
            call = SimpleNamespace(id=tc["id"], function=SimpleNamespace(name=tc["name"], arguments=tc["arguments"]))
            ours = await saf._run_tool_call(episode, env, tools, call)
            total += 1
            if normalize(ours) != normalize(recorded["content"]):
                mismatches.append({"call": tc, "ours": ours[:600], "theirs": recorded["content"][:600]})
    metrics = payload["metrics"]
    ours_metrics = saf._metrics(episode, 0.0)
    counter_diffs = {
        k: (ours_metrics[k], metrics[k])
        for k in metrics
        if k.endswith(("_retrieval_calls", "_retrieval_failures")) and ours_metrics.get(k) != metrics[k]
    }
    return {
        "id": trajectory.get("question_id"),
        "calls": total,
        "mismatches": mismatches,
        "stderr_penalty": (-env["tool_stderr_penalty"] * episode.stderr_count, metrics["tool_stderr_penalty_reward"]),
        "counter_diffs": counter_diffs,
    }


async def main() -> None:
    path, limit, search_url, terminal_url = sys.argv[1], int(sys.argv[2]), sys.argv[3], sys.argv[4]
    env = resolve_env_args(
        prime_rl_source=SRC,
        prime_rl_config="configs/experiments/search-agent-001-3-quote-v3/rl.toml",
        prime_rl_checkout="/weka/stevenc/prime-rl",
        weka_mount="/weka/oe-adapt-default",
        search_url=search_url,
        terminal_url=terminal_url,
        judge_url="http://unused/v1",
    )
    os.environ["SEARCH_ENV_ARGS"] = json.dumps(env)
    done = 0
    summary = {"trajectories": 0, "calls": 0, "mismatched_calls": 0, "stderr_mismatch": 0, "counter_mismatch": 0}
    with open(path) as f:
        for line in f:
            trajectory = json.loads(line)
            if trajectory["payload"]["metrics"]["total_tool_calls"] < 1:
                continue
            result = await replay(trajectory, env)
            summary["trajectories"] += 1
            summary["calls"] += result["calls"]
            summary["mismatched_calls"] += len(result["mismatches"])
            summary["stderr_mismatch"] += abs(result["stderr_penalty"][0] - result["stderr_penalty"][1]) > 1e-9
            summary["counter_mismatch"] += bool(result["counter_diffs"])
            print(json.dumps({k: v for k, v in result.items() if k != "mismatches"}), flush=True)
            for mm in result["mismatches"][:3]:
                print("  MISMATCH", json.dumps(mm)[:1500], flush=True)
            done += 1
            if done >= limit:
                break
    print("SUMMARY", json.dumps(summary))


asyncio.run(main())
