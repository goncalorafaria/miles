"""Trace-replaying TMax agent for the orchestration simulator.

Same contract and process model as ``examples/tmax/tmax_agent_function.run``: runs in its own
child process (``--custom-agent-function-mode subproc``), starts a "sandbox" under the same
rollout-scoped start semaphore, then alternates OpenAI chat calls through the session server
with "bash" tool executions. Durations come from a real episode profile of the production run
(scaled by SIM_COMPRESS); the requests are the canonical TITO-valid script from specs.py.
"""

import asyncio
import hashlib
import os
import pickle
import random
import time

from openai import AsyncOpenAI

from miles.rollout.agentic.rollout_limits import rollout_semaphore

_SPECS = None


def _specs():
    global _SPECS
    if _SPECS is None:
        with open(os.environ["SIM_SPECS"], "rb") as f:
            _SPECS = [s["messages"] for s in pickle.load(f)]
    return _SPECS


async def run(base_url, prompt, request_kwargs, metadata, **kwargs):
    compress = float(os.environ.get("SIM_COMPRESS", "1"))
    start_limit = int(os.environ.get("SIM_START_CONCURRENCY", "64"))
    profiles = metadata["profiles"]
    # Siblings share metadata; pick a profile from the real group by session identity.
    key = hashlib.md5(base_url.encode()).digest()
    profile = profiles[int.from_bytes(key[:4], "little") % len(profiles)]
    rng = random.Random(key)
    specs = _specs()
    turns = max(1, min(profile["turns"], len(specs)))
    tool_s = profile["env"] / max(1, profile["tools"]) if profile["tools"] else 0.0
    t0 = time.monotonic()
    async with rollout_semaphore("sim-sandbox-start", start_limit):
        await asyncio.sleep(profile["start"] / compress)
    started = time.monotonic()
    policy = AsyncOpenAI(base_url=f"{base_url}/v1", api_key="EMPTY", timeout=3600)  # production default max_retries=2
    query_s = 0.0
    query_max = 0.0
    messages = None
    for turn in range(turns):
        messages = specs[turn]
        q0 = time.monotonic()
        completion = await policy.chat.completions.create(
            model="model",
            messages=messages,
            tools=[
                {
                    "type": "function",
                    "function": {
                        "name": "bash",
                        "parameters": {
                            "type": "object",
                            "properties": {"command": {"type": "string"}},
                            "required": ["command"],
                        },
                        "description": "Run one bash command.",
                    },
                }
            ],
            extra_body=request_kwargs,
        )
        q = time.monotonic() - q0
        query_s += q
        query_max = max(query_max, q)
        assert completion.choices[0].message.tool_calls, "sim reply lost its tool call"
        if turn + 1 < turns:
            await asyncio.sleep(tool_s * rng.lognormvariate(0, 0.5) / compress)
    await policy.close()
    return {
        "agent_metrics": {
            "sim_turns": turns,
            "sim_query_s": query_s,
            "sim_query_max_s": query_max,
            "sim_start_s": started - t0,
            "sim_total_s": time.monotonic() - t0,
            "sim_end_wall": time.time(),
            "total_tool_time": 0.0,
        }
    }


async def reward(args, sample, **kwargs):
    samples = sample if isinstance(sample, list) else [sample]
    now = time.time()
    for s in samples:
        end = ((s.metadata or {}).get("agent_metrics") or {}).get("sim_end_wall")
        if end is not None:
            s.metadata["sim_post_agent_s"] = now - end
    rewards = [float(random.random() < 0.5) for _ in samples]
    return rewards if isinstance(sample, list) else rewards[0]
