# Orchestration simulator

`tests/manual/orchestration_sim/` runs the fully async agentic rollout end to end on CPU,
with no GPU, model weights or trainer, so orchestration changes can be benchmarked
against the production shape before a training run.

## What is real and what is simulated

Real, from whichever Miles tree is first on `PYTHONPATH`: `FullyAsyncRolloutFn`, the
submission scheduler and data buffer, `agentic_tool_call.generate`, subprocess agent
calls (`--custom-agent-function-mode subproc`), 32 v2 session servers (TITO, collection), the miles router, and session collection back into the rollout
process.

Simulated:

- `engine.py`: one process per engine serving pre-rendered SGLang chat completions
  (bash tool call, per-token logprobs) after
  `prefill_s + tokens * token_s`. Latency does not depend on load unless
  `--max-running` is set, so any extra latency the agent sees comes from the path
  between the agent and the engine.
- `sim_agent.py`: same contract and process model as the TMax agent function. It holds
  the rollout-scoped sandbox-start semaphore for the recorded start time, then
  alternates OpenAI chat calls with the recorded tool time. Turn counts, start and
  tool durations come from real episode groups (`make_profiles.py`).
- The trainer drains each batch immediately (`weight_version = step`).

## Running

```bash
SIM=tests/manual/orchestration_sim
export PYTHONPATH=$PWD:/path/to/sglang/python:$SIM
python $SIM/specs.py /path/to/Qwen3.5-4B-tmax-template qwen35 /tmp/specs.pkl
python $SIM/make_profiles.py --out /tmp/episodes.json /path/to/traces-*.jsonl
python $SIM/run_sim.py --label prod-1024 --out /tmp/prod-1024.json \
  --hf-checkpoint /path/to/Qwen3.5-4B-tmax-template --specs /tmp/specs.pkl \
  --profiles /tmp/episodes.json --concurrency 1024 --prefill-s 6 --token-s 0.02 \
  --duration 1500 --warmup 600 --extra "<extra miles args>"
```

`--agent-mode pool` runs the agents on the worker pool ([agent-pool.md](agent-pool.md)) instead of
one process per episode, and the CPU summary's `agents` role counts agent processes either way.
`specs.py` takes the median generated tokens per turn as a fourth argument; raise it to exercise larger per-token log-probability payloads.

`--extra` passes Miles flags, so the same command compares code versions (put another
tree first on `PYTHONPATH`) and flag sets. The summary reports time-averaged engine
running requests, engine calls/s, agent-observed versus engine time per turn
(`chat_path_overhead_per_turn_s`), post-agent (collection) latency, samples in flight,
rollout-loop round-trip time, CPU per process role and host load.

Run one system at a time and check `host_load1_mean`: about 1,000 agent processes per
system make concurrent runs contend for CPU and invalidate the comparison. With a
live system, `probe.py` times one turn directly at an engine, at the router and at a
session server to locate added latency; `py-spy record -p <pid>` then shows where it goes.
