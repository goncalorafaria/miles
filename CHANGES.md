# Agentic rollout orchestration

## What changes for a launch

Miles now runs episodes from `agentic_tool_call.generate` on persistent worker processes by default. Existing launch commands can omit `--custom-agent-function-mode pool`. To reproduce the previous process-per-episode behavior, pass `--custom-agent-function-mode subproc`.

Multiple episodes can share one worker concurrently. Keep mutable episode state and client cleanup scoped to the episode. A blocking call inside an async agent can delay other episodes assigned to that worker. See the [agent pool guide](docs/developer/agent-pool.md) for worker sizing and lifecycle.

## What changes during a rollout

- The coordinator assigns an episode to the worker with the fewest pending requests. The worker hosts the session, assembles training samples and returns them over ZeroMQ. The original prompt group still receives its results for reward processing.
- On the standalone collection path, payload transfer uses bounded concurrency. Decoding happens after the transfer slot is released. Successful collections are acknowledged in the background.
- The router forwards response bytes directly. TMax moves blocking trace writes and Docker calls off the shared event loop and closes per-episode clients.

The [CPU orchestration simulator](docs/developer/orchestration-simulator.md) exercises both execution modes with the same episode profiles. Pass `--agent-mode subproc` for the previous baseline and omit the override to measure the current default.

## Before and after at 1,024 concurrent episodes

These CPU simulator runs used eight simulated inference engines and 256 samples per batch. The before column uses a subprocess per episode. The after column uses 32 persistent workers. Ranges give both subprocess stress runs.

| Profile and metric | Before | After |
| --- | ---: | ---: |
| Production-paced replay, engine calls/s | 20.70 | 20.52 |
| Production-paced replay, mean active engine requests | 960 | 965 |
| Accelerated stress, engine calls/s | 179–185 | 171 |
| Accelerated stress, collected samples/s | 3.78–3.96 | 3.71 |
| Accelerated stress, p95 time from agent exit to collected sample | 1.07–8.69 s | 0.61 s |
| Accelerated stress, p99 rollout event-loop round trip | 164–624 ms | 23 ms |

The worker pool reduced collection and scheduling delays under stress. Engine throughput was within 1% of the baseline on the production-paced replay and 5–8% lower under accelerated load. The production-paced runs each delivered one batch, which is insufficient to estimate steady-state training step time. The saved comparisons cover 1,024 inflight episodes. A matched 512-episode result is still missing.

## Engine occupancy during rollout

Enable `--use-miles-dashboard --dump-details /path/to/run` to log engine time series.
For external vLLM servers, add `--dashboard-engine-addrs http://engine-a:8000 http://engine-b:8000`.
Supply each API server's base URL. Explicit addresses take precedence over router discovery.
The dashboard polls each server's Prometheus `/metrics` endpoint every two seconds by default
and persists the samples under the run's `dashboard/` directory.

Select `vllm_num_requests_running`, `vllm_num_requests_waiting`, or `vllm_kv_cache_usage_perc`
in the timeline. The KV value is a fraction (0.75 means 75%). Within each endpoint, the
engine series sums request counts across data-parallel ranks and averages their KV utilization.

### Miles-managed SGLang engines

With `--use-miles-router --use-wandb --sglang-enable-metrics`, the rollout
executor's W&B system monitor polls the Miles router's `/engine_metrics`.
The router reads each healthy registered engine's `/metrics` concurrently
and exports `sglang_num_running_reqs`, `sglang_num_queue_reqs`, and
`sglang_token_usage` with the worker address and original labels.
Token usage is the fraction of KV token capacity in use. An unavailable
engine contributes a gap. Engine metrics can also be viewed under
**Metrics → engine metrics** when the Miles dashboard is enabled.
