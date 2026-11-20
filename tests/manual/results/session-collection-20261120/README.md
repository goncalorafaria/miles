# Session collection investigation

**Production root cause is not yet established.** Acceptance requires reproducing
collection failure with the original code and observing the same workload succeed
with the change. A throughput improvement alone is insufficient.

Base: `cea1943fe8e6bfefe182fc4fef283518434c0340`.
Working branch: `codex/session-collection-throughput`.

## Replay method

`tests/manual/benchmark_recorded_collection.py` loads the 256 saved samples from
BF16 rollout `153.pt` (update 154). Generated spans come from the recorded loss
masks; candidate arrays are encoded as the per-turn base64 records expected by
Miles. Both variants decode these arrays, assemble turns, pack, transfer over
localhost HTTP and decode. Before controlled overload timing, both assemblers are checked against all 256
recorded samples for exact tokens, loss masks, rollout log probabilities and MC32
IDs/log probabilities. Timed requests retain only constant-cost shape checks.

The old client class and HTTP helpers are loaded directly using `git show` at the
base revision. The new client is imported from the working tree. Both retain the
120-second collection timeout. Old HTTP connection capacity is 4096, matching the
saved production arguments (512 * 8 / TP1). The new collection admission limit is
32 and server assembly concurrency is 2 per worker. File-descriptor limit is
65536 for both. Each case has fresh client/server processes, with lazy imports and
first-connection setup warmed before timing. Thread decoding is used by BOTH
actual client implementations.

Saved production configuration confirms 32 session workers. The 32-worker test
uses seeded random routing and 512 concurrent collection attempts. Single-worker
bursts are separately labeled overload tests and cannot establish production
causality. Even 512 simultaneous completions is more concentrated than 512 active
episodes spread across inference and tool calls.

This reconstructs candidate decoding and turn merging, not the original v2
session tree, tokenizer text decoding, agent process scheduling, concurrent chat
traffic or Ray data transfer. Recorded model responses are fixed. The shared host
is not an otherwise idle benchmark machine. Peak RSS includes inherited corpus
pages and must not be summed across workers as physical memory usage.

## Real Podman replay

`podman-32.jsonl`: 32/32 episodes completed, 457 real tool calls, no infrastructure
failures. Each episode received a fresh container and used recorded bash actions,
including real verifiers. No inference was run. This tests the sandbox path
separately; it is NOT an old/new collection comparison or policy evaluation.
Model delay was set to zero. The first 32 eligible traces are not a representative
random task sample. Startup includes possible image-cache effects and retries.

## Invalid comparisons

Files in `invalid-baseline/` are retained for audit only. Their old-client harness
had synchronous decoding and different cleanup behavior. Those speedups and
failure counts have been withdrawn. `synthetic-components.json` is only an
independent packing/connection microbenchmark, not an actual old/new client or
production failure reproduction.

Corrected client results and failure tracebacks are saved alongside this file.
Training remains stopped; these local experiments do not restart training.

## Corrected 32-worker comparison (one repeat)

Both variants completed all 512 requests with zero failures. This does **not** reproduce the production failure.

| Version | Completed | Wall seconds | Requests/s | p95 seconds | Max client loop stall, seconds |
|---|---:|---:|---:|---:|---:|
| old | 512 | 15.48 | 33.07 | 14.27 | 1.332 |
| new | 512 | 9.46 | 54.14 | 9.33 | 0.138 |

## Controlled failure reproduction: 2048 collections, one worker

Source: `controlled-overload2048.json`. One repeat, actual old/new clients,
original 120-second collection deadline, no injected network faults. Full-array
correctness validation occurs before load timing. All cases replay the same
256 recorded payloads eight times.

| Variant | Completed / attempted | Timeouts | Total wall seconds |
|---|---:|---:|---:|
| Old | 1734 / 2048 | 314 | 178.98 |
| All changes | 2048 / 2048 | 0 | 81.50 |
| Client changes only | 2048 / 2048 | 0 | 86.22 |
| Server changes only | 2048 / 2048 | 0 | 102.21 |

Client changes comprise bounded collection admission, connection reuse, retries
and acknowledgment after decoding. Server changes comprise bounded offloaded
assembly and cached collection results; the server-only case retains the old
array merge implementation. These are bundle ablations, not tests of each
individual change. Admission waiting is outside the new client's network timeout;
total wall time includes that wait and old timeout cleanup.

The old server's maximum event-loop stall was 89.15 seconds, versus 0.091 seconds
with all changes. This supports overload-induced starvation in this fixture.
The saved first controlled failure times out in `_post_buffer` during aiohttp
connection cleanup. The earlier inline-validation overload captured a timeout
while reading `iter_any()`, as seen in production. These are collection timeout
failures, not process crashes; identical production causality is not established.

Both isolated change bundles prevented failure in this run. Repeated trials and
production-like session/tokenizer, scheduling and concurrent traffic are still
needed before attributing the training aborts to this component. The production-
sized test above is explicitly a negative reproduction result.

The 32-worker comparison and the archived
`actual-clients-overload2048-inline-validation.json` performed full-array checks
inside concurrent requests; their timings include that extra CPU work. The latter
is not the headline controlled benchmark.

Reproduce from the repository root in the benchmark environment:

```bash
PYTHONPATH=/tmp/miles-collection-http-deps:.:/weka/gfaria/tmax-4b-miles/sglang/python \
/weka/gfaria/miles-olmo-core/.venv/bin/python \
  tests/manual/benchmark_recorded_collection.py \
  --dump /weka/gfaria/tmax-score-logprobs/traces/tmax-4b-prod-bf16-mc32-callback-20261115-8h/rollout_data/153.pt \
  --requests 2048 --repeats 1 --workers 1 --assemble-turns \
  --variants old new client_only server_only \
  --output /tmp/miles-controlled-overload2048.json
```

## Repeated 32-worker load sweep

All 12 cases completed: two repeats per version and load, reversing version order
on the second repeat. Repeated the same 256 saved payloads; validation outside
timing and unchanged 120-second network deadline. Source: `controlled-workers32-load-sweep.json`.

| Concurrent attempts | Old mean wall seconds | New mean wall seconds | Old failures / attempts across repeats | New failures / attempts across repeats |
|---|---:|---:|---:|---:|
| 1024 | 31.05 | 10.21 | 0 / 2048 | 0 / 2048 |
| 2048 | 68.14 | 21.08 | 0 / 4096 | 0 / 4096 |
| 4096 | 183.48 | 40.50 | 2097 / 8192 | 0 / 8192 |

At 4096 attempts the old version failed in BOTH repeats (1068 and 1029 failures),
while the new version completed all attempts. Wall time includes admission and
cleanup; old failed batches are not equal completed-work throughput comparisons.
This reproduces an overload failure with the production worker count, but at eight
times the production episode cap. It does not establish the cause of aborts at
512 active episodes. This remains a collection replay, not full training.
