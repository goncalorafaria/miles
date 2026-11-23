# Session collection and optional diagnostics

When an agent episode ends, `agentic_tool_call.generate` collects its training samples
from the owning session server (`OpenAIEndpointTracer.collect_samples`). Collection is
bounded at both ends and is safe to retry.

## Client side (rollout process)

`miles/rollout/generate_utils/collection_transport.py` owns one pooled aiohttp client per
rollout event loop and admits at most `--session-collection-concurrency` (default 256)
payload transfers at a time; the admission wait does not consume the 120 s network deadline.
Each collection carries a `collection_id`. A timeout or transport error is retried up to
twice with the same ID. The slot is released once the bytes arrive; the reply is then
decoded off the event loop, and only then is the session DELETEd (acknowledged) in the
background.

Only the transfer holds a slot because a finished sample keeps its
`--async-max-concurrent-samples` budget until its collection returns. In the FP8 production
run, 32 slots held through decode and a DELETE with a 10 s timeout collected at about
0.9/s with 5 MB payloads, so about 900 of 1024 budget slots sat with finished samples
waiting for collection while the engines idled.

## Server side (session server)

`miles/rollout/session/collection.py` assembles samples on a bounded worker pool
(`--session-collection-workers`, default 2 per server) under the per-session lock, so a
large merge never blocks the server's chat proxying. The packed result is retained under
its `collection_id` until the DELETE, or `--session-collection-retention` seconds, so a
retried POST returns the identical payload instead of reassembling. A different
`max_seq_len` or agent metadata for the same session before acknowledgement is rejected.
`--session-collection-processes` moves assembly into spawned processes; benchmark before
enabling it (threads had higher throughput on synthetic 32K-token sessions).

MC/top-k candidate arrays of all turns are merged with one allocation per field
(`merge_samples` in `miles/rollout/generate_utils/sample_utils.py`) instead of the former
pairwise concatenation, whose cost grew quadratically with the number of turns.

## Diagnostic dumps

With `--async-rollout-diagnostics`, `miles/ray/rollout/diagnostic_writer.py` snapshots
only basic fields (identities, prompt/response, token IDs, masks, label, reward, status,
metadata) and writes them from a spawned process. Candidate/replay arrays and training
log probabilities are not written. At most one write is pending; a busy writer skips the
next dump and `rollout/diagnostics/dropped_total`, `skipped` and `write_failures_total`
report it. Without the flag, `--save-debug-rollout-data` keeps the former synchronous,
full-field dump.

## Benchmarking the orchestration

`tests/manual/benchmark_collection_workers.py` measures server-side assembly alone. The
end-to-end orchestration simulator (fake engines and sandboxes, real rollout function,
session servers and router) is described in `docs/developer/orchestration-simulator.md`.
