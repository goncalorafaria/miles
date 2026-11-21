# Collection capacity and optional diagnostics

Agent episodes, session collection, and training delivery have separate lifetimes.
`miles/rollout/collection_capacity.py` transfers a generation credit after an agent
has finished and a bounded collection slot is available. The sample and its group
remain owned by the rollout task until collection and reward processing finish.
Cancellation and the normal task completion callback return each generation
credit exactly once.

For fully async agentic runs using sample submission:

```text
--async-max-concurrent-samples 1024
--async-max-pending-collections 128
--session-collection-concurrency 32
```

The first value limits generating trajectories; the second bounds finished
sessions admitted outside that limit, including waits for the HTTP transport.
The third bounds concurrent HTTP collections. A full collection backlog retains
generation credits until space opens. New submissions still use complete prompt
groups: with 32 siblings, 32 available credits allow one replacement group.
The finished-group buffer and policy-staleness checks continue to apply.
Setting pending collections to zero retains the previous shared capacity.
Evaluation and generators without a handoff keep their normal completion credits.

`miles/rollout/generate_hub/agentic_tool_call.py` performs the handoff;
`miles/rollout/inference_rollout/inference_rollout_common.py` owns task credits.
`miles/rollout/session/collection.py` retains retry results under the session lock
until acknowledgement or expiry. These are training payloads and are never
subject to the optional diagnostic drop policy.

## Diagnostic dumps

`miles/ray/rollout/diagnostic_writer.py` snapshots only basic fields before sending
them to a writer process: identities, prompt/response, token IDs, masks, label,
reward, status, and metadata. Flat token/mask lists are copied directly. Training
log probabilities, candidate/replay arrays, and training metadata are not copied
into diagnostic snapshots. Live training samples and checkpoint capture are
unchanged.

There is at most one pending diagnostic write. A busy writer skips the next dump
without first copying samples. `rollout/diagnostics/dropped_total`, `skipped`, and
`write_failures_total` expose missing diagnostics; optional write errors are also
logged. Shutdown waits for the accepted write. Snapshot preparation still has a
cost; this is bounded asynchronous writing, not zero-cost or zero-copy logging.

## Optional process assembly

```text
--session-collection-processes
--session-collection-workers 2
```

`miles/rollout/session/collection_process.py` initializes tokenizers and assembly
hooks in spawned workers. Only the locked session state crosses IPC, never the
server or full registry. Workers return the existing packed binary response;
there is no conversion of array payloads to JSON lists. IPC still copies data.
The default remains thread assembly. Benchmark before enabling processes: CPU
isolation can reduce event-loop stalls while reducing total collection throughput.

## Reproduce measurements

```bash
python tests/manual/benchmark_collection_diagnostics.py \
  --dump /path/to/retained/rollout.pt --samples 256 --output /tmp/diagnostics-bench
python tests/manual/benchmark_collection_workers.py \
  --sessions 128 --tokens 32768 --output /tmp/collection-bench
```

The first compares full versus basic snapshots of the same retained batch.
The second repeats finalized synthetic session records through actual assembly
and binary transport, checking recovered tokens and log probabilities. Neither
measures inference, Podman, or end-to-end training throughput.

Measured locally on 2026-11-21, CPU-only:

| Diagnostic mode, 256 retained samples | Submit | Maximum loop stall | Dump size |
| --- | ---: | ---: | ---: |
| Full fields | 6.213 s | 1.871 s | 1,119,129,929 bytes |
| Basic fields, flat-list copy | 1.423 s | 0.251 s | 51,682,569 bytes |

A busy submission took 10 microseconds and skipped its dump. These are single
paired trials; production contention may change timings substantially.

| Assembly mode, 128 concurrent sessions × 32K tokens | Sessions/s | Maximum loop stall |
| --- | ---: | ---: |
| Threads, 2 workers | 39.59 | 47.5 ms |
| Processes, 2 workers | 26.30 | 31.8 ms |

That synthetic workload favors threads for throughput. Process assembly is
available for workloads where isolation outweighs serialization cost; it is not
a claimed universal speedup.
