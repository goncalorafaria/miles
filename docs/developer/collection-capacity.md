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

In this legacy backlog mode, the handoff returns scheduler credit only; the
generation semaphore still wraps the full hook. The first value limits scheduler
credits; the second bounds finished
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

## Persistent collection processes

The optional process collector splits at agent completion, before HTTP fetch and
decode. Enable it for fully async, sample-granularity, session-server runs:

```text
--async-max-concurrent-samples 1024
--async-collection-worker
--async-collection-workers 2
--async-max-outstanding-attempts 1152
--session-collection-concurrency 32
```

`miles/rollout/generate_utils/collection_worker.py` owns persistent Ray actors,
each with its own HTTP transport, event loop, decode executor, and publication
executor. Total HTTP concurrency stays 32 (16 per actor here), not 32 per actor.
Each actor reserves one Ray CPU. This differs from server-side process assembly
(`--session-collection-processes`), which need not be enabled.

The coordinator reserves outstanding capacity before submitting a whole prompt
group. On descriptor acceptance, `agent_finished()` in
`miles/rollout/collection_capacity.py` releases both the generation semaphore
permit and scheduler credit exactly once. Group tasks still await collected
samples and reward processing. Whole groups, including already finished siblings,
retain their outstanding reservation until group delivery. The separate finished
output buffer and staleness checks remain in force. A slow collector therefore
applies explicit bounded backpressure; it cannot promise steady inference occupancy.

Workers publish results into Ray's object store and return a small reference.
The coordinator materializes the result on a dedicated receiver thread before
sending DELETE to the owning session server. NumPy buffers use object-store
transport; Python lists and metadata still require serialization. This is not
zero-copy transport for every field. DELETE remains part of outstanding work,
but no longer holds generation capacity.

Worker failure retries the same collection ID while the server retains its
response. Cancellation does not DELETE an unaccepted result. Tests kill the
owning actor, check retries and cancellation, verify v2 multiple-sample results,
and check that acknowledgement follows receiver ownership. Both one- and
two-process variants pass. A saturation test also caught and fixed an Event wakeup
bug that could retire idle workers when several woke for one queued job.

The tracer reports thread-queue wait, decode CPU/wall time, event-loop resume delay,
transport wait, and DELETE separately. Collector metrics include object-store
publication, outstanding jobs, queued jobs, and live workers.

### Aggressive overload benchmark

`tests/manual/benchmark_orchestrator_collection.py` uses the actual fully async
Miles producer, scheduler, generation permits, HTTP collection, wire decoder,
and group consumer. Generation itself is simulated with a delay; no GPU,
Podman, model, or trainer participates. Duplicate/missing delivered sample IDs,
worker survival, and outstanding limits are checked.

```bash
python tests/manual/benchmark_orchestrator_collection.py \
  --inflight 512 --modes legacy process2 --seconds 20 --repeats 3 \
  --pool-blockers 128 --blocker-seconds 1 --output /tmp/collector-overload
python tests/manual/benchmark_orchestrator_collection.py \
  --inflight 512 --modes legacy process process2 --seconds 30 \
  --dump /path/to/retained/rollout.pt --sample-index 64 \
  --output /tmp/collector-recorded
```

CPU measurements on 2026-11-21; total HTTP concurrency 32 in every mode:

| Workload | Legacy samples/s | Two processes samples/s |
| --- | ---: | ---: |
| Deliberately saturated default thread pool, mean of 3 × 20 s | 3.20 | 118.93 |
| Saved training sample replay, one 30 s trial | 132.23 | 132.27 |

Under artificial contention, process results ranged 116.80–121.60 samples/s
(~37.2× mean throughput). The legacy path held about 512 generation permits
although only ~3 agents were actually executing on average. The process path
held ~24 permits and respected the 640 outstanding-attempt limit.
This demonstrates isolation from shared executor starvation, not proof that
production currently experiences the same contention or will speed up 37×.

Without artificial contention, two processes matched the replay baseline, while
one process reached only 113.06 samples/s. Maximum coordinator loop stall in that
single replay was 232 ms legacy versus 536 ms with two processes: multiprocessing
has not eliminated all coordinator stalls. Serialization, group handling, and
bounded downstream capacity still matter. Keep the new mode opt-in until a live
paired comparison measures collected samples/s and training batch acquisition.
