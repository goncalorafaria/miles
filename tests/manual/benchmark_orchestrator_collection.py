"""Overload the real Miles producer and collector without a model or GPU.

A separate HTTP process returns recorded or synthetic packed session responses.
Legacy and process modes use the same producer, groups, payload and HTTP limit.
"""

import argparse
import asyncio
import json
import logging
import multiprocessing
import os
import resource
import statistics
import time
from collections import deque
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import ray
from aiohttp import web

from miles.rollout.base_types import GenerateFnOutput
from miles.rollout.collection_capacity import CollectionCapacity
from miles.rollout.fully_async_rollout import FullyAsyncRolloutFn
from miles.rollout.generate_utils.collection_transport import collection_transport
from miles.rollout.generate_utils.collection_worker import PersistentCollector
from miles.rollout.generate_utils.openai_endpoint_utils import OpenAIEndpointTracer
from miles.rollout.session.samples.codec import COMPUTED_FIELDS, encode_samples
from miles.rollout.submission_scheduler import SampleBackfillSubmission
from miles.utils.types import Sample


def _serve(pipe, tokens, metadata_items, delay, delete_delay, dump, sample_index):
    async def serve():
        sample = Sample(
            tokens=[1] * (tokens + 1),
            response_length=tokens,
            rollout_log_probs=[-0.1] * tokens,
            loss_mask=[1] * tokens,
            response="generated",
            status=Sample.Status.COMPLETED,
        )
        sample.rollout_routed_experts = np.zeros((tokens + 1, 8, 4), dtype=np.int32)
        sample.metadata = {"turns": [{"role": "tool", "content": str(i) * 8} for i in range(metadata_items)]}
        if dump:
            import torch

            raw = torch.load(dump, map_location="cpu", weights_only=False)
            sample = Sample.from_dict(raw["samples"][sample_index])
            del raw
        payload = encode_samples([sample], {})

        async def post(request):
            await request.read()
            await asyncio.sleep(delay)
            return web.Response(body=payload)

        async def delete(request):
            await asyncio.sleep(delete_delay)
            return web.Response(status=204)

        app = web.Application()
        app.router.add_post("/sessions/{sid}/samples", post)
        app.router.add_delete("/sessions/{sid}", delete)
        runner = web.AppRunner(app, access_log=None)
        await runner.setup()
        site = web.TCPSite(runner, "127.0.0.1", 0)
        await site.start()
        pipe.send((site._server.sockets[0].getsockname()[1], len(payload), len(sample.tokens)))
        await asyncio.Event().wait()

    asyncio.run(serve())


class Source:
    def __init__(self, siblings):
        self.siblings = siblings
        self.index = 0

    def get_samples(self, count):
        assert count == 1
        samples = [
            Sample(index=i, group_index=self.index // self.siblings, prompt="task", reward=0.5)
            for i in range(self.index, self.index + self.siblings)
        ]
        self.index += self.siblings
        return [samples]


async def trial(args, mode, inflight, url):
    collector = (
        PersistentCollector(
            capacity=inflight + args.backlog, concurrency=args.concurrency, workers=2 if mode == "process2" else 1
        )
        if mode.startswith("process")
        else None
    )
    if collector:
        await collector.start()
    state = SimpleNamespace(
        args=SimpleNamespace(
            partial_rollout=False,
            mask_offpolicy_in_partial_rollout=False,
            group_rm=False,
            sglang_router_policy="round_robin",
        ),
        aborted=False,
        collector=collector,
        sampling_params={},
        generate_fn_semaphore=asyncio.Semaphore(inflight),
        collection_capacity=CollectionCapacity(args.backlog),
    )
    live = launched = finished = 0
    client_timings = {}

    async def agent(input):
        nonlocal live, launched, finished
        live += 1
        launched += 1
        try:
            await asyncio.sleep(args.agent_seconds)
        finally:
            live -= 1
        finished += 1
        tracer = OpenAIEndpointTracer(url, str(input.sample.index), collection_concurrency=args.concurrency)
        timings = {}
        if collector:
            result = await collector.collect(tracer, input.sample, max_seq_len=None)
        elif mode == "backlog":
            async with state.collection_capacity.admit():
                result = await tracer.collect_samples(input.sample, max_seq_len=None, timings=timings)
        else:
            result = await tracer.collect_samples(input.sample, max_seq_len=None, timings=timings)
        for key, value in timings.items():
            client_timings[key] = client_timings.get(key, 0.0) + value
        if timings:
            client_timings["completed"] = client_timings.get("completed", 0) + 1
        return GenerateFnOutput(samples=result.samples[0])

    state.generate_function = agent
    fn = object.__new__(FullyAsyncRolloutFn)
    fn.args = SimpleNamespace(
        async_max_concurrent_samples=inflight, n_samples_per_prompt=args.siblings, rollout_batch_size=8
    )
    fn.state = state
    fn.data_source = Source(args.siblings)
    fn._scheduler = SampleBackfillSubmission(args.siblings)
    fn._producer_resumed = asyncio.Event()
    fn._producer_resumed.set()
    fn._curr_kv_cache_namespace = None
    fn._retry_buffer = deque()
    fn._running_tasks = []
    fn._output = asyncio.Queue(maxsize=32)
    seen = set()
    observations = []
    started = time.monotonic()

    async def consume():
        while True:
            entry = await fn._output.get()
            for sample in entry.group:
                assert sample.index not in seen, "duplicate attempt delivered"
                assert len(sample.tokens) == args.tokens + 1
                seen.add(sample.index)

    async def monitor():
        previous = time.monotonic()
        while True:
            await asyncio.sleep(0.02)
            now = time.monotonic()
            observations.append(
                dict(
                    seconds=now - started,
                    live=live,
                    held_generation=inflight - state.generate_fn_semaphore._value,
                    held_scheduler=fn._scheduler.samples_in_flight,
                    outstanding=len(fn._running_tasks) * args.siblings,
                    launched=launched,
                    finished=finished,
                    collected=len(seen),
                    loop_stall=max(0, now - previous - 0.02),
                    rss_mb=_rss_mb(),
                )
            )
            previous = now

    async def block_default_pool():
        while True:
            await asyncio.to_thread(time.sleep, args.blocker_seconds)

    blockers = [asyncio.create_task(block_default_pool()) for _ in range(args.pool_blockers)]
    producer = asyncio.create_task(fn._worker_loop())
    consumer = asyncio.create_task(consume())
    monitor_task = asyncio.create_task(monitor())
    try:
        done, _ = await asyncio.wait({producer, consumer}, timeout=args.seconds, return_when=asyncio.FIRST_COMPLETED)
        for task in done:
            task.result()
        elapsed = time.monotonic() - started
        row = dict(
            mode=mode,
            inflight=inflight,
            outstanding_budget=inflight + args.backlog if collector else None,
            seconds=elapsed,
            launched=launched,
            collected=len(seen),
            collected_per_second=len(seen) / elapsed,
            launched_per_second=launched / elapsed,
            max_loop_stall=max(x["loop_stall"] for x in observations),
            mean_live_agents=statistics.mean(x["live"] for x in observations),
            mean_held_generation=statistics.mean(x["held_generation"] for x in observations),
            mean_held_scheduler=statistics.mean(x["held_scheduler"] for x in observations),
            peak_outstanding=max(x["outstanding"] for x in observations),
            peak_coordinator_rss_mb=max(x["rss_mb"] for x in observations),
        )
        row["client_timings"] = client_timings
        if collector:
            row["worker"] = await collector.metrics()
            assert row["worker"]["live_workers"] == args.concurrency
            assert row["peak_outstanding"] <= collector.capacity
        assert all(
            0 <= x["held_generation"] <= inflight and 0 <= x["held_scheduler"] <= inflight for x in observations
        )
        return row, observations
    finally:
        for task in [producer, consumer, monitor_task, *blockers]:
            task.cancel()
        await asyncio.gather(producer, consumer, monitor_task, *blockers, return_exceptions=True)
        for item in fn._running_tasks:
            item.task.cancel()
        await asyncio.gather(*(item.task for item in fn._running_tasks), return_exceptions=True)
        if collector:
            await collector.close()
        else:
            await collection_transport(args.concurrency).close()


def _rss_mb():
    with open("/proc/self/statm") as source:
        return int(source.read().split()[1]) * os.sysconf("SC_PAGE_SIZE") / 1024**2


async def run(args, url, payload_bytes):
    rows = []
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=True)
    (output / "config.json").write_text(json.dumps(vars(args) | {"payload_bytes": payload_bytes}, indent=2))
    for repeat in range(args.repeats):
        for inflight in args.inflight:
            for mode in args.modes:
                row, observations = await trial(args, mode, inflight, url)
                row["repeat"] = repeat
                rows.append(row)
                (output / f"{mode}-{inflight}-r{repeat}.json").write_text(json.dumps(observations))
                (output / "results.json").write_text(json.dumps(rows, indent=2))
                print(json.dumps(row), flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", required=True)
    parser.add_argument("--dump", default=None)
    parser.add_argument("--sample-index", type=int, default=0)
    parser.add_argument("--inflight", type=int, nargs="+", default=[128, 512, 1024])
    parser.add_argument("--modes", nargs="+", default=["legacy", "backlog", "process"])
    parser.add_argument("--seconds", type=float, default=20)
    parser.add_argument("--repeats", type=int, default=1)
    parser.add_argument("--siblings", type=int, default=32)
    parser.add_argument("--backlog", type=int, default=128)
    parser.add_argument("--concurrency", type=int, default=32)
    parser.add_argument("--pool-blockers", type=int, default=0)
    parser.add_argument("--blocker-seconds", type=float, default=0.25)
    parser.add_argument("--tokens", type=int, default=8192)
    parser.add_argument("--metadata-items", type=int, default=4096)
    parser.add_argument("--agent-seconds", type=float, default=0.1)
    parser.add_argument("--collection-seconds", type=float, default=0.2)
    parser.add_argument("--delete-seconds", type=float, default=0.01)
    args = parser.parse_args()
    logging.basicConfig(level=logging.WARNING)
    context = multiprocessing.get_context("spawn")
    receive, send = context.Pipe(duplex=False)
    server = context.Process(
        target=_serve,
        args=(
            send,
            args.tokens,
            args.metadata_items,
            args.collection_seconds,
            args.delete_seconds,
            args.dump,
            args.sample_index,
        ),
    )
    server.start()
    try:
        port, payload_bytes, token_count = receive.recv()
        args.tokens = token_count - 1
        ray.init(
            address="local",
            num_cpus=2,
            include_dashboard=False,
            object_store_memory=512 * 1024 * 1024,
            logging_level="ERROR",
        )
        asyncio.run(run(args, f"http://127.0.0.1:{port}", payload_bytes))
    finally:
        ray.shutdown()
        server.terminate()
        server.join()
