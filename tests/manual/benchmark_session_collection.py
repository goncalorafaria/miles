"""CPU-only isolated collection benchmarks; no model or training job is launched.

Run with PYTHONPATH=. python tests/manual/benchmark_session_collection.py --output /tmp/collection.json
"""

import argparse
import asyncio
import json
import statistics
import time
from types import SimpleNamespace

import numpy as np
from aiohttp import web
from starlette.responses import Response

from miles.rollout.generate_utils.collection_transport import CollectionTransport
from miles.rollout.generate_utils.sample_utils import _merge_sample_pair, merge_samples
from miles.rollout.session.collection import SessionCollections
from miles.rollout.session.samples.codec import encode_samples
from miles.utils.http_utils import post_buffer_no_retry
from miles.utils.types import Sample


def merge_benchmark():
    results = []
    tokenizer = SimpleNamespace(decode=lambda tokens: "")
    for k in (32, 128):
        turns, tokens = [], [1]
        for i in range(32):
            tokens = tokens + ([2] * 8 if i else []) + [3] * 1024
            turns.append(
                Sample(
                    tokens=tokens,
                    response_length=1024,
                    status=Sample.Status.COMPLETED,
                    rollout_topk_token_ids=np.full((1024, k), i, dtype=np.int32),
                    rollout_topk_log_probs=np.full((1024, k), -3.0, dtype=np.float32),
                )
            )

        def old(turns=turns):
            result = turns[0]
            for turn in turns[1:]:
                result = _merge_sample_pair(result, turn, tokenizer)
            return result

        baseline = old()
        updated = merge_samples(turns, tokenizer)
        np.testing.assert_array_equal(baseline.rollout_topk_token_ids, updated.rollout_topk_token_ids)
        np.testing.assert_array_equal(baseline.rollout_topk_log_probs, updated.rollout_topk_log_probs)
        row = {"k": k, "turns": 32, "generated_tokens": 32768}
        for name, fn in (
            ("pairwise", old),
            ("single_allocation", lambda turns=turns: merge_samples(turns, tokenizer)),
        ):
            times = []
            for _ in range(7):
                t = time.perf_counter()
                fn()
                times.append(time.perf_counter() - t)
            row[name + "_median_s"] = statistics.median(times)
        results.append(row)
    return results


async def http_benchmark(*, offload, pooled, requests, tokens):
    sample = Sample(
        tokens=[1] + [2] * tokens,
        response_length=tokens,
        status=Sample.Status.COMPLETED,
        rollout_topk_token_ids=np.zeros((tokens, 32), dtype=np.int32),
        rollout_topk_log_probs=np.full((tokens, 32), -3.0, dtype=np.float32),
    )
    peers = set()
    state = SimpleNamespace(lock=asyncio.Lock())

    async def expire(_):
        pass

    owner = SessionCollections(concurrency=2, retention_seconds=60, expire=expire)

    def pack():
        return Response(encode_samples([sample], {}))

    async def handler(request):
        peers.add(request.transport.get_extra_info("peername"))
        if offload:
            result = await owner.collect("bench", state, key=None, assemble=pack)
        else:
            result = pack()
        return web.Response(body=result.body)

    app = web.Application()
    app.router.add_post("/samples", handler)
    runner = web.AppRunner(app)
    await runner.setup()
    site = web.TCPSite(runner, "127.0.0.1", 0)
    await site.start()
    port = site._server.sockets[0].getsockname()[1]
    transport = CollectionTransport(16)
    slots = asyncio.Semaphore(16)
    lag, latencies = [], []
    done = False

    async def heartbeat():
        while not done:
            t = time.perf_counter()
            await asyncio.sleep(0.005)
            lag.append(max(0.0, time.perf_counter() - t - 0.005))

    async def collect():
        async with slots:
            t = time.perf_counter()
            if pooled:
                async with transport.admit() as client:
                    data = await post_buffer_no_retry(
                        f"http://127.0.0.1:{port}/samples", {}, timeout=120, client=client
                    )
            else:
                data = await post_buffer_no_retry(f"http://127.0.0.1:{port}/samples", {}, timeout=120)
            latencies.append(time.perf_counter() - t)
            return len(data)

    ticker = asyncio.create_task(heartbeat())
    started = time.perf_counter()
    try:
        sizes = await asyncio.gather(*(collect() for _ in range(requests)))
        elapsed = time.perf_counter() - started
    finally:
        done = True
        await ticker
        await transport.close()
        await runner.cleanup()
    return dict(
        offload=offload,
        pooled=pooled,
        requests=requests,
        tokens=tokens,
        payload_bytes=sizes[0],
        elapsed_s=elapsed,
        requests_per_s=requests / elapsed,
        connections=len(peers),
        request_p50_s=float(np.percentile(latencies, 50)),
        request_p95_s=float(np.percentile(latencies, 95)),
        loop_lag_p95_s=float(np.percentile(lag, 95)),
        loop_lag_max_s=max(lag),
    )


async def main(args):
    results = {
        "scope": "CPU synthetic collection; same-process localhost HTTP, no inference or tools",
        "array_merge": merge_benchmark(),
        "http": [],
    }
    # Change one variable at a time; retain both small and full trajectory payloads.
    for tokens in (256, 32768):
        for offload, pooled in ((False, False), (False, True), (True, True)):
            row = await http_benchmark(offload=offload, pooled=pooled, requests=args.requests, tokens=tokens)
            results["http"].append(row)
            print(json.dumps(row), flush=True)
    with open(args.output, "w") as f:
        json.dump(results, f, indent=2)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--requests", type=int, default=128)
    parser.add_argument("--output", required=True)
    asyncio.run(main(parser.parse_args()))
