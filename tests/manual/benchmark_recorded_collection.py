"""Replay real rollout payloads over localhost HTTP in separate server/client processes.

Measures packing/transfer/decode, not reconstruction of original session trees,
model inference, or Podman. Baseline uses the base commit's HTTP implementation.
"""

import argparse
import asyncio
import base64
import importlib.util
import json
import multiprocessing as mp
import random
import resource
import subprocess
import tempfile
import time
import traceback
from types import SimpleNamespace

import aiohttp
import numpy as np
import torch
from aiohttp import web
from starlette.responses import Response

from miles.rollout.generate_utils.collection_transport import collection_transport
from miles.rollout.generate_utils.openai_endpoint_utils import OpenAIEndpointTracer
from miles.rollout.generate_utils.rollout_topk_logprobs import append_rollout_topk_logprobs
from miles.rollout.generate_utils.sample_utils import _merge_sample_pair, merge_samples
from miles.rollout.session.collection import SessionCollections
from miles.rollout.session.samples.codec import encode_samples
from miles.utils.types import Sample


def recorded_turns(sample):
    """Recover generated spans; tool observations remain in each next prompt.

    Text decoding and original session tree traversal are not reconstructed.
    MC arrays are encoded once outside timing, matching persisted server records.
    """
    mask = sample.loss_mask
    if not mask or not mask[0] or not mask[-1]:
        raise ValueError("Replay needs generated spans at both response boundaries")
    offset = len(sample.tokens) - sample.response_length
    records = []
    start = 0
    while start < len(mask):
        end = start + 1
        while end < len(mask) and mask[end] == mask[start]:
            end += 1
        if mask[start]:
            ids = sample.rollout_topk_token_ids[start:end]
            logps = sample.rollout_topk_log_probs[start:end]
            meta = dict(
                output_score_mode=f"mc{ids.shape[1]}",
                output_top_logprobs_shape=list(ids.shape),
                output_top_logprobs_null_prefix=0,
                output_token_logprobs=[None] * (end - start),
            )
            for key, array, dtype in (("idx", ids, "int32"), ("val", logps, "float32")):
                field = f"output_top_logprobs_{key}_flat_b64"
                meta[field] = base64.b64encode(array.tobytes()).decode()
                meta[field + "_dtype"] = dtype
            records.append((sample.tokens[: offset + end], sample.rollout_log_probs[start:end], meta))
        start = end
    return records


def assemble_turns(records, variant):
    turns = []
    for tokens, logps, meta in records:
        turn = Sample(
            tokens=tokens,
            response_length=len(logps),
            rollout_log_probs=logps,
            loss_mask=[1] * len(logps),
            status=Sample.Status.COMPLETED,
        )
        append_rollout_topk_logprobs(
            turn,
            meta,
            meta["output_top_logprobs_shape"][1],
            estimator="mc",
            array_output=True,
            copy_arrays=variant not in ("new", "merge_only"),
        )
        turns.append(turn)
    tokenizer = SimpleNamespace(decode=lambda tokens: "")
    if variant not in ("new", "merge_only"):
        result = turns[0]
        for turn in turns[1:]:
            result = _merge_sample_pair(result, turn, tokenizer)
        return result
    return merge_samples(turns, tokenizer)


def stats(values):
    return {f"p{q}": float(np.percentile(values, q)) for q in (50, 95, 99)} if values else {}


async def serve(samples, variant, pipe, records):
    lag, pack_times, peers = [], [], set()
    states = {}

    async def expire(sid):
        owner.acknowledge(sid)
        states.pop(sid, None)

    owner = SessionCollections(concurrency=2, retention_seconds=1800, expire=expire)

    async def tick():
        while True:
            start = time.perf_counter()
            await asyncio.sleep(0.005)
            lag.append(max(0, time.perf_counter() - start - 0.005))

    ticker = asyncio.create_task(tick())

    async def collect(request):
        sid = request.match_info["sid"]
        index = int(sid) % len(samples)
        peers.add(request.transport.get_extra_info("peername"))

        def pack():
            t = time.perf_counter()
            sample = assemble_turns(records[index], variant) if records else samples[index]
            data = encode_samples([sample], {})
            pack_times.append(time.perf_counter() - t)
            return Response(data)

        if variant not in ("new", "server_only"):
            reply = pack()
        else:
            state = states.setdefault(sid, SimpleNamespace(lock=asyncio.Lock()))
            reply = await owner.collect(sid, state, key=sid, assemble=pack)
        return web.Response(body=reply.body)

    async def delete(request):
        await expire(request.match_info["sid"])
        return web.Response(status=204)

    async def metrics(request):
        return web.json_response(
            dict(
                server_loop_lag_s=stats(lag),
                server_loop_lag_max_s=max(lag, default=0),
                pack_s=stats(pack_times),
                connections=len(peers),
                packed=len(pack_times),
                server_peak_rss_mib=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024,
            )
        )

    async def reset(request):
        lag.clear()
        pack_times.clear()
        peers.clear()
        return web.Response(status=204)

    app = web.Application()
    app.router.add_post("/sessions/{sid}/samples", collect)
    app.router.add_delete("/sessions/{sid}", delete)
    app.router.add_get("/metrics", metrics)
    app.router.add_post("/metrics/reset", reset)
    runner = web.AppRunner(app)
    await runner.setup()
    site = web.TCPSite(runner, "127.0.0.1", 0, backlog=2048)
    await site.start()
    pipe.send(site._server.sockets[0].getsockname()[1])
    try:
        while not pipe.poll():
            await asyncio.sleep(0.05)
    finally:
        ticker.cancel()
        await runner.cleanup()


def server(samples, variant, pipe, records):
    asyncio.run(serve(samples, variant, pipe, records))


async def clients(ports, count, variant, samples, old_tracer):
    route = random.Random(0)
    assigned_ports = [route.choice(ports) for _ in range(count)]
    pool = collection_transport(32)
    latency = []
    errors = []
    lag = []
    sizes = []

    async def tick():
        while True:
            t = time.perf_counter()
            await asyncio.sleep(0.005)
            lag.append(max(0, time.perf_counter() - t - 0.005))

    # Exclude lazy imports, connector construction and first thread startup.
    tracer_class = OpenAIEndpointTracer if variant in ("new", "client_only") else old_tracer
    await tracer_class(f"http://127.0.0.1:{ports[0]}", str(count + 65536)).collect_samples(Sample(), max_seq_len=None)
    async with aiohttp.ClientSession() as client:
        for port in ports:
            async with client.post(f"http://127.0.0.1:{port}/metrics/reset") as reply:
                reply.raise_for_status()
    ticker = asyncio.create_task(tick())

    async def one(i):
        t = time.perf_counter()
        tracer_class = OpenAIEndpointTracer if variant in ("new", "client_only") else old_tracer
        tracer = tracer_class(f"http://127.0.0.1:{assigned_ports[i]}", str(i))

        phase = "collect_samples"
        try:
            reply = await tracer.collect_samples(Sample(), max_seq_len=None)
            elapsed = time.perf_counter() - t
            phase = "validation"
            sample = reply.samples[0]
            expected = samples[i % len(samples)]

            # Corpus equality is checked before timing, not as extra CPU load
            # competing with the old client's pending network reads.
            assert len(sample.tokens) == len(expected.tokens)
            assert sample.rollout_topk_token_ids.shape == expected.rollout_topk_token_ids.shape
            sizes.append(sample.rollout_topk_token_ids.nbytes + sample.rollout_topk_log_probs.nbytes)
            latency.append(elapsed)
        except Exception as error:
            errors.append(
                dict(
                    type=type(error).__name__,
                    message=str(error),
                    phase=phase,
                    elapsed_s=time.perf_counter() - t,
                    traceback=traceback.format_exc(limit=8) if not errors else None,
                )
            )

    t = time.perf_counter()
    try:
        await asyncio.gather(*(one(i) for i in range(count)))
        elapsed = time.perf_counter() - t
        worker_metrics = []
        async with aiohttp.ClientSession() as c:
            for port in ports:
                async with c.get(f"http://127.0.0.1:{port}/metrics") as r:
                    worker_metrics.append(await r.json())
        server_metrics = dict(
            workers=len(ports),
            connections=sum(m["connections"] for m in worker_metrics),
            packed=sum(m["packed"] for m in worker_metrics),
            server_loop_lag_max_s=max(m["server_loop_lag_max_s"] for m in worker_metrics),
            server_peak_rss_mib=max(m["server_peak_rss_mib"] for m in worker_metrics),
            worker_metrics=worker_metrics,
        )
    finally:
        ticker.cancel()
        await pool.close()
    return dict(
        variant=variant,
        requests=count,
        completed=len(latency),
        errors=errors,
        elapsed_s=elapsed,
        requests_per_s=len(latency) / elapsed,
        end_to_end_s=stats(latency),
        client_loop_lag_s=stats(lag),
        client_loop_lag_max_s=max(lag, default=0),
        candidate_array_bytes=stats(sizes),
        **server_metrics,
    )


def run_case(samples, variant, count, old_tracer, result_pipe, records, workers):
    ctx = mp.get_context("fork")
    processes, pipes, ports = [], [], []
    try:
        for _ in range(workers):
            parent, child = ctx.Pipe()
            process = ctx.Process(target=server, args=(samples, variant, child, records))
            process.start()
            processes.append(process)
            pipes.append(parent)
            if not parent.poll(60):
                raise TimeoutError("server startup")
            ports.append(parent.recv())
        row = asyncio.run(clients(ports, count, variant, samples, old_tracer))
        row["client_peak_rss_mib"] = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024
        result_pipe.send(row)
    except Exception as error:
        result_pipe.send({"fatal_error": repr(error)})
    finally:
        for pipe in pipes:
            pipe.send("stop")
        for process in processes:
            process.join(10)
            if process.is_alive():
                process.terminate()
                process.join()


def main(args):
    soft, hard = resource.getrlimit(resource.RLIMIT_NOFILE)
    resource.setrlimit(resource.RLIMIT_NOFILE, (min(65536, hard), hard))
    raw = torch.load(args.dump, map_location="cpu", weights_only=False)["samples"]
    samples = [Sample.from_dict(s) for s in raw]
    del raw
    records = [recorded_turns(s) for s in samples] if args.assemble_turns else None
    if records:
        for expected, record in zip(samples, records, strict=True):
            for version in ("old", "new"):
                actual = assemble_turns(record, version)
                assert actual.tokens == expected.tokens
                assert actual.loss_mask == expected.loss_mask
                np.testing.assert_array_equal(actual.rollout_log_probs, expected.rollout_log_probs)
                np.testing.assert_array_equal(actual.rollout_topk_token_ids, expected.rollout_topk_token_ids)
                np.testing.assert_array_equal(actual.rollout_topk_log_probs, expected.rollout_topk_log_probs)
    print(json.dumps(dict(loaded_samples=len(samples), dump=args.dump)), flush=True)
    # Capture exact pre-change network helper, rather than approximating its behavior.
    source = subprocess.check_output(["git", "show", args.base + ":miles/utils/http_utils.py"])
    with tempfile.NamedTemporaryFile(suffix=".py") as f:
        f.write(source)
        f.flush()
        spec = importlib.util.spec_from_file_location("baseline_http", f.name)
        old = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(old)
    old._client_concurrency = 4096  # saved production args: 512 * 8 GPUs / TP1
    tracer_source = subprocess.check_output(
        ["git", "show", args.base + ":miles/rollout/generate_utils/openai_endpoint_utils.py"]
    )
    with tempfile.NamedTemporaryFile(suffix=".py") as f:
        f.write(tracer_source)
        f.flush()
        spec = importlib.util.spec_from_file_location("baseline_tracer", f.name)
        old_tracer_module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(old_tracer_module)
    old_tracer_module.post = old.post
    old_tracer_module.post_buffer_no_retry = old.post_buffer_no_retry
    result = dict(
        scope=__doc__,
        base=args.base,
        dump=args.dump,
        samples=len(samples),
        nofile_limit=resource.getrlimit(resource.RLIMIT_NOFILE)[0],
        tcp_backlog=2048,
        actual_client_implementations=True,
        validation_outside_timing=True,
        assembly_from_recorded_spans=args.assemble_turns,
        rows=[],
    )
    ctx = mp.get_context("fork")
    for repeat in range(args.repeats):
        for count in args.requests:
            for variant in (args.variants if repeat % 2 == 0 else list(reversed(args.variants))):
                parent, child = ctx.Pipe()
                process = ctx.Process(
                    target=run_case,
                    args=(
                        samples,
                        variant,
                        count,
                        old_tracer_module.OpenAIEndpointTracer,
                        child,
                        records,
                        args.workers,
                    ),
                )
                process.start()
                if not parent.poll(300):
                    process.terminate()
                    process.join()
                    raise TimeoutError("benchmark case exceeded 300 seconds")
                row = parent.recv()
                process.join(15)
                if "fatal_error" in row:
                    raise RuntimeError(row["fatal_error"])
                row["repeat"] = repeat
                result["rows"].append(row)
                print(json.dumps(row), flush=True)
                with open(args.output, "w") as out:
                    json.dump(result, out, indent=2)


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "--variants",
        nargs="+",
        choices=["old", "new", "server_only", "client_only", "merge_only"],
        default=["old", "new"],
    )
    p.add_argument("--workers", type=int, default=1)
    p.add_argument("--assemble-turns", action="store_true")
    p.add_argument("--dump", required=True)
    p.add_argument("--output", required=True)
    p.add_argument("--base", default="cea1943f")
    p.add_argument("--requests", type=int, nargs="+", default=[512, 1024])
    p.add_argument("--repeats", type=int, default=2)
    main(p.parse_args())
