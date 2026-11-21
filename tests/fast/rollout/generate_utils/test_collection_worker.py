import asyncio
import os
from types import SimpleNamespace

import numpy as np
import pytest
import ray
from aiohttp import web
from tests.ci.ci_register import register_cpu_ci

from miles.rollout.collection_capacity import GenerationCredit, current_credit, generation_permit
from miles.rollout.generate_utils.collection_worker import PersistentCollector
from miles.rollout.generate_utils.openai_endpoint_utils import OpenAIEndpointTracer
from miles.rollout.session.samples.codec import COMPUTED_FIELDS_V2, encode_samples
from miles.utils.types import Sample

register_cpu_ci(est_time=180, suite="stage-b-cpu", labels=["rollout"])


@pytest.mark.asyncio
@pytest.mark.parametrize("workers", [1, 2])
async def test_process_handoff_retry_and_ownership_before_delete(workers):
    ray.init(
        address="local",
        num_cpus=2,
        include_dashboard=False,
        object_store_memory=128 * 1024 * 1024,
        logging_level="ERROR",
    )
    entered = asyncio.Queue()
    release = asyncio.Event()
    accepted = False
    deletes = []
    samples = [Sample(tokens=[1, 2], response_length=1, rollout_log_probs=[-0.1], reward=0.5) for _ in range(2)]
    for sample in samples:
        sample.rollout_routed_experts = np.ones((2, 1, 1), dtype=np.int32)
    payload = encode_samples(samples, {}, fields=COMPUTED_FIELDS_V2)

    async def post(request):
        body = await request.json()
        await entered.put(body["collection_id"])
        await release.wait()
        return web.Response(body=payload)

    async def delete(request):
        assert accepted, "DELETE before coordinator accepted object-store result"
        deletes.append(request.path)
        return web.Response(status=204)

    class Receiver(PersistentCollector):
        async def _receive(self, request):
            nonlocal accepted
            reply = await super()._receive(request)
            accepted = True
            return reply

    app = web.Application()
    app.router.add_post("/sessions/{sid}/samples", post)
    app.router.add_delete("/sessions/{sid}", delete)
    runner = web.AppRunner(app)
    await runner.setup()
    site = web.TCPSite(runner, "127.0.0.1", 0)
    await site.start()
    port = site._server.sockets[0].getsockname()[1]
    collector = Receiver(capacity=4, concurrency=2, workers=workers)
    semaphore = asyncio.Semaphore(1)
    credits = []
    tracer = OpenAIEndpointTracer(
        f"http://127.0.0.1:{port}", "s", samples_wire_fields=COMPUTED_FIELDS_V2, collection_concurrency=2
    )

    async def attempt():
        credit = GenerationCredit(lambda: credits.append(True))
        token = current_credit.set(credit)
        try:
            async with generation_permit(semaphore):
                return await collector.collect(tracer, Sample(index=7), max_seq_len=10)
        finally:
            credit.release()
            current_credit.reset(token)

    task = None
    try:
        await collector.start()
        assert await collector._actor.start.remote() != os.getpid()
        task = asyncio.create_task(attempt())
        assert await asyncio.wait_for(entered.get(), 120) == tracer.collection_id
        assert credits == [True]
        assert semaphore._value == 1
        assert not task.done()
        assert deletes == []
        ray.kill(collector._owner(SimpleNamespace(collection_id=tracer.collection_id)), no_restart=False)
        assert await asyncio.wait_for(entered.get(), 120) == tracer.collection_id
        release.set()
        reply = await asyncio.wait_for(task, 120)
        assert len(reply.samples) == 2
        assert [sample.index for sample in reply.samples] == [7, 7]
        np.testing.assert_array_equal(reply.samples[0].rollout_routed_experts, np.ones((2, 1, 1), dtype=np.int32))
        assert credits == [True]
        assert len(deletes) == 1
        assert (await collector.metrics())["live_workers"] == 2
        release.clear()
        tracer = OpenAIEndpointTracer(
            f"http://127.0.0.1:{port}",
            "cancel",
            samples_wire_fields=COMPUTED_FIELDS_V2,
            collection_concurrency=2,
        )
        task = asyncio.create_task(attempt())
        assert await asyncio.wait_for(entered.get(), 30) == tracer.collection_id
        task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await task
        metrics = await collector.metrics()
        assert metrics["jobs"] == 0
        assert metrics["live_workers"] == 2
        assert len(deletes) == 1
    finally:
        release.set()
        if task is not None and not task.done():
            task.cancel()
            await asyncio.gather(task, return_exceptions=True)
        await collector.close()
        await runner.cleanup()
        ray.shutdown()
