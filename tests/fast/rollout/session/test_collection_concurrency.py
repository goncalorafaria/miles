"""Collection workers stay responsive, bounded, and replayable after disconnects."""

import asyncio
import threading
from types import SimpleNamespace

import pytest
from aiohttp import web
from starlette.responses import Response

from miles.rollout.generate_utils.collection_transport import CollectionTransport
from miles.rollout.session.collection import SessionCollections
from miles.rollout.session.errors import MessageValidationError


@pytest.mark.asyncio
async def test_worker_does_not_block_loop_and_cancellation_keeps_session_locked():
    started, release = threading.Event(), threading.Event()
    state = SimpleNamespace(lock=asyncio.Lock())
    count = 0

    def assemble():
        nonlocal count
        count += 1
        started.set()
        assert release.wait(3)
        return Response(b"exact payload")

    async def expire(sid):
        raise AssertionError("unexpected expiration")

    owner = SessionCollections(concurrency=1, retention_seconds=60, expire=expire)
    first = asyncio.create_task(owner.collect("s", state, key="id", assemble=assemble))
    try:
        async with asyncio.timeout(2):
            while not started.is_set():
                await asyncio.sleep(0.001)
        first.cancel()
        with pytest.raises(asyncio.CancelledError):
            await first
        assert state.lock.locked()
        retry = asyncio.create_task(owner.collect("s", state, key="id", assemble=assemble))
        await asyncio.sleep(0)
        release.set()
        assert (await retry).body == b"exact payload"
        assert not state.lock.locked()
        assert count == 1
        assert (await owner.collect("s", state, key="id", assemble=assemble)).body == b"exact payload"
        assert count == 1
        with pytest.raises(MessageValidationError):
            await owner.collect("s", state, key="changed", assemble=assemble)
    finally:
        release.set()
        owner.acknowledge("s")


@pytest.mark.asyncio
async def test_assembly_bound_and_retention_expiry():
    active = peak = 0
    guard = threading.Lock()
    release = threading.Event()
    expired = set()

    def assemble():
        nonlocal active, peak
        with guard:
            active += 1
            peak = max(peak, active)
        assert release.wait(3)
        with guard:
            active -= 1
        return Response(b"x")

    async def expire(sid):
        expired.add(sid)
        owner.acknowledge(sid)

    owner = SessionCollections(concurrency=2, retention_seconds=0.02, expire=expire)
    tasks = [
        asyncio.create_task(owner.collect(str(i), SimpleNamespace(lock=asyncio.Lock()), key="id", assemble=assemble))
        for i in range(10)
    ]
    try:
        async with asyncio.timeout(2):
            while peak < 2:
                await asyncio.sleep(0.001)
        assert active == 2
    finally:
        release.set()
    await asyncio.gather(*tasks)
    owner.acknowledge("0")
    async with asyncio.timeout(2):
        while len(expired) < 9:
            await asyncio.sleep(0.005)
    assert peak == 2
    assert "0" not in expired


@pytest.mark.asyncio
async def test_transport_reuses_connection_and_limits_admission():
    peers = []

    async def handler(request):
        peers.append(request.transport)
        return web.Response(body=b"payload")

    app = web.Application()
    app.router.add_post("/samples", handler)
    runner = web.AppRunner(app)
    await runner.setup()
    site = web.TCPSite(runner, "127.0.0.1", 0)
    await site.start()
    port = site._server.sockets[0].getsockname()[1]
    pool = CollectionTransport(2)
    try:
        for _ in range(3):
            async with pool.admit() as client:
                async with client.post(f"http://127.0.0.1:{port}/samples") as reply:
                    assert await reply.read() == b"payload"
        assert len({id(p) for p in peers}) == 1
        active = peak = 0
        release = asyncio.Event()

        async def collect():
            nonlocal active, peak
            async with pool.admit():
                active += 1
                peak = max(peak, active)
                await release.wait()
                active -= 1

        tasks = [asyncio.create_task(collect()) for _ in range(10)]
        await asyncio.sleep(0.01)
        assert active == 2
        tasks[-1].cancel()
        release.set()
        await asyncio.gather(*tasks, return_exceptions=True)
        assert peak == 2 and pool.users == 0
    finally:
        await pool.close()
        await runner.cleanup()
