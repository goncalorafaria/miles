import asyncio
from types import SimpleNamespace

import pytest

from miles.rollout.generate_utils.collection_worker import CollectionWorker


@pytest.mark.asyncio
async def test_single_item_wakeup_never_loses_other_workers():
    worker = CollectionWorker(capacity=64, concurrency=32)

    async def collect(request):
        await asyncio.sleep(0)
        return request.collection_id

    worker._collect = collect
    await worker.start()
    try:
        for index in range(128):
            request = SimpleNamespace(collection_id=str(index))
            await worker.enqueue(request)
            assert await worker.result(str(index)) == str(index)
            await worker.cancel(str(index))
            assert (await worker.metrics())["live_workers"] == 32
        assert (await worker.metrics())["jobs"] == 0
    finally:
        await worker.close()


@pytest.mark.asyncio
async def test_cancel_queued_items_reclaims_bounded_capacity():
    worker = CollectionWorker(capacity=8, concurrency=1)
    hold = asyncio.Event()

    async def collect(request):
        await hold.wait()
        return request.collection_id

    worker._collect = collect
    await worker.start()
    try:
        for index in range(100):
            await worker.enqueue(SimpleNamespace(collection_id=str(index)))
            await worker.cancel(str(index))
        assert (await worker.metrics())["queued"] == 0
        assert (await worker.metrics())["live_workers"] == 1
        for index in range(8):
            await worker.enqueue(SimpleNamespace(collection_id=str(index)))
        with pytest.raises(RuntimeError, match="reservation violated"):
            await worker.enqueue(SimpleNamespace(collection_id="excess"))
    finally:
        await worker.close()
