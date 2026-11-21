import asyncio

import pytest

from miles.rollout.collection_capacity import CollectionCapacity, GenerationCredit, current_credit


@pytest.mark.asyncio
async def test_release_only_after_collection_admission_and_once_on_cancellation():
    released = []
    capacity = CollectionCapacity(1)
    entered = asyncio.Queue()
    hold = asyncio.Event()

    async def episode(index):
        credit = GenerationCredit(lambda: released.append(index))
        token = current_credit.set(credit)
        try:
            async with capacity.admit():
                await entered.put(index)
                await hold.wait()
        finally:
            credit.release()
            current_credit.reset(token)

    first = asyncio.create_task(episode(1))
    assert await entered.get() == 1
    second = asyncio.create_task(episode(2))
    await asyncio.sleep(0)
    assert released == [1]
    assert capacity.active == 1
    second.cancel()
    with pytest.raises(asyncio.CancelledError):
        await second
    assert released == [1, 2]
    first.cancel()
    with pytest.raises(asyncio.CancelledError):
        await first
    assert released == [1, 2]
    assert capacity.active == 0


@pytest.mark.asyncio
async def test_burst_is_bounded_and_all_results_delivered():
    capacity = CollectionCapacity(8)
    released = []
    peak = 0

    async def episode(index):
        nonlocal peak
        credit = GenerationCredit(lambda: released.append(index))
        token = current_credit.set(credit)
        try:
            async with capacity.admit():
                peak = max(peak, capacity.active)
                await asyncio.sleep(0)
                return index
        finally:
            credit.release()
            current_credit.reset(token)

    assert await asyncio.gather(*(episode(i) for i in range(2048))) == list(range(2048))
    assert sorted(released) == list(range(2048))
    assert peak == 8
    assert capacity.active == 0


@pytest.mark.asyncio
async def test_disabled_handoff_keeps_credit_until_completion():
    released = []
    credit = GenerationCredit(lambda: released.append(1))
    token = current_credit.set(credit)
    try:
        async with CollectionCapacity(0).admit():
            assert not released
        credit.release()
        assert released == [1]
    finally:
        current_credit.reset(token)


@pytest.mark.asyncio
async def test_both_generation_limits_release_at_handoff_once():
    from miles.rollout.collection_capacity import agent_finished, generation_permit

    released = []
    credit = GenerationCredit(lambda: released.append(True))
    token = current_credit.set(credit)
    semaphore = asyncio.Semaphore(1)
    try:
        async with generation_permit(semaphore):
            assert semaphore._value == 0
            agent_finished()
            agent_finished()
            assert semaphore._value == 1
            assert released == [True]
        credit.release()
        assert semaphore._value == 1
        assert released == [True]
    finally:
        current_credit.reset(token)


@pytest.mark.asyncio
async def test_cancel_before_generation_permit_does_not_return_unacquired_permit():
    from miles.rollout.collection_capacity import generation_permit

    semaphore = asyncio.Semaphore(0)

    async def blocked():
        async with generation_permit(semaphore):
            raise AssertionError("unreachable")

    task = asyncio.create_task(blocked())
    await asyncio.sleep(0)
    task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await task
    assert semaphore._value == 0
