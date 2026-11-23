"""Rollout limits hold across a rollout's agent processes and free a slot when its holder dies."""

import asyncio
import uuid

import pytest

from miles.rollout.agentic.agent_function import call_agent_function
from miles.rollout.agentic.rollout_limits import rollout_semaphore
from miles.utils.test_utils import agent_function_stubs as stubs


async def _in_agent_processes(fn, count: int, **kwargs) -> list:
    return await asyncio.gather(*(call_agent_function(fn, mode="subproc", **kwargs) for _ in range(count)))


def _peak_overlap(spans: list[tuple[float, float]]) -> int:
    return max(sum(1 for start, end in spans if start <= t < end) for t, _ in spans)


def test_semaphore_caps_holders_across_agent_processes():
    spans = asyncio.run(_in_agent_processes(stubs.hold_semaphore, 6, name="test-cap", limit=2, hold_s=1.0))
    assert _peak_overlap(spans) <= 2
    assert max(end for _, end in spans) - min(start for start, _ in spans) >= 2.9, "six 1s holds under 2 slots"


def test_lock_serializes_holders_across_agent_processes():
    spans = asyncio.run(_in_agent_processes(stubs.hold_lock, 3, name="test-lock", hold_s=0.5))
    assert _peak_overlap(spans) == 1


def test_a_dead_holder_frees_its_slot():
    with pytest.raises(RuntimeError, match="before returning a result"):
        asyncio.run(call_agent_function(stubs.die_holding_semaphore, mode="subproc", name="test-dead"))
    asyncio.run(
        asyncio.wait_for(
            call_agent_function(stubs.hold_semaphore, mode="subproc", name="test-dead", limit=1, hold_s=0.0), 30
        )
    )


def _unique(prefix: str) -> str:
    return f"{prefix}-{uuid.uuid4().hex}"


def test_many_waiters_in_one_process_respect_the_cap():
    name, active, peak = _unique("test-many"), [0], [0]

    async def hold():
        async with rollout_semaphore(name, 2):
            active[0] += 1
            peak[0] = max(peak[0], active[0])
            await asyncio.sleep(0.02)
            active[0] -= 1

    async def main():
        await asyncio.wait_for(asyncio.gather(*(hold() for _ in range(40))), 60)

    asyncio.run(main())
    assert peak[0] == 2


@pytest.mark.parametrize("cancel_head", [True, False])
def test_a_cancelled_waiter_leaves_the_line_usable(cancel_head):
    """Cancel the waiter polling the slots, or one asleep behind it; the line must keep moving."""
    name = _unique("test-cancel")

    async def main():
        holder_in, release = asyncio.Event(), asyncio.Event()

        async def holder():
            async with rollout_semaphore(name, 1):
                holder_in.set()
                await release.wait()

        async def waiter():
            async with rollout_semaphore(name, 1):
                return True

        hold = asyncio.create_task(holder())
        await holder_in.wait()
        head = asyncio.create_task(waiter())
        await asyncio.sleep(0.5)  # the head takes the queue lock and polls
        behind = asyncio.create_task(waiter())
        await asyncio.sleep(0.5)  # the second waiter sleeps on the queue lock
        cancelled, kept = (head, behind) if cancel_head else (behind, head)
        cancelled.cancel()
        with pytest.raises(asyncio.CancelledError):
            await cancelled
        release.set()
        await hold
        assert await asyncio.wait_for(kept, 30)
        assert await asyncio.wait_for(waiter(), 30), "a later waiter still gets through"

    asyncio.run(main())
