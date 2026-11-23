"""Limits shared by a rollout process and the agent processes it starts.

Under ``--custom-agent-function-mode subproc`` every agent-function call has a
process of its own, so a module-level lock or semaphore limits nothing. These
are ``flock`` locks on files in one directory per rollout process instead: the
kernel releases a slot when its holder exits, including when the holder is killed.
"""

import asyncio
import fcntl
import os
import random
import re
import tempfile
import threading
from collections.abc import AsyncIterator, Iterator
from contextlib import asynccontextmanager, contextmanager
from pathlib import Path

# An agent process inherits its rollout process's scope through this variable.
SCOPE_ENV_VAR = "MILES_ROLLOUT_LIMITS_SCOPE"

_POLL_S = 0.2


def current_scope() -> str:
    """Name the rollout process whose limits apply here: this process, or the one that started it."""
    return os.environ.get(SCOPE_ENV_VAR) or f"pid{os.getpid()}"


@asynccontextmanager
async def rollout_semaphore(name: str, limit: int) -> AsyncIterator[None]:
    """Hold one of ``limit`` slots named ``name`` for the duration of the block."""
    if limit < 1:
        raise ValueError(f"rollout_semaphore {name!r} needs a positive limit, got {limit}")
    fd = await _acquire_any_slot(name, limit)
    try:
        yield
    finally:
        os.close(fd)  # closing the descriptor releases its lock


@contextmanager
def rollout_lock(name: str) -> Iterator[None]:
    """Hold the lock named ``name``, blocking the calling thread until it is free."""
    fd = _open_lock_file(f"{name}.lock")
    try:
        fcntl.flock(fd, fcntl.LOCK_EX)
        yield
    finally:
        os.close(fd)


async def _acquire_any_slot(name: str, limit: int) -> int:
    if (fd := _try_any_slot(name, limit)) is not None:
        return fd
    # Wait in line: only the waiter holding the queue lock polls the slots; the others sleep in the
    # kernel. With every waiter polling, a burst of ~1000 agent processes each reopening all slot
    # files several times a second saturated the rollout node's CPUs.
    queue_fd = await _lock_blocking(f"{name}.queue.lock")
    try:
        while (fd := _try_any_slot(name, limit)) is None:
            await asyncio.sleep(_POLL_S * (0.5 + random.random()))
        return fd
    finally:
        os.close(queue_fd)


def _try_any_slot(name: str, limit: int) -> int | None:
    # a random order keeps waiters from all contending for slot 0
    for slot in random.sample(range(limit), limit):
        fd = _open_lock_file(f"{name}.{slot}.lock")
        try:
            fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            os.close(fd)  # another holder has this slot
            continue
        except BaseException:
            os.close(fd)
            raise
        return fd
    return None


async def _lock_blocking(file_name: str) -> int:
    """Take an exclusive lock without blocking the loop, on a thread of its own.

    Not the default executor: inline-mode waiters would occupy its threads for as long as they
    wait. If the caller is cancelled first, the thread still takes the lock and releases it at once.
    """
    fd = _open_lock_file(file_name)
    loop = asyncio.get_running_loop()
    locked = loop.create_future()

    def deliver(error: BaseException | None) -> None:
        if locked.cancelled():
            os.close(fd)
        elif error is not None:
            locked.set_exception(error)
        else:
            locked.set_result(None)

    def wait() -> None:
        try:
            fcntl.flock(fd, fcntl.LOCK_EX)
            error = None
        except BaseException as e:  # noqa: BLE001 - handed to the waiting coroutine
            error = e
        try:
            loop.call_soon_threadsafe(deliver, error)
        except RuntimeError:  # the loop closed while waiting; nobody will use the lock
            os.close(fd)

    threading.Thread(target=wait, name=f"rollout-limit-queue-{file_name}", daemon=True).start()
    try:
        await locked
    except BaseException:
        if not locked.cancelled():
            os.close(fd)  # the thread failed: it delivered the error and kept no lock
        raise
    return fd


def _open_lock_file(file_name: str) -> int:
    return os.open(_lock_dir() / re.sub(r"[^A-Za-z0-9._-]", "_", file_name), os.O_CREAT | os.O_RDWR, 0o600)


def _lock_dir() -> Path:
    lock_dir = Path(tempfile.gettempdir()) / f"miles-rollout-limits-{current_scope()}"
    lock_dir.mkdir(exist_ok=True)
    return lock_dir
