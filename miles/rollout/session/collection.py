"""Bounded, retryable assembly of finalized session samples.

A collection ID identifies one immutable result. The session lock protects the
worker's read until packing finishes; cancelling an HTTP waiter cannot release
that lock early. DELETE acknowledges delivery. Unacknowledged results expire.
"""

import asyncio
import inspect
import logging
import time
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

from miles.rollout.session.errors import MessageValidationError

logger = logging.getLogger(__name__)


@dataclass
class _Collection:
    key: str
    task: asyncio.Task
    expiry: asyncio.TimerHandle | None = None


class SessionCollections:
    def __init__(self, *, concurrency: int, retention_seconds: float, expire: Callable):
        self._slots = asyncio.Semaphore(concurrency)
        self._retention_seconds = retention_seconds
        self._expire = expire
        self._entries: dict[str, _Collection] = {}

    async def collect(self, session_id: str, session: Any, *, key: str | None, assemble: Callable):
        entry = self._entries.get(session_id) if key is not None else None
        if entry is not None and entry.key != key:
            raise MessageValidationError("collection parameters changed before acknowledgement")
        if entry is None:
            task = asyncio.create_task(self._assemble(session_id, session, assemble))
            # Retrieve exceptions even if all HTTP waiters disconnect.
            task.add_done_callback(lambda done: None if done.cancelled() else done.exception())
            if key is not None:
                entry = _Collection(key, task)
                self._entries[session_id] = entry
                task.add_done_callback(lambda _: self._schedule_expiry(session_id, entry))
        else:
            task = entry.task
        return await asyncio.shield(task)

    async def _assemble(self, session_id, session, assemble):
        queued = time.monotonic()
        async with self._slots:
            async with session.lock:
                started = time.monotonic()
                # This task is shielded from HTTP cancellation. The lock remains
                # held until the worker stops reading the session.
                response = await asyncio.to_thread(assemble)
                if inspect.isawaitable(response):
                    response = await response
                elapsed = time.monotonic() - started
                packing = response.headers.get("Server-Timing", "")
                response.headers["Server-Timing"] = (
                    (packing + "," if packing else "") + f"collection_queue;dur={(started - queued) * 1000:.3f},"
                    f"collection_assemble_pack;dur={elapsed * 1000:.3f}"
                )
                logger.info(
                    "session_collection session=%s queue_s=%.6f assemble_pack_s=%.6f bytes=%d",
                    session_id,
                    started - queued,
                    elapsed,
                    len(response.body),
                )
                return response

    def _schedule_expiry(self, session_id, entry):
        if self._entries.get(session_id) is entry:
            entry.expiry = asyncio.get_running_loop().call_later(
                self._retention_seconds, self._start_expiry, session_id, entry
            )

    def _start_expiry(self, session_id, entry):
        if self._entries.get(session_id) is entry:
            logger.warning("Expiring unacknowledged sample collection: session=%s", session_id)
            task = asyncio.create_task(self._expire(session_id))
            task.add_done_callback(lambda done: None if done.cancelled() else done.exception())

    def acknowledge(self, session_id):
        entry = self._entries.pop(session_id, None)
        if entry is not None and entry.expiry is not None:
            entry.expiry.cancel()
