"""Generation credits can transfer to a bounded collection stage, once only."""

import asyncio
import logging
import time
from contextlib import asynccontextmanager
from contextvars import ContextVar

logger = logging.getLogger(__name__)
current_credit = ContextVar("generation_credit", default=None)


class GenerationCredit:
    def __init__(self, callback):
        self._callback = callback
        self._released = False

    def release(self):
        if not self._released:
            self._released = True
            self._callback()


class CollectionCapacity:
    def __init__(self, limit):
        if limit < 0:
            raise ValueError("collection capacity must be nonnegative")
        self._slots = asyncio.Semaphore(limit) if limit else None
        self.active = 0

    @asynccontextmanager
    async def admit(self):
        credit = current_credit.get()
        if self._slots is None or credit is None:
            yield
            return
        started = time.monotonic()
        async with self._slots:
            self.active += 1
            credit.release()
            logger.info("collection_handoff active=%d admission_wait_s=%.6f", self.active, time.monotonic() - started)
            try:
                yield
            finally:
                self.active -= 1
