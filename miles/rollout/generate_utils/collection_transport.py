"""Loop-owned HTTP pool and admission control for completed rollout payloads.

Admission bounds the payload transfers in flight. Only the transfer holds a slot: a sample
keeps its producer budget until its collection returns, so anything else done under a slot
(decoding, acknowledging) throttles how fast finished samples free that budget.
"""

import asyncio
from collections.abc import Awaitable, Callable
from contextlib import asynccontextmanager

import aiohttp


class CollectionTransport:
    def __init__(self, concurrency: int):
        if concurrency < 1:
            raise ValueError("session collection concurrency must be positive")
        self.concurrency = concurrency
        self._slots = asyncio.Semaphore(concurrency)
        self.client = None
        self.users = 0
        self.idle = None
        self._background: set[asyncio.Task] = set()

    @asynccontextmanager
    async def admit(self):
        """Hold one of ``concurrency`` transfer slots and the shared client."""
        async with self._use() as client:
            async with self._slots:
                yield client

    def spawn(self, fn: Callable[[aiohttp.ClientSession], Awaitable[None]]) -> asyncio.Task:
        """Run ``fn(client)`` in the background without a slot; the client stays open until it ends."""

        async def run():
            async with self._use() as client:
                await fn(client)

        task = asyncio.get_running_loop().create_task(run())
        self._background.add(task)  # the loop keeps only weak references to tasks
        task.add_done_callback(self._background.discard)
        return task

    @asynccontextmanager
    async def _use(self):
        self.users += 1
        if self.idle is not None:
            self.idle.cancel()
            self.idle = None
        try:
            if self.client is None:
                # Acknowledgements share the pool without a slot, so it needs room beyond the transfers.
                self.client = aiohttp.ClientSession(
                    connector=aiohttp.TCPConnector(limit=0, keepalive_timeout=30),
                    timeout=aiohttp.ClientTimeout(total=None),
                    auto_decompress=False,
                    skip_auto_headers={"Accept-Encoding"},
                    cookie_jar=aiohttp.DummyCookieJar(),
                )
            yield self.client
        finally:
            self.users -= 1
            if not self.users:
                self.idle = asyncio.get_running_loop().call_later(30, self._close_idle)

    def _close_idle(self):
        if not self.users:
            client, self.client = self.client, None
            if client is not None:
                asyncio.create_task(client.close())
            loop = asyncio.get_running_loop()
            if _transports.get(loop) is self:
                _transports.pop(loop)

    async def close(self):
        loop = asyncio.get_running_loop()
        if _transports.get(loop) is self:
            _transports.pop(loop)
        if self.idle is not None:
            self.idle.cancel()
        for task in list(self._background):
            task.cancel()
        await asyncio.gather(*self._background, return_exceptions=True)
        if self.client is not None:
            await self.client.close()
            self.client = None


_transports: dict[asyncio.AbstractEventLoop, CollectionTransport] = {}


def collection_transport(concurrency: int) -> CollectionTransport:
    loop = asyncio.get_running_loop()
    transport = _transports.get(loop)
    if transport is None:
        transport = _transports[loop] = CollectionTransport(concurrency)
    elif transport.concurrency != concurrency:
        raise ValueError("collection concurrency cannot change within a rollout event loop")
    return transport
