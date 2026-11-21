"""Loop-owned HTTP pool and admission control for completed rollout payloads."""

import asyncio
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

    @asynccontextmanager
    async def admit(self):
        self.users += 1
        if self.idle is not None:
            self.idle.cancel()
            self.idle = None
        try:
            async with self._slots:
                if self.client is None:
                    self.client = aiohttp.ClientSession(
                        connector=aiohttp.TCPConnector(limit=self.concurrency, keepalive_timeout=30),
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
