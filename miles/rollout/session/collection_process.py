"""Process-isolated assembly; only one locked session crosses the IPC boundary."""

import asyncio
import multiprocessing
from concurrent.futures import ProcessPoolExecutor
from dataclasses import fields

from starlette.responses import Response

_core = None


def _initialize(config, use_addition_r3):
    # Local import breaks core -> collection_process -> sessions -> core cycle.
    from miles.rollout.session.sessions import create_session_core

    global _core
    _core = create_session_core(None, config, use_addition_r3=use_addition_r3)


def _assemble(session_type, state, session_id, kwargs):
    session = session_type(**state)
    response = _core._assemble_samples(session_id, session, **kwargs)
    return response.body, response.status_code, dict(response.headers)


class CollectionProcess:
    def __init__(self, config, *, use_addition_r3):
        self._config = config.model_copy(update={"session_collection_processes": False})
        self._use_addition_r3 = use_addition_r3
        self._pool = None

    async def assemble(self, session_id, session, **kwargs):
        if self._pool is None:
            self._pool = ProcessPoolExecutor(
                max_workers=self._config.session_collection_workers,
                mp_context=multiprocessing.get_context("spawn"),
                initializer=_initialize,
                initargs=(self._config, self._use_addition_r3),
            )
        # Caller owns the session lock until this future settles, including IPC.
        # Do not serialize locks, the server, or the registry of other sessions.
        state = {field.name: getattr(session, field.name) for field in fields(session) if field.name != "lock"}
        future = self._pool.submit(_assemble, type(session), state, session_id, kwargs)
        body, status, headers = await asyncio.wrap_future(future)
        return Response(content=body, status_code=status, headers=headers)

    async def close(self):
        if self._pool is not None:
            await asyncio.to_thread(self._pool.shutdown, wait=True)
            self._pool = None
