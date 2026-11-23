"""One agent worker: many episodes as asyncio tasks in one long-lived process.

Copies prime-rl's env-server worker (verifiers ``serve``): instead of a fresh interpreter per
episode, a fixed pool of these processes each runs many episodes concurrently. Each worker also
copies prime-rl's in-worker interception server: it serves the session routes itself (the same
``SessionServer`` app the standalone session servers run), so an agent's chat calls get TITO and
recording in this process and go straight to the inference router. At the end of an episode the
worker assembles and encodes its samples here and returns them over zmq; nothing is fetched or
decoded by HTTP afterwards.
"""

import asyncio
import json
import logging
import os
import pickle
import socket
import sys
import time
import traceback

import setproctitle
import uvicorn
import zmq
import zmq.asyncio

from miles.rollout.agent_pool.protocol import CANCEL, REPLY, RUN, RunReply, RunRequest, WorkerSpec
from miles.rollout.agentic.agent_function import _die_with_parent
from miles.utils.function_registry import load_function
from miles.utils.logging_utils import configure_logger_raw

logger = logging.getLogger(__name__)

# Spawned as `python -c BOOTSTRAP`: take the rollout's import path before importing anything from miles.
BOOTSTRAP = (
    "import pickle, sys; spec = pickle.load(sys.stdin.buffer); sys.path[:] = spec.sys_path; "
    "from miles.rollout.agent_pool.worker import main; main(spec)"
)


def main(spec: WorkerSpec) -> None:
    _die_with_parent(spec.parent_pid)
    os.environ.update(spec.env)
    configure_logger_raw(f"agent_pool_worker{spec.index} pid={os.getpid()}")
    logging.getLogger("httpx").setLevel(logging.WARNING)
    setproctitle.setproctitle(f"miles-agent-worker-{spec.index}")
    asyncio.run(_Worker(spec).serve())


class _Worker:
    def __init__(self, spec: WorkerSpec):
        # Deferred: SessionServer pulls in the tokenizer stack, which only workers need.
        from miles.rollout.session.server import SessionServer

        self._spec = spec
        self._listener = socket.create_server(("127.0.0.1", spec.session_config.port))
        self._base_url = f"http://127.0.0.1:{self._listener.getsockname()[1]}"
        self._server = SessionServer(spec.session_config)
        self._core = self._server.app.state.session_core
        self._agent = load_function(spec.agent_function_path)
        assert self._agent is not None, f"agent function {spec.agent_function_path} not found"
        self._episodes: dict[bytes, asyncio.Task] = {}

    async def serve(self) -> None:
        http = uvicorn.Server(
            uvicorn.Config(self._server.app, log_level="warning", access_log=False, timeout_keep_alive=60)
        )
        http_task = asyncio.create_task(http.serve(sockets=[self._listener]))
        while not http.started:
            if http_task.done():
                await http_task  # surfaces the startup failure
            await asyncio.sleep(0.01)
        context = zmq.asyncio.Context()
        router = context.socket(zmq.ROUTER)
        router.setsockopt(zmq.LINGER, 0)
        router.bind(self._spec.address)
        logger.info("agent worker %d serving %s, session routes at %s", self._spec.index, self._spec.address, self._base_url)
        try:
            await self._dispatch(router)
        finally:
            router.close()
            context.term()
            http.should_exit = True

    async def _dispatch(self, router) -> None:
        while True:
            peer, request_id, method, payload = await router.recv_multipart()
            if method == RUN:
                task = asyncio.create_task(self._run(router, peer, request_id, pickle.loads(payload)))
                self._episodes[request_id] = task
                task.add_done_callback(lambda _, rid=request_id: self._episodes.pop(rid, None))
            elif method == CANCEL and (task := self._episodes.get(request_id)) is not None:
                task.cancel()

    async def _run(self, router, peer: bytes, request_id: bytes, request: RunRequest) -> None:
        try:
            reply, samples = await self._episode(request)
        except asyncio.CancelledError:
            return  # the rollout gave up on it; nobody waits for a reply
        except Exception as error:  # noqa: BLE001 - the worker machinery failed; the rollout aborts the sample
            logger.exception("agent worker episode failed")
            reply, samples = RunReply(ok=False, error=repr(error), traceback=traceback.format_exc()), b""
        await router.send_multipart([peer, request_id, REPLY, pickle.dumps(reply), samples], copy=False)

    async def _episode(self, request: RunRequest) -> tuple[RunReply, bytes]:
        timings = {"started": time.time()}
        created = await self._core.create_session(
            evaluation=request.evaluation, sampling_defaults=request.session_params, extra_key=request.extra_key
        )
        session_id = json.loads(created.body)["session_id"]
        try:
            agent_metadata = await self._call_agent(request, session_id)
            timings["agent_done"] = time.time()
            collected = await self._collect(request, session_id, agent_metadata)
            timings["collected"] = time.time()
        finally:
            await self._core.delete_session(session_id)
        reply = RunReply(
            ok=True, session_id=session_id, agent_metadata=agent_metadata, collect_status=collected.status_code
        )
        reply.timings = timings
        if collected.status_code != 200:
            reply.collect_error = bytes(collected.body).decode(errors="replace")
            return reply, b""
        return reply, collected.body

    async def _call_agent(self, request: RunRequest, session_id: str) -> dict | None:
        try:
            return await self._agent(
                base_url=f"{self._base_url}/sessions/{session_id}",
                prompt=request.prompt,
                request_kwargs=request.request_kwargs,
                metadata=request.metadata,
            )
        except Exception as error:  # noqa: BLE001 - as under subproc: an agent failure still collects its turns
            logger.warning("[session=%s] Agent function failed: %s", session_id, error, exc_info=True)
            return None

    async def _collect(self, request: RunRequest, session_id: str, agent_metadata: dict | None):
        kwargs = {"max_seq_len": request.max_seq_len}
        if request.use_v2:
            kwargs["agent_metadata"] = agent_metadata
        return await self._core.collect_samples(session_id, **kwargs)


if __name__ == "__main__":
    main(pickle.load(sys.stdin.buffer))
