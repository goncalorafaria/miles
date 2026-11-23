"""The rollout process's side of the agent worker pool.

Copies prime-rl's env-server client and pool broker in one object: one zmq DEALER per worker,
each episode sent to the worker with the fewest in flight (verifiers ``serve/pool.py``). There is
no broker process: this process already knows what each worker is running. A worker that dies
fails its episodes (the caller aborts and recycles those samples) and is started again.
"""

import asyncio
import logging
import os
import pickle
import subprocess
import sys
import uuid

import zmq
import zmq.asyncio

from miles.rollout.agent_pool.protocol import CANCEL, RUN, RunReply, RunRequest, WorkerSpec
from miles.rollout.agent_pool.worker import BOOTSTRAP
from miles.rollout.agentic import rollout_limits
from miles.rollout.session.config import SessionServerConfig, compute_session_server_config

logger = logging.getLogger(__name__)

_WATCH_INTERVAL_S = 1.0


class AgentWorkerDied(RuntimeError):
    """The worker running an episode exited before replying."""


class _WorkerHandle:
    def __init__(self, spec: WorkerSpec, context: zmq.asyncio.Context):
        self.spec = spec
        self.process = _start_process(spec)
        self.socket = context.socket(zmq.DEALER)
        self.socket.setsockopt(zmq.LINGER, 0)
        self.socket.connect(spec.address)
        self.pending: dict[bytes, asyncio.Future] = {}


class AgentPool:
    def __init__(self, specs: list[WorkerSpec]):
        self._context = zmq.asyncio.Context()
        self._workers = [_WorkerHandle(spec, self._context) for spec in specs]
        self._tasks = [asyncio.create_task(self._read(worker)) for worker in self._workers]
        self._tasks.append(asyncio.create_task(self._watch()))

    async def run(self, request: RunRequest) -> tuple[RunReply, memoryview]:
        """Run one episode on the least-loaded worker; returns its reply header and samples buffer."""
        worker = min(self._workers, key=lambda w: len(w.pending))
        request_id = uuid.uuid4().bytes
        reply = asyncio.get_running_loop().create_future()
        worker.pending[request_id] = reply
        try:
            await worker.socket.send_multipart([request_id, RUN, pickle.dumps(request)])
            return await reply
        except asyncio.CancelledError:
            if not worker.socket.closed:
                worker.socket.send_multipart([request_id, CANCEL, b""])  # the episode closes its own sandbox
            raise
        finally:
            worker.pending.pop(request_id, None)

    async def close(self) -> None:
        for task in self._tasks:
            task.cancel()
        await asyncio.gather(*self._tasks, return_exceptions=True)
        for worker in self._workers:
            worker.process.kill()
            worker.socket.close()
        self._context.term()

    async def _read(self, worker: _WorkerHandle) -> None:
        while True:
            request_id, _, header, samples = await worker.socket.recv_multipart(copy=False)
            reply = worker.pending.get(request_id.bytes)
            if reply is not None and not reply.done():
                reply.set_result((pickle.loads(header.buffer), samples.buffer))

    async def _watch(self) -> None:
        while True:
            await asyncio.sleep(_WATCH_INTERVAL_S)
            for worker in self._workers:
                if (code := worker.process.poll()) is not None:
                    logger.error("agent worker %d exited with code %s; restarting it", worker.spec.index, code)
                    for reply in worker.pending.values():
                        if not reply.done():
                            reply.set_exception(AgentWorkerDied(f"agent worker {worker.spec.index} exited ({code})"))
                    worker.process = _start_process(worker.spec)


def _start_process(spec: WorkerSpec) -> subprocess.Popen:
    process = subprocess.Popen([sys.executable, "-c", BOOTSTRAP], stdin=subprocess.PIPE)
    process.stdin.write(pickle.dumps(spec))
    process.stdin.close()
    return process


def build_worker_specs(args, *, session_config: SessionServerConfig, num_workers: int) -> list[WorkerSpec]:
    scope = rollout_limits.current_scope()
    # Linux abstract-namespace sockets: no file to clean up, and no 107-byte path limit or network
    # filesystem under $TMPDIR to trip over.
    pool_id = f"miles-agent-pool-{scope}-{uuid.uuid4().hex[:8]}"
    return [
        WorkerSpec(
            index=index,
            address=f"ipc://@{pool_id}-{index}",
            session_config=session_config.model_copy(update={"instance_id": f"agent-pool-{scope}-{index}"}),
            agent_function_path=args.custom_agent_function_path,
            parent_pid=os.getpid(),
            sys_path=list(sys.path),
            # Shared limits (sandbox starts, abort counters) stay scoped to this rollout process.
            env={rollout_limits.SCOPE_ENV_VAR: scope},
        )
        for index in range(num_workers)
    ]


_pools: dict[asyncio.AbstractEventLoop, AgentPool] = {}


def agent_pool(args) -> AgentPool:
    """This event loop's pool, started on first use (the router address is known by then)."""
    loop = asyncio.get_running_loop()
    if (pool := _pools.get(loop)) is None:
        if args.agent_pool_workers < 1:
            raise ValueError(f"--agent-pool-workers must be positive, got {args.agent_pool_workers}")
        session_config = compute_session_server_config(
            args,
            host="127.0.0.1",
            port=0,  # each worker binds its own free port
            instance_id=None,
            backend_url=f"http://{args.sglang_router_ip}:{args.sglang_router_port}",
        )
        pool = _pools[loop] = AgentPool(
            build_worker_specs(args, session_config=session_config, num_workers=args.agent_pool_workers)
        )
        logger.info("Started %d agent workers", args.agent_pool_workers)
    return pool
