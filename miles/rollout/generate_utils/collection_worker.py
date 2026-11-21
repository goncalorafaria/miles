"""Persistent process owning collection HTTP/decode, with object-store result ownership."""

import asyncio
import logging
import os
import time
import zlib
from collections import deque
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass

import ray

from miles.rollout.collection_capacity import agent_finished
from miles.rollout.generate_utils.collection_transport import collection_transport
from miles.rollout.generate_utils.openai_endpoint_utils import OpenAIEndpointTracer
from miles.utils.types import Sample

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class CollectionRequest:
    router_url: str
    session_id: str
    collection_id: str
    wire_fields: tuple[str, ...]
    sample: Sample
    max_seq_len: int | None
    agent_metadata: dict | None

    def tracer(self, concurrency):
        tracer = OpenAIEndpointTracer(
            self.router_url,
            self.session_id,
            samples_wire_fields=self.wire_fields,
            collection_concurrency=concurrency,
        )
        tracer.collection_id = self.collection_id
        return tracer


@dataclass(frozen=True)
class CollectedResult:
    reference: ray.ObjectRef


@dataclass
class _Job:
    request: CollectionRequest
    result: asyncio.Future
    task: asyncio.Task | None = None


class CollectionWorker:
    """One actor; bounded async fetches and a dedicated decode thread, no coordinator GIL."""

    def __init__(self, capacity, concurrency):
        self.capacity = capacity
        self.concurrency = concurrency
        self._jobs = {}
        self._workers = []
        self._decode = None
        self._publish = None
        self._queue = None
        self._metrics = dict(
            collected=0, fetch_decode_seconds=0.0, store_seconds=0.0, acknowledged=0, delete_seconds=0.0
        )

    async def start(self):
        if self._queue is None:
            self._queue = deque()
            self._ready = asyncio.Event()
            self._decode = ThreadPoolExecutor(max_workers=1, thread_name_prefix="collection-decode")
            self._publish = ThreadPoolExecutor(max_workers=1, thread_name_prefix="collection-publish")
            self._workers = [asyncio.create_task(self._work()) for _ in range(self.concurrency)]
        return os.getpid()

    async def enqueue(self, request):
        await self.start()
        if request.collection_id in self._jobs:
            return
        if len(self._jobs) >= self.capacity:
            raise RuntimeError("collection reservation violated: outstanding-attempt budget exceeded")
        result = asyncio.get_running_loop().create_future()
        result.add_done_callback(lambda done: None if done.cancelled() else done.exception())
        job = _Job(request, result)
        self._jobs[request.collection_id] = job
        self._queue.append(job)
        self._ready.set()

    async def _work(self):
        while True:
            while not self._queue:
                await self._ready.wait()
            job = self._queue.popleft()
            if not self._queue:
                self._ready.clear()
            try:
                if job.result.cancelled():
                    continue
                job.task = asyncio.create_task(self._collect(job.request))
                result = await job.task
                if not job.result.done():
                    job.result.set_result(result)
            except asyncio.CancelledError:
                if not job.result.done():
                    job.result.cancel()
                if asyncio.current_task().cancelling():
                    raise
            except Exception as exc:
                if not job.result.done():
                    job.result.set_exception(exc)

    async def _collect(self, request):
        fetch_started = time.monotonic()
        timings = {}
        reply = await request.tracer(self.concurrency).collect_samples(
            request.sample,
            max_seq_len=request.max_seq_len,
            agent_metadata=request.agent_metadata,
            decode_executor=self._decode,
            acknowledge=False,
            timings=timings,
        )
        started = time.monotonic()
        self._metrics["fetch_decode_seconds"] += started - fetch_started
        reference = await asyncio.get_running_loop().run_in_executor(self._publish, ray.put, reply)
        self._metrics["store_seconds"] += time.monotonic() - started
        self._metrics["collected"] += 1
        for key, value in timings.items():
            self._metrics[key] = self._metrics.get(key, 0.0) + value
        logger.info("collection_store session=%s seconds=%.6f", request.session_id, time.monotonic() - started)
        return CollectedResult(reference)

    async def result(self, collection_id):
        return await asyncio.shield(self._jobs[collection_id].result)

    async def acknowledge(self, request):
        # Called only after the coordinator has materialized/accepted the result.
        # A restarted actor can still acknowledge with the original descriptor.
        started = time.monotonic()
        try:
            await request.tracer(self.concurrency).acknowledge()
            self._metrics["acknowledged"] += 1
        finally:
            self._metrics["delete_seconds"] += time.monotonic() - started
            self._jobs.pop(request.collection_id, None)

    async def cancel(self, collection_id):
        job = self._jobs.pop(collection_id, None)
        if job is not None:
            if job.task is not None:
                job.task.cancel()
            elif job in self._queue:
                self._queue.remove(job)
                if not self._queue:
                    self._ready.clear()
            if not job.result.done():
                job.result.cancel()
        # No DELETE: a retry may need the server's retained response.

    async def metrics(self):
        return {
            **self._metrics,
            "jobs": len(self._jobs),
            "queued": len(self._queue),
            "ready": sum(job.result.done() for job in self._jobs.values()),
            "live_workers": sum(not worker.done() for worker in self._workers),
        }

    async def close(self):
        for job in list(self._jobs.values()):
            await self.cancel(job.request.collection_id)
        for worker in self._workers:
            worker.cancel()
        await asyncio.gather(*self._workers, return_exceptions=True)
        if self._queue is not None:
            await collection_transport(self.concurrency).close()
        if self._decode is not None:
            await asyncio.to_thread(self._decode.shutdown, wait=True)
            await asyncio.to_thread(self._publish.shutdown, wait=True)


class PersistentCollector:
    def __init__(self, *, capacity, concurrency, workers=1):
        self.capacity = capacity
        self.concurrency = concurrency
        self._actor = None
        self._actors = []
        self._receiver = ThreadPoolExecutor(max_workers=1, thread_name_prefix="collection-receive")
        self.workers = workers
        if not 1 <= workers <= concurrency:
            raise ValueError("collection workers must be between 1 and total concurrency")

    async def start(self):
        if not self._actors:
            worker = ray.remote(num_cpus=1, max_restarts=2, max_task_retries=0)(CollectionWorker)
            for index in range(self.workers):
                quota = self.concurrency // self.workers + (index < self.concurrency % self.workers)
                self._actors.append(
                    worker.options(max_concurrency=2 * self.capacity + quota + 8).remote(self.capacity, quota)
                )
            self._actor = self._actors[0]
        pids = await asyncio.gather(*(actor.start.remote() for actor in self._actors))
        logger.info(
            "collection_workers pids=%s capacity=%d total_concurrency=%d", pids, self.capacity, self.concurrency
        )

    def _owner(self, request):
        return self._actors[zlib.crc32(request.collection_id.encode()) % len(self._actors)]

    async def metrics(self):
        rows = await asyncio.gather(*(actor.metrics.remote() for actor in self._actors))
        keys = set().union(*(row.keys() for row in rows))
        return {**{key: sum(row.get(key, 0) for row in rows) for key in keys}, "per_worker": rows}

    async def collect(self, tracer, sample, *, max_seq_len, agent_metadata=None):
        request = CollectionRequest(
            tracer.router_url,
            tracer.session_id,
            tracer.collection_id,
            tracer.samples_wire_fields,
            sample,
            max_seq_len,
            agent_metadata,
        )
        try:
            reply = await self._receive(request)
        except BaseException:
            # Cancellation must not acknowledge an unaccepted result.
            try:
                await asyncio.shield(self._owner(request).cancel.remote(request.collection_id))
            except ray.exceptions.RayError:
                logger.warning("Collector unavailable during cancellation; session retained: %s", request.session_id)
            raise
        try:
            await self._owner(request).acknowledge.remote(request)
        except ray.exceptions.RayError:
            # Receiver owns the sample; server retention is safer than recollecting.
            logger.exception("Collection acknowledgement failed after ownership acceptance: %s", request.session_id)
        return reply

    async def _receive(self, request):
        for attempt in range(3):
            try:
                await self._owner(request).enqueue.remote(request)
                agent_finished()
                stored = await self._owner(request).result.remote(request.collection_id)
                started = time.monotonic()
                reply = await asyncio.get_running_loop().run_in_executor(self._receiver, ray.get, stored.reference)
                logger.info(
                    "collection_receive session=%s seconds=%.6f", request.session_id, time.monotonic() - started
                )
                return reply
            except (ray.exceptions.RayActorError, ray.exceptions.OwnerDiedError):
                if attempt == 2:
                    raise
                # Same collection ID: server assembly stays retryable until DELETE.
                await asyncio.sleep(0.1 * (attempt + 1))

    async def close(self):
        try:
            await asyncio.gather(*(actor.close.remote() for actor in self._actors))
        finally:
            for actor in self._actors:
                ray.kill(actor, no_restart=True)
            self._actors = []
            self._actor = None
            await asyncio.to_thread(self._receiver.shutdown, wait=True)
