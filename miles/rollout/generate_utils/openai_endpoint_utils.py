"""
Utilities for the OpenAI endpoint
"""

import asyncio
import logging
import random
import time
import uuid
from argparse import Namespace
from functools import partial

import aiohttp
import httpx

from miles.rollout.generate_utils.collection_transport import collection_transport
from miles.rollout.generate_utils.sampling_mask import should_return_sampling_mask
from miles.rollout.session.samples.codec import (
    COMPUTED_FIELDS,
    COMPUTED_FIELDS_V2,
    ROLLOUT_SAMPLING_MASK_FIELDS,
    SamplesReply,
    decode_samples_and_merge_input_sample,
)
from miles.rollout.session.types import CreateSessionRequest
from miles.utils.http_utils import post, post_buffer_no_retry
from miles.utils.types import Sample

logger = logging.getLogger(__name__)

_SESSION_REQUEST_TIMEOUT = 120
_SESSION_ACK_TIMEOUT = 10


class OpenAIEndpointTracer:
    def __init__(
        self,
        router_url: str,
        session_id: str,
        session_server_instance_id: str | None = None,
        samples_wire_fields: tuple[str, ...] = COMPUTED_FIELDS,
        agent_router_url: str | None = None,
        collection_concurrency: int = 32,
    ):
        self.collection_concurrency = collection_concurrency
        self.collection_id = uuid.uuid4().hex
        self.router_url = router_url
        self.session_id = session_id
        self.base_url = f"{router_url}/sessions/{session_id}"
        # The agent may run outside the cluster; the driver's own calls stay on base_url.
        self.agent_base_url = f"{agent_router_url or router_url}/sessions/{session_id}"
        self.session_server_instance_id = session_server_instance_id
        # The samples-wire allowlist must match the server's encode: v1 default,
        # extended under --use-session-server v2 (create() selects from args;
        # direct constructions keep v1).
        self.samples_wire_fields = samples_wire_fields

    @property
    def session_server_id(self) -> str:
        """``ip:port`` of the instance owning this session, as recorded in sample metadata."""
        return self.router_url.removeprefix("http://")

    @staticmethod
    async def create(
        args: Namespace,
        *,
        evaluation: bool = False,
        sampling_params: dict | None = None,
        extra_key: str | None = None,
    ):
        instances = getattr(args, "session_server_instances", None)
        if not instances:
            raise RuntimeError(
                "session_server_instances is not set. Pass --use-session-server to start the session server."
            )
        # The only routing decision in the system: pick the owning instance once
        # per session; every later touch of the session reuses this URL.
        instance = random.choice(instances)
        session_url = instance.url
        # Drop engine-only sampling fields before validating the session creation body.
        session_params = {
            key: value for key, value in (sampling_params or {}).items() if key in CreateSessionRequest.model_fields
        }
        body = CreateSessionRequest.model_validate(
            {**session_params, "evaluation": evaluation, "extra_key": extra_key}
        )
        use_v2 = getattr(args, "use_session_server", None) == "v2"
        samples_wire_fields = COMPUTED_FIELDS_V2 if use_v2 else COMPUTED_FIELDS
        if should_return_sampling_mask(args, sampling_params, evaluation=evaluation):
            samples_wire_fields += ROLLOUT_SAMPLING_MASK_FIELDS
        response = await post(f"{session_url}/sessions", body.model_dump(exclude_none=True), action="post")
        session_id = response["session_id"]
        return OpenAIEndpointTracer(
            collection_concurrency=getattr(args, "session_collection_concurrency", 32),
            router_url=session_url,
            session_id=session_id,
            session_server_instance_id=instance.instance_id,
            samples_wire_fields=samples_wire_fields,
            agent_router_url=instance.external_url,
        )

    async def collect_samples(
        self,
        input_sample: Sample,
        *,
        max_seq_len: int | None,
        agent_metadata: dict | None = None,
        decode_executor=None,
        acknowledge: bool = True,
        timings: dict | None = None,
    ) -> SamplesReply:
        """Fetch server-assembled training samples for this session."""
        body: dict = {"max_seq_len": max_seq_len, "collection_id": self.collection_id}
        if agent_metadata is not None:
            body["metadata"] = agent_metadata
        queued = time.monotonic()
        transport = collection_transport(self.collection_concurrency)
        # Admission waits do not consume the network deadline, and include
        # decoding so completed buffers cannot accumulate without a bound.
        async with transport.admit() as client:
            admitted = time.monotonic()
            for attempt in range(3):
                try:
                    payload = await post_buffer_no_retry(
                        f"{self.base_url}/samples",
                        body,
                        timeout=_SESSION_REQUEST_TIMEOUT,
                        client=client,
                    )
                    break
                except (TimeoutError, httpx.TransportError):
                    if attempt == 2:
                        logger.error(
                            "Collection exhausted retries; retaining session %s for recovery", self.session_id
                        )
                        raise
                    await asyncio.sleep(0.5 * (2**attempt) + random.uniform(0, 0.25))
            received = time.monotonic()
            decode_call = partial(_timed_decode, payload, input_sample, self.samples_wire_fields, received)
            if decode_executor is None:
                result, timing = await asyncio.to_thread(decode_call)
            else:
                result, timing = await asyncio.get_running_loop().run_in_executor(decode_executor, decode_call)
            resumed = time.monotonic()
            if timings is not None:
                timings.update(
                    thread_queue_seconds=timing[0],
                    decode_cpu_seconds=timing[1],
                    decode_wall_seconds=timing[2],
                    loop_resume_seconds=resumed - timing[3],
                    transfer_seconds=received - admitted,
                    transport_queue_seconds=admitted - queued,
                )
            logger.info(
                "collection_decode session=%s thread_queue_s=%.6f cpu_s=%.6f execute_wall_s=%.6f loop_resume_s=%.6f",
                self.session_id,
                timing[0],
                timing[1],
                timing[2],
                resumed - timing[3],
            )
            logger.info(
                "session_collection_client session=%s queue_s=%.6f transfer_retry_s=%.6f decode_s=%.6f bytes=%d attempts=%d",
                self.session_id,
                admitted - queued,
                received - admitted,
                time.monotonic() - received,
                len(payload),
                attempt + 1,
            )
            if acknowledge:
                delete_started = time.monotonic()
                await self._acknowledge(client)
                if timings is not None:
                    timings["delete_seconds"] = time.monotonic() - delete_started
            return result

    async def acknowledge(self):
        async with collection_transport(self.collection_concurrency).admit() as client:
            await self._acknowledge(client)

    async def _acknowledge(self, client):
        started = time.monotonic()
        try:
            async with client.delete(
                self.base_url, timeout=aiohttp.ClientTimeout(total=_SESSION_ACK_TIMEOUT)
            ) as reply:
                if reply.status not in (204, 404):
                    logger.warning(
                        "Session acknowledgement failed: session=%s status=%d", self.session_id, reply.status
                    )
        except (aiohttp.ClientError, TimeoutError) as exc:
            logger.warning("Session acknowledgement failed: session=%s error=%r", self.session_id, exc)
        finally:
            logger.info("collection_delete session=%s seconds=%.6f", self.session_id, time.monotonic() - started)


def _timed_decode(payload, input_sample, fields, queued):
    started = time.monotonic()
    cpu_started = time.thread_time()
    result = decode_samples_and_merge_input_sample(payload, input_sample, fields=fields)
    cpu = time.thread_time() - cpu_started
    finished = time.monotonic()
    return result, (started - queued, cpu, finished - started, finished)
