"""The agent worker pool (``--custom-agent-function-mode pool``) against a mock SGLang backend.

The parity test pins the point of the pool: an episode run in a worker, with its session hosted
in-process, yields the same samples as the same episode through a standalone session server.
"""

import asyncio
import uuid
from types import SimpleNamespace

import aiohttp
import pytest
from tests.ci.ci_register import register_cpu_ci

from miles.rollout.agent_pool.pool import AgentPool, AgentWorkerDied, build_worker_specs
from miles.rollout.agent_pool.protocol import RunRequest
from miles.rollout.session.config import compute_session_server_config
from miles.rollout.session.samples.codec import COMPUTED_FIELDS_V2, decode_samples_and_merge_input_sample
from miles.rollout.session.server import SessionServer
from miles.utils.http_utils import find_available_port
from miles.utils.test_utils import agent_function_stubs
from miles.utils.test_utils.mock_sglang_server import ProcessResult, with_mock_server
from miles.utils.test_utils.uvicorn_thread_server import UvicornThreadServer
from miles.utils.types import Sample

# Measured locally: 95 s for the file (worker processes each load a tokenizer).
register_cpu_ci(est_time=120, suite="stage-b-cpu", labels=["rollout"])

_STUBS = "miles.utils.test_utils.agent_function_stubs"
_PROMPT = [{"role": "user", "content": "solve it"}]


def _args(agent: str) -> SimpleNamespace:
    return SimpleNamespace(
        miles_router_timeout=30,
        hf_checkpoint="Qwen/Qwen3-0.6B",
        chat_template_path=None,
        apply_chat_template_kwargs={"enable_thinking": False},
        tito_model="default",
        use_rollout_routing_replay=False,
        use_rollout_indexer_replay=False,
        use_sampling_support_replay=False,
        rollout_top_logprobs_num=0,
        rollout_sampling_logprobs_mode="selected",
        sglang_speculative_algorithm=None,
        num_layers=None,
        moe_router_topk=None,
        save_debug_trajectory_data=None,
        lora_rank=0,
        lora_adapter_path=None,
        use_session_server="v2",
        pause_generation_mode="retract",
        session_sample_picker_path="miles.rollout.session.v2.picker_hub.drop_same_prompt_retries",
        session_sample_postprocessor_path="miles.rollout.session.v2.postprocessor_hub.default_postprocess",
        custom_agent_function_path=f"{_STUBS}.{agent}",
    )


def _request(metadata: dict | None = None) -> RunRequest:
    return RunRequest(
        prompt=_PROMPT,
        request_kwargs={"max_tokens": 16},
        metadata=metadata or {},
        evaluation=False,
        session_params={},
        extra_key=None,
        max_seq_len=None,
        use_v2=True,
    )


def _pool(args, backend_url: str, *, workers: int) -> AgentPool:
    config = compute_session_server_config(args, host="127.0.0.1", port=0, instance_id=None, backend_url=backend_url)
    return AgentPool(build_worker_specs(args, session_config=config, num_workers=workers))


@pytest.fixture
def backend():
    with with_mock_server(process_fn=lambda prompt: ProcessResult(text="echo", finish_reason="stop")) as server:
        yield server


def _decode(buffer) -> list[Sample]:
    return decode_samples_and_merge_input_sample(buffer, Sample(), fields=COMPUTED_FIELDS_V2).samples


async def _via_session_server(args, backend_url: str) -> list[Sample]:
    port = find_available_port(32000)
    config = compute_session_server_config(
        args, host="127.0.0.1", port=port, instance_id=uuid.uuid4().hex, backend_url=backend_url
    )
    server = UvicornThreadServer(SessionServer(config).app, host="127.0.0.1", port=port)
    server.start()
    try:
        url = f"http://127.0.0.1:{port}"
        async with aiohttp.ClientSession() as http:
            async with http.post(f"{url}/sessions", json={}) as reply:
                session_id = (await reply.json())["session_id"]
            agent_metadata = await agent_function_stubs.two_turn_chat(
                base_url=f"{url}/sessions/{session_id}", prompt=_PROMPT, request_kwargs={"max_tokens": 16}, metadata={}
            )
            body = {"max_seq_len": None, "metadata": agent_metadata, "collection_id": "c"}
            async with http.post(f"{url}/sessions/{session_id}/samples", json=body) as reply:
                assert reply.status == 200, await reply.text()
                return _decode(await reply.read())
    finally:
        server.stop()


@pytest.mark.asyncio
async def test_pool_episode_matches_session_server_episode(backend):
    args = _args("two_turn_chat")
    expected = await _via_session_server(args, backend.url)
    pool = _pool(args, backend.url, workers=2)
    try:
        async with asyncio.timeout(120):
            reply, buffer = await pool.run(_request())
    finally:
        await pool.close()
    assert reply.ok and reply.collect_status == 200, reply
    assert reply.agent_metadata == {"reward": 0.5, "agent_metrics": {"turns": 2}}
    actual = _decode(buffer)
    assert len(actual) == len(expected) == 1
    for field in ("tokens", "loss_mask", "rollout_log_probs", "response", "response_length", "reward", "status"):
        assert getattr(actual[0], field) == getattr(expected[0], field), field
    assert _comparable(actual[0].metadata) == _comparable(expected[0].metadata)


def _comparable(metadata: dict) -> dict:
    """Metadata without what differs between any two runs: ids the backend draws and wall-clock times."""
    out = {k: v for k, v in metadata.items() if k not in ("session_server_instance_id", "session_id")}
    out["leaf"] = {k: v for k, v in out["leaf"].items() if k != "response_id"}
    out["lifecycle"] = [entry["turn"] for entry in out["lifecycle"]]
    return out


@pytest.mark.asyncio
async def test_pool_runs_many_episodes_concurrently_on_few_workers(backend):
    pool = _pool(_args("two_turn_chat"), backend.url, workers=2)
    try:
        async with asyncio.timeout(180):
            replies = await asyncio.gather(*(pool.run(_request()) for _ in range(24)))
    finally:
        await pool.close()
    assert all(reply.ok and reply.collect_status == 200 for reply, _ in replies)
    assert len({reply.session_id for reply, _ in replies}) == 24


@pytest.mark.asyncio
async def test_worker_death_fails_its_episodes_and_the_worker_restarts(backend):
    pool = _pool(_args("exit_process"), backend.url, workers=1)
    try:
        async with asyncio.timeout(120):
            with pytest.raises(AgentWorkerDied):
                await pool.run(_request())
            with pytest.raises(AgentWorkerDied):  # served by the restarted worker, which dies again
                await pool.run(_request())
    finally:
        await pool.close()


@pytest.mark.asyncio
async def test_cancelling_an_episode_cancels_it_in_the_worker(backend, tmp_path):
    marker, started = tmp_path / "cancelled", tmp_path / "started"
    pool = _pool(_args("block_until_pool_cancels"), backend.url, workers=1)
    try:
        episode = asyncio.create_task(
            pool.run(_request({"cancel_marker": str(marker), "started_marker": str(started)}))
        )
        async with asyncio.timeout(60):  # the worker loads its tokenizer before it starts the agent
            while not started.exists():
                await asyncio.sleep(0.1)
        episode.cancel()
        with pytest.raises(asyncio.CancelledError):
            await episode
        async with asyncio.timeout(30):
            while not marker.exists():
                await asyncio.sleep(0.1)
        assert marker.read_text() == "cleaned up"
    finally:
        await pool.close()
