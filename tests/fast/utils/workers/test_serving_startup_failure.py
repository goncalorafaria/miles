"""A confirmed Ray worker death must escape the initial serving barrier."""

import asyncio
from types import SimpleNamespace

import pytest
from tests.fast.ray.rollout.conftest import make_args
from tests.fast.utils.workers.fake_ray import FakeRayCluster
from tests.fast.utils.workers.test_ray_worker_manager_liveness import (
    _kill_worker_process,
    _launch,
    _make_spec,
    _scan_all_live_cells,
)

from miles.ray.rollout import rollout_server
from miles.ray.rollout.inference_controller import InferenceController
from miles.ray.rollout.rollout_server import RolloutServer
from miles.utils.context_lock import ContextLock
from miles.utils.workers.types import WorkerCommBackend
from miles.utils.workers.worker_provider.base import CellInfo
from miles.utils.workers.worker_provider.ray import RayWorkerProvider


def _provider(get_infos):
    async def remote(*, pool_ids):
        return get_infos(pool_ids=pool_ids)

    return RayWorkerProvider(SimpleNamespace(get_cell_infos=SimpleNamespace(remote=remote)), pool_ids=["engine"])


def _server(provider, *, name="default", cells=None):
    return RolloutServer(
        server_cells=cells or {},
        args=make_args(colocate=False),
        context_lock=ContextLock("InferenceController"),
        engine_provider=provider,
        model_name=name,
        init_expected_num_cells=1,
    )


async def test_confirmed_death_reaches_startup_barrier_and_restart_clears_it(fake_ray_cluster: FakeRayCluster):
    spec = _make_spec("engine").model_copy(update={"meta": lambda _: {"model_id": "default"}})
    manager = await _launch([spec], comm_backend=WorkerCommBackend.RPC)
    provider = _provider(manager.get_cell_infos)
    await provider.check_startup_failure(group_id="default")
    _kill_worker_process(fake_ray_cluster, handle_index=0)
    await _scan_all_live_cells(manager)
    info = manager.get_cell_infos(pool_ids=["engine"])["engine-00000"]
    assert not info.alive
    assert "exited unexpectedly" in info.failure_reason
    # This failed worker can disappear before the controller ever sees a live
    # cell. Querying the manager's retained failure still catches it.
    srv = _server(provider)
    with pytest.raises(RuntimeError, match="Serving startup failed.*engine-00000"):
        await asyncio.wait_for(srv.wait_init_expected_num_cells(check_worker_failures=True), timeout=1)
    await manager.start_cells(["engine-00000"])
    assert manager.get_cell_infos(pool_ids=["engine"])["engine-00000"].failure_reason is None
    await provider.check_startup_failure(group_id="default")
    await manager.stop_cells(["engine-00000"])
    await provider.check_startup_failure(group_id="default")


@pytest.mark.parametrize(
    "alive,reason,model",
    [
        (False, None, "default"),  # Pending/unlaunched is not a confirmed death.
        (False, "worker died", "other"),
        (True, "stale failure", "default"),
    ],
)
async def test_pending_or_unrelated_cells_do_not_abort_startup(alive, reason, model):
    info = CellInfo(
        cell_id="engine-0",
        pool_id="engine",
        alive=alive,
        worker_names=[],
        workers_hash="v1",
        meta={"model_id": model},
        failure_reason=reason,
    )
    await _provider(lambda **_: {info.cell_id: info}).check_startup_failure(group_id="default")


async def test_known_failure_after_polling_is_not_retried(monkeypatch):
    monkeypatch.setattr(rollout_server, "WAIT_CELLS_INITIAL_DELAY_SECONDS", 0)
    monkeypatch.setattr(rollout_server, "WAIT_CELLS_MAX_DELAY_SECONDS", 0)
    attempts = []

    async def check_startup_failure(**_):
        attempts.append(None)
        if len(attempts) == 2:
            raise RuntimeError("worker died after startup began")

    srv = _server(SimpleNamespace(check_startup_failure=check_startup_failure))
    with pytest.raises(RuntimeError, match="worker died after startup began"):
        await asyncio.wait_for(srv.wait_init_expected_num_cells(check_worker_failures=True), timeout=1)
    assert len(attempts) == 2


async def test_takeover_wait_does_not_apply_initial_failure_policy(monkeypatch):
    monkeypatch.setattr(rollout_server, "WAIT_CELLS_INITIAL_DELAY_SECONDS", 0)
    monkeypatch.setattr(rollout_server, "WAIT_CELLS_MAX_DELAY_SECONDS", 0)

    async def check_startup_failure(**_):
        raise AssertionError("takeover must allow a worker replacement")

    srv = _server(SimpleNamespace(check_startup_failure=check_startup_failure))
    waiting = asyncio.create_task(srv.wait_init_expected_num_cells())
    await asyncio.sleep(0)
    assert not waiting.done()
    srv.server_cells["replacement"] = SimpleNamespace(is_pending_weights_or_serving=True)
    await asyncio.wait_for(waiting, timeout=1)


async def test_controller_cancels_other_model_waits_on_failure():
    started, cancelled = asyncio.Event(), asyncio.Event()

    async def waiting(**kwargs):
        assert kwargs["check_worker_failures"]
        started.set()
        try:
            await asyncio.Event().wait()
        finally:
            cancelled.set()

    async def failed(**kwargs):
        await started.wait()
        raise RuntimeError("default engine died")

    controller = InferenceController(make_args(), engine_provider=None, router_providers=[])
    controller.servers = {
        "default": SimpleNamespace(wait_init_expected_num_cells=failed),
        "eval": SimpleNamespace(wait_init_expected_num_cells=waiting),
    }
    with pytest.raises(RuntimeError, match="default engine died"):
        await asyncio.wait_for(controller.wait_expected_num_cells(check_worker_failures=True), timeout=1)
    assert cancelled.is_set()


def test_older_cell_payload_remains_readable():
    info = CellInfo.model_validate(
        {
            "cell_id": "engine-0",
            "pool_id": "engine",
            "alive": False,
            "worker_names": [],
            "workers_hash": "v1",
            "meta": {},
        }
    )
    assert info.failure_reason is None
