"""Diagnostic writes preserve arrays and surface failures without owning live samples."""

import asyncio
from concurrent.futures import Future
from types import SimpleNamespace

import numpy as np
import pytest
import torch

from miles.ray.rollout.diagnostic_writer import DiagnosticWriter
from miles.utils.types import Sample


@pytest.mark.asyncio
async def test_real_process_snapshot_roundtrip(tmp_path):
    args = SimpleNamespace(save_debug_rollout_data=str(tmp_path / "{rollout_id}.pt"), save_debug_trajectory_data=None)
    sample = Sample(index=0, tokens=[1, 2], response_length=1)
    sample.metadata = {"ids": np.array([[3, 4]], dtype=np.int32), "values": np.array([[-1.0, -2.0]], dtype=np.float32)}
    writer = DiagnosticWriter(args)
    try:
        await writer.submit([sample], rollout_id=1, metadata={"marker": 7}, trainer_model_id=None)
        sample.tokens[0] = 99
        sample.metadata["ids"][:] = 99
    finally:
        await writer.close()
    data = torch.load(tmp_path / "1.pt", weights_only=False)
    assert data["metadata"] == {"marker": 7}
    assert data["samples"][0]["tokens"] == [1, 2]
    np.testing.assert_array_equal(data["samples"][0]["metadata"]["ids"], [[3, 4]])
    np.testing.assert_array_equal(data["samples"][0]["metadata"]["values"], [[-1.0, -2.0]])


@pytest.mark.asyncio
async def test_busy_writer_drops_before_touching_samples():
    writer = DiagnosticWriter(SimpleNamespace(save_debug_rollout_data="unused", save_debug_trajectory_data=None))
    pending = Future()
    writer._pending = pending
    metrics = await asyncio.wait_for(
        writer.submit([_MustNotCopy()], rollout_id=1, metadata={}, trainer_model_id=None), timeout=1
    )
    assert metrics["rollout/diagnostics/dropped_total"] == 1
    assert metrics["rollout/diagnostics/skipped"] == 1
    assert writer._pool is None
    assert not pending.cancelled()
    pending.set_result(1.0)
    await writer.close()


@pytest.mark.asyncio
async def test_optional_write_failure_is_logged_without_failing_training(caplog):
    writer = DiagnosticWriter(SimpleNamespace(save_debug_rollout_data="unused", save_debug_trajectory_data=None))
    writer._pending = Future()
    writer._pending.set_exception(OSError("disk full"))
    await writer.close()
    assert "disk full" in caplog.text
    assert writer._failures == 1


class _MustNotCopy:
    def __deepcopy__(self, memo):
        raise AssertionError("Training arrays must not be copied into diagnostics")


@pytest.mark.asyncio
async def test_basic_snapshot_omits_training_arrays_without_touching_live_sample(tmp_path):
    args = SimpleNamespace(save_debug_rollout_data=str(tmp_path / "{rollout_id}.pt"), save_debug_trajectory_data=None)
    sample = Sample(index=2, tokens=[1, 2], response_length=1, reward=0.5, metadata={"task": "example"})
    payload = _MustNotCopy()
    sample.rollout_log_probs = payload
    sample.rollout_routed_experts = payload
    sample.train_metadata = {"large": payload}
    writer = DiagnosticWriter(args)
    try:
        await writer.submit([sample], rollout_id=2, metadata={}, trainer_model_id=None)
    finally:
        await writer.close()
    saved = torch.load(tmp_path / "2.pt", weights_only=False)["samples"][0]
    assert saved["tokens"] == [1, 2]
    assert saved["reward"] == 0.5
    assert saved["metadata"] == {"task": "example"}
    assert saved["rollout_log_probs"] is None
    assert saved["rollout_routed_experts"] is None
    assert saved["train_metadata"] is None
    assert sample.rollout_log_probs is payload
    assert sample.rollout_routed_experts is payload
