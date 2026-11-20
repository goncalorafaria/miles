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
    sample.rollout_topk_token_ids = np.array([[3, 4]], dtype=np.int32)
    sample.rollout_topk_log_probs = np.array([[-1.0, -2.0]], dtype=np.float32)
    writer = DiagnosticWriter(args)
    try:
        await writer.submit([sample], rollout_id=1, metadata={"marker": 7}, trainer_model_id=None)
        sample.tokens[0] = 99
        sample.rollout_topk_token_ids[:] = 99
    finally:
        await writer.close()
    data = torch.load(tmp_path / "1.pt", weights_only=False)
    assert data["metadata"] == {"marker": 7}
    assert data["samples"][0]["tokens"] == [1, 2]
    np.testing.assert_array_equal(data["samples"][0]["rollout_topk_token_ids"], [[3, 4]])
    np.testing.assert_array_equal(data["samples"][0]["rollout_topk_log_probs"], [[-1.0, -2.0]])


@pytest.mark.asyncio
async def test_backpressure_yields_and_does_not_submit_a_second_write():
    writer = DiagnosticWriter(SimpleNamespace(save_debug_rollout_data="unused", save_debug_trajectory_data=None))
    pending = Future()
    writer._pending = pending
    waiting = asyncio.create_task(writer.submit([], rollout_id=1, metadata={}, trainer_model_id=None))
    await asyncio.sleep(0)
    assert not waiting.done()
    assert writer._pool is None
    waiting.cancel()
    with pytest.raises(asyncio.CancelledError):
        await waiting
    assert not pending.cancelled()
    pending.set_result(1.0)
    await writer.close()


@pytest.mark.asyncio
async def test_write_failure_is_not_silently_lost():
    writer = DiagnosticWriter(SimpleNamespace(save_debug_rollout_data="unused", save_debug_trajectory_data=None))
    writer._pending = Future()
    writer._pending.set_exception(OSError("disk full"))
    with pytest.raises(OSError, match="disk full"):
        await writer.close()
