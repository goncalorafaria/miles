"""One immutable diagnostic snapshot at a time, written outside the executor loop."""

import asyncio
import copy
import multiprocessing
import time
from concurrent.futures import ProcessPoolExecutor
from types import SimpleNamespace

from miles.ray.rollout.debug_data import save_debug_rollout_data
from miles.utils.types import Sample


def _snapshot_samples(samples):
    """Copy diagnostic fields only; never traverse training/replay arrays."""
    fields = (
        "index",
        "group_index",
        "rollout_id",
        "prompt",
        "tokens",
        "response",
        "response_length",
        "label",
        "reward",
        "loss_mask",
        "status",
        "metadata",
    )
    return [Sample(**copy.deepcopy({name: getattr(sample, name) for name in fields})) for sample in samples]


def _write(args, samples, rollout_id, metadata, trainer_model_id):
    started = time.monotonic()
    save_debug_rollout_data(args, samples, rollout_id, False, metadata, trainer_model_id)
    return time.monotonic() - started


class DiagnosticWriter:
    def __init__(self, args):
        self._args = SimpleNamespace(
            save_debug_rollout_data=args.save_debug_rollout_data,
            save_debug_trajectory_data=args.save_debug_trajectory_data,
        )
        self._pool = None
        self._pending = None
        self._lock = asyncio.Lock()

    async def submit(self, samples, *, rollout_id, metadata, trainer_model_id):
        async with self._lock:
            started = time.monotonic()
            metrics = {}
            if self._pending is not None:
                # Shield the writer: cancelling the caller must not lose write errors.
                metrics["rollout/diagnostics/write_seconds"] = await asyncio.shield(asyncio.wrap_future(self._pending))
                self._pending = None
            metrics["rollout/diagnostics/backpressure_seconds"] = time.monotonic() - started
            started = time.monotonic()
            # Hooks may mutate Samples after this returns. Never write live objects.
            copying = asyncio.create_task(asyncio.to_thread(_snapshot_samples, samples))
            try:
                snapshot = await asyncio.shield(copying)
                metadata = copy.deepcopy(metadata)
            except asyncio.CancelledError:
                # Do not release ownership while a thread still reads these Samples.
                await copying
                raise
            metrics["rollout/diagnostics/snapshot_seconds"] = time.monotonic() - started
            if self._pool is None:
                self._pool = ProcessPoolExecutor(max_workers=1, mp_context=multiprocessing.get_context("spawn"))
            self._pending = self._pool.submit(_write, self._args, snapshot, rollout_id, metadata, trainer_model_id)
            return metrics

    async def close(self):
        async with self._lock:
            try:
                if self._pending is not None:
                    await asyncio.shield(asyncio.wrap_future(self._pending))
                    self._pending = None
            finally:
                if self._pool is not None:
                    await asyncio.to_thread(self._pool.shutdown, wait=True)
                    self._pool = None
