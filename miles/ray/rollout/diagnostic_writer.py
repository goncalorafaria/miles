"""One immutable diagnostic snapshot at a time, written outside the executor loop."""

import asyncio
import copy
import logging
import multiprocessing
import time
from concurrent.futures import ProcessPoolExecutor
from types import SimpleNamespace

from miles.ray.rollout.debug_data import save_debug_rollout_data
from miles.utils.types import Sample

logger = logging.getLogger(__name__)


def _snapshot_samples(samples):
    """Copy diagnostic fields only; never traverse training/replay arrays."""
    fields = (
        "index",
        "group_index",
        "rollout_id",
        "prompt",
        "response",
        "response_length",
        "label",
        "reward",
        "status",
        "metadata",
    )
    return [
        Sample(
            **copy.deepcopy({name: getattr(sample, name) for name in fields}),
            # Flat integer lists need a new container, not recursive traversal.
            tokens=list(sample.tokens) if sample.tokens is not None else None,
            loss_mask=list(sample.loss_mask) if sample.loss_mask is not None else None,
        )
        for sample in samples
    ]


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
        self._dropped = 0
        self._failures = 0

    async def submit(self, samples, *, rollout_id, metadata, trainer_model_id):
        async with self._lock:
            started = time.monotonic()
            metrics = {}
            if self._pending is not None and not self._pending.done():
                self._dropped += 1
                return {"rollout/diagnostics/dropped_total": self._dropped, "rollout/diagnostics/skipped": 1}
            if self._pending is not None:
                try:
                    metrics["rollout/diagnostics/write_seconds"] = self._pending.result()
                except Exception:
                    self._failures += 1
                    logger.exception("Optional diagnostic write failed")
                self._pending = None
            metrics["rollout/diagnostics/dropped_total"] = self._dropped
            metrics["rollout/diagnostics/write_failures_total"] = self._failures
            metrics["rollout/diagnostics/skipped"] = 0
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
            try:
                self._pending = self._pool.submit(_write, self._args, snapshot, rollout_id, metadata, trainer_model_id)
            except Exception:
                self._failures += 1
                metrics["rollout/diagnostics/write_failures_total"] = self._failures
                logger.exception("Optional diagnostic submission failed")
            return metrics

    async def close(self):
        async with self._lock:
            try:
                if self._pending is not None:
                    try:
                        await asyncio.shield(asyncio.wrap_future(self._pending))
                    except Exception:
                        self._failures += 1
                        logger.exception("Optional diagnostic write failed during shutdown")
                    self._pending = None
            finally:
                if self._pool is not None:
                    await asyncio.to_thread(self._pool.shutdown, wait=True)
                    self._pool = None
