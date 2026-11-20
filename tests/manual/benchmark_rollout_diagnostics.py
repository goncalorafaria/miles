"""Replay saved rollout diagnostics; measure loop stalls, not training throughput."""

import argparse
import asyncio
import json
import time
from pathlib import Path
from types import SimpleNamespace

import torch

from miles.ray.rollout.debug_data import save_debug_rollout_data
from miles.ray.rollout.diagnostic_writer import DiagnosticWriter
from miles.utils.types import Sample


async def run(args):
    raw = torch.load(args.dump, map_location="cpu", weights_only=False)
    samples = [Sample.from_dict(s) for s in raw["samples"][: args.samples]]
    del raw
    rows = []
    for mode in ("sync_protocol2", "sync_protocol5", "background_protocol5"):
        folder = Path(args.output) / mode
        folder.mkdir(parents=True, exist_ok=True)
        config = SimpleNamespace(
            save_debug_rollout_data=str(folder / "{rollout_id}.pt"), save_debug_trajectory_data=None
        )
        lag = []

        async def heartbeat(lag=lag):
            while True:
                t = time.monotonic()
                await asyncio.sleep(0.01)
                lag.append(max(0.0, time.monotonic() - t - 0.01))

        ticker = asyncio.create_task(heartbeat())
        await asyncio.sleep(0.02)
        started = time.monotonic()
        if mode == "background_protocol5":
            writer = DiagnosticWriter(config)
            try:
                metrics = await writer.submit(samples, rollout_id=0, metadata={}, trainer_model_id=None)
                submitted = time.monotonic() - started
            finally:
                await writer.close()
        else:
            original_save = torch.save
            if mode == "sync_protocol2":

                def old_save(*a, _save=original_save, **kw):
                    kw["pickle_protocol"] = 2
                    return _save(*a, **kw)

                torch.save = old_save
            try:
                save_debug_rollout_data(config, samples, 0, False)
            finally:
                torch.save = original_save
            submitted = time.monotonic() - started
            metrics = {}
        elapsed = time.monotonic() - started
        await asyncio.sleep(0.02)
        ticker.cancel()
        await asyncio.gather(ticker, return_exceptions=True)
        row = dict(
            mode=mode,
            samples=len(samples),
            submit_seconds=submitted,
            total_seconds=elapsed,
            max_loop_stall_seconds=max(lag),
            bytes=(folder / "0.pt").stat().st_size,
            **metrics
        )
        rows.append(row)
        print(json.dumps(row), flush=True)
        # Compare complete saved samples outside timing; remove generated large files afterwards.
        loaded = torch.load(folder / "0.pt", weights_only=False)["samples"]
        import numpy as np

        for want, got in zip(samples, loaded, strict=True):
            assert want.tokens == got["tokens"]
            np.testing.assert_array_equal(want.rollout_topk_token_ids, got["rollout_topk_token_ids"])
            np.testing.assert_array_equal(want.rollout_topk_log_probs, got["rollout_topk_log_probs"])
        del loaded
        (folder / "0.pt").unlink()
        (Path(args.output) / "results.json").write_text(json.dumps(rows, indent=2))


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--dump", required=True)
    p.add_argument("--samples", type=int, default=32)
    p.add_argument("--output", required=True)
    asyncio.run(run(p.parse_args()))
