"""Paired full/basic diagnostic snapshots from one retained batch; no GPU required."""

import argparse
import asyncio
import copy
import json
import time
from pathlib import Path
from types import SimpleNamespace

import torch

from miles.ray.rollout import diagnostic_writer
from miles.utils.types import Sample


async def run(args):
    raw = torch.load(args.dump, map_location="cpu", weights_only=False)
    samples = [Sample.from_dict(s) for s in raw["samples"][: args.samples]]
    del raw
    original = diagnostic_writer._snapshot_samples
    rows = []
    for mode in ("full", "basic"):
        diagnostic_writer._snapshot_samples = copy.deepcopy if mode == "full" else original
        target = Path(args.output) / mode
        target.mkdir(parents=True, exist_ok=True)
        writer = diagnostic_writer.DiagnosticWriter(
            SimpleNamespace(save_debug_rollout_data=str(target / "{rollout_id}.pt"), save_debug_trajectory_data=None)
        )
        stalls = []

        async def heartbeat():
            while True:
                start = time.monotonic()
                await asyncio.sleep(0.01)
                stalls.append(max(0, time.monotonic() - start - 0.01))

        ticker = asyncio.create_task(heartbeat())
        await asyncio.sleep(0.02)
        start = time.monotonic()
        try:
            metrics = await writer.submit(samples, rollout_id=0, metadata={}, trainer_model_id=None)
            submit = time.monotonic() - start
            skip_start = time.monotonic()
            skipped = await writer.submit(samples, rollout_id=1, metadata={}, trainer_model_id=None)
            skip_time = time.monotonic() - skip_start
        finally:
            await writer.close()
        total = time.monotonic() - start
        await asyncio.sleep(0.02)
        ticker.cancel()
        await asyncio.gather(ticker, return_exceptions=True)
        row = dict(
            mode=mode,
            samples=len(samples),
            submit_seconds=submit,
            total_seconds=total,
            max_loop_stall_seconds=max(stalls),
            bytes=(target / "0.pt").stat().st_size,
            busy_submit_seconds=skip_time,
            skipped=skipped.get("rollout/diagnostics/skipped"),
            **metrics
        )
        saved = torch.load(target / "0.pt", weights_only=False)["samples"]
        for live, result in zip(samples, saved, strict=True):
            assert live.tokens == result["tokens"]
            assert live.reward == result["reward"]
            if mode == "basic":
                assert result["rollout_log_probs"] is None
                assert result["train_metadata"] is None
        del saved
        (target / "0.pt").unlink()
        rows.append(row)
        print(json.dumps(row), flush=True)
        (Path(args.output) / "results.json").write_text(json.dumps(rows, indent=2))
    diagnostic_writer._snapshot_samples = original


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dump", required=True)
    parser.add_argument("--samples", type=int, default=256)
    parser.add_argument("--output", required=True)
    asyncio.run(run(parser.parse_args()))
