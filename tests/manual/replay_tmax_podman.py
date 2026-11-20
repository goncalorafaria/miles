"""Replay recorded TMax bash calls against real, fresh gateway containers.

This measures infrastructure load, not policy quality: actions remain fixed even
when live observations differ. No model inference or training is performed.
Run with PYTHONPATH=.:examples/tmax and the normal gateway client installed.
"""

import argparse
import asyncio
import json
import os
import time
import uuid
from pathlib import Path

from tmax_gateway import TmaxGatewaySandbox


def load_traces(path, count):
    paths = sorted(path.glob("traces-*.jsonl")) if path.is_dir() else [path]
    records = []
    for source in paths:
        with source.open() as stream:
            for line in stream:
                trace = json.loads(line)
                turns = []
                for message in trace.get("messages", []):
                    if message.get("role") != "assistant":
                        continue
                    commands = []
                    for call in message.get("tool_calls") or []:
                        function = call.get("function", {})
                        if function.get("name") != "bash":
                            continue
                        args = function.get("arguments", {})
                        try:
                            args = json.loads(args) if isinstance(args, str) else args
                            command = args.get("command")
                        except (ValueError, AttributeError):
                            continue
                        if isinstance(command, str):
                            commands.append(command)
                    turns.append(commands)
                if any(turns) and trace.get("image") and trace.get("task_id"):
                    records.append((trace, turns))
                if len(records) >= count:
                    return records
    return records


async def replay(args):
    # Isolate locks and container ownership from evaluation/training processes.
    os.environ["MILES_ROLLOUT_LIMITS_SCOPE"] = "collection-replay-" + uuid.uuid4().hex
    os.environ["TMAX_GATEWAY_URL"] = args.gateway_url
    records = load_traces(args.traces, args.episodes)
    if len(records) < args.episodes:
        raise ValueError(f"Only {len(records)} eligible traces, requested {args.episodes}")
    slots = asyncio.Semaphore(args.concurrency)
    output = args.output.open("w")
    started = time.perf_counter()

    async def episode(index, trace, turns):
        queued = time.perf_counter()
        async with slots:
            t = time.perf_counter()
            row = dict(
                index=index,
                task_id=trace["task_id"],
                queue_s=t - queued,
                tool_calls=0,
                tool_s=0.0,
                simulated_model_wait_s=0.0,
                recorded_reward=trace.get("reward"),
                status="started",
            )
            # Traces only have aggregate model duration, not per-turn timestamps.
            delay = (
                args.model_delay_scale * trace.get("metrics", {}).get("model_query_time_sum", 0) / max(1, len(turns))
            )
            phase = "startup"
            try:
                async with TmaxGatewaySandbox(trace["image"], str(args.task_data / trace["task_id"])) as box:
                    row["startup_s"] = time.perf_counter() - t
                    row["instance_id"] = box.instance_id
                    for commands in turns:
                        phase = "model_delay"
                        if delay:
                            await asyncio.sleep(delay)
                            row["simulated_model_wait_s"] += delay
                        submitted = False
                        for command in commands:
                            phase = "tool"
                            before = time.perf_counter()
                            _, submitted = await box.bash(command)
                            row["tool_s"] += time.perf_counter() - before
                            row["tool_calls"] += 1
                            if submitted:
                                phase = "verifier"
                                before = time.perf_counter()
                                row["reward"] = await box.run_tests()
                                row["verifier_s"] = time.perf_counter() - before
                                break
                        if submitted:
                            break
                    phase = "cleanup"
                row["status"] = "completed"
            except Exception as error:
                row.update(status="error", failed_phase=phase, error_type=type(error).__name__, error=str(error))
            finally:
                row["elapsed_s"] = time.perf_counter() - t
                output.write(json.dumps(row) + "\n")
                output.flush()
                print(json.dumps({k: v for k, v in row.items() if k != "error"}), flush=True)

    try:
        await asyncio.gather(*(episode(i, *record) for i, record in enumerate(records)))
    finally:
        output.close()
    print(
        json.dumps(
            dict(
                episodes=len(records),
                total_s=time.perf_counter() - started,
                scope="real Podman, recorded actions; no session collection yet",
            )
        ),
        flush=True,
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--traces", type=Path, required=True)
    parser.add_argument("--task-data", type=Path, required=True)
    parser.add_argument("--gateway-url", required=True)
    parser.add_argument("--episodes", type=int, default=2)
    parser.add_argument("--concurrency", type=int, default=2)
    parser.add_argument("--model-delay-scale", type=float, default=1.0)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.episodes < 1 or args.concurrency < 1 or args.model_delay_scale < 0:
        parser.error("episodes/concurrency must be positive and delay scale nonnegative")
    asyncio.run(replay(args))
