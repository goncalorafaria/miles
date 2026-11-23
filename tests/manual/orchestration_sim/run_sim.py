"""End-to-end orchestration simulator for the TMax fully-async rollout (no GPU, no training).

Production code under test (whatever miles tree is first on PYTHONPATH):
  FullyAsyncRolloutFn -> generate_and_rm_group -> agentic_tool_call.generate
  -> subproc agent processes -> 32 session servers (v2, TITO, selected-token logprobs)
  -> miles router -> engines, then session collection back into the rollout process.

Simulated: engines (engine.py, load-independent latency), sandboxes and tool calls
(sim_agent.py, durations replayed from the production traces), and the trainer
(drains each batch immediately, publishing weight_version = step).

Reports engine occupancy (time-averaged running requests), episodes in flight,
agent-observed per-turn latency vs engine service time (= orchestration overhead on
the chat path), post-agent latency (collection), loop lag and throughput.
"""

import argparse
import asyncio
import json
import multiprocessing
import os
import random
import statistics
import subprocess
import sys
import tempfile
import threading
import time
import urllib.request
from pathlib import Path
from unittest.mock import patch

HERE = Path(__file__).resolve().parent


def _free_port():
    import socket

    s = socket.socket()
    s.bind(("127.0.0.1", 0))
    port = s.getsockname()[1]
    s.close()
    return port


def _wait(url, timeout=180):
    deadline = time.time() + timeout
    while time.time() < deadline:
        try:
            urllib.request.urlopen(url, timeout=2).read()
            return
        except Exception:
            time.sleep(0.5)
    raise TimeoutError(url)


def build_args(bench, prompt_path, router_port):
    from miles.utils.arguments import parse_args

    argv = [
        "sim",
        "--train-backend",
        "fsdp",
        "--no-enable-sample-ownership-checker",
        "--hf-checkpoint",
        bench.hf_checkpoint,
        "--prompt-data",
        str(prompt_path),
        "--input-key",
        "prompt",
        "--label-key",
        "label",
        "--metadata-key",
        "metadata",
        "--rollout-shuffle",
        "--num-rollout",
        "100000",
        "--rollout-batch-size",
        str(bench.batch),
        "--n-samples-per-prompt",
        str(bench.siblings),
        "--global-batch-size",
        str(bench.batch * bench.siblings),
        "--rollout-temperature",
        "1.0",
        "--rollout-max-response-len",
        "16384",
        "--max-seq-len",
        "67584",
        "--rollout-num-gpus",
        str(bench.engines),
        "--rollout-num-gpus-per-engine",
        "1",
        "--use-miles-router",
        "--sglang-router-ip",
        "127.0.0.1",
        "--sglang-router-port",
        str(router_port),
        "--custom-generate-function-path",
        "miles.rollout.generate_hub.agentic_tool_call.generate",
        "--custom-agent-function-path",
        "sim_agent.run",
        "--custom-agent-function-mode",
        bench.agent_mode,
        "--custom-rm-path",
        "sim_agent.reward",
        "--use-session-server",
        "v2",
        "--session-server-workers",
        str(bench.session_servers),
        "--tito-model",
        "qwen35",
        "--fully-async",
        "--max-weight-staleness",
        "32",
        "--async-data-buffer-capacity-factor",
        "4",
        "--async-max-concurrent-samples",
        str(bench.concurrency),
        *bench.extra.split(),
    ]
    with patch.object(sys, "argv", argv):
        return parse_args()


def start_processes(bench):
    procs = []
    engine_ports = [_free_port() for _ in range(bench.engines)]
    for i, port in enumerate(engine_ports):
        p = subprocess.Popen(
            [
                sys.executable,
                str(HERE / "engine.py"),
                bench.specs,
                str(port),
                str(bench.prefill_s),
                str(bench.token_s),
                str(bench.jitter),
                str(bench.max_running),
                str(bench.compress),
                str(i),
            ],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )
        procs.append(p)
    for port in engine_ports:
        _wait(f"http://127.0.0.1:{port}/health")
    return procs, engine_ports


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--label", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--hf-checkpoint", required=True, help="tokenizer of the Qwen3.5 TMax model (tito qwen35)")
    parser.add_argument("--specs", required=True, help="turn script from specs.py")
    parser.add_argument("--profiles", required=True, help="episode groups from make_profiles.py")
    parser.add_argument("--engines", type=int, default=8)
    parser.add_argument("--session-servers", type=int, default=32)
    parser.add_argument("--concurrency", type=int, default=1024)
    parser.add_argument("--batch", type=int, default=8)
    parser.add_argument("--siblings", type=int, default=32)
    parser.add_argument("--compress", type=float, default=1.0)
    parser.add_argument("--prefill-s", type=float, default=2.0)
    parser.add_argument("--token-s", type=float, default=0.02)
    parser.add_argument("--jitter", type=float, default=0.5)
    parser.add_argument("--max-running", type=int, default=0)
    parser.add_argument("--start-concurrency", type=int, default=64)
    parser.add_argument("--agent-mode", default="subproc")
    parser.add_argument("--duration", type=float, default=900)
    parser.add_argument("--warmup", type=float, default=300)
    parser.add_argument("--extra", default="")
    bench = parser.parse_args()

    os.environ["SIM_SPECS"] = bench.specs
    os.environ["SIM_COMPRESS"] = str(bench.compress)
    os.environ["SIM_START_CONCURRENCY"] = str(bench.start_concurrency)

    import ray

    ray.init(
        address="local",
        num_cpus=32,
        include_dashboard=False,
        logging_level="ERROR",
        object_store_memory=8 * 1024**3,
        runtime_env={"env_vars": {"PYTHONPATH": os.environ.get("PYTHONPATH", "")}},
    )

    from miles.rollout.base_types import RolloutFnConstructorInput, RolloutFnTrainInput
    from miles.rollout.inference_rollout.compatibility import call_rollout_function, load_rollout_function
    from miles.rollout.session.config import compute_session_server_config
    from miles.rollout.session.server import run_session_server
    from miles.rollout.session.types import SessionServerInstance
    from miles.router.config import compute_miles_router_config
    from miles.router.router import run_router
    from miles.utils import http_utils
    from miles.utils.async_utils import run, submit
    from miles.utils.function_registry import load_function

    groups = json.load(open(bench.profiles))
    random.Random(0).shuffle(groups)
    ctx = multiprocessing.get_context("spawn")
    children = []
    engine_procs, engine_ports = start_processes(bench)
    try:
        with tempfile.TemporaryDirectory(prefix="orch-sim-") as tmp:
            prompt_path = Path(tmp) / "prompts.jsonl"
            with prompt_path.open("w") as f:
                for i, g in enumerate(groups):
                    profiles = [{k: e[k] for k in ("turns", "start", "env", "tools")} for e in g]
                    f.write(
                        json.dumps(
                            {
                                "prompt": [{"role": "user", "content": f"task {i}"}],
                                "label": "",
                                "metadata": {"profiles": profiles, "task_id": f"t{i}"},
                            }
                        )
                        + "\n"
                    )
            router_port = _free_port()
            args = build_args(bench, prompt_path, router_port)
            router_cfg = compute_miles_router_config(
                args, host="127.0.0.1", port=router_port, num_engines=bench.engines
            )
            router = ctx.Process(target=run_router, args=(router_cfg,), daemon=True)
            router.start()
            children.append(router)
            _wait(f"http://127.0.0.1:{router_port}/list_workers")
            for port in engine_ports:
                req = urllib.request.Request(
                    f"http://127.0.0.1:{router_port}/add_worker?url=http://127.0.0.1:{port}", method="POST"
                )
                urllib.request.urlopen(req).read()
            instances = []
            for i in range(bench.session_servers):
                port = _free_port()
                cfg = compute_session_server_config(
                    args,
                    host="127.0.0.1",
                    port=port,
                    instance_id=f"sim-{i}",
                    backend_url=f"http://127.0.0.1:{router_port}",
                )
                p = ctx.Process(target=run_session_server, args=(cfg,), daemon=True)
                p.start()
                children.append(p)
                instances.append((port, SessionServerInstance(addr=f"127.0.0.1:{port}", instance_id=f"sim-{i}")))
            for port, _ in instances:
                _wait(f"http://127.0.0.1:{port}/health", timeout=300)
            args.session_server_instances = [inst for _, inst in instances]
            http_utils.init_http_client(args)
            rollout = load_rollout_function(
                RolloutFnConstructorInput(args=args, data_source=load_function(args.data_source_path)(args)),
                args.rollout_function_path,
            )
            roles = {
                "router": [router.pid],
                "session": [p.pid for p in children[1:]],
                "engine": [p.pid for p in engine_procs],
                "driver": [os.getpid()],
            }
            result = drive(
                bench, args, rollout, engine_ports, call_rollout_function, RolloutFnTrainInput, run, submit, roles
            )
            Path(bench.out).write_text(json.dumps(result, indent=1))
            print(json.dumps(result["summary"], indent=1), flush=True)
    except BaseException:
        import traceback

        traceback.print_exc()
        sys.stdout.flush()
        sys.stderr.flush()
    finally:
        for p in engine_procs:
            p.kill()
        for p in children:
            p.kill()
        ray.shutdown()
        os._exit(0)


def _role_cpu(roles):
    import psutil

    out = {}
    for role, pids in roles.items():
        per = []
        for pid in pids:
            try:
                t = psutil.Process(pid).cpu_times()
                per.append(t.user + t.system)
            except Exception:
                per.append(0.0)
        out[role] = per
    workers = []
    for proc in psutil.process_iter(["cmdline"]):
        try:
            cmd = " ".join(proc.info["cmdline"] or [])
            if "CollectionWorker" in cmd:
                t = proc.cpu_times()
                workers.append(t.user + t.system)
        except Exception:
            pass
    out["collector"] = workers
    out["agents"] = [_agent_cpu_seconds()]
    return out


def _agent_cpu_seconds():
    """CPU of every agent process so far: live ones (per-episode subprocesses or pool workers) plus the
    per-episode subprocesses this driver already reaped (the only children that exit during a run)."""
    import psutil

    me = psutil.Process()
    reaped = me.cpu_times().children_user + me.cpu_times().children_system
    live = 0.0
    for proc in me.children(recursive=True):
        try:
            cmd = " ".join(proc.cmdline())
            if "agent_function" in cmd or "agent_pool" in cmd or "miles-agent-worker" in cmd:
                t = proc.cpu_times()
                live += t.user + t.system + t.children_user + t.children_system
        except psutil.Error:
            pass
    return reaped + live


def _collection_state():
    """Samples whose agent has finished but whose collection has not: they still hold producer budget."""
    from miles.rollout.generate_utils import collection_transport as ct

    transport = ct._transports.get(asyncio.get_running_loop())
    if transport is None:
        return dict(collecting=0, collection_admitted=0)
    admitted = transport.concurrency - transport._slots._value
    return dict(collecting=transport.users, collection_admitted=admitted)


def drive(bench, args, rollout, engine_ports, call_rollout_function, RolloutFnTrainInput, run, submit, roles):
    observations = []
    steps = []
    stop = threading.Event()
    t_start = time.monotonic()

    def engine_stats():
        rows = []
        for port in engine_ports:
            rows.append(json.loads(urllib.request.urlopen(f"http://127.0.0.1:{port}/sim_metrics", timeout=10).read()))
        return rows

    async def rollout_state():
        fn = rollout
        sched = getattr(fn, "_scheduler", None)
        sem = fn.state.generate_fn_semaphore
        lag0 = time.perf_counter()
        await asyncio.sleep(0)
        return dict(
            samples_in_flight=getattr(sched, "samples_in_flight", None),
            running_groups=len(fn._running_tasks),
            gen_held=args.sglang_server_concurrency * args.rollout_num_gpus - sem._value
            if hasattr(sem, "_value")
            else None,
            loop_yield_s=time.perf_counter() - lag0,
            buffer=len(getattr(fn._output, "_buffer", [])),
            **_collection_state(),
        )

    def sampler():
        prev = None
        while not stop.wait(2.0):
            t = time.monotonic() - t_start
            try:
                es = engine_stats()
            except Exception:
                es = None
            try:
                t0 = time.perf_counter()
                rs = submit(rollout_state()).result(timeout=30)
                rs["loop_rtt_s"] = time.perf_counter() - t0
            except Exception as e:
                rs = {"error": repr(e)}
            row = dict(t=t, rollout=rs, cpu=_role_cpu(roles), load1=os.getloadavg()[0])
            if es is not None:
                cur = dict(
                    area=sum(e["area_running"] for e in es),
                    calls=sum(e["calls"] for e in es),
                    service=sum(e["service_s"] for e in es),
                    queue=sum(e["queue_s"] for e in es),
                    t=t,
                )
                row["engine_running_now"] = sum(e["running"] for e in es)
                row["engine_queued_now"] = sum(e["queued"] for e in es)
                if prev is not None:
                    dt = cur["t"] - prev["t"]
                    row["engine_running_avg"] = (cur["area"] - prev["area"]) / dt
                    row["calls_per_s"] = (cur["calls"] - prev["calls"]) / dt
                prev = cur
                row["engine_cum"] = cur
            observations.append(row)

    th = threading.Thread(target=sampler, daemon=True)
    th.start()
    step = 0
    samples_meta = []
    try:
        while time.monotonic() - t_start < bench.duration:
            s0 = time.monotonic()
            out = call_rollout_function(rollout, RolloutFnTrainInput(rollout_id=step, weight_version=step))
            groups = out.samples
            for group in groups:
                for traj in group:
                    for sample in traj if isinstance(traj, list) else [traj]:
                        md = sample.metadata or {}
                        am = md.get("agent_metrics") or (md.get("agent") or {}).get("agent_metrics") or {}
                        samples_meta.append(
                            dict(
                                t=time.monotonic() - t_start,
                                status=str(sample.status),
                                turns=am.get("sim_turns"),
                                query_s=am.get("sim_query_s"),
                                query_max_s=am.get("sim_query_max_s"),
                                start_s=am.get("sim_start_s"),
                                total_s=am.get("sim_total_s"),
                                post_agent_s=md.get("sim_post_agent_s"),
                                tokens=len(sample.tokens) if sample.tokens is not None else None,
                                response_length=sample.response_length,
                            )
                        )
            steps.append(
                dict(step=step, t=time.monotonic() - t_start, seconds=time.monotonic() - s0, groups=len(groups))
            )
            last = observations[-1] if observations else {}
            print(
                f"[{bench.label}] step {step} t={time.monotonic() - t_start:.0f}s {time.monotonic() - s0:.1f}s "
                f"groups={len(groups)} engine_running={last.get('engine_running_now')} "
                f"in_flight={last.get('rollout', {}).get('samples_in_flight')} "
                f"collecting={last.get('rollout', {}).get('collecting')}",
                flush=True,
            )
            del out, groups
            step += 1
    finally:
        stop.set()
        th.join(timeout=10)
    return dict(
        config=vars(bench),
        steps=steps,
        observations=observations,
        samples=samples_meta,
        summary=summarize(bench, steps, observations, samples_meta),
    )


def summarize(bench, steps, observations, samples):
    w = bench.warmup
    obs = [o for o in observations if o["t"] >= w and "engine_running_avg" in o]
    smp = [s for s in samples if s["t"] >= w]
    st_ = [s for s in steps if s["t"] >= w]
    span = (obs[-1]["t"] - obs[0]["t"]) if len(obs) > 1 else float("nan")
    calls = (obs[-1]["engine_cum"]["calls"] - obs[0]["engine_cum"]["calls"]) if len(obs) > 1 else 0
    svc = (obs[-1]["engine_cum"]["service"] - obs[0]["engine_cum"]["service"]) if len(obs) > 1 else 0

    def q(xs, p):
        return sorted(xs)[min(len(xs) - 1, int(p * len(xs)))] if xs else None

    ok = [s for s in smp if s["query_s"] is not None and s["turns"]]
    per_turn_query = [s["query_s"] / s["turns"] for s in ok]
    post = [s["post_agent_s"] for s in smp if s["post_agent_s"] is not None]
    rs = [o["rollout"] for o in obs if "samples_in_flight" in o.get("rollout", {})]
    cpu = {}
    if len(obs) > 1:
        for role in obs[0]["cpu"]:
            a, b = obs[0]["cpu"][role], obs[-1]["cpu"][role]
            if len(a) == len(b) and a:
                deltas = [(y - x) / span for x, y in zip(a, b, strict=True)]
                cpu[role] = dict(total_cores=sum(deltas), max_process_cores=max(deltas))
    return dict(
        cpu=cpu,
        host_load1_mean=statistics.mean(o.get("load1", 0) for o in obs) if obs else None,
        host_load1_max=max((o.get("load1", 0) for o in obs), default=None),
        label=bench.label,
        window_s=span,
        engine_running_mean=statistics.mean(o["engine_running_avg"] for o in obs) if obs else None,
        engine_calls_per_s=calls / span if span else None,
        engine_service_per_call_s=svc / calls if calls else None,
        agent_query_per_turn_s_mean=statistics.mean(per_turn_query) if per_turn_query else None,
        chat_path_overhead_per_turn_s=(statistics.mean(per_turn_query) - svc / calls)
        if per_turn_query and calls
        else None,
        post_agent_s_p50=q(post, 0.5),
        post_agent_s_p95=q(post, 0.95),
        post_agent_s_max=max(post) if post else None,
        samples_in_flight_mean=statistics.mean(
            r["samples_in_flight"] for r in rs if r["samples_in_flight"] is not None
        )
        if rs
        else None,
        running_groups_mean=statistics.mean(r["running_groups"] for r in rs) if rs else None,
        collecting_mean=statistics.mean(r.get("collecting", 0) for r in rs) if rs else None,
        collecting_max=max((r.get("collecting", 0) for r in rs), default=None),
        loop_rtt_p99_s=q([o["rollout"].get("loop_rtt_s", 0) for o in obs], 0.99),
        loop_rtt_max_s=max((o["rollout"].get("loop_rtt_s", 0) for o in obs), default=None),
        steps=len(st_),
        samples_per_s=len(smp) / span if span else None,
        aborted=sum("ABORTED" in s["status"] for s in smp),
        failed=sum("FAILED" in s["status"] for s in smp),
    )


if __name__ == "__main__":
    main()
