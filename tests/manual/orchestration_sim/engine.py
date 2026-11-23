"""Simulated SGLang engine: serves pre-rendered TMax chat completions after a modeled delay.

The model is deliberately simple and load-independent unless ``max_running`` is set:
each request holds a slot for ``prefill_s + completion_tokens * token_s`` (scaled by the
time compression), FIFO-queued beyond ``max_running``. The engine therefore adds no
contention of its own; any extra latency the agent observes comes from the path between
the agent and this process (session server, router, rollout process).

GET /sim_metrics returns running/queued counts and cumulative service statistics.
"""

import asyncio
import json
import pickle
import random
import sys
import time

import orjson
from aiohttp import web


def main(specs_path, port, prefill_s, token_s, jitter, max_running, compress, seed):
    specs = pickle.load(open(specs_path, "rb"))
    bodies = [s["response_body"] for s in specs]
    tokens = [s["completion_tokens"] for s in specs]
    rng = random.Random(seed)
    slots = asyncio.Semaphore(max_running) if max_running > 0 else None
    stats = dict(running=0, queued=0, calls=0, service_s=0.0, queue_s=0.0, area_running=0.0, area_t0=time.monotonic())

    def account():
        now = time.monotonic()
        stats["area_running"] += stats["running"] * (now - stats["area_t0"])
        stats["area_t0"] = now

    async def chat(request):
        body = await request.read()
        payload = orjson.loads(body)
        turn = min((len(payload["messages"]) - 1) // 2, len(bodies) - 1)
        service = (prefill_s + tokens[turn] * token_s) * rng.lognormvariate(0, jitter) / compress
        queued_at = time.monotonic()
        stats["queued"] += 1
        if slots is not None:
            await slots.acquire()
        try:
            stats["queued"] -= 1
            account()
            stats["running"] += 1
            stats["queue_s"] += time.monotonic() - queued_at
            await asyncio.sleep(service)
        finally:
            account()
            stats["running"] -= 1
            if slots is not None:
                slots.release()
        stats["calls"] += 1
        stats["service_s"] += service
        return web.Response(body=bodies[turn], content_type="application/json")

    async def metrics(_):
        account()
        return web.Response(text=json.dumps(stats), content_type="application/json")

    async def ok(_):
        return web.json_response({"status": "ok"})

    app = web.Application(client_max_size=1 << 30)
    app.router.add_post("/v1/chat/completions", chat)
    app.router.add_get("/sim_metrics", metrics)
    app.router.add_get("/health", ok)
    app.router.add_get("/health_generate", ok)
    app.router.add_post("/abort_request", ok)
    app.router.add_get("/get_server_info", ok)
    web.run_app(app, host="127.0.0.1", port=port, access_log=None, print=None, keepalive_timeout=600)


if __name__ == "__main__":
    a = sys.argv[1:]
    main(a[0], int(a[1]), float(a[2]), float(a[3]), float(a[4]), int(a[5]), float(a[6]), int(a[7]))
