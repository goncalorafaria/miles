"""Time one chat turn at the engine, the router and a session server of a live run_sim system.

python probe.py <session_port> <router_port> <engine_port> <specs.pkl>
"""

import json
import pickle
import sys
import time
import urllib.request

ss, router, engine, specs_path = sys.argv[1:5]
specs = pickle.load(open(specs_path, "rb"))
msgs = specs[0]["messages"]
tools = [
    {
        "type": "function",
        "function": {
            "name": "bash",
            "description": "Run one bash command.",
            "parameters": {"type": "object", "properties": {"command": {"type": "string"}}, "required": ["command"]},
        },
    }
]
body = json.dumps({"model": "m", "messages": msgs, "tools": tools}).encode()


def post(url, data):
    t = time.perf_counter()
    r = urllib.request.urlopen(
        urllib.request.Request(url, data=data, headers={"content-type": "application/json"}), timeout=600
    )
    r.read()
    return time.perf_counter() - t


for _ in range(3):
    engine_s = post(f"http://127.0.0.1:{engine}/v1/chat/completions", body)
    router_s = post(f"http://127.0.0.1:{router}/v1/chat/completions", body)
    t = time.perf_counter()
    request = urllib.request.Request(
        f"http://127.0.0.1:{ss}/sessions", data=b"{}", headers={"content-type": "application/json"}
    )
    sid = json.loads(urllib.request.urlopen(request).read())["session_id"]
    create_s = time.perf_counter() - t
    chat_s = post(f"http://127.0.0.1:{ss}/sessions/{sid}/v1/chat/completions", body)
    print(f"engine {engine_s:.3f}s  router {router_s:.3f}s  session create {create_s:.3f}s  chat {chat_s:.3f}s")
