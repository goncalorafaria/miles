"""Wait for this experiment's search services and print their gateway's URL (tmax_beaker_entry.sh exports it as
SEARCH_GATEWAY_URL).

    python examples/tmax_search/beaker/wait_for_services.py <endpoint-files dir> <search service> <judge model> [timeout s]

The gateway publishes ``endpoint_gateway_*`` records in the head registry (literegistry.coop.endpoints: name, uri,
published_at; refreshed every 10 s, stale after 30 s) only while it answers its health check. Once a fresh record's
gateway answers, every service is probed through it, as Prime-RL's own readiness gates do
(prime_rl_beaker/coordination.py probe-gateway, service_guard.py): a search query, a terminal pipeline and a judge
completion. The judge loads a 27B model, so this can take tens of minutes.

Exits 1 after the timeout (default 7200 s): the Miles task then fails, and Beaker's failure propagation stops the
service tasks.
"""

import json
import sys
import time
import urllib.request
from pathlib import Path

_FRESH_S = 30


def gateway(endpoint_dir: Path) -> str | None:
    """The newest fresh gateway record's URI."""
    records = []
    for path in endpoint_dir.glob("endpoint_gateway_*"):
        try:
            records.append(json.loads(path.read_text()))
        except (OSError, json.JSONDecodeError):
            continue
    fresh = [r for r in records if time.time() - float(r.get("published_at", 0)) < _FRESH_S]
    return max(fresh, key=lambda r: r["published_at"])["uri"].rstrip("/") if fresh else None


def post(url: str, body: dict, timeout: float = 60) -> dict:
    request = urllib.request.Request(url, json.dumps(body).encode(), {"content-type": "application/json"})
    with urllib.request.urlopen(request, timeout=timeout) as response:
        return json.load(response)


def ready(url: str, search_service: str, judge_model: str) -> str | None:
    """None when every service answers, else what does not."""
    probes = {
        "search": lambda: post(
            f"{url}/search", {"mode": "query", "query": "Barack Obama", "num_results": 1, "model_path": search_service}
        )["success"],
        "terminal": lambda: post(f"{url}/terminal", {"contents": "a\nb", "command": "wc -l", "truncation": 100})[
            "success"
        ],
        "judge": lambda: post(
            f"{url}/v1/chat/completions",
            {"model": judge_model, "messages": [{"role": "user", "content": "Say OK."}], "max_tokens": 8},
            timeout=300,
        )["choices"],
    }
    for name, probe in probes.items():
        try:
            if not probe():
                return name
        except Exception as e:  # noqa: BLE001 - not up yet
            return f"{name} ({type(e).__name__}: {str(e)[:200]})"
    return None


def main() -> None:
    endpoint_dir, search_service, judge_model = Path(sys.argv[1]), sys.argv[2], sys.argv[3]
    deadline = time.time() + float(sys.argv[4] if len(sys.argv) > 4 else 7200)
    last_report = 0.0
    while time.time() < deadline:
        url = gateway(endpoint_dir)
        missing = "gateway record" if url is None else ready(url, search_service, judge_model)
        if missing is None:
            print(url)
            return
        if time.time() - last_report > 120:
            print(f"waiting for the search services: {missing}", file=sys.stderr, flush=True)
            last_report = time.time()
        time.sleep(10)
    print(f"the search services were not ready in time ({missing})", file=sys.stderr)
    sys.exit(1)


if __name__ == "__main__":
    main()
