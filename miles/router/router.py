import asyncio
import json
import logging
import re
import sys

import aiohttp
import setproctitle
import uvicorn
from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse
from starlette.responses import Response

from miles.router.config import MilesRouterConfig
from miles.utils.logging_utils import configure_logger_raw
from miles.utils.workers.argv_utils import parse_config_argv

logger = logging.getLogger(__name__)


# Names accepted by the W&B OpenMetrics scraper and the Miles dashboard.
_OCCUPANCY_GAUGE = re.compile(
    r"^sglang:(num_running_reqs|num_queue_reqs|token_usage)(?:\{(.*)\})?(\s+.+)$"
)


def _engine_gauges(worker: str, payload: str) -> str:
    # Preserve the engine's label set (including DP rank) and sample value.
    # Filtering before parsing avoids constructing objects for every histogram bucket.
    worker_label = f"worker_addr={json.dumps(worker)}"
    lines = []
    for line in payload.splitlines():
        if match := _OCCUPANCY_GAUGE.fullmatch(line):
            name, labels, value = match.groups()
            labels = f"{worker_label},{labels}" if labels else worker_label
            lines.append(f"sglang_{name}{{{labels}}}{value}\n")
    return "".join(lines)


def run_router(config: MilesRouterConfig):
    """
    Run the Miles router with the specified configuration.
    """
    # Spawned as a fresh interpreter, so it inherits no logging config.
    configure_logger_raw("miles_router")
    # Visible to `pkill -9 miles`; without this the daemon inherits "python".
    setproctitle.setproctitle("miles-router")

    # Initialize the router with tokenizer and lazy worker initialization
    miles_router = MilesRouter(config, verbose=False)

    # Start the server
    # Idle keep-alive closure well after the session servers' 5s client expiry, so a pooled
    # connection is never closed by the router while a client is reusing it.
    uvicorn.run(
        miles_router.app, host=config.host, port=config.port, log_level="info", access_log=False, timeout_keep_alive=60
    )


class MilesRouter:
    def __init__(self, config: MilesRouterConfig, verbose=False):
        """Initialize the miles-router with SGLang router address"""
        self.config = config
        self.verbose = verbose

        self.app = FastAPI()
        self.app.router.on_startup.append(self._start_background_health_check)

        # URL -> Active Request Count (load state)
        self.worker_request_counts: dict[str, int] = {}
        # URL -> Consecutive Failures
        self.worker_failure_counts: dict[str, int] = {}
        # Quarantined workers excluded from routing pool
        self.dead_workers: set[str] = set()

        # aiohttp, not httpx: httpcore's pool rescans every connection on each request start and
        # close, so with hundreds of in-flight generations the router saturated its one core on pool
        # bookkeeping and queued requests for tens of seconds before they reached an engine.
        # Created lazily on the serving loop (a ClientSession is bound to the loop that made it).
        self._client: aiohttp.ClientSession | None = None
        self._client_loop: asyncio.AbstractEventLoop | None = None

        self._setup_routes()

    @property
    def client(self) -> aiohttp.ClientSession:
        loop = asyncio.get_running_loop()
        if self._client is None or self._client.closed or self._client_loop is not loop:
            self._client = aiohttp.ClientSession(
                connector=aiohttp.TCPConnector(limit=self.config.max_connections, limit_per_host=0),
                # Per phase, as httpx.Timeout(timeout) was: waiting for a pooled connection, then each read.
                timeout=aiohttp.ClientTimeout(total=None, connect=self.config.timeout, sock_read=self.config.timeout),
                # Bodies pass through byte-for-byte, together with their Content-Encoding.
                auto_decompress=False,
                cookie_jar=aiohttp.DummyCookieJar(),
            )
            self._client_loop = loop
        return self._client

    def _setup_routes(self):
        """Setup all the HTTP routes except catch-all proxy"""
        # sglang-router api
        self.app.post("/add_worker")(self.add_worker)
        self.app.post("/remove_worker")(self.remove_worker)
        self.app.get("/list_workers")(self.list_workers)
        self.app.get("/engine_metrics")(self.engine_metrics)
        # Catch-all route for proxying to SGLang - must be registered LAST
        self.app.api_route("/{path:path}", methods=["GET", "POST", "PUT", "DELETE"])(self.proxy)

    async def _start_background_health_check(self):
        asyncio.create_task(self._health_check_loop())

    async def _check_worker_health(self, url):
        """Encapsulated health check logic for better maintainability."""
        try:
            async with self.client.get(f"{url}/health", timeout=aiohttp.ClientTimeout(total=5.0)) as response:
                await response.read()
            if response.status == 200:
                return url, True
            logger.debug(f"[miles-router] Worker {url} is unhealthy (Status: {response.status})")
        except Exception as e:
            logger.debug(f"[miles-router] Worker {url} health check failed: {e}")
        return url, False

    async def _health_check_loop(self):
        """Background loop to monitor worker health and adjust routing pool."""
        interval = self.config.health_check_interval
        threshold = self.config.health_check_failure_threshold

        while True:
            try:
                await asyncio.sleep(interval)

                urls = [u for u in self.worker_request_counts if u not in self.dead_workers]
                if not urls:
                    continue

                results = await asyncio.gather(*(self._check_worker_health(url) for url in urls))

                for url, is_healthy in results:
                    if url not in self.worker_request_counts:
                        continue
                    if not is_healthy:
                        failures = self.worker_failure_counts.get(url, 0) + 1
                        self.worker_failure_counts[url] = failures

                        if failures >= threshold:
                            logger.warning(
                                f"[miles-router] Worker {url} failed {threshold} consecutive health checks. Marking as DEAD."
                            )
                            self.dead_workers.add(url)
                            # TODO (chenyang): Connect back 'dead' workers requires a mechanism to sync
                            # model versions to avoid off-policy issues from stale weights, since these
                            # dead workers' parameters may not be refitted.
                    else:
                        self.worker_failure_counts[url] = 0

                logger.debug(
                    f"[miles-router] Health check complete. {len(self.worker_request_counts) - len(self.dead_workers)} workers healthy."
                )

            except asyncio.CancelledError:
                logger.warning("[miles-router] Background health check loop is being cancelled.")
                raise
            except Exception as e:
                logger.error(f"[miles-router] Unexpected error in health check loop: {e}", exc_info=True)
                await asyncio.sleep(5)

    async def engine_metrics(self) -> Response:
        workers = [url for url in self.worker_request_counts if url not in self.dead_workers]
        samples = await asyncio.gather(*(self._fetch_engine_gauges(url) for url in workers))
        header = (
            "# TYPE sglang_num_running_reqs gauge\n"
            "# TYPE sglang_num_queue_reqs gauge\n"
            "# TYPE sglang_token_usage gauge\n"
        )
        return Response(header + "".join(samples), media_type="text/plain; version=0.0.4")

    async def _fetch_engine_gauges(self, url: str) -> str:
        try:
            async with self.client.get(
                f"{url}/metrics", timeout=aiohttp.ClientTimeout(total=5), auto_decompress=True
            ) as response:
                response.raise_for_status()
                payload = await response.text()
            return await asyncio.to_thread(_engine_gauges, url, payload)
        except (aiohttp.ClientError, asyncio.TimeoutError, UnicodeError) as exc:
            logger.debug("Skipping engine metrics from %s: %s", url, exc)
            return ""

    async def proxy(self, request: Request, path: str):
        """Proxy all other requests to the SGLang router"""
        result = await self.do_proxy(request, path)
        return self.build_proxy_response(result)

    async def do_proxy(
        self,
        request: Request,
        path: str,
        body: bytes | None = None,
        headers: dict | None = None,
    ) -> dict:
        """Core proxy logic. Returns dict with request_body, response_body, status_code, headers."""
        worker_url = self._use_url()
        url = f"{worker_url}/{path}"

        if body is None:
            body = await request.body()
        if headers is None:
            headers = dict(request.headers)
        if body is not None:
            headers = {k: v for k, v in headers.items() if k.lower() not in ("content-length", "transfer-encoding")}

        try:
            async with self.client.request(
                request.method, url, data=body, headers=headers, allow_redirects=False
            ) as response:
                content = await response.read()
            return {
                "request_body": body,
                "response_body": content,
                "status_code": response.status,
                # Lower-case keys, as httpx returned them.
                "headers": {k.lower(): v for k, v in response.headers.items()},
            }
        finally:
            self._finish_url(worker_url)

    def build_proxy_response(self, result: dict) -> Response:
        """Build HTTP response from proxy result."""
        content = result["response_body"]
        status_code = result["status_code"]
        headers = result["headers"]
        headers = {
            k: v
            for k, v in headers.items()
            if k.lower() not in ("content-length", "transfer-encoding", "server", "date")
        }
        # Pass the body through unchanged: re-parsing and re-serializing multi-megabyte generate
        # replies (per-token logprobs, score arrays) cost the router's single core for no change.
        return Response(content=content, status_code=status_code, headers=headers)

    async def add_worker(self, request: Request):
        """Add a new worker to the router.
        Supports providing the URL via query string or JSON body.
        Examples:
        - POST /add_worker?url=http://127.0.0.1:10090
        - POST /add_worker  with body {"url": "http://127.0.0.1:10090"}
        """
        worker_url = await self._parse_worker_url(request)
        if not worker_url:
            return JSONResponse(
                status_code=400, content={"error": "worker_url is required (use query ?url=... or JSON body)"}
            )

        # Add if new, keep a simple request count per worker
        if worker_url not in self.worker_request_counts:
            self.worker_request_counts[worker_url] = 0
            self.worker_failure_counts[worker_url] = 0
            self.dead_workers.discard(worker_url)
            if self.verbose:
                print(f"[miles-router] Added new worker: {worker_url}")

        return {"status": "success", "worker_urls": self.worker_request_counts}

    async def remove_worker(self, request: Request):
        """Remove a worker from the router, using the same URL conventions as add_worker."""
        worker_url = await self._parse_worker_url(request)
        if worker_url is None:
            return JSONResponse(
                status_code=400, content={"error": "worker_url is required (use query ?url=... or JSON body)"}
            )

        self.worker_request_counts.pop(worker_url, None)
        self.worker_failure_counts.pop(worker_url, None)
        self.dead_workers.discard(worker_url)
        logger.info(f"[miles-router] Removed worker: {worker_url}")

        return {"status": "success", "worker_urls": self.worker_request_counts}

    async def _parse_worker_url(self, request: Request) -> str | None:
        if worker_url := request.query_params.get("url") or request.query_params.get("worker_url"):
            return worker_url

        body = await request.body()
        payload = json.loads(body) if body else {}
        return payload.get("url") or payload.get("worker_url")

    async def list_workers(self, request: Request):
        """List all registered workers"""
        return {"urls": list(self.worker_request_counts.keys())}

    def _use_url(self):
        """Select worker URL with minimal active requests."""

        if not self.dead_workers:
            # Healthy path: select from all workers
            url = min(self.worker_request_counts, key=self.worker_request_counts.get)
        else:
            # Degraded path: select from workers not in dead_workers
            valid_workers = (w for w in self.worker_request_counts if w not in self.dead_workers)
            try:
                url = min(valid_workers, key=self.worker_request_counts.get)
            except ValueError:
                raise RuntimeError("No healthy workers available in the pool") from None

        self.worker_request_counts[url] += 1
        return url

    def _finish_url(self, url: str) -> None:
        """Mark the request to the given URL as finished"""
        if (count := self.worker_request_counts.get(url)) is None or count == 0:
            logger.info(f"[miles-router] Request to {url} finished after the worker was deregistered; ignoring")
            return
        self.worker_request_counts[url] = count - 1


if __name__ == "__main__":
    run_router(parse_config_argv(MilesRouterConfig, sys.argv[1:]))
