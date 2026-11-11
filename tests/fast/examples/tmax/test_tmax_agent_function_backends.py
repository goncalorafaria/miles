"""tmax_agent_function on the podman backend, where literegistry-podman-client is not installed (the entry script
installs it only for the gateway backend)."""

import importlib
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[4] / "examples" / "tmax"))

import tmax_sandbox  # noqa: E402


def test_the_podman_backend_runs_without_the_gateway_client(monkeypatch):
    monkeypatch.setitem(sys.modules, "literegistry_podman_client", None)  # any import of it fails, as on a podman job
    monkeypatch.delitem(sys.modules, "tmax_gateway", raising=False)
    monkeypatch.delitem(sys.modules, "tmax_agent_function", raising=False)
    monkeypatch.setenv("TMAX_SANDBOX_BACKEND", "podman")
    tmax_sandbox._settings.cache_clear()
    try:
        agent_function = importlib.import_module("tmax_agent_function")
        box = agent_function._make_sandbox(image="hamishi740/swerl-tmax-v3:test", task_dir="/tmp/task")
    finally:
        tmax_sandbox._settings.cache_clear()
    assert isinstance(box, tmax_sandbox.TmaxSandbox)
