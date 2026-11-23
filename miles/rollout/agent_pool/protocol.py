"""Wire protocol between the rollout process and its agent worker pool.

Modeled on prime-rl's env server (verifiers ``serve``): the rollout process holds one zmq DEALER
per worker and each worker one ROUTER. Requests are ``[request_id, method, payload]``; a worker
answers ``[request_id, b"reply", header, samples]``, where ``samples`` is the safetensors buffer
``encode_samples`` produced (sent without a copy) and ``header`` the small pickled ``RunReply``.

Both ends are processes of the same rollout on the same host, started from the same tree, so
pickle is the payload codec, as for subproc agent calls (``miles.rollout.agentic.agent_function``).
"""

from dataclasses import dataclass, field
from typing import Any

RUN = b"run"
CANCEL = b"cancel"
REPLY = b"reply"


@dataclass(frozen=True)
class WorkerSpec:
    """Everything a worker needs to start: built once by the rollout process."""

    index: int
    address: str  # zmq ipc:// endpoint the worker binds
    session_config: Any  # miles.rollout.session.config.SessionServerConfig, bound to 127.0.0.1:<worker port>
    agent_function_path: str
    parent_pid: int
    sys_path: list[str]
    env: dict[str, str]


@dataclass(frozen=True)
class RunRequest:
    """One episode: a session on the worker's in-process session core, the agent, then assembly."""

    prompt: Any
    request_kwargs: dict
    metadata: dict
    evaluation: bool
    session_params: dict  # CreateSessionRequest fields (sampling defaults)
    extra_key: str | None
    max_seq_len: int | None
    use_v2: bool


@dataclass
class RunReply:
    """Small header of a reply; the assembled samples travel as the next frame."""

    ok: bool
    session_id: str | None = None
    agent_metadata: dict | None = None
    # A failure of the worker machinery (not of the agent, whose errors are logged and tolerated):
    error: str | None = None
    traceback: str | None = None
    # The assembly's HTTP-equivalent status: 200 carries a samples buffer, anything else its body as text.
    collect_status: int | None = None
    collect_error: str | None = None
    timings: dict = field(default_factory=dict)
