# Agent worker pool

By default Miles runs agentic episodes the way prime-rl's env server does:
a fixed set of long-lived worker processes (`--agent-pool-workers`), each running
many episodes as asyncio tasks, instead of a fresh Python process per episode (`subproc`).

Each worker hosts the session routes in-process (the same `SessionServer` app the standalone
session servers run), so an agent's chat calls get TITO tokenization and recording in the worker
and go straight to the inference router. When an episode ends, the worker assembles and encodes
its samples and sends them to the rollout process over zmq (DEALER/ROUTER, one socket pair per
worker); the rollout process only decodes them. There is no session-server hop, no
`/samples` collection request and no acknowledgement on this path.

- Code: `miles/rollout/agent_pool/` (`pool.py` rollout side, `worker.py`, `protocol.py`);
  `generate_in_agent_pool` in `miles/rollout/generate_hub/agentic_tool_call.py`.
- Agent contract: unchanged (`run(base_url, prompt, request_kwargs, metadata)`). Many episodes
  now share a process and an event loop, so an agent must not block the loop or rely on
  per-process module state; cross-episode limits keep using `miles.rollout.agentic.rollout_limits`,
  scoped to the rollout process as under `subproc`.
- Failure handling: an agent error is logged and its turns are still collected, as under `subproc`.
  A worker that exits fails its in-flight episodes (their samples are aborted and recycled) and is
  restarted. Cancelling an episode cancels its task in the worker.
- Tests: `tests/fast/rollout/agent_pool/test_agent_pool.py`, including parity with the
  session-server path on a mock SGLang backend.
