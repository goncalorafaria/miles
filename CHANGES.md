# Agentic rollout orchestration

## What changes for a launch

Miles now runs episodes from `agentic_tool_call.generate` on persistent worker processes by default. Existing launch commands can omit `--custom-agent-function-mode pool`. To reproduce the previous process-per-episode behavior, pass `--custom-agent-function-mode subproc`.

Multiple episodes can share one worker concurrently. Keep mutable episode state and client cleanup scoped to the episode. A blocking call inside an async agent can delay other episodes assigned to that worker. See the [agent pool guide](docs/developer/agent-pool.md) for worker sizing and lifecycle.

## What changes during a rollout

- The coordinator assigns an episode to the worker with the fewest pending requests. The worker hosts the session, assembles training samples and returns them over ZeroMQ. The original prompt group still receives its results for reward processing.
- On the standalone collection path, payload transfer uses bounded concurrency. Decoding happens after the transfer slot is released. Successful collections are acknowledged in the background.
- The router forwards response bytes directly. TMax moves blocking trace writes and Docker calls off the shared event loop and closes per-episode clients.

The [CPU orchestration simulator](docs/developer/orchestration-simulator.md) exercises both execution modes with the same episode profiles. Pass `--agent-mode subproc` for the previous baseline and omit the override to measure the current default.
