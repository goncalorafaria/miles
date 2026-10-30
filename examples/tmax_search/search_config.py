"""The search environment's settings, read from the Prime-RL experiment the port reproduces.

A Prime-RL search experiment configures its environment in ``rl.toml``: ``[orchestrator.train.source.legacy.args]``
(the arguments of ``search_agent.environment.load_environment``) and ``[...legacy.extra_env_kwargs]``. That table is
the single source of truth here too. The launcher reads it from the pinned Prime-RL source (``resolve_env_args``),
resolves the files it names and the services it calls, and hands the result to the agent and reward processes as the
``SEARCH_ENV_ARGS`` environment variable (JSON), which they read with ``env_args``.

Paths in the table are the Prime-RL job's: its checkout (``run.yaml``'s ``working_dir``, e.g. /weka/stevenc/prime-rl)
maps to the pinned source, the rest of its ``/weka`` mount (``oe-adapt-default``) to where this job mounts that
bucket. The service URLs are rewritten as Prime-RL's own launcher does in coupled runs
(``prime_rl_beaker/multinode.py`` ``prepare_rl_runtime``): ``<gateway>/search``, ``<gateway>/terminal`` and
``<gateway>/v1``.
"""

import json
import os
from functools import cache
from pathlib import Path
from typing import Any

import tomllib

ENV_ARGS_VAR = "SEARCH_ENV_ARGS"
_LEGACY_ENV_ID = "search_agent.environment"

# The load_environment arguments the port implements, with the defaults load_environment gives them
# (environments/search_agent/search_agent/environment.py), for a table that omits them.
_DEFAULTS: dict[str, Any] = {
    "dataset": None,
    "split": "train",
    "judge_model_path": "Qwen/Qwen3.8-27B",
    "judge_enabled": True,
    "local_search_model_path": None,
    "judge_timeout": 240,
    "judge_max_retries": 3,
    "judge_mode": "gateway",
    "judge_policy_model": "Qwen/Qwen3.5-4B",
    "judge_seed": 42,
    "judge_prompts_path": None,
    "timeout": 65,
    "max_retries": 3,
    "terminal_truncation": 2000,
    "max_turns": None,
    "max_tool_calls": 150,
    "tool_stderr_penalty": 0.05,
    "missing_final_answer_penalty": -1.0,
    "token_budget_warning_fraction": 0.1,
    "turn_budget_warning_fraction": 0.1,
    "prompt": {},
    "tools": {},
}
# Arguments accepted because they cannot change this recipe's episodes: the service URLs are replaced by the job's
# gateway, and webterminal_max_content_length only caps pages fetched through Jina, which a local search model path
# turns off (literegistry_tool_client/fetch.py).
_REPLACED_OR_INERT = {"search_server_url", "terminal_server_url", "judge_server_url", "webterminal_max_content_length"}
# load_environment's default judge prompts (prime_rl_common/search_defaults.json), under search_agent/resources.
_JUDGE_PROMPTS_RESOURCE = (
    "environments/search_agent/search_agent/resources/judge_templates/browsecomp-open-instruct.json"
)


def resolve_env_args(
    *,
    prime_rl_source: str,
    prime_rl_config: str,
    prime_rl_checkout: str,
    weka_mount: str,
    search_url: str,
    terminal_url: str,
    judge_url: str,
) -> dict[str, Any]:
    """The environment arguments of the Prime-RL experiment ``prime_rl_config`` (relative to ``prime_rl_source``),
    resolved for this job: defaults filled in, files read, services pointed at ``*_url``."""
    source = Path(prime_rl_source)
    config = tomllib.loads((source / prime_rl_config).read_text())
    train_source = config["orchestrator"]["train"]["source"]
    assert len(train_source) == 1, f"{prime_rl_config}: expected one training source, got {len(train_source)}"
    legacy = train_source[0]["legacy"]
    assert legacy["id"] == _LEGACY_ENV_ID, f"{prime_rl_config}: environment {legacy['id']!r}, not {_LEGACY_ENV_ID!r}"
    raw = dict(legacy.get("args", {}))
    unknown = set(raw) - set(_DEFAULTS) - _REPLACED_OR_INERT
    assert not unknown, f"{prime_rl_config} sets environment arguments the port does not implement: {sorted(unknown)}"
    extra = dict(legacy.get("extra_env_kwargs", {}))
    assert set(extra) <= {"max_total_completion_tokens"}, f"unsupported extra_env_kwargs: {sorted(extra)}"
    args = _DEFAULTS | {k: v for k, v in raw.items() if k in _DEFAULTS}

    def local(path: str) -> Path:
        """A path of the Prime-RL job, in this job's filesystem."""
        if path.startswith(prime_rl_checkout.rstrip("/") + "/"):
            return source / path[len(prime_rl_checkout.rstrip("/")) + 1 :]
        assert path.startswith("/weka/"), f"{path}: not under the Prime-RL job's /weka mount"
        return Path(weka_mount) / path[len("/weka/") :]

    # The behaviors this port implements; load_environment supports more.
    assert args["judge_enabled"] and args["judge_mode"] == "extract_then_judge", "only extract_then_judge is ported"
    tools, prompt = args.pop("tools"), args.pop("prompt")
    search, webterminal = tools.get("search", {}), tools.get("webterminal", {})
    assert tools.get("document_reference", "url") == "url", "only URL document references are ported"
    assert search.get("interface") == "query_only", "only the query_only search interface is ported"
    assert set(tools) <= {"search", "webterminal", "document_reference"}, f"unknown tools keys: {sorted(tools)}"
    assert set(prompt) <= {"template_path", "final_warning_path"}, f"unknown prompt keys: {sorted(prompt)}"
    assert prompt.get("template_path") and prompt.get("final_warning_path"), "the port needs both prompt files"
    max_turns = args["max_turns"] or args["max_tool_calls"] + 1
    assert max_turns >= args["max_tool_calls"] + 1, "max_turns must reserve a final-answer turn"

    judge_prompts_path = args.pop("judge_prompts_path")
    judge_prompts = local(judge_prompts_path) if judge_prompts_path else source / _JUDGE_PROMPTS_RESOURCE
    return {
        **args,
        "dataset": str(local(args["dataset"])),
        "max_turns": max_turns,
        "max_total_completion_tokens": int(extra.get("max_total_completion_tokens", -1)),
        "search_server_url": search_url,
        "terminal_server_url": terminal_url,
        "judge_server_url": judge_url,
        "template": json.loads(local(prompt["template_path"]).read_text()),
        # Instruction files are read stripped (search_agent/webterminal_config.py read_instruction).
        "final_warning": local(prompt["final_warning_path"]).read_text().strip(),
        "search_num_results": int(search.get("num_results", 10)),
        "search_description": local(search["description_path"]).read_text().strip(),
        "webterminal_description": local(webterminal["description_path"]).read_text().strip(),
        "judge_prompts": json.loads(judge_prompts.read_text()),
    }


@cache
def env_args() -> dict[str, Any]:
    """The resolved arguments, in an agent or reward process (the launcher exports them)."""
    return json.loads(os.environ[ENV_ARGS_VAR])
