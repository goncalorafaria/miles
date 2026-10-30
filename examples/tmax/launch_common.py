"""Launcher helpers shared by the TMax launchers: examples/tmax/run_tmax_qwen3_5.py and
examples/tmax_search/run_search_qwen3_5.py."""

import dataclasses
import os
import shlex
from pathlib import Path

import tmax_wandb
import yaml

from miles.utils.external_utils import command_utils
from miles.utils.typer_utils import SCRIPT_ENV_VAR_PREFIX


def fetch_command(hf_repo: str, dst: str) -> str:
    """Download a Hugging Face repo into dst, or copy a local HF checkpoint there once: the model dir keeps it after
    the run directory it came from is cleaned up."""
    if not hf_repo.startswith("/"):
        return f"hf download {hf_repo} --local-dir {dst}"
    return f"if [ ! -d {dst} ]; then rm -rf {dst}.tmp && cp -r {hf_repo} {dst}.tmp && mv {dst}.tmp {dst}; fi"


def wandb_args(script_file: str, *, run_id: str, exp_name: str, project: str, entity: str) -> str:
    """Default wandb flags (on when WANDB_API_KEY is set) with this experiment's overrides applied in place.

    A named experiment logs all its restarts into one run (group and run id from exp_name), which a resume then
    continues, instead of one run per Beaker retry.
    """
    tokens = shlex.split(command_utils.get_default_wandb_args(script_file, run_id=run_id))
    if not tokens:
        return ""
    # The launcher logs this command line (and prints it again if the job fails), so the API key stays off it:
    # without --wandb-key, wandb reads WANDB_API_KEY, which every process of the job inherits.
    if "--wandb-key" in tokens:
        del tokens[tokens.index("--wandb-key") : tokens.index("--wandb-key") + 2]
    overrides = {"--wandb-project": project, "--wandb-team": entity}
    if exp_name:
        overrides |= {
            "--wandb-group": exp_name,
            "--wandb-run-id": tmax_wandb.run_id(exp_name),
        }
    for flag, value in overrides.items():
        if not value:
            continue
        if flag in tokens:
            tokens[tokens.index(flag) + 1] = value
        else:
            tokens += [flag, value]
    return shlex.join(tokens) + " "


def apply_yaml_config(argv: list[str], script_args: type) -> list[str]:
    """Accept ``<command> <experiment>.yaml [--flag value ...]``, as open-instruct's ArgumentParserPlus does.

    The YAML's keys (fields of the ``script_args`` dataclass) become MILES_SCRIPT_* env vars, which dataclass_cli
    reads as defaults, so flags on the command line still override them, and an already-exported env var wins over
    the file. A top-level ``beaker:`` section is for beaker/submit.py and ignored here.
    """
    configs = [arg for arg in argv[1:] if arg.endswith((".yaml", ".yml"))]
    if not configs:
        return argv
    assert len(configs) == 1, f"pass at most one experiment config, got {configs}"
    config = yaml.safe_load(Path(configs[0]).read_text()) or {}
    config.pop("beaker", None)
    unknown = set(config) - {f.name for f in dataclasses.fields(script_args)}
    assert not unknown, f"{configs[0]} sets unknown fields: {sorted(unknown)}"
    for key, value in config.items():
        if value is not None:
            env_value = str(value).lower() if isinstance(value, bool) else str(value)
            os.environ.setdefault(f"{SCRIPT_ENV_VAR_PREFIX}{key.upper()}", env_value)
    return [arg for arg in argv if arg != configs[0]]
