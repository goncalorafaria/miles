"""The search services' Beaker tasks: Prime-RL's own, as its coupled launcher renders them.

A Prime-RL search experiment runs its services as tasks of the training experiment (prime_rl_beaker/coupled.py): a
Redis and a LiteRegistry gateway, terminal and quote-BM25 search replicas on Jupiter CPUs, and the vLLM judge on one
GPU. Each task boots from the pinned Prime-RL source (beaker/runtime/entrypoint.sh), registers in a head registry of
endpoint files on weka, and runs under ``prime_rl_beaker.coupled worker``, which exits 0 once ``<state-dir>/complete``
exists.

``services_template.json`` holds those tasks, taken from an accepted Prime-RL submission rendered by the pinned source
(``derive``), with its run-specific values replaced by ``{PLACEHOLDERS}``. ``service_tasks`` fills them for a Miles run
with the values of the Prime-RL experiment's ``run.yaml`` (corpus, index, service name, replica counts): the Miles task
reads the gateway's address from the same endpoint files (wait_for_services.py) and writes ``complete`` when training
succeeds (tmax_beaker_entry.sh).

    python examples/tmax_search/beaker/services.py derive <accepted beaker.json>   # regenerate the template
"""

import json
import re
import sys
from pathlib import Path

import yaml

TEMPLATE = Path(__file__).with_name("services_template.json")
SERVICE_TASKS = ("redis", "gateway", "terminal", "localsearch", "judge")


def derive(spec_path: str) -> dict:
    """Template the service tasks of an accepted coupled Prime-RL submission (its beaker.json)."""
    spec = json.loads(Path(spec_path).read_text())
    tasks = [t for t in spec["tasks"] if t["name"] in SERVICE_TASKS]
    assert [t["name"] for t in tasks] == list(SERVICE_TASKS), [t["name"] for t in tasks]
    text = json.dumps(tasks)
    (source,) = set(re.findall(r"(/weka/[^\"' ]+/prime-rl-sources/([0-9a-f]{64})/source)", text))
    head_registry = re.search(r"file://(/weka/[^\"' ]+)/runtime/endpoint-files", text).group(1)
    run_name = Path(head_registry).name
    (identity,) = set(re.findall(r"[a-z]+-" + re.escape(run_name) + r"-[a-z0-9-]+", text))
    (lifecycle,) = set(re.findall(re.escape(head_registry) + r"/runtime/lifecycle/[0-9a-f]+", text))
    env = {e["name"]: e.get("value") for t in tasks for e in t.get("envVars", [])}
    search = next(t for t in tasks if t["name"] == "localsearch")
    search_env = re.search(
        r"LOCAL_SEARCH_CORPUS_JSONL=(\S+?); LUCENE_INDEX_DIR=(\S+?); LITEREGISTRY_SERVICE_NAME=(\S+?);",
        json.dumps(search["command"]).replace("\\", ""),
    )
    corpus, index, service_name = search_env.groups()
    replacements = [
        (lifecycle, "{STATE_DIR}"),
        (source[0], "{PRIME_RL_SOURCE}"),
        (identity, "{IDENTITY}"),
        (env["WANDB_CACHE_DIR"], "{RUN_DIR}/wandb-cache"),
        (env["WANDB_DATA_DIR"], "{RUN_DIR}/wandb-staging"),
        (head_registry, "{RUN_DIR}"),
        (corpus, "{SEARCH_CORPUS}"),
        (index, "{SEARCH_INDEX}"),
        (service_name, "{SEARCH_SERVICE_NAME}"),
        (env["RETRIEVAL_NVME_CACHE_DIR"], "{NVME_CACHE_DIR}"),
    ]
    for old, new in replacements:
        text = text.replace(old, new)
    tasks = json.loads(text)
    for task in tasks:
        task.pop("replicas", None)
        for e in task.get("envVars", []):
            if "secret" in e:
                e["secret"] = "{HF_TOKEN_SECRET}"
    return {
        "derived_from": {"spec": spec_path, "prime_rl_source_id": source[1], "run": run_name},
        "tasks": tasks,
    }


def service_tasks(
    *,
    prime_rl_source: str,
    prime_rl_run_yaml: str,
    run_dir: str,
    state_dir: str,
    identity: str,
    hf_token_secret: str,
    context: dict,
    replicas: dict[str, int],
    nvme_cache_dir: str,
) -> list[dict]:
    """The template's tasks for one Miles run. Paths are the service tasks' own, under their /weka (oe-adapt-default).

    ``replicas`` overrides the run.yaml's terminal / localsearch counts (smoke tests run few); ``context`` gives the
    tasks' ``priority`` and ``min_runtime``."""
    template = json.loads(TEMPLATE.read_text())
    assert f"prime-rl-sources/{template['derived_from']['prime_rl_source_id']}/source" in prime_rl_source, (
        f"{TEMPLATE.name} was rendered by Prime-RL source {template['derived_from']['prime_rl_source_id']}, "
        f"not {prime_rl_source}: regenerate it (derive) from a submission of that source"
    )
    service = yaml.safe_load(Path(prime_rl_run_yaml).read_text())["service"]
    assert service.get("local_search_backend") == "quote_bm25_v1", "the template's search tasks are quote_bm25_v1"
    values = {
        "{PRIME_RL_SOURCE}": prime_rl_source,
        "{STATE_DIR}": state_dir,
        "{RUN_DIR}": run_dir,
        "{IDENTITY}": identity,
        "{SEARCH_CORPUS}": service["local_search_corpus_jsonl"],
        "{SEARCH_INDEX}": service["local_search_index_dir"],
        "{SEARCH_SERVICE_NAME}": service["local_search_service_name"],
        "{NVME_CACHE_DIR}": nvme_cache_dir,
        "{HF_TOKEN_SECRET}": hf_token_secret,
    }
    text = json.dumps(template["tasks"])
    for placeholder, value in values.items():
        text = text.replace(placeholder, value)
    unfilled = set(re.findall(r"(?<!\$)\{[A-Z_]+\}", text))  # ${NAME} are the tasks' own shell variables
    assert not unfilled, f"unfilled template placeholders: {sorted(unfilled)}"
    tasks = json.loads(text)
    counts = {
        "terminal": replicas.get("terminal", service["terminal_replicas"]),
        "localsearch": replicas.get("localsearch", service["local_search_replicas"]),
    }
    judge_model = service["generation_model"]
    for task in tasks:
        assert task["name"] != "judge" or f"--model={judge_model}" in json.dumps(task["command"]), judge_model
        if task["name"] in counts:
            task["replicas"] = counts[task["name"]]
        # The Miles task's priority and minimum runtime (Prime-RL gives the judge 8 h, the CPU services 0 h; a smoke
        # runs without one, on unallocated capacity).
        task["context"] = {**task["context"], "priority": context["priority"], "minRuntime": context["min_runtime"]}
    return tasks


if __name__ == "__main__":
    command, spec_path = sys.argv[1:3]
    assert command == "derive", __doc__
    TEMPLATE.write_text(json.dumps(derive(spec_path), indent=1) + "\n")
    print(f"wrote {TEMPLATE}")
