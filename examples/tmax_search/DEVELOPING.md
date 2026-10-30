# Developing the search port

What the example does and how it maps to Prime-RL is in [README.md](README.md); the TMax port's process model, Beaker
tooling and conventions, which this port reuses, are in [../tmax/DEVELOPING.md](../tmax/DEVELOPING.md).

## The pinned Prime-RL source

The port runs Prime-RL code from a sealed copy of steven-prime-rl (`a5fcb924…`, identical to its `main` at `6d48bbe2`
in every directory the port uses), at `/weka/oe-adapt-default/shashankg/miles_search/prime-rl-sources/<id>/source`:
the agent and reward import `literegistry_tool_client` and `search_agent.{protocol,tools,tool_metrics}` from it (the
launcher puts them on `PYTHONPATH`), the launcher reads the experiment's `rl.toml`, prompt files and dataset from it,
and the Beaker service tasks boot from it. Moving to a newer Prime-RL means a new sealed copy, `prime_rl_source` in the
configs, and `beaker/services_template.json` regenerated from an accepted submission of that source:

```bash
python examples/tmax_search/beaker/services.py derive <prime-rl run>/submissions/<id>/beaker.json
```

## Local checks

Miles itself does not run on the dev box (see the TMax DEVELOPING.md); the environment does. Prime-RL's services run
from Steven's Beaker images under podman (`beaker image pull` needs `DOCKER_HOST=unix:///run/podman/podman.sock`),
booted from the pinned source the way the Beaker tasks boot:

```bash
SRC=/weka/shashankg/miles_search/prime-rl-sources/a5fcb924645d9bd8f326efd5087c8de6447c83fe17c6d07c36e99ee736d30a37/source
# Terminal (the tools image); it insists on a registry, a local sqlite one does
podman run -d --name search-terminal-local --network host -v /weka/oe-adapt-default:/weka:ro \
    docker.io/library/primebeaker-tools:local bash $SRC/beaker/runtime/entrypoint.sh launcher \
    literegistry terminal --host=0.0.0.0 --port=18010 --registry=sqlite:////tmp/lr.sqlite3
# Search (the quote-retrieval image), served from weka: its NVMe cache needs a real NVMe device
podman run -d --name search-local --network host -v /weka/oe-adapt-default:/weka:ro \
    -e JAVA_HOME=/weka/stevenc/tmax/lucene_files/jdk-21 -e SEARCH_WORKERS=4 -e SEARCH_MAX_PENDING=64 \
    -e RETRIEVAL_JAR=/weka/stevenc/s2orc/dolma/s2orcforolmo/index_fs_v1/.venv/lib/python3.12/site-packages/pyserini/resources/jars/anserini-1.4.0-fatjar.jar \
    -e LOCAL_SEARCH_CORPUS_JSONL=/weka/stevenc/data-dr/search_agents/bc_rl_v1/collection/part-00000.jsonl \
    -e LUCENE_INDEX_DIR=/weka/stevenc/prime-rl/indexes/bc-rl-v1/quote_v3_fast_v1 \
    docker.io/library/prime-rl-quote-retrieval:local bash $SRC/beaker/runtime/entrypoint.sh retrieval \
    python -m uvicorn prime_rl_retrieval.server:create_app --factory --host 0.0.0.0 --port 18011
# Policy and judge (one L40S each; the judge in fp8)
vllm serve Qwen/Qwen3.5-4B --port 8012 --served-model-name model --language-model-only \
    --enable-auto-tool-choice --tool-call-parser qwen3_coder --reasoning-parser qwen3 --max-model-len 131072
vllm serve Qwen/Qwen3.8-27B --port 8021 --language-model-only --quantization fp8 --max-model-len 32768 \
    --reasoning-parser qwen3
```

`tools/` checks the port against Prime-RL's own trajectories (`<run>/artifacts/<attempt>/trajectories/train/step_N.jsonl`
of a Prime-RL search run) and against live episodes; each takes the pinned source from `PRIME_RL_SOURCE` and needs
`PYTHONPATH=examples/tmax_search:$SRC/deps/literegistry/literegistry_tool_client/src:$SRC/environments/search_agent:.`:

| Tool | Checks |
|---|---|
| `replay_tools.py <step.jsonl> <n> <search url> <terminal url>` | Replays each trajectory's tool calls through the port's tool layer and diffs the observations, stderr penalties and tool-outcome counters |
| `judge_parity.py <step.jsonl> <n> <judge url> <tokenizer>` | Scores the trajectories with the port's reward function and diffs format verdicts, judge verdicts and rewards |
| `run_episodes.py <out> <questions> <samples> <policy url> <search url> <terminal url>` | Runs live episodes through `search_agent_function.run` against a vLLM policy, keeping each turn's token ids |
| `score_episodes.py <episodes> <judge url> <tokenizer>` | Scores them with the port's reward function |

## Tests

```bash
uv run --no-project --python /stage/.venv/bin/python --with "pytest-asyncio==0.24.0" python -m pytest -q --noconftest \
    -p no:cacheprovider -o addopts="" tests/fast/examples/tmax_search
```

They need the pinned source on weka and skip without it.
