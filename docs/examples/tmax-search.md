---
title: "Search-Agent RL"
description: "Search-agent RL on BrowseComp-style questions with a local search corpus and an LLM judge, Prime-RL's recipe on the TMax setup."
# Generated from examples/tmax_search/README.md by scripts/tools/sync_example_docs.py. Edit that README, not this file.
---
GRPO for a search agent on BrowseComp-style questions, the Miles port of steven-prime-rl's Prime-RL experiment
`configs/experiments/search-agent-bc-rl-v1-1-fp32-lr5e-7`. Qwen/Qwen3.5-4B, thinking on, answers 1,215 questions
(BC-RL v1.1) with two tools over a 258,484-document local corpus:

- `search(query)`: quote-BM25 over the corpus, 10 results, each a title, a 180-character snippet and its URL;
- `webterminal(command)`: `browse <url> | rg -n -i pattern`: the page comes from the same corpus (never the live web) and
  the pipeline runs on a stateless terminal service (rg, grep, sed, head, ...), output cut at 10,000 characters.

Qwen/Qwen3.8-27B judges the final answer (extract it, then compare it with the reference), and the reward is

    reward = judge (0 or 1) - 0.05 x model-caused webterminal errors - 1 if the episode ends without a final answer

The port runs on the TMax port's Miles setup ([../tmax/README.md](/examples/tmax)): the session server, fully async
training and the Beaker tooling. Everything of Prime-RL's environment that does not import `verifiers` runs unchanged
from a pinned copy of the Prime-RL source, as do its search, terminal and judge services.

## Files

| File | Role |
|---|---|
| `run_search_qwen3_5.py` | Launcher: `prepare` (prompt JSONL from the experiment's dataset and prompt, model download and conversion) and `execute`. Its fields are Prime-RL's trainer and orchestrator settings |
| `search_config.py` | Reads the environment's settings from the Prime-RL experiment's `rl.toml`, maps its paths and service URLs into this job, and hands them to the agent and reward processes (`SEARCH_ENV_ARGS`) |
| `search_agent_function.py` | The episode loop, a port of Prime-RL's `WebTerminalJudgeSearchAgentEnv` turn loop; tools through Prime-RL's own clients |
| `search_reward.py` | The reward: Prime-RL's format gates (run from its source) and its two-stage judge (ported), on the session server's sample |
| `search_loss.py` | Prime-RL's `group_token_mean` loss aggregation through `--custom-convert-samples-to-train-data-path` |
| `search_metrics.py` | Per-rollout W&B metrics: stop conditions, tool outcomes, judge outcomes, reward components, zero-variance groups |
| `configs/` | Experiment configs: `smoke_search_4b.yaml` (4 GPUs, bf16 logits), its twin `smoke_search_4b_fp32_head.yaml`, that on the commit-named image `smoke_search_4b_fp32_head_imgd0831a475b.yaml`, and `prod_4node_search_4b.yaml` (Prime-RL's scale, 4 nodes) |
| `beaker/services.py`, `beaker/services_template.json` | Prime-RL's service tasks (Redis, gateway, terminal, search, judge) for the Beaker experiment |
| `beaker/wait_for_services.py` | The head waits for the services and finds their gateway |

## How it maps to Prime-RL

The environment's settings are not copied: `search_config.resolve_env_args` reads them from the experiment's
`rl.toml` (`[orchestrator.train.source.legacy]`) in the pinned source and rejects any it does not implement. The
trainer's settings are the launcher's defaults:

| Prime-RL (`rl.toml`) | Miles (launcher) |
|---|---|
| `[trainer.loss] type default, dppo_mask_low/high 0.2, kl_tau 0` | `--policy-loss-type dppo_tv --dppo-tv-threshold 0.2` on the rollout log-probs; the same mask (the sampled token's probability moving past 0.2 in the advantage's direction), no ratio clip |
| `aggregation group_token_mean` | `search_loss.py`: Miles' sample-mean reducer with per-sample denominators `N_g * G / B` |
| `[orchestrator.group_sampling] zero_variance_filter, sample_full_batch false`: all-equal groups masked, not refilled | `filter_zero_std: false`: such a group has advantage 0 and still counts in the group mean |
| `batch_size 256, group_size 32` | `rollout_batch_size 8, n_samples_per_prompt 32`; GRPO mean baseline, no std normalization |
| `[trainer.optim] adamw lr 5e-7, betas 0.95/0.95, eps 1e-12, max_norm 1, constant` | `--lr 5e-7 --adam-beta1 0.95 --adam-beta2 0.95 --adam-eps 1e-12 --clip-grad 1.0`, constant |
| `optimization_dtype / reduce_dtype float32` | Megatron's fp32 master weights and Adam state; `--accumulate-allreduce-grads-in-fp32` |
| `fused_lm_head_fp32_logits` and inference `enable_fp32_lm_head` | `fp32_lm_head` (`--fp32-lm-head --sglang-enable-fp32-lm-head`): the same fp32 GEMM of the bf16 operands; the trainer computes the response rows' logits 2048 at a time, as Prime-RL does |
| `max_off_policy_steps 4, max_inflight_episodes 512`, in-flight weight updates | `--fully-async --max-weight-staleness 4 --async-max-concurrent-samples 512 --pause-generation-mode in_place` |
| `seq_len 131072, max_completion_tokens 32768` | `--max-seq-len 131072 --rollout-max-response-len 32768` |
| 16 learner + 16 rollout GPUs + 1 judge GPU | `async_rollout_gpus 16` on 4 nodes, the judge a 1-GPU task of the experiment |
| `renderer qwen3.5, thinking_retention tool_cycle` | Miles' `qwen35` template with `preserve_thinking`: every turn's reasoning stays in the context, as Prime-RL's tool-cycle bridging keeps it with one user turn |
| `ckpt.interval 20` | `save_interval 20, hf_snapshot_every 20` |

## Services on Beaker

A config with a `beaker.search_services` section runs Prime-RL's own service tasks in the same Beaker experiment as
the Miles task, as Prime-RL's coupled launcher does: Redis and the LiteRegistry gateway, terminal replicas and
quote-BM25 search replicas on Jupiter CPUs, and the vLLM judge on one Holmes GPU. `beaker/services_template.json` is
those tasks as rendered by the pinned Prime-RL source for an accepted submission, with the run-specific values as
placeholders (regenerate it with `python examples/tmax_search/beaker/services.py derive <beaker.json>` when the pinned
source changes). `submit.py` fills it with the experiment's `run.yaml` (corpus, index, service name, replica counts,
which `search_services.replicas` can override) and this run's state directory on `oe-adapt-default`.

- **Discovery.** The services register in a head registry of endpoint files under
  `<run_root>/<exp_name>/runtime/endpoint-files`. The Miles head waits until the gateway's record is fresh and a search
  query, a terminal pipeline and a judge completion answer through it (`wait_for_services.py`, up to 2 h), then exports
  `SEARCH_GATEWAY_URL`; the agents call `<gateway>/search`, `<gateway>/terminal` and `<gateway>/v1`.
- **Lifecycle.** When training succeeds the head writes `complete` in the submission's lifecycle directory, and every
  service task exits 0. A failed or preempted Miles task stops the services (failure propagation), and a failed Redis,
  gateway or judge stops the Miles task (Prime-RL's critical services); the terminal and search replicas fail alone.
- **Shared inputs.** The pinned Prime-RL source is a copy under `/weka/oe-adapt-default/shashankg/miles_search/`; the
  corpus, index, JDK and Anserini jar are Steven's, read in place.

## Run

```bash
python examples/tmax/beaker/submit.py examples/tmax_search/configs/smoke_search_4b.yaml [--dry-run]
```

Locally (no Miles), the services run from Steven's images under podman and the agent loop against vLLM: see
[DEVELOPING.md](https://github.com/radixark/miles/blob/main/examples/tmax_search/DEVELOPING.md).

## Validation

Local, against the pinned source's services and Prime-RL's own trajectories of its nearest run
(`search-agent-001-3-quote-v3`: same data, corpus and environment, lr 4e-6, bf16):

| Check | Result |
|---|---|
| Tool calls of 100 Prime-RL trajectories (steps 1 and 200) replayed through the port's tool layer | 4,852 of 4,875 observations identical; the rest differ because the pinned terminal service is newer than that run's (it reports `error_type` and accepts `\|` in quoted patterns). Stderr penalties identical in all 100 |
| Port's reward on 120 Prime-RL trajectories (step 1), the judge Qwen3.8-27B in fp8 | Same format verdicts in 120 of 120, same judge verdicts in 112 of 112, same reward in 120 of 120 |
| 16 live episodes of Qwen/Qwen3.5-4B (8 questions x 2) through the port's loop | All end with a final answer, 42 turns on average (Prime-RL step 1: 41); mean reward 0.55, 62% correct |

On Holmes, the smoke configs (3 steps of 2 questions x 8 rollouts, B300s); every task of each exited 0 once the
Miles task marked the run complete:

| Smoke | Steps (grad norm) | Train/rollout log-prob gap | Episodes judged (correct) |
|---|---|---|---|
| `smoke_search_4b` (bf16 logits), `01M662W57MFXW38HEYWDQJYAHN` | 0.27, 0.37, 0.35 | 0.011–0.013 | 92 of 93 (45) |
| `smoke_search_4b_fp32_head`, `01M667F63PRPE8ZXA9YB9J1HX9` | 0.23, 0.52, 0.39 | 0.0078, 0.0081, 0.0080 | 83 of 84 (35) |
| `smoke_search_4b_fp32_head_imgd0831a475b` (image `tmax-miles-train-d0831a475b`, no min_runtime), `01M78B4DB8BB6T8ASJNCTDMT69` | 0.44, 0.35, 0.34 | 0.0073, 0.0079, 0.0081 | 102 of 102 (48) |

The fp32 head lowers the gap by about 30%; the training GPUs peaked at about 100 of 268 GB. Materializing the fp32
logits instead (the first version of `--fp32-lm-head`) needed 110 GB more on the second step's log-prob pass and ran
out of memory.

## Known differences

- **Tool schema tokens.** Miles renders the tools through SGLang's `Tool` model, as SGLang's own chat serving does: each
  tool's `description` comes before its `name`, and `"defer_loading": null` is added. On a Prime-RL step-1 trajectory
  the first prompt is 1,124 tokens against Prime-RL's 1,112, and those two tool lines are the only difference.
- **Judge failures.** A judge that never parses drops its whole group (Miles' missing-reward filter); Prime-RL drops
  only that rollout and keeps a group with two valid ones.
- **Length-truncated turns** end the episode, unjudged: the session server cannot extend a turn without `<|im_end|>`.
  Prime-RL runs a truncated turn's complete tool calls and goes on.
- **Staleness.** Miles drops groups more than 4 weight versions old at training time; Prime-RL cancels episodes that
  saw more than 4 updates.
- **Tool-call parsing** is SGLang's `qwen3_coder` parser, not Prime-RL's renderer.
