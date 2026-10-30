"""Search-agent RL for Qwen3.5: Prime-RL's search-agent recipe on the TMax port's Miles setup.

=====================

Trains Qwen/Qwen3.5-4B (thinking on) to answer BrowseComp-style questions with two tools, a BM25 ``search`` over a
local corpus and a ``webterminal`` that pipes a corpus page through rg/sed/head; a Qwen3.8-27B judge scores the final
answer. It reproduces a Prime-RL search experiment, by default steven-prime-rl's
``configs/experiments/search-agent-bc-rl-v1-1-fp32-lr5e-7``: the environment's settings are read from that
experiment's ``rl.toml`` in the pinned Prime-RL source (``search_config``), the trainer's are this launcher's fields,
each set as Prime-RL's ``[trainer]`` and ``[orchestrator]`` tables set them (the mapping is in README.md).

Requires:
  * the pinned Prime-RL source (--prime-rl-source), for the environment's tool clients, prompts and judge templates;
  * the search, terminal and judge services behind a LiteRegistry gateway (--gateway-url, or SEARCH_GATEWAY_URL,
    which beaker/tmax_beaker_entry.sh exports for the services its experiment runs), or each service's URL.

`prepare` writes the prompt JSONL from the experiment's dataset and prompt template, downloads the model and converts
it to Megatron torch_dist. `execute` submits the training job.

=====================

Args:
  --model-name: a key of _RECIPES (default: Qwen3.5-4B, Prime-RL's Qwen/Qwen3.5-4B).
  --prime-rl-source: the pinned Prime-RL source tree (a sealed copy of steven-prime-rl).
  --prime-rl-config: the experiment's rl.toml, relative to --prime-rl-source.
  --prime-rl-checkout / --prime-rl-weka-mount: where the Prime-RL job had its checkout and its /weka, to map the
      experiment's paths into this job (default: /weka/stevenc/prime-rl and /weka/oe-adapt-default).
  --gateway-url: the services' LiteRegistry gateway (default: inherit SEARCH_GATEWAY_URL); or --search-url,
      --terminal-url and --judge-url for services without one.
  --rollout-batch-size / --n-samples-per-prompt: groups x rollouts per step (default: 8 x 32, Prime-RL's
      batch_size 256 / group_size 32).
  --max-seq-len: per-episode token budget, the prompt included (default: 131072, orchestrator seq_len).
  --rollout-max-response-len: per-turn generation cap (default: 32768, max_completion_tokens).
  --lr / --adam-beta1 / --adam-beta2 / --adam-eps: AdamW (default: 5e-7, 0.95, 0.95, 1e-12; constant schedule).
  --dppo-threshold: DPPO's trust region on the sampled token's probability (default: 0.2, dppo_mask_low/high).
  --group-token-mean / --no-group-token-mean: Prime-RL's loss aggregation (default: on; off is Miles' sample mean).
  --filter-zero-std: drop all-equal groups and refill (default: off: Prime-RL masks them without refilling).
  --fp32-lm-head / --no-fp32-lm-head: fp32 logits in the trainer and the engines (default: on).
  --async-rollout-gpus: rollout GPUs of a fully-async run (default: 16, Prime-RL's 16 inference GPUs).
  --max-weight-staleness: drop groups more than this many versions old (default: 4, max_off_policy_steps).
  --async-max-concurrent-samples: episodes in flight (default: 512, max_inflight_episodes).
  --log-kl-to-base / --no-log-kl-to-base: log the KL to the base model (default: on).
  --save-interval / --hf-snapshot-every: checkpoints and kept HF snapshots (default: 20 / 20, ckpt.interval).
  --trace-dir: append one JSONL record per episode and judgment under <trace-dir>/<exp-name> (default: off).
  --model-dir / --data-dir / --output-dir: checkpoint / dataset / output directories.

=====================

    python examples/tmax_search/run_search_qwen3_5.py prepare examples/tmax_search/configs/smoke_search_4b.yaml
    python examples/tmax_search/run_search_qwen3_5.py execute examples/tmax_search/configs/smoke_search_4b.yaml
"""

import hashlib
import json
import os
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

import typer
from search_config import ENV_ARGS_VAR, resolve_env_args

_SCRIPT_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(_SCRIPT_DIR.parent / "tmax"))
from launch_common import apply_yaml_config, fetch_command, wandb_args  # noqa: E402

from miles.utils.external_utils import command_utils  # noqa: E402

_MODEL_NAMES = Literal["Qwen3.5-4B"]


@dataclass(frozen=True)
class _Recipe:
    hf_repo: str
    megatron_model_type: str


_RECIPES: dict[str, _Recipe] = {
    # Prime-RL's model and renderer: stock Qwen/Qwen3.5-4B, thinking on (Miles' qwen35 template opens <think>).
    "Qwen3.5-4B": _Recipe("Qwen/Qwen3.5-4B", "qwen3.5-4B"),
}

app = typer.Typer()


@dataclass
class ScriptArgs(command_utils.ExecuteTrainConfig):
    run_id: str = command_utils.create_run_id()
    model_name: _MODEL_NAMES = "Qwen3.5-4B"
    prime_rl_source: str = (
        "/weka/oe-adapt-default/shashankg/miles_search/prime-rl-sources/"
        "a5fcb924645d9bd8f326efd5087c8de6447c83fe17c6d07c36e99ee736d30a37/source"
    )
    prime_rl_config: str = "configs/experiments/search-agent-bc-rl-v1-1-fp32-lr5e-7/rl.toml"
    prime_rl_checkout: str = "/weka/stevenc/prime-rl"
    prime_rl_weka_mount: str = "/weka/oe-adapt-default"
    # The services' gateway; empty inherits SEARCH_GATEWAY_URL. Each *_url, if set, wins over the gateway's route.
    gateway_url: str = ""
    search_url: str = ""
    terminal_url: str = ""
    judge_url: str = ""
    num_gpus_per_node: int = 8
    rollout_batch_size: int = 8
    n_samples_per_prompt: int = 32
    num_rollout: int = 200
    max_seq_len: int = 131072
    rollout_max_response_len: int = 32768
    tensor_model_parallel_size: int = 1
    context_parallel_size: int = 1
    max_tokens_per_gpu: int = 131072
    optimizer_cpu_offload: bool = False
    lr: float = 5e-7
    adam_beta1: float = 0.95
    adam_beta2: float = 0.95
    adam_eps: float = 1e-12
    dppo_threshold: float = 0.2
    group_token_mean: bool = True
    filter_zero_std: bool = False
    fp32_lm_head: bool = True
    async_rollout_gpus: int = 16
    max_weight_staleness: int = 4
    async_max_concurrent_samples: int = 512
    async_data_buffer_capacity_factor: float = 1.0
    sglang_mem_fraction_static: float = 0.8
    num_prompts: int = 0
    log_kl_to_base: bool = True
    save_interval: int = 20
    hf_snapshot_every: int = 20
    trace_dir: str = ""
    dump_training_data: bool = False
    session_server: Literal["v1", "v2"] = "v2"
    wandb_project: str = ""
    wandb_entity: str = ""
    exp_name: str = ""
    extra_args: str = ""
    data_dir: str = "/root/datasets"
    model_dir: str = "/root/models"
    megatron_path: str = "/root/Megatron-LM"

    def __post_init__(self):
        assert self.hf_snapshot_every >= 0 and self.hf_snapshot_every % self.save_interval == 0, (
            f"hf_snapshot_every ({self.hf_snapshot_every}) must be 0 or a multiple of save_interval "
            f"({self.save_interval}): HF snapshots are written at saves"
        )
        assert self.async_rollout_gpus > 0, "the search recipe trains fully async, as Prime-RL does"

    @property
    def recipe(self) -> _Recipe:
        return _RECIPES[self.model_name]

    @property
    def checkpoint_dir(self) -> str:
        return f"{self.output_dir}/{self.exp_name or self.run_id}/checkpoints"

    @property
    def trace_run_dir(self) -> str:
        return f"{self.trace_dir}/{self.exp_name or self.run_id}" if self.trace_dir else ""

    @property
    def experiment_dir(self) -> Path:
        return Path(self.prime_rl_source) / self.prime_rl_config.rsplit("/", 1)[0]

    @property
    def prompt_data(self) -> str:
        suffix = f"_first{self.num_prompts}" if self.num_prompts else ""
        return f"{self.data_dir}/search/{self.experiment_dir.name}{suffix}.jsonl"

    @property
    def train_layout(self) -> tuple[int, int]:
        """(actor nodes, actor GPUs per node): one split node, or whole nodes for each side, training first."""
        train_gpus = self.num_nodes * self.num_gpus_per_node - self.async_rollout_gpus
        assert train_gpus > 0, f"async_rollout_gpus={self.async_rollout_gpus} leaves no GPU for training"
        if self.num_nodes == 1:
            return 1, train_gpus
        assert train_gpus % self.num_gpus_per_node == 0, "on several nodes, async_rollout_gpus must be whole nodes"
        return train_gpus // self.num_gpus_per_node, self.num_gpus_per_node

    def env_args(self) -> dict:
        """The experiment's environment arguments, resolved for this job (search_config)."""
        gateway = (self.gateway_url or os.environ.get("SEARCH_GATEWAY_URL", "")).rstrip("/")
        urls = {
            "search_url": self.search_url or (gateway and f"{gateway}/search"),
            "terminal_url": self.terminal_url or (gateway and f"{gateway}/terminal"),
            "judge_url": self.judge_url or (gateway and f"{gateway}/v1"),
        }
        assert all(urls.values()), f"set --gateway-url (or SEARCH_GATEWAY_URL), or every service URL: {urls}"
        return resolve_env_args(
            prime_rl_source=self.prime_rl_source,
            prime_rl_config=self.prime_rl_config,
            prime_rl_checkout=self.prime_rl_checkout,
            weka_mount=self.prime_rl_weka_mount,
            **urls,
        )


@app.command()
@command_utils.dataclass_cli
def prepare(args: ScriptArgs):
    U = args.create_backend()
    _write_prompt_data(args)
    U.exec_command_cpu(fetch_command(args.recipe.hf_repo, f"{args.model_dir}/{args.model_name}"))
    U.convert_checkpoint(
        model_name=args.model_name,
        megatron_model_type=args.recipe.megatron_model_type,
        num_gpus_per_node=args.num_gpus_per_node,
        dir_dst=args.model_dir,
        hf_checkpoint=f"{args.model_dir}/{args.model_name}",
        megatron_path=args.megatron_path,
    )


def _write_prompt_data(args: ScriptArgs) -> None:
    """One JSONL row per question: the experiment's prompt template rendered with it (search_agent/data.py), plus
    the id, question and answer the judge needs. The dataset is checked against the experiment's manifest."""
    env = resolve_env_args(
        prime_rl_source=args.prime_rl_source,
        prime_rl_config=args.prime_rl_config,
        prime_rl_checkout=args.prime_rl_checkout,
        weka_mount=args.prime_rl_weka_mount,
        search_url="-",
        terminal_url="-",
        judge_url="-",
    )
    dataset = Path(env["dataset"])
    manifest = args.experiment_dir / "corpus-manifest.json"
    if manifest.exists():
        expected = json.loads(manifest.read_text())["training_dataset"]["sha256"]
        actual = hashlib.sha256(dataset.read_bytes()).hexdigest()
        assert actual == expected, f"{dataset}: sha256 {actual}, the experiment's manifest records {expected}"
    rows = [json.loads(line) for line in dataset.read_text().splitlines() if line.strip()]
    if args.num_prompts:
        rows = rows[: args.num_prompts]
    out = Path(args.prompt_data)
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w") as f:
        for row in rows:
            question = row["question"].strip()
            prompt = [
                {"role": m["role"], "content": m["content"].format(question=question)}
                for m in env["template"]["chat_template"]
            ]
            metadata = {"id": row["id"], "question": question, "answer": row["answer"]}
            f.write(json.dumps({"prompt": prompt, "metadata": metadata}) + "\n")


@app.command()
@command_utils.dataclass_cli
def execute(args: ScriptArgs):
    U = args.create_backend()
    env = args.env_args()

    ckpt_args = (
        f"--hf-checkpoint {args.model_dir}/{args.model_name} "
        f"--ref-load {args.model_dir}/{args.model_name}_torch_dist "
        f"--load {args.checkpoint_dir} "
        f"--save {args.checkpoint_dir} "
        f"--save-interval {args.save_interval} "
    )
    if args.hf_snapshot_every:
        ckpt_args += f"--save-hf {args.checkpoint_dir}/hf/step_{{rollout_id}} "

    # [orchestrator] batch_size 256 = 8 groups x group_size 32; [orchestrator.train.sampling] temperature 1, top_p 1,
    # top_k -1, max_completion_tokens 32768; seq_len 131072.
    rollout_args = (
        f"--prompt-data {args.prompt_data} "
        "--input-key prompt "
        "--metadata-key metadata "
        "--rollout-shuffle "
        f"--num-rollout {args.num_rollout} "
        f"--rollout-batch-size {args.rollout_batch_size} "
        f"--n-samples-per-prompt {args.n_samples_per_prompt} "
        "--rollout-temperature 1.0 "
        "--rollout-top-p 1.0 "
        "--rollout-top-k -1 "
        f"--rollout-max-response-len {args.rollout_max_response_len} "
        f"--max-seq-len {args.max_seq_len} "
        f"--global-batch-size {args.rollout_batch_size * args.n_samples_per_prompt} "
        "--balance-data "
    )

    group_filter = "nonzero_std_filter" if args.filter_zero_std else "missing_reward_filter"
    agent_args = (
        "--custom-generate-function-path miles.rollout.generate_hub.agentic_tool_call.generate "
        "--custom-agent-function-path search_agent_function.run "
        "--custom-rm-path search_reward.reward_func "
        f"--use-session-server {args.session_server} "
        "--tito-model qwen35 "
        f"--dynamic-sampling-filter-path search_metrics.{group_filter} "
        "--custom-rollout-log-function-path search_metrics.log_rollout_data "
    )

    perf_args = (
        f"--tensor-model-parallel-size {args.tensor_model_parallel_size} "
        "--sequence-parallel "
        "--pipeline-model-parallel-size 1 "
        f"--context-parallel-size {args.context_parallel_size} "
        "--recompute-granularity full "  # [trainer.model.ac] mode full, freq 1
        "--recompute-method uniform "
        "--recompute-num-layers 1 "
        "--use-dynamic-batch-size "
        f"--max-tokens-per-gpu {args.max_tokens_per_gpu} "
    )

    # [trainer.loss] type default, kl_tau 0, dppo_mask_low/high 0.2: a mask on the sampled token's probability moving
    # past the threshold in the advantage's direction, no ratio clip, anchored on the inference log-probs.
    grpo_args = (
        "--advantage-estimator grpo "
        "--disable-grpo-std-normalization "
        "--use-rollout-logprobs "
        "--entropy-coef 0.0 "
        "--observe-training-entropy "
        f"--policy-loss-type dppo_tv --dppo-tv-threshold {args.dppo_threshold} "
    )
    if args.group_token_mean:
        # [trainer.loss] aggregation group_token_mean, on Miles' sample-mean reducer (search_loss.py)
        grpo_args += "--custom-convert-samples-to-train-data-path search_loss.convert_samples_to_train_data "
    else:
        grpo_args += "--calculate-per-token-loss "
    if args.log_kl_to_base:
        grpo_args += "--use-kl-loss --kl-loss-coef 0.0 --kl-loss-type k3 "

    # [trainer.optim] adamw, max_norm 1.0, weight_decay 0; [trainer.scheduler] constant. Megatron keeps fp32 master
    # weights and Adam state with bf16 compute ([trainer.model] optimization_dtype float32).
    optimizer_args = (
        "--optimizer adam "
        f"--lr {args.lr} "
        "--lr-decay-style constant "
        "--weight-decay 0.0 "
        f"--adam-beta1 {args.adam_beta1} "
        f"--adam-beta2 {args.adam_beta2} "
        f"--adam-eps {args.adam_eps} "
        "--clip-grad 1.0 "
    )
    if args.optimizer_cpu_offload:
        optimizer_args += "--optimizer-cpu-offload --overlap-cpu-optimizer-d2h-h2d --use-precision-aware-optimizer "

    # SGLang TP>1 produces garbage output for Qwen3.5 (sgl-project/sglang#21039), hence one GPU per engine.
    sglang_args = (
        "--rollout-num-gpus-per-engine 1 "
        f"--sglang-mem-fraction-static {args.sglang_mem_fraction_static} "
        "--sglang-reasoning-parser qwen3 "
        "--sglang-tool-call-parser qwen3_coder "
    )
    if args.fp32_lm_head:
        # [trainer.model] fused_lm_head_fp32_logits and [inference] enable_fp32_lm_head (its default): both sides.
        sglang_args += "--sglang-enable-fp32-lm-head "

    misc_args = (
        "--attention-dropout 0.0 "
        "--hidden-dropout 0.0 "
        "--accumulate-allreduce-grads-in-fp32 "  # [trainer.model] reduce_dtype float32
        "--attention-softmax-in-fp32 "
        "--attention-backend flash "
        f"--actor-num-nodes {args.train_layout[0]} "
        f"--actor-num-gpus-per-node {args.train_layout[1]} "
        f"--num-gpus-per-node {args.num_gpus_per_node} "
        # [orchestrator] max_off_policy_steps 4, max_inflight_episodes 512; in-flight weight updates.
        "--fully-async "
        f"--max-weight-staleness {args.max_weight_staleness} "
        f"--async-data-buffer-capacity-factor {args.async_data_buffer_capacity_factor} "
        f"--async-max-concurrent-samples {args.async_max_concurrent_samples} "
        "--pause-generation-mode in_place "
        "--update-weight-transfer-mode broadcast "
        f"--rollout-num-gpus {args.async_rollout_gpus} "
    )
    if args.fp32_lm_head:
        misc_args += "--fp32-lm-head "
    if args.num_nodes > 1:
        misc_args += "--pin-rollout-manager-to-head "
    if args.dump_training_data:
        assert args.trace_dir, "dump_training_data writes next to the traces: set trace_dir"
        misc_args += (
            f"--save-debug-rollout-data {args.trace_run_dir}/rollout_data/{{rollout_id}}.pt "
            f"--save-debug-train-data {args.trace_run_dir}/train_data/{{rollout_id}}_{{rank}}.pt "
        )

    wandb = wandb_args(
        __file__, run_id=args.run_id, exp_name=args.exp_name, project=args.wandb_project, entity=args.wandb_entity
    )
    train_args = (
        f"{ckpt_args} {rollout_args} {agent_args} {optimizer_args} {grpo_args} {wandb} {perf_args} "
        f"{sglang_args} {misc_args} {args.extra_args} "
    )

    source = Path(args.prime_rl_source)
    prime_rl_paths = [
        source / "deps/literegistry/literegistry_tool_client/src",  # literegistry_tool_client
        source / "environments/search_agent",  # search_agent.{protocol,tools,tool_metrics}
    ]
    U.execute_train(
        train_args=train_args,
        num_gpus_per_node=args.num_gpus_per_node,
        megatron_model_type=args.recipe.megatron_model_type,
        megatron_path=args.megatron_path,
        train_script="train_async.py",
        extra_env_vars={
            "PYTHONPATH": ":".join(
                [
                    args.megatron_path,
                    str(_SCRIPT_DIR),
                    str(_SCRIPT_DIR.parent / "tmax"),
                    str(command_utils.repo_base_dir),
                ]
                + [str(p) for p in prime_rl_paths]
            ),
            ENV_ARGS_VAR: json.dumps(env),
            "SEARCH_TRACE_DIR": args.trace_run_dir,
        },
    )


if __name__ == "__main__":
    sys.argv = apply_yaml_config(sys.argv, ScriptArgs)
    app()
