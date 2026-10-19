"""TMax Terminal-RL for Qwen3.5 MoE models (default Qwen3.5-35B-A3B): run_tmax_qwen3_5.py plus expert parallelism.

=====================

The same recipe as run_tmax_qwen3_5.py (dataset, sandboxes, agent loop, GRPO, configs, every flag documented
there), for the sparse Qwen3.5 checkpoints. On top of it this launcher:

  * shards the experts across GPUs (--expert-model-parallel-size; tensor parallelism defaults to 1);
  * replays the rollout's expert routing in training, R3 (--use-rollout-routing-replay), on by default;
  * can train the checkpoint's MTP head and draft with it in rollouts (EAGLE), off by default;
  * picks SGLang's MoE runner, which B300 needs for in-place weight updates.

The -tmax-template recipe carries hamishivi/Qwen3.5-9B's chat-template change, which training does not read
(README.md, Findings) but evals do.

=====================

Args (in addition to run_tmax_qwen3_5.py's):
  --model-name: student, a key of _MOE_RECIPES (default: Qwen3.5-35B-A3B-tmax-template).
  --expert-model-parallel-size: Megatron EP; TP x CP x EP must divide the training GPUs (default: 8).
  --rollout-routing-replay: R3, train with the experts the rollout picked (default: on). In synchronous mode,
      where --pause-generation-mode stays retract, the session server returns every turn's whole routing record
      instead of only the new rows: correct, just larger responses.
  --mtp-training: train the MTP layer (--enable-mtp-training, loss factor 0.2) and decode with it as an EAGLE
      draft (default: off).
  --sglang-moe-runner-backend: SGLang MoE kernels, empty for SGLang's default (default: flashinfer_cutlass, which
      upstream's Qwen3.5-35B-A3B recipe requires on B300).

=====================

  python examples/tmax/run_tmax_qwen3_5_moe.py prepare examples/tmax/configs/smoke_moe_sync.yaml
  python examples/tmax/run_tmax_qwen3_5_moe.py execute examples/tmax/configs/smoke_moe_sync.yaml

On Beaker set ``launcher: examples/tmax/run_tmax_qwen3_5_moe.py`` in the config's beaker: section.
"""

import sys
from dataclasses import dataclass
from typing import Literal

import run_tmax_qwen3_5 as dense
import typer

from miles.utils.external_utils import command_utils

_MOE_MODEL_NAMES = Literal["Qwen3.5-35B-A3B", "Qwen3.5-35B-A3B-tmax-template"]

_MOE_RECIPES: dict[str, dense._Recipe] = {
    "Qwen3.5-35B-A3B": dense._Recipe("Qwen/Qwen3.5-35B-A3B", "qwen3.5-35B-A3B", enable_thinking=True),
    # Qwen/Qwen3.5-35B-A3B with hamishivi/Qwen3.5-9B's chat-template change (past turns keep their reasoning)
    "Qwen3.5-35B-A3B-tmax-template": dense._Recipe("shatu/Qwen3.5-35B-A3B", "qwen3.5-35B-A3B", enable_thinking=True),
}

app = typer.Typer()


@dataclass
class MoeScriptArgs(dense.ScriptArgs):
    model_name: _MOE_MODEL_NAMES = "Qwen3.5-35B-A3B-tmax-template"
    opd_teacher: _MOE_MODEL_NAMES | None = None
    tensor_model_parallel_size: int = 1
    expert_model_parallel_size: int = 8
    rollout_routing_replay: bool = True
    mtp_training: bool = False
    sglang_moe_runner_backend: str = "flashinfer_cutlass"

    def recipe_for(self, model_name: str) -> dense._Recipe:
        return _MOE_RECIPES[model_name]

    def model_specific_train_args(self) -> str:
        out = f"--expert-model-parallel-size {self.expert_model_parallel_size} --expert-tensor-parallel-size 1 "
        if self.rollout_routing_replay:
            out += "--use-rollout-routing-replay "
        if self.sglang_moe_runner_backend:
            out += f"--sglang-moe-runner-backend {self.sglang_moe_runner_backend} "
        # The model script builds the MTP layer (--mtp-num-layers 1), and conversion keeps it; without
        # --enable-mtp-training Miles drops it from the trained model, and the engines decode without a draft.
        if self.mtp_training:
            out += (
                "--enable-mtp-training "
                "--mtp-loss-scaling-factor 0.2 "
                "--sglang-speculative-algorithm EAGLE "
                "--sglang-speculative-num-steps 2 "
                "--sglang-speculative-eagle-topk 1 "
                "--sglang-speculative-num-draft-tokens 3 "
            )
        return out


@app.command()
@command_utils.dataclass_cli
def prepare(args: MoeScriptArgs):
    dense.run_prepare(args)


@app.command()
@command_utils.dataclass_cli
def execute(args: MoeScriptArgs):
    dense.run_execute(args)


if __name__ == "__main__":
    sys.argv = dense.apply_yaml_config(sys.argv, MoeScriptArgs)
    app()
