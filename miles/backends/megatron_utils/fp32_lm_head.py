"""``--fp32-lm-head`` in the Megatron backend: the actor's output layer hands the loss its hidden states.

SGLang's ``--enable-fp32-lm-head`` multiplies the bf16 hidden states by the bf16 LM-head weight with an fp32 result,
where Megatron's output layer rounds its logits to bf16. Instead of materializing fp32 logits for every token (twice the
bf16 head's ``[tokens, vocab]`` memory, which did not fit 131k-token micro-batches of Qwen3.5-4B on a B300),
``install_fp32_lm_head`` makes the output layer return its input and record its weight, and the loss computes the fp32
logits of the response rows a chunk at a time (``loss/hub/fp32_lm_head_log_probs.py``). The weight, its gradient buffer
and the weight sync stay bf16; its gradient reaches ``main_grad`` through Megatron DDP's usual hook.
"""

from collections.abc import Sequence

import torch
from megatron.core.tensor_parallel.layers import ColumnParallelLinear
from megatron.core.utils import unwrap_model

from miles.backends.training_utils.loss.hub.fp32_lm_head_log_probs import set_lm_head_weight


def install_fp32_lm_head(model_chunks: Sequence[torch.nn.Module]) -> int:
    """Make the output layer of every chunk that has one return its hidden states; returns how many it changed."""
    installed = 0
    for chunk in unwrap_model(list(model_chunks)):
        output_layer = getattr(chunk, "output_layer", None)
        if output_layer is None:  # not the last pipeline stage
            continue
        _check_supported(output_layer)
        # The instance attribute shadows ColumnParallelLinear._forward_impl, the one call that runs the GEMM;
        # ColumnParallelLinear.forward passes it keywords only.
        output_layer._forward_impl = _hidden_states_and_weight
        installed += 1
    return installed


def _check_supported(layer: torch.nn.Module) -> None:
    if type(layer)._forward_impl is not ColumnParallelLinear._forward_impl:
        raise ValueError(f"--fp32-lm-head needs a ColumnParallelLinear output layer, got {type(layer).__name__}")
    if layer.bias is not None:
        raise ValueError("--fp32-lm-head does not support an output layer with a bias")


def _hidden_states_and_weight(
    *,
    input: torch.Tensor,
    weight: torch.Tensor,
    bias: torch.Tensor | None,
    gradient_accumulation_fusion: bool,
    allreduce_dgrad: bool,
    sequence_parallel: bool,
    grad_output_buffer: list | None,
    wgrad_deferral_limit: int | None,
    tp_group: torch.distributed.ProcessGroup | None,
) -> torch.Tensor:
    """In place of ``linear_with_grad_accumulation_and_async_allreduce``: the input, unchanged, and the weight (the
    tied embedding's when shared) recorded for the loss. Tensor parallelism is rejected at launch, so there is no
    communication to keep."""
    del gradient_accumulation_fusion, grad_output_buffer, wgrad_deferral_limit, tp_group
    assert bias is None and not sequence_parallel and not allreduce_dgrad, "--fp32-lm-head needs TP 1"
    set_lm_head_weight(weight)
    return input
