"""fp32 logits from Megatron's output layer, computed the way SGLang's ``--enable-fp32-lm-head`` computes them.

SGLang's ``LogitsProcessor._compute_lm_head`` multiplies the bf16 hidden states by the bf16 LM-head weight with
``torch.mm(..., out_dtype=torch.float32)``: fp32 accumulation and an fp32 result. Megatron's output layer runs the
same GEMM but rounds its result to bf16, which Miles only upcasts afterwards. ``install_fp32_lm_head`` swaps in
SGLang's GEMM. The tensor-parallel communication around it is Megatron's own region functions, and the weight, its
gradient buffer and the weight sync stay bf16.

The backward is the bf16 head's: the fp32 logits gradient is cast to bf16 a few thousand rows at a time (no bf16
copy of the whole ``[tokens, vocab / TP]`` gradient next to the fp32 one), and the weight gradient accumulates in
fp32 straight into ``main_grad`` under ``--gradient-accumulation-fusion``, as Megatron's linear does.
"""

from collections.abc import Sequence

import torch
from megatron.core.tensor_parallel.layers import ColumnParallelLinear
from megatron.core.tensor_parallel.mappings import (
    copy_to_tensor_model_parallel_region,
    gather_from_sequence_parallel_region,
)
from megatron.core.utils import unwrap_model

try:  # what Megatron's own linear backward returns as the placeholder weight gradient
    from transformer_engine.pytorch.module.base import get_dummy_wgrad
except ImportError:
    get_dummy_wgrad = None

# Logits-gradient rows cast to bf16 at once: 4096 x 248320 x 2 B = 2 GB for Qwen3.5 at TP 1.
_BACKWARD_CHUNK_ROWS = 4096


def install_fp32_lm_head(model_chunks: Sequence[torch.nn.Module]) -> int:
    """Make the output layer of every chunk that has one return fp32 logits; returns how many it changed."""
    installed = 0
    for chunk in unwrap_model(list(model_chunks)):
        output_layer = getattr(chunk, "output_layer", None)
        if output_layer is None:  # not the last pipeline stage
            continue
        _check_supported(output_layer)
        # The instance attribute shadows ColumnParallelLinear._forward_impl, the one call that runs the GEMM;
        # ColumnParallelLinear.forward passes it keywords only.
        output_layer._forward_impl = _fp32_logits_linear
        installed += 1
    return installed


def fp32_logits_gemm(hidden: torch.Tensor, weight: torch.Tensor) -> torch.Tensor:
    """``hidden @ weight.T`` as fp32: SGLang's fp32 LM head, with its fallback for CPU or mixed dtypes."""
    if hidden.is_cuda and hidden.dtype == weight.dtype and hidden.dtype in (torch.float16, torch.bfloat16):
        return torch.mm(hidden, weight.t(), out_dtype=torch.float32)
    return torch.mm(hidden.float(), weight.float().t())


def _check_supported(layer: torch.nn.Module) -> None:
    if type(layer)._forward_impl is not ColumnParallelLinear._forward_impl:
        raise ValueError(f"--fp32-lm-head needs a ColumnParallelLinear output layer, got {type(layer).__name__}")
    unsupported = [
        name
        for name, enabled in (
            ("a bias", layer.bias is not None),
            ("--defer-embedding-wgrad-compute", layer.config.defer_embedding_wgrad_compute),
            ("--delay-wgrad-compute", getattr(layer.config, "delay_wgrad_compute", False)),
        )
        if enabled
    ]
    if unsupported:
        raise ValueError(f"--fp32-lm-head does not support an output layer with {', '.join(unsupported)}")


def _fp32_logits_linear(
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
    """``linear_with_grad_accumulation_and_async_allreduce`` with an fp32 result and the same communication."""
    del wgrad_deferral_limit  # read only with grad_output_buffer, which _check_supported rules out
    assert bias is None and grad_output_buffer is None
    assert not hasattr(weight, "__fsdp_param__"), "--fp32-lm-head does not support Megatron-FSDP"
    if sequence_parallel:  # gather the sequence shards; their gradient is reduce-scattered back
        input = gather_from_sequence_parallel_region(input, tensor_parallel_output_grad=True, group=tp_group)
    elif allreduce_dgrad:  # every TP rank holds the whole input; its gradient is all-reduced
        input = copy_to_tensor_model_parallel_region(input, group=tp_group)
    return _Fp32LogitsLinear.apply(input, weight, gradient_accumulation_fusion)


class _Fp32LogitsLinear(torch.autograd.Function):
    """fp32 ``input @ weight.T`` from bf16 operands; the backward GEMMs are the bf16 head's."""

    @staticmethod
    def forward(ctx, input, weight, gradient_accumulation_fusion):
        ctx.save_for_backward(input, weight)
        ctx.gradient_accumulation_fusion = gradient_accumulation_fusion
        logits = fp32_logits_gemm(input.reshape(-1, input.size(-1)), weight)
        return logits.view(*input.shape[:-1], weight.size(0))

    @staticmethod
    def backward(ctx, grad_logits):
        input, weight = ctx.saved_tensors
        need_input_grad, need_weight_grad = ctx.needs_input_grad[:2]
        into_main_grad = need_weight_grad and ctx.gradient_accumulation_fusion and hasattr(weight, "main_grad")
        grad_input = torch.empty_like(input, memory_format=torch.contiguous_format) if need_input_grad else None
        weight_grad = None
        if need_weight_grad:
            weight_grad = weight.main_grad if into_main_grad else weight.new_zeros(weight.shape, dtype=torch.float32)
        # chunk along dim 0 (the sequence, for Megatron's [s, b, h]) so a transposed bshd gradient needs no full copy
        rows_per_step = max(1, input[0].numel() // input.size(-1)) if input.dim() > 1 else 1
        step = max(1, _BACKWARD_CHUNK_ROWS // rows_per_step)
        for start in range(0, input.size(0), step):
            chunk = slice(start, start + step)
            # the bf16 gradient a bf16 head's backward receives, made contiguous for the GEMMs
            grad_rows = grad_logits[chunk].to(input.dtype, memory_format=torch.contiguous_format)
            grad_rows = grad_rows.view(-1, weight.size(0))
            if grad_input is not None:
                torch.mm(grad_rows, weight, out=grad_input[chunk].view(-1, input.size(-1)))
            if weight_grad is not None:
                _addmm_(weight_grad, grad_rows.t(), input[chunk].reshape(-1, input.size(-1)))
        if not need_weight_grad:
            grad_weight = None
        elif into_main_grad:
            grad_weight = _main_grad_placeholder(weight, input.dtype)
        else:
            grad_weight = weight_grad.to(weight.dtype)
        return grad_input, grad_weight, None


def _addmm_(acc: torch.Tensor, a: torch.Tensor, b: torch.Tensor) -> None:
    """``acc += a @ b`` for bf16 ``a`` and ``b``, accumulating in ``acc``'s dtype without an ``[a, b]`` temporary on CUDA."""
    if acc.dtype == a.dtype:
        acc.addmm_(a, b)
    elif acc.is_cuda:
        torch.addmm(acc, a, b, out_dtype=acc.dtype, out=acc)
    else:
        acc.add_(a.to(acc.dtype) @ b.to(acc.dtype))


def _main_grad_placeholder(weight: torch.Tensor, dtype: torch.dtype) -> torch.Tensor | None:
    """What Megatron's linear returns after adding the weight gradient to ``main_grad`` itself."""
    if not hasattr(weight, "grad_added_to_main_grad"):
        return None
    # Megatron DDP's grad hook needs a gradient to fire; the flag tells it the gradient is already in main_grad.
    # zero_out_wgrad marks an embedding tied on the same stage: autograd adds this placeholder to the embedding's
    # real gradient, so it must be zeros.
    weight.grad_added_to_main_grad = True
    zero = getattr(weight, "zero_out_wgrad", False)
    if get_dummy_wgrad is not None:
        return get_dummy_wgrad(list(weight.main_grad.shape), dtype, zero=zero)
    return (torch.zeros if zero else torch.empty)(weight.main_grad.shape, dtype=dtype, device=weight.device)
