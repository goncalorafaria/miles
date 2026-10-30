"""``--fp32-lm-head``'s output GEMM: SGLang's fp32 logits forward, the bf16 head's backward (CPU, TP 1)."""

import pytest
import torch

pytest.importorskip("megatron.core")

from miles.backends.megatron_utils import fp32_lm_head
from miles.backends.megatron_utils.fp32_lm_head import _fp32_logits_linear, fp32_logits_gemm

_S, _B, _H, _V = 37, 2, 64, 1000


def _inputs(seed=0):
    g = torch.Generator().manual_seed(seed)
    hidden = torch.randn(_S, _B, _H, generator=g).bfloat16().requires_grad_(True)
    weight = (torch.randn(_V, _H, generator=g) * 0.05).bfloat16().requires_grad_(True)
    return hidden, weight


def _apply(hidden, weight, *, fusion=False):
    return _fp32_logits_linear(
        input=hidden,
        weight=weight,
        bias=None,
        gradient_accumulation_fusion=fusion,
        allreduce_dgrad=False,
        sequence_parallel=False,
        grad_output_buffer=None,
        wgrad_deferral_limit=None,
        tp_group=None,
    )


def _loss(logits, targets):
    # [s, b, V] -> [b, s, V] as GPTModel returns it, so the gradient arrives transposed (bshd)
    return -torch.log_softmax(logits.transpose(0, 1), dim=-1).gather(-1, targets[..., None]).sum()


def test_forward_is_the_fp32_gemm_of_the_bf16_operands():
    hidden, weight = _inputs()
    logits = _apply(hidden, weight)
    assert logits.dtype == torch.float32 and logits.shape == (_S, _B, _V)
    expected = (hidden.detach().reshape(-1, _H).double() @ weight.detach().double().t()).view(_S, _B, _V)
    torch.testing.assert_close(logits.double(), expected, rtol=0, atol=1e-5)
    # not the bf16 head: its rounding is far above the fp32 GEMM's
    assert (logits - (hidden @ weight.t()).float()).abs().max() > 1e-3


@pytest.mark.parametrize("chunk_rows", [1, 5, 4096])
def test_backward_is_the_bf16_heads_backward_of_the_fp32_logits(monkeypatch, chunk_rows):
    monkeypatch.setattr(fp32_lm_head, "_BACKWARD_CHUNK_ROWS", chunk_rows)
    hidden, weight = _inputs()
    targets = torch.randint(0, _V, (_B, _S), generator=torch.Generator().manual_seed(1))
    _loss(_apply(hidden, weight), targets).backward()

    logits = fp32_logits_gemm(hidden.detach().reshape(-1, _H), weight.detach()).view(_S, _B, _V).requires_grad_(True)
    _loss(logits, targets).backward()
    grad16 = logits.grad.reshape(-1, _V).bfloat16().double()  # what a bf16 head's backward is handed
    torch.testing.assert_close(hidden.grad, (grad16 @ weight.detach().double()).view(_S, _B, _H).bfloat16())
    torch.testing.assert_close(weight.grad, (grad16.t() @ hidden.detach().reshape(-1, _H).double()).bfloat16())


@pytest.mark.parametrize("tied", [False, True])
def test_fused_weight_gradient_goes_to_main_grad(tied):
    hidden, weight = _inputs()
    reference_hidden, reference_weight = _inputs()
    weight.main_grad = torch.full(weight.shape, 0.5)
    weight.grad_added_to_main_grad = False
    if tied:  # Megatron marks an embedding shared with the output layer on one pipeline stage
        weight.zero_out_wgrad = True
    targets = torch.randint(0, _V, (_B, _S), generator=torch.Generator().manual_seed(1))
    _loss(_apply(hidden, weight, fusion=True), targets).backward()
    _loss(_apply(reference_hidden, reference_weight), targets).backward()

    assert weight.grad_added_to_main_grad
    torch.testing.assert_close(weight.main_grad, 0.5 + reference_weight.grad.float(), rtol=1e-2, atol=1e-2)
    if tied:
        assert torch.count_nonzero(weight.grad) == 0  # added to the embedding's gradient by autograd
    torch.testing.assert_close(hidden.grad, reference_hidden.grad)


def test_frozen_weight_gets_no_gradient():
    hidden, weight = _inputs()
    weight.requires_grad_(False)
    _apply(hidden, weight).sum().backward()
    assert weight.grad is None and hidden.grad is not None
