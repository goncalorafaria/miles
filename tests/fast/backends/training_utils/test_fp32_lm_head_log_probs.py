"""``--fp32-lm-head``'s chunked LM head: log-probs, entropy and gradients equal to the materialized fp32 logits' (CPU)."""

import pytest
import torch

from miles.backends.training_utils.loss.hub import fp32_lm_head_log_probs as op

_N, _H, _V, _PAD = 23, 16, 50, 14  # 50 real tokens, 14 rows of vocab padding


def _inputs(seed=0):
    g = torch.Generator().manual_seed(seed)
    hidden = torch.randn(_N, _H, generator=g).bfloat16().requires_grad_(True)
    weight = (torch.randn(_V + _PAD, _H, generator=g) * 0.3).bfloat16().requires_grad_(True)
    rows = torch.tensor([1, 2, 3, 5, 8, 9, 10, 11, 15, 20, 21])
    targets = torch.randint(0, _V, rows.shape, generator=g)
    return hidden, weight, rows, targets


def _reference(hidden, weight, rows, targets, temperature, with_entropy):
    """The materialized fp32 logits of the bf16 operands, in fp64."""
    logits = (hidden[rows].double() @ weight[:_V].double().t()) / temperature
    log_softmax = torch.log_softmax(logits, dim=-1)
    log_probs = log_softmax.gather(1, targets.unsqueeze(-1)).squeeze(-1)
    entropy = -(log_softmax.exp() * log_softmax).sum(-1) if with_entropy else None
    return log_probs, entropy


@pytest.mark.parametrize("chunk_rows", [1, 4, 2048])
@pytest.mark.parametrize("temperature", [1.0, 0.7])
def test_forward_and_backward_match_the_materialized_head(monkeypatch, chunk_rows, temperature):
    monkeypatch.setattr(op, "_CHUNK_ROWS", chunk_rows)
    hidden, weight, rows, targets = _inputs()
    log_probs, entropy = op.fp32_lm_head_log_probs_and_entropy(
        hidden, rows, targets, weight, vocab_size=_V, temperature=temperature, with_entropy=True
    )
    ref_hidden = hidden.detach().double().requires_grad_(True)
    ref_weight = weight.detach().double().requires_grad_(True)
    ref_log_probs, ref_entropy = _reference(ref_hidden, ref_weight, rows, targets, temperature, True)
    torch.testing.assert_close(log_probs.double(), ref_log_probs, rtol=0, atol=1e-5)
    torch.testing.assert_close(entropy.double(), ref_entropy, rtol=0, atol=1e-5)

    g = torch.Generator().manual_seed(1)
    grad_lp, grad_ent = torch.randn(rows.shape, generator=g), torch.randn(rows.shape, generator=g)
    (log_probs * grad_lp).sum().add((entropy * grad_ent).sum()).backward()
    (ref_log_probs * grad_lp.double()).sum().add((ref_entropy * grad_ent.double()).sum()).backward()
    # the bf16 head's backward: bf16 logits gradient, bf16 hidden-state gradient, fp32-accumulated weight gradient
    torch.testing.assert_close(hidden.grad.double(), ref_hidden.grad, rtol=2e-2, atol=2e-2)
    torch.testing.assert_close(weight.grad.double(), ref_weight.grad, rtol=2e-2, atol=2e-2)
    assert hidden.grad[[0, 4, 22]].abs().sum() == 0  # rows that are not scored
    assert weight.grad[_V:].abs().sum() == 0  # vocab padding


def test_observed_entropy_has_no_gradient():
    hidden, weight, rows, targets = _inputs()
    log_probs, entropy = op.fp32_lm_head_log_probs_and_entropy(
        hidden, rows, targets, weight, vocab_size=_V, with_entropy=True, entropy_requires_grad=False
    )
    assert not entropy.requires_grad
    log_probs.sum().backward()
    ref_hidden = hidden.detach().double().requires_grad_(True)
    ref_log_probs, _ = _reference(ref_hidden, weight.detach().double(), rows, targets, 1.0, False)
    ref_log_probs.sum().backward()
    torch.testing.assert_close(hidden.grad.double(), ref_hidden.grad, rtol=2e-2, atol=2e-2)


def test_the_recorded_weight_is_the_one_scored():
    _, weight, _, _ = _inputs()
    op.set_lm_head_weight(weight)
    assert op.lm_head_weight() is weight
