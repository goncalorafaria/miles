"""Per-token log-probs and entropy from the final hidden states, the LM head's logits computed in fp32 a chunk at a time.

With ``--fp32-lm-head`` the actor's output layer returns its input, the bf16 hidden states, instead of bf16 logits, and
records its weight (``set_lm_head_weight``). ``fp32_lm_head_log_probs_and_entropy`` then multiplies a chunk of response
rows by the bf16 weight the way SGLang's ``--enable-fp32-lm-head`` does (``torch.mm(..., out_dtype=torch.float32)``: fp32
accumulation and an fp32 result), takes the log-softmax over the true vocabulary, and recomputes each chunk in the
backward pass, so the ``[tokens, vocab]`` logits never exist at once: about 2 GB of fp32 logits per chunk instead of the
whole bf16 tensor. open-instruct's ``--lm_head_fp32`` and Prime-RL's ``fused_lm_head_fp32_logits`` compute the same in
their chunked losses.

The statistics are the fused log-prob op's (fused_log_probs.py): per row the max ``m``, ``d = (z - m) / T``,
``log S = logsumexp(d)``, the target's ``d_y`` and the softmax mean ``mu`` of ``d``; ``log p = d_y - log S`` and
``H = log S - mu``. The backward writes ``(g * (onehot(y) - p) - c * p * (d - mu)) / T`` per chunk, then the bf16 head's
two GEMMs: the hidden-state gradient in bf16, the weight gradient accumulated in fp32.

Tensor parallelism is not supported (the launch rejects it): every rank holds the whole vocabulary.
"""

from dataclasses import dataclass

import torch
from torch import Tensor

# Rows per chunk: 2048 x 248320 x 4 B = 2 GB of fp32 logits for Qwen3.5, a little over twice that at the peak.
_CHUNK_ROWS = 2048

# The output layer's weight of the forward that produced the hidden states now being scored. The loss of a micro-batch
# runs right after its forward, in the same process, so the latest one is the right one.
_lm_head_weight: Tensor | None = None


def set_lm_head_weight(weight: Tensor) -> None:
    global _lm_head_weight
    _lm_head_weight = weight


def lm_head_weight() -> Tensor:
    assert _lm_head_weight is not None, "--fp32-lm-head: no output-layer forward has recorded the LM-head weight"
    return _lm_head_weight


def fp32_logits(hidden: Tensor, weight: Tensor) -> Tensor:
    """``hidden @ weight.T`` as fp32: SGLang's fp32 LM head, with its fallback for CPU or mixed dtypes."""
    if hidden.is_cuda and hidden.dtype == weight.dtype and hidden.dtype in (torch.float16, torch.bfloat16):
        return torch.mm(hidden, weight.t(), out_dtype=torch.float32)
    return torch.mm(hidden.float(), weight.float().t())


def fp32_lm_head_log_probs_and_entropy(
    hidden: Tensor,
    rows: Tensor,
    targets: Tensor,
    weight: Tensor,
    *,
    vocab_size: int | None = None,
    temperature: float = 1.0,
    with_entropy: bool = False,
    entropy_requires_grad: bool = True,
) -> tuple[Tensor, Tensor | None]:
    """Fp32 ``[R]`` log-probs of ``targets`` (and entropies) at ``rows`` of ``hidden @ weight.T / temperature``.

    ``hidden`` is ``[N, H]``, ``weight`` the ``[V, H]`` LM head (vocab padding past ``vocab_size`` is outside the
    softmax), ``rows`` are unique.
    """
    assert (
        hidden.dim() == 2 and weight.dim() == 2 and hidden.size(1) == weight.size(1)
    ), f"need [N, H] hidden states and a [V, H] weight, got {tuple(hidden.shape)} and {tuple(weight.shape)}"
    assert rows.shape == targets.shape, f"{tuple(rows.shape)} rows vs {tuple(targets.shape)} targets"
    options = _Options(
        vocab=vocab_size or weight.size(0),
        temperature=float(temperature) if temperature > 0 else 1.0,
        with_entropy=with_entropy,
        entropy_requires_grad=with_entropy and entropy_requires_grad,
    )
    log_probs, entropy = _Fp32LmHeadLogProbs.apply(hidden, weight, rows.long(), targets.long(), options)
    return log_probs, (entropy if with_entropy else None)


@dataclass(frozen=True)
class _Options:
    vocab: int
    temperature: float
    with_entropy: bool
    entropy_requires_grad: bool


def _shifted_logits(hidden_rows: Tensor, weight: Tensor, options: _Options) -> tuple[Tensor, Tensor]:
    """``d = (z - m) / T`` of a chunk, and its row max ``m`` of ``z / T``."""
    logits = fp32_logits(hidden_rows, weight)
    if options.temperature != 1.0:
        logits.div_(options.temperature)
    row_max = logits.max(dim=-1).values
    return logits.sub_(row_max.unsqueeze(-1)), row_max


class _Fp32LmHeadLogProbs(torch.autograd.Function):
    @staticmethod
    def forward(ctx, hidden, weight, rows, targets, options: _Options):
        vocab_weight = weight[: options.vocab]
        hidden_rows = hidden.index_select(0, rows)
        n = rows.numel()
        log_sum_exp = torch.empty(n, dtype=torch.float32, device=hidden.device)
        target_logit = torch.empty_like(log_sum_exp)
        mean_logit = torch.empty_like(log_sum_exp) if options.with_entropy else None
        for start in range(0, n, _CHUNK_ROWS):
            chunk = slice(start, start + _CHUNK_ROWS)
            shifted, _ = _shifted_logits(hidden_rows[chunk], vocab_weight, options)
            log_sum_exp[chunk] = torch.logsumexp(shifted, dim=-1)
            target_logit[chunk] = shifted.gather(1, targets[chunk].unsqueeze(-1)).squeeze(-1)
            if mean_logit is not None:
                probs = shifted.sub(log_sum_exp[chunk].unsqueeze(-1)).exp_()
                mean_logit[chunk] = (probs * shifted).sum(dim=-1)
        log_probs = target_logit - log_sum_exp
        entropy = log_sum_exp - mean_logit if mean_logit is not None else log_probs.new_empty(0)
        if not options.entropy_requires_grad:
            ctx.mark_non_differentiable(entropy)
            mean_logit = None  # the backward needs it only for the entropy
        ctx.save_for_backward(hidden_rows, weight, rows, targets, log_sum_exp, mean_logit)
        ctx.hidden_shape = hidden.shape
        ctx.options = options
        return log_probs, entropy

    @staticmethod
    def backward(ctx, grad_log_probs, grad_entropy):
        hidden_rows, weight, rows, targets, log_sum_exp, mean_logit = ctx.saved_tensors
        options: _Options = ctx.options
        vocab_weight = weight[: options.vocab]
        need_hidden, need_weight = ctx.needs_input_grad[:2]
        grad_rows = torch.empty_like(hidden_rows) if need_hidden else None
        grad_vocab_weight = (
            torch.zeros(vocab_weight.shape, dtype=torch.float32, device=weight.device) if need_weight else None
        )
        if grad_log_probs is None:
            grad_log_probs = torch.zeros_like(log_sum_exp)
        use_entropy = mean_logit is not None and grad_entropy is not None and grad_entropy.numel() > 0
        for start in range(0, rows.numel(), _CHUNK_ROWS):
            chunk = slice(start, start + _CHUNK_ROWS)
            shifted, _ = _shifted_logits(hidden_rows[chunk], vocab_weight, options)
            probs = shifted.sub(log_sum_exp[chunk].unsqueeze(-1)).exp()
            # g * (onehot(y) - p)
            grad_logits = probs * -grad_log_probs[chunk].unsqueeze(-1)
            grad_logits.scatter_add_(1, targets[chunk].unsqueeze(-1), grad_log_probs[chunk].unsqueeze(-1))
            if use_entropy:  # - c * p * (d - mu)
                shifted.sub_(mean_logit[chunk].unsqueeze(-1)).mul_(probs).mul_(grad_entropy[chunk].unsqueeze(-1))
                grad_logits.sub_(shifted)
            if options.temperature != 1.0:
                grad_logits.div_(options.temperature)
            # the bf16 head's backward: a bf16 gradient into both GEMMs
            grad_logits = grad_logits.to(hidden_rows.dtype)
            if grad_rows is not None:
                torch.mm(grad_logits, vocab_weight, out=grad_rows[chunk])
            if grad_vocab_weight is not None:
                _addmm_(grad_vocab_weight, grad_logits.t(), hidden_rows[chunk])
        grad_hidden = None
        if grad_rows is not None:
            grad_hidden = grad_rows.new_zeros(ctx.hidden_shape).index_copy_(0, rows, grad_rows)
        grad_weight = None
        if grad_vocab_weight is not None:
            grad_weight = torch.zeros_like(weight)
            grad_weight[: options.vocab] = grad_vocab_weight.to(weight.dtype)
        return grad_hidden, grad_weight, None, None, None


def _addmm_(acc: Tensor, a: Tensor, b: Tensor) -> None:
    """``acc += a @ b`` for bf16 ``a`` and ``b``, accumulating in ``acc``'s fp32 without an fp32 copy of either."""
    if acc.is_cuda and a.dtype == b.dtype and a.dtype != acc.dtype:
        torch.addmm(acc, a, b, out_dtype=acc.dtype, out=acc)
    else:
        acc.add_(a.to(acc.dtype) @ b.to(acc.dtype))
