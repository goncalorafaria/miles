"""Bound vocabulary-sized memory by checkpointing projection and normalization together."""

import torch
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint

from miles.backends.training_utils.loss.hub.score_centering.selected_log_probs import selected_log_probs_and_entropy


def chunked_head_log_probs(
    hidden: torch.Tensor,
    token_ids: torch.Tensor,
    *,
    weight: torch.Tensor,
    bias: torch.Tensor | None = None,
    head_chunk_size: int,
    **normalization_options,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Return selected scores, retaining no full-vocabulary activations between chunks.

    Backward recomputes each projection. Only hidden states, shared head weights,
    IDs and selected outputs survive forward. The head is an unsharded affine
    projection; model-specific output transforms must be handled by the caller.
    """
    if head_chunk_size <= 0:
        raise ValueError("head_chunk_size must be positive")
    if normalization_options.get("group") is not None:
        raise ValueError("Chunked affine head requires an unsharded vocabulary")

    def project(h, ids, w, b):
        return selected_log_probs_and_entropy(F.linear(h, w, b), ids, **normalization_options)

    if hidden.size(0) == 0:
        return project(hidden, token_ids, weight, bias)
    parts = [
        checkpoint(project, h, ids, weight, bias, use_reentrant=False)
        for h, ids in zip(hidden.split(head_chunk_size), token_ids.split(head_chunk_size), strict=True)
    ]
    return torch.cat([part[0] for part in parts]), torch.cat([part[1] for part in parts])
