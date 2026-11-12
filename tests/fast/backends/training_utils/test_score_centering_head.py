"""Projection chunking must preserve the real loss/gradient and bound saved vocabulary tensors."""

import runpy
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch
from tests.fast.backends.training_utils.test_score_centering_pipeline import single_rank  # noqa: F401
from tests.fast.fixtures.score_centering_fixtures import _args

from miles.backends.training_utils.data.context_parallel import get_sum_of_sample_mean
from miles.backends.training_utils.loss.hub.score_centering.chunked_head import chunked_head_log_probs
from miles.backends.training_utils.loss.hub.score_centering.selected_log_probs import selected_log_probs_and_entropy
from miles.backends.training_utils.loss.hub.score_centering_loss import (
    score_centering_loss_from_hidden,
    score_centering_loss_function,
)
from miles.backends.training_utils.loss.objective import loss_function
from miles.utils.score_centering import validate_score_centering_args


@pytest.mark.parametrize("chunk_size,rows", [(2, 7), (4096, 4101)])
def test_checkpointed_projection_matches_dense_and_saves_no_vocab_activations(chunk_size, rows):
    torch.manual_seed(19)
    hidden = torch.randn(rows, 5, dtype=torch.float64, requires_grad=True)
    weight = torch.randn(137, 5, dtype=torch.float64, requires_grad=True)
    bias = torch.randn(137, dtype=torch.float64, requires_grad=True)
    ids = torch.randint(0, 133, (rows, 129))
    ids[0, -1] = -1
    options = dict(vocab_size=133, temperature=0.7, with_entropy=True)
    dense = selected_log_probs_and_entropy(torch.nn.functional.linear(hidden, weight, bias), ids, **options)
    expected_grad = torch.autograd.grad(dense[0].square().mean() + dense[1].mean(), (hidden, weight, bias))
    saved_shapes = []

    def pack(tensor):
        saved_shapes.append(tuple(tensor.shape))
        return tensor

    with torch.autograd.graph.saved_tensors_hooks(pack, lambda tensor: tensor):
        actual = chunked_head_log_probs(hidden, ids, weight=weight, bias=bias, head_chunk_size=chunk_size, **options)
    actual_grad = torch.autograd.grad(actual[0].square().mean() + actual[1].mean(), (hidden, weight, bias))
    for result, expected in zip(actual + actual_grad, dense + expected_grad, strict=True):
        torch.testing.assert_close(result, expected)
    assert not any(len(shape) == 2 and shape[-1] == 137 for shape in saved_shapes)


@pytest.mark.parametrize("layout", ["thd", "bshd"])
@pytest.mark.parametrize("estimator", ["topk", "mc"])
@pytest.mark.usefixtures("single_rank")
def test_chunked_dppo_loss_and_gradients_match_dense(layout, estimator):
    torch.manual_seed(31)
    tokens = torch.tensor([0, 1, 2, 3, 4, 5, 6])
    ids = torch.tensor([[2, 3, 4], [1, 3, 5], [-1, -1, -1], [4, 5, 6], [5, 6, 7]])
    if estimator == "mc":
        ids[:, 1] = ids[:, 0]
    batch = dict(
        total_lengths=[7],
        response_lengths=[5],
        max_seq_lens=[7],
        unconcat_tokens=[tokens],
        loss_masks=[torch.tensor([1, 1, 0, 1, 1])],
        rollout_log_probs=[torch.tensor([-1.0, -2.0, float("nan"), -1.3, -2.2])],
        advantages=[torch.tensor([1.0, -1.0, float("nan"), 0.5, -0.5])],
        rollout_topk_token_ids=[ids],
        rollout_topk_log_probs=[torch.full((5, 3), -2.0)],
        ref_log_probs=[torch.full((5,), -2.1)],
    )
    args = _args(
        qkv_format=layout,
        vocab_size=8,
        score_centering_estimator=estimator,
        policy_loss_type="dppo_tv",
        dppo_tv_threshold=0.1,
        entropy_coef=0.01,
        use_kl_loss=True,
        kl_loss_coef=0.05,
        kl_loss_type="k2",
        use_unbiased_kl=True,
        calculate_per_token_loss=False,
        global_batch_size=1,
        recompute_loss_function=False,
    )
    hidden = torch.randn(1, 7, 4, requires_grad=True)
    weight = torch.randn(10, 4, requires_grad=True)
    bias = torch.randn(10, requires_grad=True)
    reduce = get_sum_of_sample_mean([7], [5], batch["loss_masks"], qkv_format=layout, max_seq_lens=[7])
    expected, expected_metrics = score_centering_loss_function(
        args, batch, torch.nn.functional.linear(hidden, weight, bias), reduce
    )
    expected_grad = torch.autograd.grad(expected, (hidden, weight, bias))
    actual, metrics = score_centering_loss_from_hidden(
        args, batch, hidden, reduce, weight=weight, bias=bias, head_chunk_size=2
    )
    actual_grad = torch.autograd.grad(actual, (hidden, weight, bias))
    torch.testing.assert_close(actual, expected)
    for key in expected_metrics:
        torch.testing.assert_close(metrics[key], expected_metrics[key])
    assert metrics["sc_head_chunk_size"] == 2
    assert metrics["sc_head_chunks"] == 3
    for value, reference in zip(actual_grad, expected_grad, strict=True):
        torch.testing.assert_close(value, reference)
    assert not actual_grad[0][0, 3].any()  # tool observation
    assert not actual_grad[1][8:].any()  # padded vocabulary

    # The shared Megatron normalization path must accept the hidden-state loss.
    def impl(a, b, h, reduction):
        return score_centering_loss_from_hidden(a, b, h, reduction, weight=weight, bias=bias, head_chunk_size=2)

    scaled, _, _ = loss_function(args, batch, 3, hidden, apply_megatron_loss_scaling=True, loss_impl=impl)
    torch.testing.assert_close(scaled, expected * 3)


@pytest.mark.parametrize(
    "override",
    [
        dict(train_backend="fsdp"),
        dict(tensor_model_parallel_size=2),
        dict(context_parallel_size=2),
        dict(pipeline_model_parallel_size=2),
        dict(loss_type="policy_loss"),
        dict(recompute_loss_function=True),
    ],
)
def test_head_chunking_rejects_unsupported_launches(override):
    with pytest.raises(ValueError, match="head chunking"):
        validate_score_centering_args(
            _args(**(dict(train_backend="megatron", score_centering_head_chunk_size=4096) | override))
        )


@pytest.mark.usefixtures("single_rank")
def test_megatron_output_processor_captures_tied_head_and_scores_reference():
    # Load the lightweight adapter without importing Megatron's CUDA package initializer.
    path = Path(__file__).resolve().parents[4] / "miles/backends/megatron_utils/score_centering_head.py"
    cls = runpy.run_path(str(path))["ChunkedScoreHead"]
    adapter = cls(2)
    torch.manual_seed(7)
    hidden = torch.randn(7, 1, 4)
    layer = torch.nn.Linear(4, 8)
    tied_weight = torch.randn(8, 4)
    states = adapter.capture(
        hidden_states=hidden, output_layer=layer, output_weight=tied_weight, config=SimpleNamespace(use_mup=False)
    )
    assert adapter.weight is tied_weight
    args = _args()
    tokens = torch.tensor([0, 1, 2, 3, 4, 5, 6])
    result = adapter.log_probs(
        states, args=args, unconcat_tokens=[tokens], total_lengths=[7], response_lengths=[5], with_entropy=True
    )
    logits = torch.nn.functional.linear(hidden[1:6, 0], tied_weight, layer.bias) / args.rollout_temperature
    lp = logits.log_softmax(-1)
    torch.testing.assert_close(result["log_probs"][0], lp.gather(-1, tokens[2:, None]).squeeze(-1))
    torch.testing.assert_close(result["entropy"][0], -(lp.exp() * lp).sum(-1))
    assert not result["log_probs"][0].requires_grad
