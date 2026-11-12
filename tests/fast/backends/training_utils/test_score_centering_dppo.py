"""DPPO centering must cancel its weighted score, including candidate-side gates."""

import pytest
import torch

from miles.backends.training_utils.loss.hub.math_utils import compute_dppo_tv_policy_loss, dppo_tv_score_weight
from miles.backends.training_utils.loss.hub.score_centering import ScoreCenteringInputs, score_centering_loss


@pytest.mark.parametrize("advantage", [-1.0, 1.0])
@pytest.mark.parametrize("estimator", ["topk", "mc"])
@pytest.mark.parametrize("count", [32, 64, 128])
def test_constant_advantage_cancels_in_expectation(advantage, estimator, count):
    # MC counts exactly match q, so this deterministic expectation test needs no tolerance on sampling noise.
    q = torch.tensor([0.5, 0.25, 0.125, 0.125])
    logits = torch.tensor([0.1, 0.15, 0.5, 0.25]).log().requires_grad_()
    logp = logits.log_softmax(-1)
    ids = torch.arange(4) if estimator == "topk" else torch.repeat_interleave(torch.arange(4), (q * count).long())
    head = logp[ids].expand(4, -1)
    rollout = q.log().requires_grad_()
    advantages = torch.full((4,), advantage)
    native, _ = compute_dppo_tv_policy_loss(rollout.detach() - logp, logp, rollout.detach(), advantages, 0.1)
    native_gradient = torch.autograd.grad((q * native).sum(), logits, retain_graph=True)[0]
    assert native_gradient.abs().max() > 0.01  # The gate leaves a nonzero expectation to center.
    loss, _ = score_centering_loss(
        ScoreCenteringInputs(
            train_log_probs=logp,
            train_head_log_probs=head,
            rollout_log_probs=rollout,
            rollout_head_log_probs=rollout[ids].expand(4, -1),
            head_mask=torch.ones_like(head, dtype=torch.bool),
            advantages=advantages,
            estimator=estimator,
            dppo_tv_threshold=0.1,
        )
    )
    gradient, rollout_gradient = torch.autograd.grad((q * loss).sum(), (logits, rollout), allow_unused=True)
    torch.testing.assert_close(gradient, torch.zeros_like(gradient), atol=2e-7, rtol=0)
    assert rollout_gradient is None


@pytest.mark.parametrize("advantage", [-1.0, 1.0])
def test_candidate_coefficient_matches_native_autograd_at_extremes(advantage):
    logp = torch.tensor([-0.1, -2.0, -35.0, -0.01, -2.0], requires_grad=True)
    logq = torch.tensor([-2.0, -0.1, -0.1, -35.0, -2.0])
    advantages = torch.full_like(logp, advantage)
    native, _ = compute_dppo_tv_policy_loss(logq - logp, logp, logq, advantages, 0.1)
    expected = -torch.autograd.grad(native.sum(), logp)[0] / advantage
    actual = dppo_tv_score_weight(logp, logq, advantages, 0.1)
    torch.testing.assert_close(actual, expected)


def test_ungated_full_head_retains_native_dppo_value_and_gradient():
    logits = torch.tensor([0.2, -0.3, 0.4], requires_grad=True)
    logp = logits.log_softmax(-1)
    q = torch.tensor([0.3, 0.3, 0.4]).log()
    loss, metrics = score_centering_loss(
        ScoreCenteringInputs(
            train_log_probs=logp[:1],
            train_head_log_probs=logp[None],
            rollout_log_probs=q[:1],
            rollout_head_log_probs=q[None],
            head_mask=torch.ones((1, 3), dtype=torch.bool),
            advantages=torch.ones(1),
            dppo_tv_threshold=1.0,
        )
    )
    native, _ = compute_dppo_tv_policy_loss(q[:1] - logp[:1], logp[:1], q[:1], torch.ones(1), 1.0)
    torch.testing.assert_close(loss, native)
    torch.testing.assert_close(
        torch.autograd.grad(loss.sum(), logits, retain_graph=True)[0], torch.autograd.grad(native.sum(), logits)[0]
    )
    torch.testing.assert_close(metrics.correction, torch.zeros(1), atol=1e-7, rtol=0)
