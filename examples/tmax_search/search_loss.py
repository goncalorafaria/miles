"""Prime-RL's ``group_token_mean`` loss aggregation, through ``--custom-convert-samples-to-train-data-path``.

Prime-RL (src/prime_rl/trainer/rl/loss.py, aggregation "group_token_mean") averages the per-token loss over each
group's tokens, then over the step's groups:

    L = 1/G * sum_g 1/N_g * sum_{i in g} sum_t m_it * l_it          (N_g: loss tokens of group g)

Miles' sample-mean reducer (no ``--calculate-per-token-loss``) computes 1/B * sum_i (sum_t m_it * l_it) / d_i over the
B samples of a step, where d_i is ``rollout_mask_sums[i]``. With d_i = N_g * G / B the two agree. This hook runs Miles'
own conversion and rewrites ``rollout_mask_sums`` so. Miles stores them as integers, so d_i is rounded: a relative error
of at most 0.5 / d_i in each group's weight (d_i = N_g / 32 with groups of 32, so under 1e-3 for any group of 32
one-turn episodes and about 1e-5 for typical ones).

With the zero-variance filter off this is also Prime-RL's "mask without refill": a group whose rewards are all equal has
advantage 0, adds nothing to the sum, and still counts in G.
"""

from collections import defaultdict

from miles.ray.rollout.train_data_conversion import (
    convert_samples_to_train_data as miles_convert_samples_to_train_data,
)
from miles.utils.function_registry import load_function
from miles.utils.types import Sample


def convert_samples_to_train_data(args, samples: list[Sample]) -> dict:
    assert not args.calculate_per_token_loss, "group_token_mean needs Miles' sample-mean reducer"
    post_process = (
        load_function(args.custom_reward_post_process_path) if args.custom_reward_post_process_path else None
    )
    train_data = miles_convert_samples_to_train_data(
        args,
        samples,
        metadata={},
        custom_convert_samples_to_train_data_func=None,
        custom_reward_post_process_func=post_process,
    )
    train_data["rollout_mask_sums"] = group_token_mean_denominators(
        [s.group_index for s in samples], train_data["loss_masks"]
    )
    return train_data


def group_token_mean_denominators(group_indices: list[int], loss_masks: list[list[int]]) -> list[int]:
    """d_i = N_g * G / B for each sample i of group g, rounded (at least 1)."""
    assert all(g is not None for g in group_indices), "group_token_mean needs every sample's group_index"
    group_tokens: dict[int, int] = defaultdict(int)
    for group, mask in zip(group_indices, loss_masks, strict=True):
        group_tokens[group] += sum(mask)
    scale = len(group_tokens) / len(group_indices)
    return [max(1, round(group_tokens[group] * scale)) for group in group_indices]
