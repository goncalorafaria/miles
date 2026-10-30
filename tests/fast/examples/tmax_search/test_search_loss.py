"""Prime-RL's group_token_mean through Miles' per-sample denominators (search_loss.py)."""

import pytest

from tests.fast.examples.tmax_search.conftest_paths import REPO  # noqa: F401  (import paths)

search_loss = pytest.importorskip("search_loss")


def test_denominators_give_group_token_mean():
    # 2 groups of 2 samples: group 0 has 3 + 1 loss tokens, group 1 has 10 + 6.
    groups = [0, 0, 1, 1]
    masks = [[1, 1, 1], [1, 0], [1] * 10, [1] * 6 + [0, 0]]
    denominators = search_loss.group_token_mean_denominators(groups, masks)
    assert denominators == [round(4 * 2 / 4), round(4 * 2 / 4), round(16 * 2 / 4), round(16 * 2 / 4)]

    # Miles' sample-mean reducer: 1/B * sum_i (sum_t m*l) / d_i, here with l = 1 per token.
    miles = sum(sum(m) / d for m, d in zip(masks, denominators, strict=True)) / len(masks)
    # Prime-RL: 1/G * sum_g 1/N_g * sum tokens of g, = 1 for per-token losses of 1.
    assert miles == pytest.approx(1.0)


def test_every_sample_needs_its_group():
    with pytest.raises(AssertionError):
        search_loss.group_token_mean_denominators([0, None], [[1], [1]])
