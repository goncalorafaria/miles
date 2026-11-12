"""The TMax configs must retain DPPO-TV while enabling the actual score-centering consumer."""

import shlex
from pathlib import Path
from types import SimpleNamespace

import pytest
import yaml
from tests.fast.examples.tmax.test_run_tmax_qwen3_5 import launcher
from tests.fast.fixtures.score_centering_fixtures import _args

from miles.utils.arguments import validate_policy_loss_type_args
from miles.utils.rollout_topk_logprobs import validate_rollout_topk_logprobs_args
from miles.utils.score_centering import validate_score_centering_args


@pytest.mark.parametrize("layout", ["smoke", "prod"])
@pytest.mark.parametrize("mode,estimator", [("sct32", "topk"), ("scmc32", "mc")])
def test_tmax_combined_config_reaches_train_command(monkeypatch, layout, mode, estimator):
    path = Path(__file__).resolve().parents[4] / "examples/tmax/configs" / f"{layout}_4b_dppo_tv_{mode}.yaml"
    config = yaml.safe_load(path.read_text())
    config.pop("beaker")
    config["gateway_url"] = "http://test-gateway"
    captured = {}
    monkeypatch.setattr(
        launcher.ScriptArgs,
        "create_backend",
        lambda self: SimpleNamespace(execute_train=lambda **kwargs: captured.update(kwargs)),
    )
    launcher.execute(**config)
    argv = shlex.split(captured["train_args"])
    for flag, value in (
        ("--loss-type", "score_centering"),
        ("--policy-loss-type", "dppo_tv"),
        ("--score-centering-estimator", estimator),
        ("--rollout-top-logprobs-num", "32"),
    ):
        assert argv.count(flag) == 1 and argv[argv.index(flag) + 1] == value
    assert "--rollout-score-array" in argv and "--use-rollout-logprobs" in argv
    args = _args(
        policy_loss_type="dppo_tv",
        dppo_tv_threshold=0.1,
        use_rollout_logprobs=True,
        rollout_score_array=True,
        rollout_top_logprobs_num=32,
        score_centering_estimator=estimator,
    )
    validate_policy_loss_type_args(args)
    validate_rollout_topk_logprobs_args(args)
    validate_score_centering_args(args)


@pytest.mark.parametrize("mode", ["tis", "mis"])
def test_dppo_cannot_silently_mix_extra_importance_weights(mode):
    with pytest.raises(ValueError, match="own ratio"):
        validate_score_centering_args(_args(policy_loss_type="dppo_tv", score_centering_is=mode))
