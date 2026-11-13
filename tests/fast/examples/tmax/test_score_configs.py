"""The TMax configs must retain DPPO-TV while enabling the actual score-centering consumer."""

import argparse
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
@pytest.mark.parametrize(
    "mode,estimator", [("sct32", "topk"), ("scmc32", "mc"), ("sct128", "topk"), ("scmc32_nvfp4", "mc")]
)
def test_tmax_combined_config_reaches_train_command(monkeypatch, layout, mode, estimator):
    path = Path(__file__).resolve().parents[4] / "examples/tmax/configs" / f"{layout}_4b_dppo_tv_{mode}.yaml"
    if mode == "sct128" and layout == "prod":
        pytest.skip("SCT128 is qualified with the smoke layout first")
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
        ("--rollout-top-logprobs-num", "128" if mode == "sct128" else "32"),
    ):
        assert argv.count(flag) == 1 and argv[argv.index(flag) + 1] == value
    assert "--rollout-score-array" in argv and "--use-rollout-logprobs" in argv
    if mode in ("sct128", "scmc32_nvfp4"):
        assert argv[argv.index("--score-centering-head-chunk-size") + 1] == "4096"
    if mode == "scmc32_nvfp4":
        assert argv[argv.index("--sglang-quantization") + 1] == "nvfp4_dense_online"
    args = _args(
        policy_loss_type="dppo_tv",
        dppo_tv_threshold=0.1,
        use_rollout_logprobs=True,
        rollout_score_array=True,
        rollout_top_logprobs_num=128 if mode == "sct128" else 32,
        score_centering_head_chunk_size=4096 if mode in ("sct128", "scmc32_nvfp4") else 0,
        train_backend="megatron",
        score_centering_estimator=estimator,
    )
    validate_policy_loss_type_args(args)
    validate_rollout_topk_logprobs_args(args)
    validate_score_centering_args(args)


@pytest.mark.parametrize("mode", ["tis", "mis"])
def test_dppo_cannot_silently_mix_extra_importance_weights(mode):
    with pytest.raises(ValueError, match="own ratio"):
        validate_score_centering_args(_args(policy_loss_type="dppo_tv", score_centering_is=mode))


@pytest.mark.parametrize("precision", ["bf16", "nvfp4"])
def test_nvfp4_production_topology_and_scheduling(precision):
    path = Path(__file__).resolve().parents[4] / f"examples/tmax/configs/prod_4b_dppo_tv_scmc32_{precision}.yaml"
    config = yaml.safe_load(path.read_text())
    assert "--use-miles-router" in shlex.split(config["extra_args"])
    beaker = config["beaker"]
    assert config["num_nodes"] == beaker["nodes"] == 2
    assert config["num_gpus_per_node"] == beaker["gpus"] == 8
    assert config["async_rollout_gpus"] == 8
    assert config["rollout_batch_size"] * config["n_samples_per_prompt"] == 256
    assert beaker["priority"] == "urgent" and beaker["preemptible"] is True
    assert "min_runtime" not in beaker


def test_nvfp4_is_accepted_by_serving_cli():
    from sglang.srt.server_args import ServerArgs

    parser = argparse.ArgumentParser()
    ServerArgs.add_cli_args(parser)
    args = parser.parse_args(["--model-path", "/local/checkpoint", "--quantization", "nvfp4_dense_online"])
    assert args.quantization == "nvfp4_dense_online"


def test_dense_nvfp4_config_constructs_and_loads_fused_bf16_shards():
    import torch
    from sglang.srt.layers.linear import MergedColumnParallelLinear
    from sglang.srt.layers.quantization.nvfp4_dense_online import NvFp4DenseOnlineConfig, NvFp4DenseOnlineLinearMethod
    from sglang.srt.layers.quantization.unquant import UnquantizedLinearMethod

    config = NvFp4DenseOnlineConfig()
    assert config.get_scaled_act_names() == []
    layer = MergedColumnParallelLinear(
        input_size=336,
        output_sizes=[80, 80],
        bias=False,
        params_dtype=torch.bfloat16,
        quant_config=config,
        tp_rank=0,
        tp_size=1,
    )
    assert isinstance(layer.quant_method, NvFp4DenseOnlineLinearMethod)
    assert not isinstance(layer.quant_method, UnquantizedLinearMethod)
    gate = torch.full((80, 336), 2.0, dtype=torch.bfloat16)
    up = torch.full((80, 336), 3.0, dtype=torch.bfloat16)
    layer.weight.weight_loader(layer.weight, gate, 0)
    layer.weight.weight_loader(layer.weight, up, 1)
    torch.testing.assert_close(layer.weight, torch.cat((gate, up)))
