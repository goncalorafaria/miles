"""Blackwell validation: dense NVFP4 numerics, CUDA graphs, and real 4B reload.

Run with TMAX_SC_MODEL set to the local BF16 TMax checkpoint. No HF credentials
or checkpoint conversion are needed. This validates publication, not terminal RL.
"""

import base64
import os
from pathlib import Path

import numpy as np
import pytest
import torch
import torch.nn.functional as F
from flashinfer import fp4_quantize
from safetensors import safe_open
from sglang.srt.layers.quantization.nvfp4_dense_online import NvFp4DenseOnlineLinearMethod, _inverse_global_scale
from sglang.test.quant_ref_utils import dequantize_nvfp4_to_dtype


@pytest.mark.parametrize("shape", [(1, 256, 512), (5, 160, 336), (64, 1024, 1024)])
def test_dense_kernel_and_graph_reload(shape):
    m, n, k = shape
    method = NvFp4DenseOnlineLinearMethod()
    layer = torch.nn.Module()
    with torch.device("cuda"):
        method.create_weights(layer, k, [n // 2, n - n // 2], k, n, torch.bfloat16)
    layer.weight.data.normal_(std=0.1)
    method.process_weights_after_loading(layer)
    x = torch.randn(m, k, device="cuda", dtype=torch.bfloat16) * 0.1
    padded_x = F.pad(x, (0, layer.nvfp4_pad_k))
    scale = _inverse_global_scale(padded_x).reciprocal()
    packed, sf = fp4_quantize(padded_x, scale)
    x_ref = dequantize_nvfp4_to_dtype(packed, sf, scale, torch.float32)
    w_ref = dequantize_nvfp4_to_dtype(
        layer.nvfp4_weight,
        layer.nvfp4_weight_scale,
        layer.nvfp4_weight_descale.reciprocal(),
        torch.float32,
    )
    expected = (x_ref @ w_ref.T)[:, :n]
    actual = method.apply(layer, x)
    torch.testing.assert_close(actual.float(), expected, atol=0.015, rtol=0.015)
    relative_error = (actual.float() - F.linear(x.float(), layer.weight.float())).norm() / expected.norm()
    assert relative_error < 0.25
    # Capture once; updated packed buffers must be visible through the same graph.
    stream = torch.cuda.Stream()
    stream.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(stream):
        for _ in range(3):
            method.apply(layer, x)
    torch.cuda.current_stream().wait_stream(stream)
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        captured = method.apply(layer, x)
    ptrs = {name: p.data_ptr() for name, p in layer.named_parameters()}
    for factor in (0.5, 2.0, 0.0):
        layer.weight.data.mul_(factor)
        method.process_weights_after_loading(layer)
        assert ptrs == {name: p.data_ptr() for name, p in layer.named_parameters()}
        graph.replay()
        torch.testing.assert_close(captured, method.apply(layer, x))
        assert torch.isfinite(captured).all()
    assert captured.count_nonzero() == 0


def _check_result(result):
    if isinstance(result, dict):
        assert result.get("success", True), result
    elif isinstance(result, tuple):
        assert result[0], result


def test_real_model_publication():
    from sglang import Engine
    from sglang.srt.managers.io_struct import GenerateReqInput

    model = os.environ.get("TMAX_SC_MODEL")
    if not model:
        pytest.skip("TMAX_SC_MODEL is required for the real 4B validation")
    engine = Engine(
        model_path=model,
        quantization="nvfp4_dense_online",
        dtype="bfloat16",
        context_length=1024,
        mem_fraction_static=0.5,
        max_running_requests=4,
        max_total_tokens=4096,
        max_mamba_cache_size=8,
        cuda_graph_max_bs_decode=4,
        attention_backend="triton",
        sampling_backend="pytorch",
        disable_flashinfer_autotune=True,
        watchdog_timeout=900,
    )
    try:
        # A down projection is dense, unfused, and materially affects logits.
        candidates = []
        for path in sorted(Path(model).glob("*.safetensors")):
            with safe_open(path, framework="pt", device="cpu") as checkpoint:
                for name in checkpoint.keys():
                    if name.endswith("layers.0.mlp.down_proj.weight"):
                        candidates.append((name, checkpoint.get_tensor(name).cuda()))
        assert len(candidates) == 1, [name for name, _ in candidates]
        name, original = candidates[0]

        def generate(mode):
            request = GenerateReqInput(
                text="The result of 17 times 23 is",
                sampling_params={
                    "temperature": 1.0,
                    "top_p": 1.0,
                    "top_k": -1,
                    "max_new_tokens": 8,
                    "ignore_eos": True,
                    "sampling_seed": 47,
                    "score_centering_samples": 32 if mode == "mc" else 0,
                    "array_output_logprobs": True,
                },
                return_logprob=True,
                top_logprobs_num=32,
                return_flat_raw_output_top_logprobs=True,
                return_flat_raw_top_logprobs_b64=True,
            )
            generator = engine.tokenizer_manager.generate_request(request, None)
            result = engine.loop.run_until_complete(generator.__anext__())
            meta = result["meta_info"]
            assert meta["output_score_mode"] == ("mc32" if mode == "mc" else "topk")
            assert meta["output_top_logprobs_shape"] == [8, 32]
            values = np.frombuffer(base64.b64decode(meta["output_top_logprobs_val_flat_b64"]), dtype="<f4")
            indices = np.frombuffer(base64.b64decode(meta["output_top_logprobs_idx_flat_b64"]), dtype="<i4")
            assert values.size == indices.size == 256
            assert np.isfinite(values).all() and (values <= 0).all()
            assert (indices >= 0).all()
            print({"mode": mode, "shape": meta["output_top_logprobs_shape"]}, flush=True)
            return result

        before = generate("mc")
        for factor in (0.0, 1.0):
            _check_result(engine.begin_weight_update())
            _check_result(engine.update_weights_from_tensor([(name, original * factor)]))
            _check_result(engine.end_weight_update())
            after = generate("mc")
            before_lp = before["meta_info"]["output_token_logprobs"][0][0]
            after_lp = after["meta_info"]["output_token_logprobs"][0][0]
            assert np.isfinite(after_lp)
            if factor == 0:
                assert abs(after_lp - before_lp) > 1e-5
            else:
                assert after_lp == pytest.approx(before_lp, abs=2e-3)
        generate("topk")
    finally:
        engine.shutdown()
