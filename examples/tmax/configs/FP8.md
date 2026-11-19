# TMax FP8 blockwise inference preset

`prod_4b_dppo_tv_scmc32_fp8_block_kv.yaml` uses a derived checkpoint exported and validated by the preparation job before production submission. The launch artifacts and validation status are in `/weka/gfaria/tmax-4b-miles/validation/fp8-block-kv-20261119/`.

It preserves the BF16 control's TMax workload, DPPO-TV, MC32 array transfer, 4096-token head chunks and lag 32. This preset uses 1024 concurrent episodes, 131072 chunked prefill tokens, and 262144 max prefill tokens. Training and reference weights remain BF16. The trailing `--hf-checkpoint` overrides the rollout HF source while the launcher's BF16 reference path stays unchanged. The production job has a fresh output name and explicitly initializes training from the original BF16 distributed checkpoint through `--pretrained-checkpoint`; the FP8 HF export is for inference.

## Required checkpoint preparation before launch

Create a separate FP8 export from the chosen BF16 TMax checkpoint. Its `config.json` must include:

```json
{
  "quantization_config": {
    "quant_method": "fp8",
    "fmt": "e4m3",
    "activation_scheme": "dynamic",
    "weight_block_size": [128, 128]
  }
}
```

This is metadata for actual FP8 weights and corresponding `weight_scale_inv` tensors, not a way to relabel BF16 weights. Add checkpoint-specific `modules_to_not_convert` exclusions for embeddings, head, norms, convolution and small recurrent gating projections. Match the exclusions to the serialized module names and SGLang's fused projection mapping. Do not quantize those modules indiscriminately with the generic converter: Qwen3.5 contains non-matrix convolution weights and projections not covered by Miles' FP8 publication processor.

SGLang's `Fp8Config` rejects blockwise weights unless the checkpoint is FP8-serialized. Plain online `--quantization fp8` on a BF16 checkpoint does not establish 128x128 block quantization. Miles' `miles/backends/megatron_utils/megatron_to_hf/processors/quantizer_fp8.py` reads this metadata and emits blockwise weights and scales during publication; validate Qwen3.5 fused projection handling with a real weight update before launching training.

## KV cache and validation

`--sglang-kv-cache-dtype fp8_e4m3` requests FP8 attention KV storage. It does not quantize the recurrent state of the linear-attention layers. Weight block sizes do not configure KV-cache scaling. Validate the selected attention backend, KV scales and numerical accuracy on the installed server version; do not assume requesting the cache dtype supplies calibrated scales.

Before production: export and inspect tensor shapes/dtypes, start one inference engine, confirm blockwise configuration and FP8 KV allocation, compare fixed-prompt log probabilities against BF16, and verify publication changes/restores outputs. Consult `launch-state.json` in the validation directory for the actual GPU validation result; successful configuration parsing alone is not a GPU validation. Scheduling minimum runtime belongs to the Beaker experiment specification, not this YAML's preemptible flag.

## Validation observed

The one-engine GPU validation passed generation, MC32/top-k array checks, and a single-projection publication change/restore check using the Miles FP8 processor. This is not a completed full training update or an accuracy comparison against BF16.

A production attempt with 262144 chunked prefill tokens failed with an illegal CUDA memory access during prefill graph capture. The current 131072-token attempt passed graph capture and reached initial weight publication. Successful rollouts and completed training updates remain to be verified.
