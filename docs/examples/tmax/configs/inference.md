---
title: "TMax inference configs and score logprobs"
description: "TMax BF16 and NVFP4 serving configs and generated-token SCT/SCMC array requests."
# Generated from examples/tmax/configs/inference/README.md by scripts/tools/sync_example_docs.py. Edit that README, not this file.
---
These are **native SGLang server configs**, loaded with `sglang serve --config`.
They are not training configs for `run_tmax_qwen3_5.py` or the Beaker submitter.
The matching BF16 and NVFP4 variants use one inference GPU, the TMax context
limit and parsers. No trainer, session server or sandbox is launched.

## Serving port

The TMax Dockerfile pins SGLang at `7b6993a310d0dbb67929b51582d1729a6295e417`.
It now applies [sglang_score_logprobs.patch](https://github.com/radixark/miles/blob/main/docker/patch/sglang_score_logprobs.patch)
before installing SGLang. This ports the existing SCT and SCMC serving work
without upgrading that base. Build a new image as described in the
[TMax image instructions](/examples/tmax/beaker#images); existing images do not
contain the patch. Commit local changes before using the image builder, whose
context comes from `git archive HEAD`.

The patch adds generated-token MC32/64/128 probes, preserving duplicate draws
and their exact normalized rollout log probabilities. Each position draws
independently with replacement. SCT uses the existing top-k logprob extraction.
Both support preserving float32 logprob and int32 token-ID arrays through
scheduler/detokenizer IPC and base64 encoding at the HTTP boundary. Response
token sampling still uses the request's normal sampling settings.

The same options now also pass through SGLang's chat schema for TMax session
server v2. See [score centering with DPPO-TV](https://github.com/radixark/miles/blob/main/examples/tmax/SCORE_CENTERING.md) for the
training configs, estimator definition, array alignment and validation scope.

## BF16 and NVFP4

Run from the Miles repository in an environment containing the patched SGLang:

```bash
SGLANG_RETURN_ORIGINAL_LOGPROB=0 sglang serve \
  --config examples/tmax/configs/inference/sglang_bf16.yaml \
  --model-path /path/to/Qwen3.5-4B-tmax-template
```

For NVFP4, use the same BF16 source checkpoint with the patched runtime:

```bash
SGLANG_RETURN_ORIGINAL_LOGPROB=0 sglang serve \
  --config examples/tmax/configs/inference/sglang_nvfp4.yaml \
  --model-path /path/to/Qwen3.5-4B-tmax-template
```

The `nvfp4_dense_online` method converts dense weights on load and after RL
publication. It uses W4A4 dense GEMMs and retains BF16 staging weights. See
[dense NVFP4 with MC32](https://github.com/radixark/miles/blob/main/examples/tmax/SCORE_CENTERING.md#dense-nvfp4-with-mc32) for the
implementation, precision coverage, runtime image, and validation status. Both
server configs use the same port and are intended for sequential comparisons.

## Requests and array decoding

[score_requests.json](https://github.com/radixark/miles/blob/main/examples/tmax/configs/inference/score_requests.json) contains `baseline`, `sct32`,
`sct64`, `sct128`, `scmc32`, `scmc64` and `scmc128`. All request only generated
positions. The baseline retains sampled-token logprobs, as RL already needs
those, but requests no candidate array. Use the same requests for either dtype:

```bash
jq '.scmc32' examples/tmax/configs/inference/score_requests.json \
  | curl --fail-with-body http://127.0.0.1:30000/generate \
      -H 'Content-Type: application/json' --data-binary @- > /tmp/tmax-scores.json
```

The prompt is a simple serving smoke prompt, not a TMax episode or a throughput
benchmark. For agent prompts, render the TMax chat template and send its token
IDs through `input_ids` instead of the example's `text`.

`top_logprobs_num` is the candidate width. In SCMC mode it must equal
`sampling_params.score_centering_samples`; it does not select the top-k tokens.
In all examples `sampling_params.top_k = -1` leaves response sampling unfiltered.
SCMC requires a sampling seed, positive temperature and unfiltered sampling.
Array output requires nonstreaming generation and does not support speculative
decoding or beam search.

```python
import base64
import json
import numpy as np

with open('/tmp/tmax-scores.json') as f:
    meta = json.load(f)['meta_info']
shape = tuple(meta['output_top_logprobs_shape'])
assert meta['output_top_logprobs_null_prefix'] == 0
assert meta['output_top_logprobs_val_flat_b64_dtype'] == 'float32'
assert meta['output_top_logprobs_idx_flat_b64_dtype'] == 'int32'
logprobs = np.frombuffer(
    base64.b64decode(meta['output_top_logprobs_val_flat_b64']), dtype='<f4'
).reshape(shape)
token_ids = np.frombuffer(
    base64.b64decode(meta['output_top_logprobs_idx_flat_b64']), dtype='<i4'
).reshape(shape)
print(meta['output_score_mode'], logprobs.shape, token_ids.shape)
```

The mode is `topk` for SCT or `mc32`/`mc64`/`mc128` for SCMC. Preserve duplicate
MC IDs: they are independent samples. Values are normalized over the rollout
vocabulary, not renormalized over the returned candidates.

For an SCT packing comparison, remove `array_output_logprobs` from sampling
parameters and both `return_flat_raw_*` request flags to exercise legacy nested
JSON. Keep all sampling settings and candidate widths identical. SCMC currently
requires the array path. Base64 decoding allocates a byte buffer;
`np.frombuffer` then views that buffer without per-candidate Python objects.

## Validation

Both server YAML files parse through the pinned SGLang `prepare_server_args`;
all seven request examples pass request normalization and sampling validation.
The Docker patch was applied to a temporary Git index at the pinned base; its
resulting tree exactly matches the ported checkout. Pinned Ruff checks pass.

Against the ported source, the existing serving test files yielded **58 passed,
24 skipped** on CPU (CUDA cases skipped):

```bash
python -m pytest -q --noconftest \
  test/registered/sampling/test_mc32.py \
  test/registered/unit/managers/test_flat_raw_top_logprobs.py
```

These paths are relative to the SGLang checkout. The tests cover request
validation, duplicate preservation, array IPC and flat response packing. The
CUDA cases additionally check normalized scores, draw distributions and response
RNG isolation. CPU success does not qualify the Triton kernels on the new base,
NVFP4 loading, or the TMax session-to-trainer path. No new throughput or overhead
numbers have been measured for this port.
