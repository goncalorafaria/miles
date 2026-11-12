# TMax score centering with DPPO-TV

The `goncalof/tmax-score-logprobs` branch combines the existing TMax DPPO-TV
policy objective with generated-token score centering. The baseline remains
unchanged unless `--loss-type score_centering` is enabled.

## Configs

- [SCT32 smoke](configs/smoke_4b_dppo_tv_sct32.yaml)
- [SCMC32 smoke](configs/smoke_4b_dppo_tv_scmc32.yaml)
- [SCT32 production layout](configs/prod_4b_dppo_tv_sct32.yaml)
- [SCMC32 production layout](configs/prod_4b_dppo_tv_scmc32.yaml)

The smoke configs retain the existing 4-GPU, 3-update TMax recipe. Production
configs retain 256 responses per update and the 64K episode context. They use
BF16 inference; the [NVFP4 inference-only configs](configs/inference/README.md)
still require dense checkpoint conversion and weight-publication support for RL.

Build the patched image using the [existing image procedure](beaker/README.md#images)
and replace `beaker.image` in the selected config with its immutable ID. The
checked-in image name is a placeholder, not a published image. The config uses
the user's output/trace directories and gateway registry; supply your own Docker
registry credentials if its sandbox images require them. The model and task data
are the prepared, shared TMax inputs. These configs have not yet completed a
full terminal-agent Megatron smoke run.

`extra_args` enables:

```text
--loss-type score_centering
--policy-loss-type dppo_tv                 # emitted by policy_loss: dppo_tv
--dppo-tv-threshold 0.1                    # emitted by the existing launcher
--use-rollout-logprobs                     # emitted by the existing launcher
--score-centering-estimator topk           # or mc
--rollout-top-logprobs-num 32               # also supports 64/128
--rollout-score-array
--score-centering-is none
```

DPPO-TV already supplies the importance ratio and gate. Additional TIS/MIS
weighting is rejected rather than silently changing their composition. To test
64 or 128, change the candidate width in the config's `extra_args`. Response
sampling remains unfiltered; the width controls extraction, not top-k sampling.
`SGLANG_RETURN_ORIGINAL_LOGPROB=0` must reach the inference processes.

## Loss definition

Let `p` be the current trainer distribution, `q` the recorded rollout
distribution, `A` the detached token advantage and `r=p/q`. The existing DPPO-TV
mask rejects an action when `abs(p-q)>threshold` and the proposed update pushes
farther from the rollout policy. Its sampled loss remains `-A*r*mask`, including
the existing numerical exponent clamp. The centering coefficient `w` is the
actual derivative coefficient of that surrogate with respect to `log p`, so it
also becomes zero outside the native exponent-clamp gradient range.

The combined loss adds `A*C`; its gradient is:

```text
-A * [w(y) * grad(log p(y)) - estimated E_{v~q}[w(v) * grad(log p(v))]]
```

Candidate weights and recorded rollout probabilities are detached. The DPPO-TV
mask is evaluated on each candidate using that token's advantage sign. A masked
sampled action may still receive a centering correction; masking the entire
corrected loss would change this estimator. No straight-through loss expression
or CUDA synchronization is introduced.

- **SCMC:** draw K independent samples with replacement from `q` and use
  `C = mean(stopgrad(w(v)) * log p(v))`. Duplicate IDs are retained. This is a
  Monte Carlo estimate of the weighted-score expectation; it does not invent a
  probability mass or tail from the K draws.
- **SCT:** use the recorded top-K head and
  `C = sum_head(stopgrad(q(v)*w(v)-p(v)) * log p(v))`. This models the unobserved
  tail as ungated importance sampling (`q*w=p`). It is exact at full vocabulary
  coverage, and approximate if DPPO gating or the exponent guard affects omitted
  tail tokens. Existing non-DPPO SCT retains its original rescaled-tail formula.

## Data path

1. Session v2 resolves the request with the run's estimator and candidate width.
   The SGLang chat schema forwards `array_output_logprobs` and
   `score_centering_samples` into sampling parameters. MC needs a seed; when the
   caller omits one, the request receives a fresh seed which is recorded with it.
2. SGLang keeps candidates as typed arrays through scheduler/detokenizer IPC,
   then base64 encodes them at the HTTP boundary. The response mode, shape,
   dtype and byte lengths are checked before training; an old server cannot
   silently return top-k when MC was requested.
3. Session collection decodes arrays, trims them with TITO output tokens and
   inserts masked padding for tool observations. The v2 sample codec and training
   batch use the existing typed candidate fields. MC duplicates survive both.
4. The chunked selected-logprob trainer scores sampled and candidate token IDs,
   computes the combined loss, and uses the existing loss reduction, entropy and
   reference-KL handling. MC diagnostics report mean probe probabilities instead
   of labeling repeated draws as head probability mass.

## Validation scope

The CPU regression selection passed 339 tests (31 skipped), including the
existing score-centering modes, DPPO-TV, session codecs, generated-token array
transport and SGLang chat schema. Tests cover both advantage signs, native
exponent guards, detached rollout probabilities, full-vocabulary cancellation,
MC duplicate counts, tool masking and three optimizer updates through the
session-array-to-training path for K=32/64/128. The two Ray-dependent cases need
the repository's `conftest.py`; all other selected cases also run with
`--noconftest`.

A separate opt-in check lives in
[`tests/manual/test_tmax_score_centering_training.py`](../../tests/manual/test_tmax_score_centering_training.py).
It requires `TMAX_SC_MODEL`, `TMAX_SC_ENDPOINT` and a separate trainer GPU. It
uses real SGLang generations, the session-v2 sample codec and the Miles combined
loss for three HF-model optimizer updates in each of the six modes. Rewards
are synthetic +1/-1 and no training weights are published back to the server.
It is neither a terminal reward evaluation nor a Megatron production run.

The SGLang port also passed all **86 tests on a Holmes GPU**, including all 24
CUDA cases, in experiment `01M78P16591G6P96CKB9M7SKJY`.

The real-model check passed all **six modes and 18 optimizer updates** on
2026-11-12 in experiment `01M78PF4BP4SYMB1TAVCX96CW9` (exit code 0): SCT and
SCMC at K=32/64/128, using the prepared `Qwen3.5-4B-tmax-template` checkpoint
with BF16 inference. One Holmes B300 served SGLang and a second trained the
model. Each mode used two live generations of 64 response tokens and three
updates through the actual Miles DPPO-TV score-centering loss. All losses and
gradient norms were finite, and every update had a nonzero gradient. The pytest
phase completed in 171.31 seconds, including model loading and cold training
kernel compilation; this is a correctness check, not an overhead benchmark.

The first real-model attempt exposed a tokenizer API mismatch (chat-template
output was a `BatchEncoding`); explicitly requesting token IDs fixed it before
the successful run. The successful source revisions were Miles `55fbb24` and
SGLang `34aff9cd3c`. Result dataset `01M78PF4BXXH1X7WRSQ6BBV8K4` contains the
JUnit report, server log and success marker. The shared trainer model retains
updates between modes while the rollout server keeps its initial weights, so
this also exercises scoring against a stale rollout policy. It does not cover
Megatron parallelism, terminal rewards, weight publication or NVFP4 inference.
