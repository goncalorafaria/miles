# TMax score centering with DPPO-TV

The `goncalof/tmax-score-logprobs` branch combines the existing TMax DPPO-TV
policy objective with generated-token score centering. The baseline remains
unchanged unless `--loss-type score_centering` is enabled.

## Configs

- [SCT128 smoke, 4096-position LM-head chunks](configs/smoke_4b_dppo_tv_sct128.yaml)
- [SCT32 smoke](configs/smoke_4b_dppo_tv_sct32.yaml)
- [SCMC32 smoke](configs/smoke_4b_dppo_tv_scmc32.yaml)
- [SCT32 production layout](configs/prod_4b_dppo_tv_sct32.yaml)
- [SCMC32 production layout](configs/prod_4b_dppo_tv_scmc32.yaml)

The smoke configs retain the existing 4-GPU, 3-update TMax recipe. Production
configs retain 256 responses per update and the 64K episode context. They use
BF16 inference unless the explicit NVFP4 variant is selected; see the dense
NVFP4 publication path below.

The SCT128 smoke config pins the patched runtime `01M78W2R0WT8YC53P4V3H3RHCH`,
built with [Dockerfile.score-logprobs](docker/Dockerfile.score-logprobs) from the
working TMax base image. For the other configs, build the patched image using
the [existing image procedure](beaker/README.md#images) and replace their
placeholder `beaker.image` with its immutable ID. The config uses
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

## Chunked LM-head projection

The SCT128 smoke config enables `--score-centering-head-chunk-size 4096` and
`--log-probs-chunk-size 4096`. The first bounds the projection itself: Megatron's
output processor returns final hidden states instead of full-sequence logits.
The loss projects response positions in chunks, checkpoints projection and
normalization together, and retains only sampled/candidate log probabilities.
Backward recomputes vocabulary tensors one chunk at a time. The second knob
bounds the normalization within each projection chunk. Setting only the second
knob does not avoid allocating full-sequence vocabulary logits.

The head adapter also handles reference/actor forward-only scoring, preserving
the TMax reference-KL diagnostic. Shared loss scaling, tool masks, entropy and
DPPO-TV remain in the same Miles loss path. The implementation currently supports
an unsharded affine head (TP=CP=PP=1, with data parallelism supported), no LoRA,
FP8, MTP, sampling-support replay or overlapping parameter gathers. Unsupported
launch combinations fail validation. `sc_head_chunk_size` and `sc_head_chunks`
make the active training path visible in the metrics. No CUDA synchronization
was added.

The SCT128 recipe keeps the existing TMax smoke workload: 2 trainer + 2 inference
GPUs, 32 terminal responses per update, three updates, the same prepared TMax
model/task data, episode limits, sandbox gateway and BF16 inference. NVFP4 serving
configs now have a separate publication validation job. This is the full TMax entrypoint, not
the separate HF synthetic-reward probe below.

## Dense NVFP4 with MC32

[prod_4b_dppo_tv_scmc32_nvfp4.yaml](configs/prod_4b_dppo_tv_scmc32_nvfp4.yaml)
keeps the production TMax workload: two 8-GPU nodes, eight trainer and eight
inference GPUs, 256 terminal responses per update, 500 updates, DPPO-TV, MC32
arrays, and 4096-position LM-head chunks. The matching
[smoke config](configs/smoke_4b_dppo_tv_scmc32_nvfp4.yaml) uses two trainer and two
inference GPUs for three updates. Both request Holmes with urgent priority,
preemptible scheduling, zero retries, and no minimum runtime.

The patched `nvfp4_dense_online` SGLang method accepts the original BF16 TMax
checkpoint and the trainer's existing BF16 publication stream. The native
`end_weight_update` hook quantizes each completed fused dense matrix into NVFP4
weights and FP8 block scales. Dense GEMMs quantize activations dynamically and
use FlashInfer CUTLASS W4A4 on Blackwell. Embeddings, LM head, normalization,
convolution/state parameters, and KV cache retain their existing precision.
There is no separate converted checkpoint or trainer quantization metadata.

BF16 loader weights remain resident alongside packed weights. This preserves
partial/bucketed loads and fused projection loaders but adds memory compared
with packed-only inference. Packed weights/scales are refreshed in place so
CUDA graphs see newly published values. No host synchronization is added.
This implementation targets dense BF16/FP16 source models; it rejects MoE and
already-quantized checkpoint inputs.

[sglang_nvfp4_dense_online.patch](../../docker/patch/sglang_nvfp4_dense_online.patch)
is included in both Docker build paths. Runtime `01M793QBYA81CQXDHVFBNS2QCN`
contains it and the score-array patch on the unchanged SGLang base.
The first Blackwell validation,
[01M7911C7QQ3NP1Z88T3YF1AW8](https://beaker.org/ex/01M7911C7QQ3NP1Z88T3YF1AW8),
passed all three independent dequantized-reference GEMM and repeated CUDA-graph
reload cases. Its real 4B test failed before engine startup because the test
used the obsolete `cuda_graph_max_bs` keyword. The test now uses
`cuda_graph_max_bs_decode`; all 13 Engine keywords were checked against the
pinned runtime schema.

The second validation,
[01M7939AV1GJVQBCZY4DEKSNRQ](https://beaker.org/ex/01M7939AV1GJVQBCZY4DEKSNRQ),
also passed the three kernel cases but failed during server construction: the
config was missing the required `get_scaled_act_names()` method. This is fixed
in the current runtime. Local checks instantiate the config and real column,
row, and merged-column layers; a regression test loads both fused BF16 shards.
The third validation loaded the real model and captured CUDA graphs, then failed
because the test's weight suffix also matched an unused MTP weight. The fixture
now selects the actor weight explicitly; MTP remains disabled. The
[fourth validation](https://beaker.org/ex/01M794707AVS3P3HFTFWM9W0E5) completed
MC32 array generation before and after weight updates, but its restoration check
compared log probabilities of sampled tokens, which can differ across requests.
The revised test compares token-aligned first-position SCT distributions for the
same prompt, checks that zeroing the weight changes the distribution, and checks
restoration within the original 0.002 tolerance. MC32 array checks still run before
and after each update. Real-model publication remains unverified until this passes.

The [BF16 control](configs/prod_4b_dppo_tv_scmc32_bf16.yaml) matches the two-node
NVFP4 production config, runtime image, MC32 arrays, DPPO-TV, 4K head chunks,
256 responses/update, and scheduling. Only the run name and inference precision
arguments differ. The BF16 control launches independently; NVFP4 production
remains gated on all four validation tests passing without skips.

Matched runs pinned to Miles `924bc88` and runtime
`01M793QBYA81CQXDHVFBNS2QCN`:
- [BF16 two-node production](https://beaker.org/ex/01M793SQKE64ADTNCJQ1K5EQHN):
  both replicas started; the TMax training entrypoint launched, with training results pending.
- [NVFP4 publication retry](https://beaker.org/ex/01M794KXY7XJ97FNVB2FMX9Z2Z):
  scheduled; the separate two-node NVFP4 run submits only after this passes.


## TMax SCT128 run

The full terminal-RL smoke was submitted as
[01M78W4XJXMJBAV6HY5YR6PBC3](https://beaker.org/ex/01M78W4XJXMJBAV6HY5YR6PBC3),
pinned to Miles `e709405` and runtime `01M78W2R0WT8YC53P4V3H3RHCH`.
It runs `smoke_4b_dppo_tv_sct128.yaml` through the normal TMax Beaker entrypoint.
The submitted spec maps this checkout's `/weka/gfaria` prefix to the job's
`/weka/oe-adapt-default/gfaria` mount, runs focused Megatron preflight tests,
and has a two-hour job timeout. At submission it was queued; submission is not
a completed-training result.

Projection, loss/gradient parity, masking, reference scoring and config tests
passed on CPU (45 passed, one unconfigured production variant skipped). The
projection test crosses the 4096-position boundary and checks that checkpointing
does not retain vocabulary-sized activations. The existing Megatron model tests
require CUDA libraries unavailable in the CPU container, so those are included
in the GPU job's preflight instead.

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
