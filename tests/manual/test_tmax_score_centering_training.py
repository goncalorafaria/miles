"""Opt-in real-model check: live SGLang chat -> session-v2 sample codec -> Miles loss -> HF optimizer.

Set TMAX_SC_MODEL to a local TMax HF checkpoint and TMAX_SC_ENDPOINT to a patched
SGLang server. Run with CUDA_VISIBLE_DEVICES selecting the separate trainer GPU.
This is a short synthetic-reward integration check, not a Megatron terminal RL run.
"""

import os
from types import SimpleNamespace

import pytest
import requests
import torch
import transformers
from tests.fast.backends.training_utils.test_score_centering_pipeline import single_rank  # noqa: F401
from tests.fast.fixtures.score_centering_fixtures import _args
from tests.fast.fixtures.session_fixtures import make_session_server_config

from miles.backends.training_utils.data.context_parallel import get_sum_of_sample_mean
from miles.backends.training_utils.loss.hub.losses import get_loss_function
from miles.ray.rollout.train_data_conversion import convert_samples_to_train_data
from miles.rollout.session.request_args import prepare_chat_request
from miles.rollout.session.samples.codec import (
    COMPUTED_FIELDS_V2,
    decode_samples_and_merge_input_sample,
    encode_samples,
)
from miles.rollout.session.samples.merge import compute_samples_from_openai_records
from miles.rollout.session.types import SessionRecord
from miles.utils.types import Sample


@pytest.fixture(scope="module")
def live_model():
    path, endpoint = os.environ.get("TMAX_SC_MODEL"), os.environ.get("TMAX_SC_ENDPOINT")
    if not path or not endpoint:
        pytest.skip("set TMAX_SC_MODEL and TMAX_SC_ENDPOINT for real-model training")
    assert torch.cuda.is_available(), "A separate trainer GPU is required"
    config = transformers.AutoConfig.from_pretrained(path, local_files_only=True)
    cls = getattr(transformers, config.architectures[0])
    model = cls.from_pretrained(path, torch_dtype=torch.bfloat16, attn_implementation="sdpa", local_files_only=True)
    model.to("cuda").train()
    model.config.use_cache = False
    tokenizer = transformers.AutoTokenizer.from_pretrained(path, local_files_only=True)
    return model, tokenizer, endpoint.rstrip("/")


@pytest.mark.parametrize("estimator", ["topk", "mc"])
@pytest.mark.parametrize("k", [32, 64, 128])
@pytest.mark.usefixtures("single_rank")
def test_generated_arrays_train_real_model(live_model, estimator, k):
    model, tokenizer, endpoint = live_model
    config = make_session_server_config(
        rollout_top_logprobs_num=k,
        rollout_score_array=True,
        score_centering_estimator=estimator,
        use_session_server="v2",
    )
    args = _args(
        rollout_top_logprobs_num=k,
        rollout_temperature=1.0,
        rollout_score_array=True,
        score_centering_estimator=estimator,
        policy_loss_type="dppo_tv",
        dppo_tv_threshold=0.1,
        save_debug_trajectory_data=None,
        sglang_speculative_algorithm=None,
        ci_test=True,
        qkv_format="bshd",
        vocab_size=getattr(model.config, "text_config", model.config).vocab_size,
    )
    samples = []
    for index, question in enumerate(("What is 17 times 23?", "What is 13 times 19?")):
        messages = [{"role": "user", "content": question}]
        prompt = tokenizer.apply_chat_template(messages, tokenize=True, return_dict=False, add_generation_prompt=True)
        body = prepare_chat_request(
            {"model": "tmax-score-smoke", "messages": messages, "max_tokens": 64, "seed": 17 + index},
            SimpleNamespace(resolve_request_args=lambda request, **kwargs: request),
            config=config,
            turn_args=None,
        ).body
        body["input_ids"] = prompt  # The session renderer supplies these after request validation.
        response = requests.post(f"{endpoint}/v1/chat/completions", json=body, timeout=300)
        response.raise_for_status()
        record = SessionRecord(
            timestamp=2.0,
            request_timestamp=1.0,
            method="POST",
            path="v1/chat/completions",
            status_code=200,
            request=body,
            response=response.json(),
        )
        sample = compute_samples_from_openai_records(args, [record], tokenizer)[0]
        sample.reward, sample.index, sample.group_index = (1.0 if index == 0 else -1.0), index, 0
        samples.append(sample)
    wire = encode_samples(samples, {}, fields=COMPUTED_FIELDS_V2)
    samples = decode_samples_and_merge_input_sample(wire, Sample(), fields=COMPUTED_FIELDS_V2).samples
    batch = convert_samples_to_train_data(args, samples, {}, None, None)
    total = [len(s.tokens) for s in samples]
    maximum = max(total)
    tokens = torch.zeros((len(samples), maximum), dtype=torch.long, device="cuda")
    mask = torch.zeros_like(tokens)
    for i, sample in enumerate(samples):
        tokens[i, : total[i]] = torch.tensor(sample.tokens, device="cuda")
        mask[i, : total[i]] = 1
    batch.update(
        total_lengths=total,
        max_seq_lens=[maximum] * len(samples),
        unconcat_tokens=[torch.tensor(s.tokens, device="cuda") for s in samples],
        loss_masks=[torch.tensor(s.loss_mask, device="cuda") for s in samples],
        rollout_log_probs=[torch.tensor(s.rollout_log_probs, device="cuda") for s in samples],
        advantages=[
            torch.full((s.response_length,), 1.0 if i == 0 else -1.0, device="cuda") for i, s in enumerate(samples)
        ],
    )
    reduce = get_sum_of_sample_mean(
        total, batch["response_lengths"], batch["loss_masks"], qkv_format="bshd", max_seq_lens=batch["max_seq_lens"]
    )
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-5)
    for step in range(3):
        optimizer.zero_grad(set_to_none=True)
        logits = model(input_ids=tokens, attention_mask=mask, use_cache=False).logits
        loss, metrics = get_loss_function(args)(args, batch, logits, reduce)
        assert torch.isfinite(loss)
        loss.backward()
        norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0, error_if_nonfinite=True)
        assert norm.item() > 0
        optimizer.step()
        print(
            {
                "estimator": estimator,
                "k": k,
                "step": step,
                "response_tokens": sum(batch["response_lengths"]),
                "loss": loss.item(),
                "grad_norm": norm.item(),
                "correction": metrics["sc_correction"].item(),
            },
            flush=True,
        )
    optimizer.zero_grad(set_to_none=True)
