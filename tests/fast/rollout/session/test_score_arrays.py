"""Array scores survive session-v2 preparation, trimming, tools, and binary sample transport."""

import base64
import json
from types import SimpleNamespace

import httpx
import numpy as np
import pytest
import torch
from tests.fast.backends.training_utils.test_score_centering_pipeline import single_rank  # noqa: F401
from tests.fast.fixtures.score_centering_fixtures import _args, _Tokenizer
from tests.fast.fixtures.session_fixtures import make_session_server_config

from miles.backends.training_utils.data.context_parallel import get_sum_of_sample_mean
from miles.backends.training_utils.loss.hub.losses import get_loss_function
from miles.ray.rollout.train_data_conversion import convert_samples_to_train_data, split_train_data_by_dp_raw
from miles.rollout.generate_utils.rollout_topk_logprobs import (
    append_rollout_topk_logprobs,
    validate_rollout_topk_logprobs_sample,
)
from miles.rollout.generate_utils.sample_utils import merge_samples
from miles.rollout.session.request_args import prepare_chat_request
from miles.rollout.session.samples.codec import (
    COMPUTED_FIELDS_V2,
    decode_samples_and_merge_input_sample,
    encode_samples,
)
from miles.rollout.session.samples.merge import compute_samples_from_openai_records
from miles.rollout.session.types import SessionRecord
from miles.router.config import MilesRouterConfig
from miles.router.router import MilesRouter
from miles.utils.types import Sample


def _meta(k, mode, n=2):
    ids = np.tile(np.arange(k, dtype=np.int32) if mode == "topk" else np.zeros(k, dtype=np.int32), (n, 1))
    logps = np.full((n, k), -np.log(160), dtype=np.float32)
    return {
        "output_token_logprobs": [(-float(np.log(160)), 0, None)] * n,
        "output_score_mode": "topk" if mode == "topk" else f"mc{k}",
        "output_top_logprobs_shape": [n, k],
        "output_top_logprobs_null_prefix": 0,
        "output_top_logprobs_idx_flat_b64": base64.b64encode(ids.tobytes()).decode(),
        "output_top_logprobs_val_flat_b64": base64.b64encode(logps.tobytes()).decode(),
        "output_top_logprobs_idx_flat_b64_dtype": "int32",
        "output_top_logprobs_val_flat_b64_dtype": "float32",
    }


@pytest.mark.parametrize("k", [32, 64, 128])
@pytest.mark.parametrize("mode", ["topk", "mc"])
@pytest.mark.usefixtures("single_rank")
def test_multiturn_array_session_transport(k, mode):
    config = make_session_server_config(
        rollout_top_logprobs_num=k, rollout_score_array=True, score_centering_estimator=mode, use_session_server="v2"
    )
    tokenizer = SimpleNamespace(resolve_request_args=lambda request, **kwargs: request)
    prepared = prepare_chat_request({}, tokenizer, config=config, turn_args=None).body
    assert prepared["array_output_logprobs"] is True
    assert prepared["score_centering_samples"] == (k if mode == "mc" else 0)
    if mode == "mc":
        assert isinstance(prepared["seed"], int)
    records = [
        SessionRecord(
            timestamp=2.0,
            request_timestamp=1.0,
            method="POST",
            path="v1/chat/completions",
            status_code=200,
            request={**prepared, "input_ids": prompt},
            response={"choices": [{"meta_info": _meta(k, mode), "finish_reason": "stop"}]},
        )
        for prompt in ([1, 2], [1, 2, 0, 7])
    ]
    samples = compute_samples_from_openai_records(
        _args(
            rollout_top_logprobs_num=k,
            rollout_score_array=True,
            score_centering_estimator=mode,
            save_debug_trajectory_data=None,
            sglang_speculative_algorithm=None,
        ),
        records,
        _Tokenizer(),
        accumulated_token_ids=[1, 2, 0, 7, 0, 0],
        max_trim_tokens=1,
    )
    merged = merge_samples(samples, _Tokenizer())
    assert merged.loss_mask == [1, 0, 1, 1]
    validate_rollout_topk_logprobs_sample(merged, k, estimator=mode)
    # The v2 codec carries candidates as typed tensors, never lists of Python numbers.
    wire = encode_samples([merged], {}, fields=COMPUTED_FIELDS_V2)
    restored = decode_samples_and_merge_input_sample(wire, Sample()).samples[0]
    np.testing.assert_array_equal(restored.rollout_topk_token_ids, merged.rollout_topk_token_ids)
    np.testing.assert_array_equal(restored.rollout_topk_log_probs, merged.rollout_topk_log_probs)
    assert restored.rollout_topk_token_ids.dtype == np.int32
    if mode == "mc":
        assert (restored.rollout_topk_token_ids[0] == 0).all()
    # Exercise the real sample -> train batch -> loss -> optimizer path, including tool masks.
    restored.reward, restored.index, restored.group_index = 1.0, 0, 0
    args = _args(
        rollout_temperature=1.0,
        vocab_size=160,
        ci_test=True,
        rollout_top_logprobs_num=k,
        rollout_score_array=True,
        score_centering_estimator=mode,
        policy_loss_type="dppo_tv",
        dppo_tv_threshold=0.1,
    )
    data = convert_samples_to_train_data(args, [restored], {}, None, None)
    batch = split_train_data_by_dp_raw(args, data, dp_size=1)[0]
    batch.update(
        total_lengths=[len(restored.tokens)],
        unconcat_tokens=[torch.tensor(restored.tokens)],
        loss_masks=[torch.tensor(restored.loss_mask)],
        rollout_log_probs=[torch.tensor(restored.rollout_log_probs)],
        advantages=[torch.ones(restored.response_length)],
    )
    reduce = get_sum_of_sample_mean(batch["total_lengths"], batch["response_lengths"], batch["loss_masks"])
    torch.manual_seed(7)
    model = torch.nn.Sequential(torch.nn.Embedding(160, 16), torch.nn.Linear(16, 160))
    optimizer = torch.optim.SGD(model.parameters(), lr=0.01)
    initial = model[1].weight.detach().clone()
    for _ in range(3):
        optimizer.zero_grad()
        logits = model(torch.tensor([restored.tokens]))
        logits.retain_grad()
        loss, metrics = get_loss_function(args)(args, batch, logits, reduce)
        assert torch.isfinite(loss) and torch.isfinite(metrics["sc_correction"])
        loss.backward()
        assert all(torch.isfinite(p.grad).all() for p in model.parameters())
        assert logits.grad[0, 2].count_nonzero() == 0  # The tool-observation target is untrained.
        optimizer.step()
    assert not torch.equal(initial, model[1].weight)


@pytest.mark.parametrize(
    "change",
    [
        {"output_score_mode": "topk"},
        {"output_top_logprobs_shape": [1, 32]},
        {"output_top_logprobs_val_flat_b64_dtype": "float64"},
        {"output_top_logprobs_idx_flat_b64": "AA=="},
    ],
)
def test_array_decoder_rejects_wrong_server_contract(change):
    with pytest.raises(ValueError):
        append_rollout_topk_logprobs(Sample(), _meta(32, "mc") | change, 32, estimator="mc", array_output=True)


@pytest.mark.asyncio
@pytest.mark.parametrize("mode", ["mc", "topk"])
@pytest.mark.parametrize("k", [32, 64, 128])
async def test_score_array_request_and_response_survive_miles_router(mode, k):
    config = make_session_server_config(
        rollout_top_logprobs_num=k, rollout_score_array=True, score_centering_estimator=mode
    )
    tokenizer = SimpleNamespace(resolve_request_args=lambda request, **kwargs: request)
    prepared = prepare_chat_request({}, tokenizer, config=config, turn_args=None).body
    body = json.dumps(prepared).encode()
    response_body = json.dumps({"choices": [{"meta_info": _meta(k, mode)}]}).encode()

    async def backend(request):
        assert request.url.path == "/v1/chat/completions"
        assert request.content == body
        forwarded = json.loads(request.content)
        assert forwarded["score_centering_samples"] == (k if mode == "mc" else 0)
        assert forwarded["array_output_logprobs"] is True
        return httpx.Response(200, content=response_body, headers={"content-type": "application/json"})

    router = MilesRouter(
        MilesRouterConfig(
            host="127.0.0.1",
            port=0,
            max_connections=8,
            timeout=10,
            health_check_interval=60,
            health_check_failure_threshold=3,
        )
    )
    await router.client.aclose()
    router.client = httpx.AsyncClient(transport=httpx.MockTransport(backend))
    router.worker_request_counts["http://backend"] = 0
    try:
        async with httpx.AsyncClient(
            transport=httpx.ASGITransport(app=router.app), base_url="http://router"
        ) as client:
            response = await client.post(
                "/v1/chat/completions", content=body, headers={"content-type": "application/json"}
            )
        assert response.status_code == 200
        assert response.json() == json.loads(response_body)
        sample = Sample()
        append_rollout_topk_logprobs(
            sample, response.json()["choices"][0]["meta_info"], k, estimator=mode, array_output=True
        )
        assert sample.rollout_topk_token_ids.shape == (2, k)
        assert sample.rollout_topk_log_probs.dtype == np.float32
    finally:
        await router.client.aclose()
