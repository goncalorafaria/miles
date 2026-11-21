import pytest
from tests.ci.ci_register import register_cpu_ci
from tests.fast.fixtures.session_fixtures import make_session_server_config
from tests.fast.rollout.session.test_samples import _make_record
from tokenizers import Tokenizer
from tokenizers.models import WordLevel
from transformers import PreTrainedTokenizerFast

from miles.rollout.session.collection_process import CollectionProcess
from miles.rollout.session.linear_trajectory import LinearTrajectory
from miles.rollout.session.sessions import create_session_core
from miles.rollout.session.v2.session_state import SessionStateV2

register_cpu_ci(est_time=100, suite="stage-b-cpu", labels=["rollout"])


@pytest.mark.asyncio
@pytest.mark.parametrize("version", ["v1", "v2"])
async def test_spawned_assembly_matches_inline_bytes(tmp_path, version):
    tokenizer = PreTrainedTokenizerFast(
        tokenizer_object=Tokenizer(WordLevel({"[UNK]": 0, "hello": 1}, unk_token="[UNK]")),
        unk_token="[UNK]",
    )
    tokenizer.chat_template = "{% for message in messages %}{{ message['content'] }}{% endfor %}"
    tokenizer.save_pretrained(tmp_path)
    config = make_session_server_config(
        hf_checkpoint=str(tmp_path),
        use_session_server=version,
        session_sample_picker_path="miles.rollout.session.v2.picker_hub.drop_same_prompt_retries",
        session_sample_postprocessor_path="miles.rollout.session.v2.postprocessor_hub.default_postprocess",
    )
    core = create_session_core(None, config)
    session = SessionStateV2() if version == "v2" else LinearTrajectory()
    if version == "v1":
        session.records.append(_make_record([1], [1, 1]))
        session.trajectory_token_ids.append([1, 1, 1])
    else:
        parent = None
        for index in range(2):
            prompt = [1] * (1 + index * 3)
            parent = session.tree.create_node(
                parent,
                delta_messages=[{"role": "user", "content": "hello"}, {"role": "assistant", "content": "hello"}],
                token_ids=prompt + [1, 1],
                completion_span=(len(prompt), len(prompt) + 2),
                committed_at=float(index),
                response_id=str(index),
                record=_make_record(prompt, [1, 1]),
                finish_reason="stop",
            )
    expected = core._assemble_samples("s", session, max_seq_len=1024)
    assert expected.status_code == 200, expected.body
    worker = CollectionProcess(config, use_addition_r3=False)
    try:
        actual = await worker.assemble("s", session, max_seq_len=1024)
        assert actual.body == expected.body
        assert actual.status_code == expected.status_code
    finally:
        await worker.close()
