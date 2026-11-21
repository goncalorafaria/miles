"""Burst identical finalized sessions through thread/process assembly; no inference."""

import argparse
import asyncio
import json
import time
from pathlib import Path

from tests.fast.fixtures.session_fixtures import make_session_server_config
from tests.fast.rollout.session.test_samples import _make_record
from tokenizers import Tokenizer
from tokenizers.models import WordLevel
from transformers import PreTrainedTokenizerFast

from miles.rollout.session.linear_trajectory import LinearTrajectory
from miles.rollout.session.samples.codec import decode_samples_and_merge_input_sample
from miles.rollout.session.sessions import create_session_core
from miles.utils.types import Sample


async def run(args):
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=True)
    tokenizer = PreTrainedTokenizerFast(
        tokenizer_object=Tokenizer(WordLevel({"[UNK]": 0, "hello": 1}, unk_token="[UNK]")), unk_token="[UNK]"
    )
    tokenizer.chat_template = "{% for message in messages %}{{ message['content'] }}{% endfor %}"
    tokenizer.save_pretrained(output / "tokenizer")
    record = _make_record([1], [1] * args.tokens)
    rows = []
    for process in (False, True):
        core = create_session_core(
            None,
            make_session_server_config(
                hf_checkpoint=str(output / "tokenizer"),
                session_collection_processes=process,
                session_collection_workers=2,
            ),
        )
        state = LinearTrajectory(
            messages=[{"role": "user", "content": "hello"}, {"role": "assistant", "content": "hello"}],
            records=[record],
            trajectory_token_ids=[[1] * (args.tokens + 1)],
        )
        warm = await core.collections.collect(
            "warm", state, key=None, assemble=lambda: core._dispatch_assembly("s", state, max_seq_len=None)
        )
        expected = decode_samples_and_merge_input_sample(warm.body, Sample()).samples
        assert expected and len(expected[0].tokens) == args.tokens + 1
        stalls = []

        async def tick():
            while True:
                start = time.monotonic()
                await asyncio.sleep(0.01)
                stalls.append(max(0, time.monotonic() - start - 0.01))

        async def collect(i):
            state = LinearTrajectory(
                messages=[{"role": "user", "content": "hello"}, {"role": "assistant", "content": "hello"}],
                records=[record],
                trajectory_token_ids=[[1] * (args.tokens + 1)],
            )
            result = await core.collections.collect(
                str(i), state, key=None, assemble=lambda: core._dispatch_assembly("s", state, max_seq_len=None)
            )
            assert result.status_code == 200
            return result.body

        ticker = asyncio.create_task(tick())
        await asyncio.sleep(0.02)
        start = time.monotonic()
        results = await asyncio.gather(*(collect(i) for i in range(args.sessions)))
        elapsed = time.monotonic() - start
        await asyncio.sleep(0.02)
        ticker.cancel()
        await asyncio.gather(ticker, return_exceptions=True)
        for body in results:
            actual = decode_samples_and_merge_input_sample(body, Sample()).samples[0]
            assert actual.tokens == expected[0].tokens
            assert actual.rollout_log_probs == expected[0].rollout_log_probs
        row = dict(
            mode="process" if process else "thread",
            sessions=args.sessions,
            tokens=args.tokens,
            seconds=elapsed,
            sessions_per_second=args.sessions / elapsed,
            max_loop_stall_seconds=max(stalls),
        )
        rows.append(row)
        print(json.dumps(row), flush=True)
        if core.collection_process:
            await core.collection_process.close()
        (output / "results.json").write_text(json.dumps(rows, indent=2))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", required=True)
    parser.add_argument("--sessions", type=int, default=128)
    parser.add_argument("--tokens", type=int, default=32768)
    asyncio.run(run(parser.parse_args()))
