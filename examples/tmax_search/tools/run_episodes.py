"""Run search episodes through the port's agent function against a local vLLM policy and the local services.

Without a Miles session server there is no training sample, so each turn's completion token ids are captured from
vLLM (return_token_ids) to score the episodes later (score_episodes.py).

Usage: run_episodes.py <out.jsonl> <num_questions> <samples_per_question> <policy_base_url> <search_url> <terminal_url>
"""

import asyncio
import json
import os
import sys
import time

import openai
import search_agent_function as saf
from search_config import resolve_env_args

SRC = os.environ.get(
    "PRIME_RL_SOURCE",
    "/weka/oe-adapt-default/shashankg/miles_search/prime-rl-sources/"
    "a5fcb924645d9bd8f326efd5087c8de6447c83fe17c6d07c36e99ee736d30a37/source",
)


class _Recorder:
    """An AsyncOpenAI stand-in that records each completion's token ids per episode."""

    def __init__(self, sink: list, **kwargs):
        self._client = openai.AsyncOpenAI(**kwargs)
        self._sink = sink
        self.chat = self
        self.completions = self

    async def create(self, **kwargs):
        kwargs.setdefault("extra_body", {})["return_token_ids"] = True
        response = await self._client.chat.completions.create(**kwargs)
        extra = response.choices[0].model_extra or {}
        self._sink.append(extra.get("token_ids"))
        return response


async def main() -> None:
    out, n_questions, n_samples, policy, search_url, terminal_url = sys.argv[1:7]
    env = resolve_env_args(
        prime_rl_source=SRC,
        prime_rl_config="configs/experiments/search-agent-bc-rl-v1-1-fp32-lr5e-7/rl.toml",
        prime_rl_checkout="/weka/stevenc/prime-rl",
        weka_mount="/weka/oe-adapt-default",
        search_url=search_url,
        terminal_url=terminal_url,
        judge_url="http://unused/v1",
    )
    os.environ["SEARCH_ENV_ARGS"] = json.dumps(env)
    rows = [json.loads(line) for line in open(env["dataset"])][: int(n_questions)]
    sem = asyncio.Semaphore(32)

    async def episode(row):
        question = row["question"].strip()
        prompt = [
            {"role": m["role"], "content": m["content"].format(question=question)}
            for m in env["template"]["chat_template"]
        ]
        token_ids: list = []
        async with sem:
            t0 = time.time()
            # run() builds its client before its first await, so nothing else runs between the patch and its use.
            saf.AsyncOpenAI = lambda **kw: _Recorder(token_ids, **kw)
            result = await saf.run(
                base_url=policy,
                prompt=prompt,
                request_kwargs={"temperature": 1.0, "top_p": 1.0, "top_k": -1, "max_tokens": 32768},
                metadata={"id": row["id"], "question": question, "answer": row["answer"], "max_seq_len": 131072},
            )
        result["turn_token_ids"] = token_ids
        result["wall"] = time.time() - t0
        return result

    jobs = [episode(row) for row in rows for _ in range(int(n_samples))]
    with open(out, "w") as f:
        for coro in asyncio.as_completed(jobs):
            result = await coro
            f.write(json.dumps(result) + "\n")
            f.flush()
            m = result["agent_metrics"]
            print(
                f"stop={m['stop_condition']} turns={m['turns']} tools={m['tool_calls']} "
                f"search={m['search_calls']} web={m['webterminal_calls']} stderr={m['stderr_count']} "
                f"tokens={m['total_tokens']} wall={result['wall']:.0f}s",
                flush=True,
            )


asyncio.run(main())
