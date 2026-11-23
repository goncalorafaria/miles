"""Build a TITO-valid synthetic TMax conversation script.

Every simulated episode replays the first T turns of the same script: turn k's request is
the canonical history plus one tool observation, and its reply is a pre-rendered SGLang
chat completion (bash tool call, per-token logprobs).
"""

import json
import pickle
import sys

import numpy as np

BASH_TOOL = {
    "type": "function",
    "function": {
        "name": "bash",
        "description": "Run one bash command.",
        "parameters": {
            "type": "object",
            "properties": {"command": {"type": "string"}},
            "required": ["command"],
        },
    },
}


def _repeat_text(tokenizer, token_id, n):
    text = tokenizer.decode([token_id] * n, skip_special_tokens=False)
    assert len(tokenizer.encode(text, add_special_tokens=False)) == n
    return text


def build(hf_checkpoint, tito_model, *, turns, prompt_tokens, output_tokens, obs_tokens):
    from miles.utils.chat_template_utils import get_tito_tokenizer, resolve_fixed_chat_template
    from miles.utils.processing_utils import load_tokenizer

    template, template_kwargs = resolve_fixed_chat_template(tito_model)
    tokenizer = load_tokenizer(hf_checkpoint, chat_template_path=template, trust_remote_code=True)
    tito = get_tito_tokenizer(tokenizer, tokenizer_type=tito_model, chat_template_kwargs=template_kwargs)
    token_id = next(
        t
        for text in (" x", " a", " the")
        for t in tokenizer.encode(text, add_special_tokens=False)
        if tokenizer.encode(tokenizer.decode([t]), add_special_tokens=False) == [t]
    )
    history = [
        {"role": "user", "content": "Please solve this task:" + _repeat_text(tokenizer, token_id, prompt_tokens)}
    ]
    tools = [BASH_TOOL]
    template_args = tito.default_template_args(tools)
    out = []
    for turn in range(turns):
        if turn:
            history = history + [
                {
                    "role": "tool",
                    "tool_call_id": f"call_{turn - 1:05d}",
                    "content": _repeat_text(tokenizer, token_id, obs_tokens[turn % len(obs_tokens)]),
                }
            ]
        n_out = output_tokens[turn % len(output_tokens)]
        assistant = {
            "role": "assistant",
            "content": _repeat_text(tokenizer, token_id, n_out),
            "tool_calls": [
                {
                    "id": f"call_{turn:05d}",
                    "type": "function",
                    "function": {"name": "bash", "arguments": json.dumps({"command": f"echo {turn}"})},
                }
            ],
        }
        prompt_text = tito.apply_chat_template(
            history, add_generation_prompt=True, tokenize=False, template_args=template_args
        )
        full_text = tito.apply_chat_template(
            history + [assistant], add_generation_prompt=False, tokenize=False, template_args=template_args
        )
        assert full_text.startswith(prompt_text)
        completion = tokenizer.encode(full_text[len(prompt_text) :], add_special_tokens=False)
        n = len(completion)
        logps = [-((i % 97) + 1) / 128.0 for i in range(n)]
        meta = {
            "completion_tokens": n,
            "prompt_tokens": 0,
            "cached_tokens": 0,
            "output_token_logprobs": [[lp, t] for lp, t in zip(logps, completion, strict=True)],
            "finish_reason": {"type": "stop"},
        }
        response = {
            "id": f"sim-{turn}",
            "object": "chat.completion",
            "created": 0,
            "model": "sim",
            "choices": [{"index": 0, "message": assistant, "finish_reason": "tool_calls", "meta_info": meta}],
            "usage": {"prompt_tokens": 0, "completion_tokens": n, "total_tokens": n},
        }
        out.append(
            dict(
                messages=json.loads(json.dumps(history)),
                response_body=json.dumps(response, separators=(",", ":")).encode(),
                completion_tokens=n,
            )
        )
        history = history + [assistant]
    return out


if __name__ == "__main__":
    hf, tito_model, path = sys.argv[1:4]
    # Median generated tokens per turn. The default 220 gives ~0.8 MB collection payloads per
    # episode; the FP8 production run collected 5 MB at p50 and 11 MB at p90 (10-19K response
    # tokens per episode), which needs ~1400 here.
    output_median = float(sys.argv[4]) if len(sys.argv) > 4 else 220
    rng = np.random.default_rng(1)
    output_tokens = [int(x) for x in np.clip(rng.lognormal(np.log(output_median), 0.8, 64), 16, 8000)]
    obs_tokens = [int(x) for x in np.clip(rng.lognormal(np.log(110), 0.9, 64), 8, 4000)]
    specs = build(
        hf, tito_model, turns=64, prompt_tokens=1200, output_tokens=output_tokens, obs_tokens=obs_tokens
    )
    pickle.dump(specs, open(path, "wb"))
    print(
        "turns",
        len(specs),
        "mean completion",
        sum(s["completion_tokens"] for s in specs) / len(specs),
        "mean body KB",
        sum(len(s["response_body"]) for s in specs) / len(specs) / 1024,
    )
