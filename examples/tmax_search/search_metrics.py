"""Per-rollout W&B metrics of the search port: the episode, tool and judge metrics Prime-RL's search runs log.

Two hooks, both in the rollout process, as in ``tmax_metrics``:

* ``nonzero_std_filter`` / ``missing_reward_filter`` (``--dynamic-sampling-filter-path``) record every finished group
  before Miles' filter decides;
* ``log_rollout_data`` (``--custom-rollout-log-function-path``) logs, once per trained batch, what was recorded since
  the previous batch, plus the trained samples' lengths and advantages (``tmax_metrics.trained_metrics``).

Keys, all under ``rollout/`` (plotted against ``rollout/step``):

* ``rollout/agent/*``: stop conditions (Prime-RL's ``stop_condition/*``), turns, tool calls by tool, tokens, time;
* ``rollout/tools/{search,webterminal}/*``: calls per episode and failures per call by category (Prime-RL's
  ``*_retrieval_calls`` / ``*_retrieval_failures`` / ``*_retrieval_<category>_failures``);
* ``rollout/judge/*``: why episodes were not judged (``skipped/<reason>``, e.g. ``format:think_close_count:0``), the
  invalid-judgment rate, and the reward components (``judge_answer``, ``tool_stderr_penalty_reward``,
  ``missing_final_answer_penalty_reward``);
* ``rollout/unfiltered/*``: each group's correct-answer rate before the filter, and ``frac_zero_variance``, the groups
  Prime-RL masks (all rewards equal).
"""

import threading
from collections import Counter
from statistics import mean

from search_agent.tool_metrics import TOOL_FAILURE_CATEGORIES
from tmax_metrics import trained_metrics, unfiltered_metrics

from miles.rollout.filter_hub.base_types import FilterOutput, iter_samples
from miles.rollout.filter_hub.common_filters import apply_missing_reward_filter, apply_reward_nonzero_std_filter
from miles.utils.metric_utils import compute_rollout_step
from miles.utils.tracking_utils import tracking

_lock = threading.Lock()
_groups: list[dict] = []  # per finished group: correct rate, zero variance, and its episodes' records

_MEAN_KEYS = (
    "turns",
    "tool_calls",
    "search_calls",
    "webterminal_calls",
    "stderr_count",
    "completion_tokens",
    "total_tokens",
    "terminal_turn_tokens",
    "model_query_time_sum",
    "env_execution_time_sum",
    "total_time",
)
_REWARD_COMPONENTS = ("judge_answer", "tool_stderr_penalty_reward", "missing_final_answer_penalty_reward")


def nonzero_std_filter(args, samples, **kwargs) -> FilterOutput:
    """Miles' zero-variance filter (drop and refill), after recording the group."""
    _record_group(args, samples)
    return apply_reward_nonzero_std_filter(args, samples, **kwargs)


def missing_reward_filter(args, samples, **kwargs) -> FilterOutput:
    """Miles' missing-reward filter alone, after recording the group: with the zero-variance filter off (Prime-RL's
    masking, see search_loss.py) a group whose judge failed is dropped, every other group trains."""
    _record_group(args, samples)
    return apply_missing_reward_filter(args, samples, **kwargs)


def log_rollout_data(rollout_id, args, samples, rollout_extra_metrics, rollout_time) -> bool:
    with _lock:
        groups = list(_groups)
        _groups.clear()
    episodes = [e for g in groups for e in g["episodes"]]
    metrics = _episode_metrics(episodes) | _group_metrics(groups) | trained_metrics(args, rollout_id, samples)
    if metrics:
        metrics["rollout/step"] = compute_rollout_step(args, rollout_id)
        tracking.log(args, metrics, step_key="rollout/step")
    return False


def _record_group(args, samples) -> None:
    episodes = []
    for s in iter_samples(samples):
        metadata = s.metadata or {}
        if "agent_metrics" in metadata:
            episodes.append(
                {
                    "agent": metadata["agent_metrics"],
                    "judge": metadata.get("judge") or {},
                    "components": metadata.get("reward_components"),
                }
            )
    rewards = [s.get_reward_value(args) for s in iter_samples(samples) if s.reward is not None]
    correct = [e["components"]["judge_answer"] for e in episodes if e["components"]]
    group = {
        "episodes": episodes,
        "correct_rate": mean(correct) if correct else None,
        "zero_variance": bool(rewards) and max(rewards) - min(rewards) <= 1e-6,
    }
    with _lock:
        _groups.append(group)


def _episode_metrics(episodes: list[dict]) -> dict[str, float]:
    if not episodes:
        return {}
    n = len(episodes)
    agent = [e["agent"] for e in episodes]
    metrics = {"rollout/agent/episodes": n}
    for key in _MEAN_KEYS:
        values = [a[key] for a in agent if key in a]
        if values:
            metrics[f"rollout/agent/{key}_mean"] = mean(values)
    for condition, count in Counter(a["stop_condition"] for a in agent).items():
        metrics[f"rollout/agent/stop_condition/{condition}"] = count / n

    for tool in ("search", "webterminal"):
        calls = sum(a.get(f"{tool}_retrieval_calls", 0) for a in agent)
        metrics[f"rollout/tools/{tool}/calls_per_episode"] = calls / n
        if calls:
            for key in ("failures", *(f"{category}_failures" for category in TOOL_FAILURE_CATEGORIES)):
                metrics[f"rollout/tools/{tool}/{key}_per_call"] = (
                    sum(a.get(f"{tool}_retrieval_{key}", 0) for a in agent) / calls
                )

    judges = [e["judge"] for e in episodes if e["judge"]]
    if judges:
        metrics["rollout/judge/invalid_rate"] = mean(not j["valid"] for j in judges)
        metrics["rollout/judge/invoked_rate"] = mean(bool(j.get("invoked")) for j in judges)
        for reason, count in Counter(j.get("skipped") or "judged" for j in judges).items():
            metrics[f"rollout/judge/skipped/{reason}"] = count / len(judges)
    components = [e["components"] for e in episodes if e["components"]]
    for key in _REWARD_COMPONENTS:
        if components:
            metrics[f"rollout/judge/{key}_mean"] = mean(c[key] for c in components)
    return metrics


def _group_metrics(groups: list[dict]) -> dict[str, float]:
    if not groups:
        return {}
    metrics = unfiltered_metrics([g["correct_rate"] for g in groups if g["correct_rate"] is not None])
    metrics["rollout/unfiltered/frac_zero_variance"] = mean(g["zero_variance"] for g in groups)
    return metrics
