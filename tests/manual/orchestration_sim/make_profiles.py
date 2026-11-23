"""Turn TMax trace JSONL files into episode-group profiles for run_sim.py.

  python make_profiles.py --out episodes.json traces-host-a.jsonl [traces-host-b.jsonl ...]
  python make_profiles.py --out collapsed.json --collapsed   # one-turn, no-tool episodes

A group is the episodes of one task_id in one trace file (siblings of one prompt); only
complete groups of --siblings are kept, so per-group straggler structure is preserved.
"""

import argparse
import collections
import json
import random


def from_traces(paths, siblings):
    groups = []
    for path in paths:
        by_task = collections.defaultdict(list)
        with open(path) as f:
            for line in f:
                try:
                    row = json.loads(line)
                except json.JSONDecodeError:
                    continue
                m = row["metrics"]
                by_task[row["task_id"]].append(
                    dict(
                        turns=m["turns"],
                        start=m["sandbox_start_time"],
                        env=m["env_execution_time_sum"],
                        tools=m.get("tool_calls", 0),
                    )
                )
        groups += [g for g in by_task.values() if len(g) == siblings]
    return groups


def collapsed(n_groups, siblings, seed=7):
    rng = random.Random(seed)
    return [
        [dict(turns=1, start=rng.lognormvariate(2.08, 0.4), env=0.0, tools=0) for _ in range(siblings)]
        for _ in range(n_groups)
    ]


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", required=True)
    parser.add_argument("--siblings", type=int, default=32)
    parser.add_argument("--collapsed", action="store_true")
    parser.add_argument("traces", nargs="*")
    a = parser.parse_args()
    groups = collapsed(400, a.siblings) if a.collapsed else from_traces(a.traces, a.siblings)
    json.dump(groups, open(a.out, "w"))
    print(f"{len(groups)} groups -> {a.out}")
