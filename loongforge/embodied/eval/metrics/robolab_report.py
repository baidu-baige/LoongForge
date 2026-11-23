# Copyright 2026 The LoongForge Authors.
# SPDX-License-Identifier: Apache-2.0

"""Summarize LoongForge RoboLab results and compare with the public leaderboard.

Usage: python3 -m loongforge.embodied.eval.metrics.robolab_report <output_dir> [cosmos3|pi05]

Difficulty / capability tags come from each episode record (written by the RoboLab runner).
Older result files lack them; then ``ROBOLAB_ROOT`` (a RoboLab checkout) is read instead.
"""

import glob
import json
import os
import sys

ATTR_TO_CATEGORY = {
    "size": "visual",
    "color": "visual",
    "semantics": "visual",
    "spatial": "relational",
    "conjunction": "relational",
    "counting": "relational",
    "stacking": "procedural",
    "sorting": "procedural",
    "reorientation": "procedural",
    "affordance": "procedural",
}

# Attributes the leaderboard counts as "procedural": stacking + reorientation + affordance
# (22 tasks). This repo's metadata also tags 12 "sorting" tasks as procedural, so the local
# group has 34 tasks. Both rows are printed; only ``procedural*`` matches the leaderboard.
LB_PROCEDURAL_ATTRS = frozenset({"stacking", "reorientation", "affordance"})

# Public leaderboard (10 eps/task), research.nvidia.com/labs/srl/projects/robolab/leaderboard.html.
LEADERBOARDS = {
    "cosmos3": {  # Cosmos3-Nano-Policy
        "all": (441, 1200),
        "visual": (219, 840),
        "relational": (207, 420),
        "simple": (260, 640),
        "moderate": (138, 390),
        "complex": (43, 170),
        "procedural_lb": (60, 220),
    },
    "pi05": {  # pi05_droid_jointpos (π0.5), read 2026-11-20
        "all": (336, 1200),
        "visual": (200, 840),
        "relational": (142, 420),
        "simple": (190, 640),
        "moderate": (123, 390),
        "complex": (23, 170),
        "procedural_lb": (48, 220),
    },
}
LEADERBOARD = LEADERBOARDS["cosmos3"]


def _meta_path() -> str | None:
    root = os.environ.get("ROBOLAB_ROOT", "")
    if not root:
        return None
    return os.path.join(root, "robolab", "tasks", "_metadata", "task_metadata.json")


def load(out_dir):
    """Return ``(meta, records)``; ``meta`` may be empty when records carry their own tags."""
    meta = {}
    path = _meta_path()
    if path and os.path.exists(path):
        meta = {m["task_name"]: m for m in json.load(open(path))}
    recs = []
    for path in glob.glob(f"{out_dir}/worker_*/results.jsonl"):
        for line in open(path):
            line = line.strip()
            if line.startswith("{"):
                recs.append(json.loads(line))
    if not meta and any("difficulty_label" not in r for r in recs):
        raise SystemExit(
            "results predate per-episode task tags; set ROBOLAB_ROOT to the RoboLab checkout"
        )
    return meta, recs


def task_meta(meta, record):
    """Task metadata for one record: its own tags first, then the RoboLab metadata file."""
    own = {k: record[k] for k in ("difficulty_label", "task_attributes") if k in record}
    if own:
        return {"difficulty_label": own.get("difficulty_label"), "attributes": own.get("task_attributes")}
    return meta.get(record.get("task_suite"), {})


def record_attributes(record_meta):
    """Raw attribute tags of one task (``"stacking, semantics"`` -> ``{"stacking", "semantics"}``)."""
    return {a.strip() for a in str(record_meta.get("attributes", "")).split(",") if a.strip()}


def categories(record_meta):
    return {ATTR_TO_CATEGORY[a] for a in record_attributes(record_meta) if a in ATTR_TO_CATEGORY}


def line(name, k, n, label=None):
    if n == 0:  # a smoke run may not cover every group
        return
    lb = LEADERBOARD.get(name)
    lb_txt = f" | leaderboard {lb[0]}/{lb[1]} = {lb[0] / lb[1]:.1%}" if lb else ""
    print(f"{label or name:<12} {k:>4}/{n:<4} {k / n:>6.1%}{lb_txt}")


def main():
    global LEADERBOARD
    if len(sys.argv) < 2:
        raise SystemExit(__doc__)
    out_dir = sys.argv[1]
    LEADERBOARD = LEADERBOARDS[sys.argv[2] if len(sys.argv) > 2 else "cosmos3"]
    meta, recs = load(out_dir)
    dedup = {r["episode_id"]: r for r in recs}
    print(f"records={len(recs)} unique_episodes={len(dedup)}")
    recs = list(dedup.values())
    tasks = sorted({r["task_suite"] for r in recs})
    total_tasks = len(meta) or len({r["task_suite"] for r in recs})
    print(f"tasks_with_results={len(tasks)} / {total_tasks}")

    k = sum(int(r["success"]) for r in recs)
    print()
    line("all", k, len(recs))
    for label in ("simple", "moderate", "complex"):
        sub = [r for r in recs if task_meta(meta, r).get("difficulty_label") == label]
        line(label, sum(int(r["success"]) for r in sub), len(sub))
    for cat in ("visual", "relational", "procedural"):
        sub = [r for r in recs if cat in categories(task_meta(meta, r))]
        line(cat, sum(int(r["success"]) for r in sub), len(sub))
    # Leaderboard-comparable procedural: dropping the "sorting" tasks leaves its 22 tasks.
    lb_proc = [
        r for r in recs
        if record_attributes(task_meta(meta, r)) & LB_PROCEDURAL_ATTRS
        and "sorting" not in record_attributes(task_meta(meta, r))
    ]
    line("procedural_lb", sum(int(r["success"]) for r in lb_proc), len(lb_proc), label="procedural*")
    if lb_proc:
        print("  (* leaderboard definition: stacking/reorientation/affordance, sorting excluded)")

    failures = [r for r in recs if not int(r["success"])]
    reasons = {}
    for r in failures:
        reasons[r.get("failure_reason")] = reasons.get(r.get("failure_reason"), 0) + 1
    print("\nfailure reasons:", reasons)

    print("\nper-task success rate (worst 10):")
    per_task = {}
    for r in recs:
        per_task.setdefault(r["task_suite"], []).append(int(r["success"]))
    ranked = sorted(per_task.items(), key=lambda kv: sum(kv[1]) / len(kv[1]))
    for name, values in ranked[:10]:
        print(f"  {name:<45} {sum(values)}/{len(values)}")
    print("\nper-task success rate (best 5):")
    for name, values in ranked[::-1][:5]:
        print(f"  {name:<45} {sum(values)}/{len(values)}")


if __name__ == "__main__":
    main()
