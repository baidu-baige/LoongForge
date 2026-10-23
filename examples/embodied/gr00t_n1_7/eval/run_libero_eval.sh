#!/usr/bin/env bash
# Copyright 2026 The LoongForge Authors.
# SPDX-License-Identifier: Apache-2.0
#
# Public: GR00T-N1.7 LIBERO task-success, serial evaluation of one or more suites.
#
# - Each suite YAML (configs/libero/libero_<suite>.yaml) holds only the fixed eval
#   protocol (benchmark.libero_protocol: gr00t_official, 720 steps, chunk 8, ...);
#   the machine/artifact paths below are injected at runtime, so no YAML needs editing.
# - One policy server per suite on a single GPU, one LIBERO env; suites, tasks and
#   episodes all run serially. The orchestrator starts and stops the server itself.
# - gr00t_official writes every invocation to a fresh
#   <run>/<suite>/gr00t_official/<run-id>/ (resume/restart are ignored).
#
# Usage:
#   bash run_libero_eval.sh                                         # 4 suites x 10 tasks x 20 episodes
#   SUITES=libero_10 EPISODES=2 MAX_TASKS=1 bash run_libero_eval.sh # smoke test
#   CKPT_PATH=/path/to/ckpt STATS_PATH=/path/to/stats.json SUITES=libero_10 bash run_libero_eval.sh
set -euo pipefail

# ---- Paths injected into the suite YAMLs at runtime ----
COSMOS_PATH=${COSMOS_PATH:-"/path/to/gr00t/Cosmos-Reason2-2B"}
# CKPT_PATH / STATS_PATH may contain {suite} (e.g. .../libero_{suite}/checkpoints/steps_20000);
# it is replaced by the suite name without the libero_ prefix (10, goal, object, spatial).
CKPT_PATH=${CKPT_PATH:-"/path/to/GR00T-N1.7-LIBERO/libero_{suite}"}
STATS_PATH=${STATS_PATH:-"/path/to/GR00T-N1.7-LIBERO/libero_{suite}/statistics.json"}
LOONGFORGE_PATH=${LOONGFORGE_PATH:-"/path/to/LoongForge"}
OUT_ROOT=${OUT_ROOT:-"/path/to/reports"}
SERVER_PY=${SERVER_PY:-"/usr/bin/python"}

# ---- Other run parameters ----
CONFIG_DIR=${CONFIG_DIR:-"$LOONGFORGE_PATH/examples/embodied/gr00t_n1_7/eval/configs/libero"}
BENCHMARK_PYTHON=${BENCHMARK_PYTHON:-"/path/to/bin/python"}
RUN_TAG=${RUN_TAG:-"libero_serial_$(date +%Y%m%d_%H%M%S)"}
GPU=${GPU:-0}
SUITES=${SUITES:-"libero_spatial libero_object libero_goal libero_10"}
EPISODES=${EPISODES:-20}       # per task; 10 tasks x 20 = the official 200 denominator
MAX_TASKS=${MAX_TASKS:-0}      # 0 = all 10 tasks of the suite
SEED=${SEED:-42}
MUJOCO_GL=${MUJOCO_GL:-egl}    # osmesa in the YAMLs; egl where osmesa is not installed

export PYTHONPATH="$LOONGFORGE_PATH:$LOONGFORGE_PATH/ops/cuda_source/gr00t_n1_7_op:${PYTHONPATH:-}"
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 TOKENIZERS_PARALLELISM=false
export CUDA_VISIBLE_DEVICES="$GPU"
# The policy server is a child process and inherits these. COSMOS_LOCAL_PATH loads
# the Qwen3-VL processor from disk instead of the gated HF repo.
export COSMOS_LOCAL_PATH="$COSMOS_PATH"
# LIBERO's benchmark/__init__.py torch.load()s its own vendored init_states
# pickles without weights_only=False, which torch >= 2.6 rejects.
export TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD=1
# robosuite asserts MUJOCO_EGL_DEVICE_ID is listed in CUDA_VISIBLE_DEVICES.
if [ "$MUJOCO_GL" = "egl" ]; then export MUJOCO_EGL_DEVICE_ID=${MUJOCO_EGL_DEVICE_ID:-$GPU}; fi

read -ra SUITE_ARR <<< "$SUITES"
RUN_DIR="$OUT_ROOT/$RUN_TAG"
mkdir -p "$RUN_DIR/configs"

py() { "$BENCHMARK_PYTHON" -W ignore "$@"; }

render_cfg() {   # render_cfg <template.yaml> <out.yaml> key.sub=value ...
    py -c 'import sys, yaml
src, dst, *items = sys.argv[1:]
cfg = yaml.safe_load(open(src))
for item in items:
    key, value = item.split("=", 1)
    *parents, leaf = key.split(".")
    node = cfg
    for name in parents: node = node.setdefault(name, {})
    node[leaf] = yaml.safe_load(value)
yaml.safe_dump(cfg, open(dst, "w"), sort_keys=False)' "$@"
}

# LIBERO reads ${LIBERO_CONFIG_PATH:-~/.libero}/config.yaml on import and, when it
# is missing, asks on stdin whether to use a custom dataset path -- EOFError in a
# non-interactive run. Answer "n" once so LIBERO writes its default config, with
# paths resolved from the LIBERO package installed in BENCHMARK_PYTHON.
libero_cfg="${LIBERO_CONFIG_PATH:-$HOME/.libero}/config.yaml"
if [ ! -f "$libero_cfg" ]; then
    echo "Creating LIBERO default config: $libero_cfg"
    echo n | "$BENCHMARK_PYTHON" -W ignore -c "import libero.libero" > /dev/null
    [ -f "$libero_cfg" ] || { echo "ERROR: failed to create $libero_cfg" >&2; exit 1; }
fi

for suite in "${SUITE_ARR[@]}"; do
    echo "===== $suite ====="
    short=${suite#libero_}
    ckpt=${CKPT_PATH//\{suite\}/$short}
    stats=${STATS_PATH//\{suite\}/$short}
    cfg="$RUN_DIR/configs/$suite.yaml"      # fixed YAML + runtime-injected paths
    sdir="$RUN_DIR/$suite"
    mkdir -p "$sdir"
    [ -e "$ckpt" ] || { echo "ERROR: checkpoint not found: $ckpt" >&2; exit 1; }
    [ -f "$stats" ] || { echo "ERROR: statistics not found: $stats" >&2; exit 1; }
    render_cfg "$CONFIG_DIR/$suite.yaml" "$cfg" \
        "benchmark.max_tasks=$MAX_TASKS" "benchmark.episodes_per_task=$EPISODES" \
        "model.base_model_path=$COSMOS_PATH" "model.model_name=$COSMOS_PATH" \
        "server.python=$SERVER_PY" "server.log=$sdir/policy_server.log" \
        "server.ckpt_path=$ckpt" "server.dataset_statistics_path=$stats" \
        "server.loongforge_root=$LOONGFORGE_PATH" \
        "env.eval_root=$LOONGFORGE_PATH/loongforge/embodied/eval" \
        "env.mujoco_gl=$MUJOCO_GL" "env.pyopengl_platform=$MUJOCO_GL" \
        "run.output_dir=$sdir" "run.seed=$SEED"
    "$BENCHMARK_PYTHON" -m loongforge.embodied.eval.orchestrator.run --config "$cfg" \
        > "$sdir/eval.log" 2>&1 \
        || { echo "ERROR: $suite failed, see $sdir/eval.log" >&2; exit 1; }
    echo "  done: $(find "$sdir/gr00t_official" -name suite_summary.csv | head -n1)"
done

echo "===== Summary ====="
py - "$RUN_DIR" "$EPISODES" "${SUITE_ARR[@]}" <<'PY' | tee "$RUN_DIR/summary.txt"
import json, pathlib, sys
run_dir, n, suites = pathlib.Path(sys.argv[1]), int(sys.argv[2]), sys.argv[3:]
w = max(len(s) for s in suites)
rates = []
for s in suites:
    runs = sorted((run_dir / s / "gr00t_official").glob("*/tasks.jsonl"))
    if len(runs) != 1:
        raise SystemExit(f"INVALID: {s} has {len(runs)} gr00t_official runs (expected 1)")
    tasks = [json.loads(l) for l in runs[0].read_text().splitlines() if l.strip()]
    bad = [t["task_id"] for t in tasks if t["status"] != "OK" or t["n_episodes"] != n]
    if bad:
        raise SystemExit(f"INVALID: {s} tasks {bad} failed or lack {n} episodes")
    rows = [json.loads(l) for l in (runs[0].parent / "results.jsonl").read_text().splitlines() if l.strip()]
    ok = sum(int(r["success"]) for r in rows)
    rates.append(ok / len(rows))
    print(f"{s:<{w}}  {ok}/{len(rows)}  {ok / len(rows):.2%}")
print(f"{'MACRO_AVERAGE':<{w}}  {sum(rates) / len(rates):.2%}")
PY
echo "===== Done: $RUN_DIR ====="
