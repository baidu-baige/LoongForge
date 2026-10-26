#!/usr/bin/env bash
# Copyright 2026 The LoongForge Authors.
# SPDX-License-Identifier: Apache-2.0
#
# Public: GR00T-N1.7 SimplerEnv WidowX (Bridge) task-success, serial evaluation of
# the 7 official Bridge tasks x EPISODES episodes.
#
# - Each task YAML ($CONFIG_DIR/<task>.yaml) holds only the fixed eval protocol plus
#   the eval protocol; the machine/artifact paths below are injected at runtime,
#   so no YAML needs editing.
# - One policy server per task (port taken from that task's YAML) shares a single
#   GPU with one render env; tasks and episodes run serially.
# - The reset seed of episode i is SEED + i.
#
# Usage:
#   bash run_simplerenv_bridge_eval.sh                                           # 7 tasks x 100 episodes
#   EPISODES=2 TASKS=widowx_carrot_on_plate bash run_simplerenv_bridge_eval.sh   # smoke test
#   CKPT_PATH=/path/to/checkpoint bash run_simplerenv_bridge_eval.sh             # another checkpoint
set -euo pipefail

# ---- Paths injected into the task YAMLs at runtime ----
COSMOS_PATH=${COSMOS_PATH:-"/path/to/gr00t/Cosmos-Reason2-2B"}
CKPT_PATH=${CKPT_PATH:-"/path/to/bridge/checkpoints"}
STATS_PATH=${STATS_PATH:-"/path/to/datasets/stats.json"}
LOONGFORGE_PATH=${LOONGFORGE_PATH:-"/path/to/LoongForge"}
SIMPLERENV_ROOT=${SIMPLERENV_ROOT:-"/path/to/SimplerEnv"}
OUT_ROOT=${OUT_ROOT:-"/path/to/reports"}
NVIDIA_LIB_DIR=${NVIDIA_LIB_DIR:-"/usr/lib64"}
NVIDIA_ICD_JSON=${NVIDIA_ICD_JSON:-"/etc/vulkan/icd.d/nvidia_icd.json"}
SERVER_PY=${SERVER_PY:-"/usr/bin/python"}

# ---- Other run parameters ----
CONFIG_DIR=${CONFIG_DIR:-"$LOONGFORGE_PATH/examples/embodied/gr00t_n1_7/eval/configs/simplerenv/bridge"}
BENCHMARK_PYTHON=${BENCHMARK_PYTHON:-"/path/to/bin/python"}
RUN_TAG=${RUN_TAG:-"bridge_official7_serial_$(date +%Y%m%d_%H%M%S)"}
GPU=${GPU:-0}
TASKS=${TASKS:-"widowx_open_drawer widowx_close_drawer widowx_spoon_on_towel widowx_put_eggplant_in_basket"\
" widowx_carrot_on_plate widowx_stack_cube widowx_put_eggplant_in_sink"}
EPISODES=${EPISODES:-100}
SEED=${SEED:-0}

export PYTHONPATH="$LOONGFORGE_PATH:$LOONGFORGE_PATH/ops/cuda_source/gr00t_n1_7_op:${PYTHONPATH:-}"
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 TOKENIZERS_PARALLELISM=false
# Required on the render side:
#   LD_PRELOAD mesa libGLdispatch -> otherwise `import sapien` fails with _glapi_tls_Current
#   GROOT_DISABLE_RT_DENOISER=1   -> otherwise the OptiX denoiser fails and drawer tasks drop to 0
export LD_PRELOAD="/lib/x86_64-linux-gnu/libGLdispatch.so.0${LD_PRELOAD:+:$LD_PRELOAD}"
export GROOT_DISABLE_RT_DENOISER=1

read -ra TASK_ARR <<< "$TASKS"
RUN_DIR="$OUT_ROOT/$RUN_TAG"
mkdir -p "$RUN_DIR/servers" "$RUN_DIR/configs"

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

SERVER_PID=""
stop_server() {
    [ -n "$SERVER_PID" ] || return 0
    kill -- "-$SERVER_PID" 2>/dev/null || kill "$SERVER_PID" 2>/dev/null || true
    sleep 3
    kill -9 -- "-$SERVER_PID" 2>/dev/null || true
    SERVER_PID=""
}
trap stop_server EXIT

start_server() {   # start_server <task.yaml> <log>
    local port
    port="$(py -c 'import sys, yaml; print(yaml.safe_load(open(sys.argv[1]))["server"]["port"])' "$1")"
    CUDA_VISIBLE_DEVICES="$GPU" setsid "$SERVER_PY" \
        -m loongforge.embodied.eval.servers.loongforge_server --config "$1" > "$2" 2>&1 &
    SERVER_PID=$!
    until timeout 2 bash -c "cat </dev/null >/dev/tcp/127.0.0.1/$port" 2>/dev/null; do
        kill -0 "$SERVER_PID" 2>/dev/null || { echo "ERROR: server exited, see $2" >&2; exit 1; }
        sleep 5
    done
    echo "  server ready: port=$port gpu=$GPU pid=$SERVER_PID"
}

for task in "${TASK_ARR[@]}"; do
    echo "===== $task ====="
    cfg="$RUN_DIR/configs/$task.yaml"        # fixed YAML + runtime-injected paths
    slog="$RUN_DIR/servers/$task.log"
    tdir="$RUN_DIR/$task"
    ecfg="$tdir/configs/current.yaml"        # rewritten for every episode
    mkdir -p "$tdir/configs"
    render_cfg "$CONFIG_DIR/$task.yaml" "$cfg" \
        "model.base_model_path=$COSMOS_PATH" "model.model_name=$COSMOS_PATH" \
        "server.python=$SERVER_PY" "server.log=$slog" \
        "server.ckpt_path=$CKPT_PATH" "server.dataset_statistics_path=$STATS_PATH" \
        "server.loongforge_root=$LOONGFORGE_PATH" \
        "env.eval_root=$LOONGFORGE_PATH/loongforge/embodied/eval" "env.simplerenv_root=$SIMPLERENV_ROOT" \
        "env.ld_library_path=$NVIDIA_LIB_DIR" "env.nvidia_lib_dir=$NVIDIA_LIB_DIR" \
        "env.nvidia_icd_json=$NVIDIA_ICD_JSON"
    start_server "$cfg" "$slog"
    for ep in $(seq 0 $((EPISODES - 1))); do
        render_cfg "$cfg" "$ecfg" "run.output_dir=$tdir" "run.seed=$SEED" "run.episode_idx=$ep"
        CUDA_VISIBLE_DEVICES="$GPU" "$BENCHMARK_PYTHON" \
            -m loongforge.embodied.eval.orchestrator.runners.simplerenv_runner \
            --config "$ecfg" >> "$tdir/episodes.log" 2>&1 \
            || { echo "ERROR: $task episode $ep failed, see $tdir/episodes.log" >&2; exit 1; }
        echo "  episode $((ep + 1))/$EPISODES done, successes so far: $(grep -c '"success": 1' "$tdir/results.jsonl")"
    done
    stop_server
done

echo "===== Summary ====="
py - "$RUN_DIR" "$EPISODES" "${TASK_ARR[@]}" <<'PY' | tee "$RUN_DIR/summary.txt"
import json, pathlib, sys
run_dir, n, tasks = pathlib.Path(sys.argv[1]), int(sys.argv[2]), sys.argv[3:]
w = max(len(t) for t in tasks)
rates = []
for t in tasks:
    rows = [json.loads(l) for l in (run_dir / t / "results.jsonl").read_text().splitlines() if l.strip()]
    if len(rows) != n or len({r["episode_id"] for r in rows}) != n:
        raise SystemExit(f"INVALID: {t} has {len(rows)} records (expected {n} unique)")
    s = sum(int(r["success"]) for r in rows)
    rates.append(s / n)
    print(f"{t:<{w}}  {s}/{n}  {s / n:.2%}")
print(f"{'MACRO_AVERAGE':<{w}}  {sum(rates) / len(rates):.2%}")
PY
echo "===== Done: $RUN_DIR ====="
