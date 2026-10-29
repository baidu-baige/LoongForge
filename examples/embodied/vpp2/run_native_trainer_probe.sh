#!/usr/bin/env bash
# Copyright 2026 The LoongForge Authors.
# SPDX-License-Identifier: Apache-2.0
# TINY model + real LIBERO pipeline; NOT full14B/8x32 acceptance.
set -euo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
SIDE=loongforge
: "${VPP2_ASSETS_ROOT:?Set shared real asset root}"
: "${VPP2_BASE_ROOT:?Set official checkout root}"
: "${VPP2_LOONG_ROOT:?Set LoongForge checkout root (diagnostic helper imports)}"
: "${OUTPUT_DIR:?Use a NEW output directory}"
PYTHON_BIN="${PYTHON_BIN:-python}"
export PYTHONPATH="$VPP2_LOONG_ROOT:$VPP2_BASE_ROOT/src:$VPP2_BASE_ROOT/scripts/repro${PYTHONPATH:+:$PYTHONPATH}"
export CUBLAS_WORKSPACE_CONFIG=:4096:8 PYTHONHASHSEED=42 FLASH_ATTENTION_DETERMINISTIC=1
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-2}"
export DEEPSPEED_CONFIG="$VPP2_LOONG_ROOT/examples/embodied/vpp2/zero2_bf16.json"
[[ ! -e "$OUTPUT_DIR" ]] || { echo "Refusing existing output: $OUTPUT_DIR" >&2; exit 2; }
[[ $# == 0 ]] || { echo 'Use environment overrides' >&2; exit 2; }
STEPS="${TRAIN_ITERS:-100}"
WORLD="${GPUS_PER_NODE:-2}"
BATCH="${PER_DEVICE_BATCH_SIZE:-2}"
RUN_ARGS=(--side "$SIDE" --output "$OUTPUT_DIR" --steps "$STEPS" --batch "$BATCH")
if [[ "$SIDE" == base ]]; then
  [[ "${STRATEGY:-deepspeed}" == deepspeed ]] || { echo 'BASE probe supports ZeRO2 only' >&2; exit 2; }
  TMP_CONFIG="$(mktemp /tmp/vpp2-probe-accelerate-XXXXXX.yaml)"
  trap 'rm -f "$TMP_CONFIG"' EXIT
  cat > "$TMP_CONFIG" <<EOF
compute_environment: LOCAL_MACHINE
distributed_type: DEEPSPEED
deepspeed_config:
  deepspeed_config_file: $DEEPSPEED_CONFIG
  zero3_init_flag: false
mixed_precision: null
num_machines: 1
num_processes: $WORLD
machine_rank: 0
rdzv_backend: static
same_network: true
use_cpu: false
EOF
  "$PYTHON_BIN" -m accelerate.commands.launch --config_file "$TMP_CONFIG" \
    --main_process_port "${MASTER_PORT:-29661}" "$HERE/native_trainer_probe.py" "${RUN_ARGS[@]}"
else
  RUN_ARGS+=(--strategy "${STRATEGY:-deepspeed}" --save-interval "${SAVE_INTERVAL:-0}")
  if [[ -n "${RESUME_FROM:-}" ]]; then RUN_ARGS+=(--resume-from "$RESUME_FROM"); fi
  "$PYTHON_BIN" -m torch.distributed.run --standalone --nproc_per_node="$WORLD" \
    "$HERE/native_trainer_probe.py" "${RUN_ARGS[@]}"
fi
