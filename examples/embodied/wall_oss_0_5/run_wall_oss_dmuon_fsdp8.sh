#!/usr/bin/env bash
# Copyright 2026 The LoongForge Authors.
# SPDX-License-Identifier: Apache-2.0
#
# Wall-OSS-0.5 LoongForge Embodied 8-GPU FSDP + DMuon training launcher.
#
# Prerequisite: generate the LeRobot norm stats (libero_norm_stats.json, holding
# per-dim mean/std/q01/q99) from the source dataset, then point NORM_STATS_PATH
# at it. Run once before training:
#   DATASET_PATH=/workspace/datasets/libero \
#     OUTPUT_PATH=/workspace/datasets/libero_norm_stats.json \
#     bash examples/embodied/wall_oss_0_5/compute_norm_stats.sh
#   export NORM_STATS_PATH=/workspace/datasets/libero_norm_stats.json
#Usage:
#   bash examples/embodied/wall_oss_0_5/run_wall_oss_dmuon_fsdp8.sh
#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

export LOONGFORGE_PATH=${LOONGFORGE_PATH:-"$(cd "$SCRIPT_DIR/../../.." && pwd)"}
export LOCAL_VLA_ARTIFACTS_ROOT=${LOCAL_VLA_ARTIFACTS_ROOT:-"/ssd2/loongforge_embodied_ci/vla_artifacts"}

WALL_OSS_OPS_SRC=${WALL_OSS_OPS_SRC:-"${LOONGFORGE_PATH}/ops/cuda_source/wall_oss_05_ops"}

build_wall_oss_ops() {
    if python -c "import wall_oss_05_ops" > /dev/null 2>&1; then
        echo "Wall-OSS CUDA operators already installed: wall_oss_05_ops"
        return
    fi

    if [[ ! -f "${WALL_OSS_OPS_SRC}/setup.py" ]]; then
        echo "wall_oss_05_ops is not installed and its sources were not found at ${WALL_OSS_OPS_SRC}." >&2
        echo "Set WALL_OSS_OPS_SRC to the wall_oss_05_ops directory and rerun." >&2
        exit 1
    fi

    echo "Installing Wall-OSS CUDA operators from ${WALL_OSS_OPS_SRC}"
    NVCC_THREADS=${NVCC_THREADS:-${MAX_JOBS:-8}} \
        pip install --no-build-isolation -e "${WALL_OSS_OPS_SRC}"
}

build_wall_oss_ops

export NO_ALBUMENTATIONS_UPDATE=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export NCCL_PROTO='broadcast:simple;allgather:simple'
export NCCL_MIN_NCHANNELS=${NCCL_MIN_NCHANNELS:-10}
export NCCL_MAX_NCHANNELS=${NCCL_MAX_NCHANNELS:-10}
export NCCL_BUFFSIZE=${NCCL_BUFFSIZE:-2097152}
export LOONGFORGE_WALL_SKIP_UNUSED_LM_HEAD=1

REPO=${REPO:-"$LOONGFORGE_PATH"}
RUN_ROOT=${OUTPUT_ROOT:-"$LOONGFORGE_PATH/outputs"}
RUN_NAME=${RUN_NAME:-wall_oss_0_5}
RUN_DIR="$RUN_ROOT/$RUN_NAME"

mkdir -p "$RUN_DIR"

GPUS_PER_NODE=8
CUDA_ID=${CUDA_VISIBLE_DEVICES:-0,1,2,3,4,5,6,7}
MASTER_ADDR=localhost
MASTER_PORT="${MASTER_PORT:-29731}"
NNODES=1
NODE_RANK=0

export CUDA_VISIBLE_DEVICES="$CUDA_ID"

DISTRIBUTED_ARGS=(
    --nproc_per_node "$GPUS_PER_NODE"
    --nnodes "$NNODES"
    --node_rank "$NODE_RANK"
    --master_addr "$MASTER_ADDR"
    --master_port "$MASTER_PORT"
)

MODEL_CONFIG_ARGS=(
    --model-name wall_oss_0_5
)

DATA_ARGS=(
    --dataset-format lerobot_datasets
    --dataset-strategy wall_oss_0_5
    --dataset-path "${DATA_PATH:-"$LOCAL_VLA_ARTIFACTS_ROOT/wall_oss_0_5/datasets/libero"}"
    --tokenizer-path "${TOKENIZER_PATH:-"$LOCAL_VLA_ARTIFACTS_ROOT/wall_oss_0_5/models/Qwen2.5-VL-3B-Instruct"}"
    --num-workers 16
)

TRAINING_ARGS=(
    --trainer-type FinetuneTrainer
    --train-iters "${TRAIN_ITERS:-40}"
    --per-device-batch-size "${PER_DEVICE_BATCH:-32}"
    --gradient-accumulation-steps 1
    --seed 10222
    --output-dir "$RUN_DIR/output"
    --pretrained-checkpoint "${PRETRAINED_CHECKPOINT:-"$LOCAL_VLA_ARTIFACTS_ROOT/wall_oss_0_5/models/wall-oss-0.5"}"
    --lr-base 5.0e-05
    --lr-warmup-iters 20
    --min-lr 1.0e-06
    --custom-lr-lambda
    --optimizer dmuon
    --dmuon-muon-lr 0.02
    --dmuon-momentum 0.95
    --dmuon-ns-steps 5
    --dmuon-muon-weight-decay 0.0
    --dmuon-adamw-lr 1.0e-03
    --adam-beta1 0.9
    --adam-beta2 0.999
    --adam-eps 1.0e-08
    --weight-decay 1.0e-02
    --dmuon-ns-backend gram
    --dmuon-ns-coefficients default
    --dmuon-nesterov
    --dmuon-forward-prefetch-depth 12
    --dmuon-adamw-foreach
    --clip-grad 1.0
    --save-format safetensors
    --no-save-training-state
    --save-interval 0
)

FSDP_ARGS=(
    --distributed-strategy fsdp
    --dtype bfloat16
    --fsdp-no-wrap-modules Conv3d,Linear,Embedding
    --fsdp-reshard-default false
    --fsdp-original-param-dtype fp32
    --fsdp-unshard-param-dtype bf16
    --fsdp-reduce-dtype bf16
    --no-fsdp-cast-forward-inputs
    --fsdp-root-optimizer-prefetch
)

LOGGING_ARGS=(
    --log-interval 1
    --detail-log-interval "${DETAIL_LOG_INTERVAL:-0}"
    --wandb-mode disabled
)

OVERRIDE_ARGS=(
    data.norm_stats_path="${NORM_STATS_PATH:-"$LOCAL_VLA_ARTIFACTS_ROOT/wall_oss_0_5/datasets/libero_norm_stats.json"}"
)

cd "$REPO"

PYTHONPATH="$REPO${PYTHONPATH:+:$PYTHONPATH}" \
    torchrun "${DISTRIBUTED_ARGS[@]}" \
    "$REPO/loongforge/embodied/train.py" \
    "${MODEL_CONFIG_ARGS[@]}" \
    "${DATA_ARGS[@]}" \
    "${TRAINING_ARGS[@]}" \
    "${FSDP_ARGS[@]}" \
    "${LOGGING_ARGS[@]}" \
    "${OVERRIDE_ARGS[@]}" 2>&1 | tee "$RUN_DIR/train.log"
