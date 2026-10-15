#!/usr/bin/env bash
# Copyright 2026 The LoongForge Authors.
# SPDX-License-Identifier: Apache-2.0

# ============================================================================
# run_lingbot_vla_v2_finetune.sh - lingbot_vla_v2 custom_dp training launch script (Embodied)
#
# custom_dp training: compute parameters are fully replicated, optimizer
# master/state are owned per rank, gradients are reduced to owners and updated
# parameters are synced back. This is NOT PyTorch FSDP1 FULL_SHARD (that path
# is incompatible with Muon's complete-matrix Newton-Schulz).
#
# Defaults reproduce the validated best configuration: 8 GPUs, micro batch 10,
# GBS 80, bf16 gradient reduction (model.grad_reduce_dtype), gradient/parameter
# sync overlap enabled. Model fields this run does not pin (use_compile,
# gradient_checkpointing) come from configs/models/embodied/lingbot_vla_v2.yaml.
#
# Usage:
#   bash run_lingbot_vla_v2_finetune.sh                                  # GBS80 best config, 20 iters
#   bash run_lingbot_vla_v2_finetune.sh > run.log 2>&1                   # capture output (caller's choice)
#   MICRO_BS=16 bash run_lingbot_vla_v2_finetune.sh                      # GBS128, GA1
#   MICRO_BS=8 GRAD_ACCUM=2 bash run_lingbot_vla_v2_finetune.sh          # GBS128, GA2
#   bash run_lingbot_vla_v2_finetune.sh --train-iters 50000              # override a training flag
#   bash run_lingbot_vla_v2_finetune.sh model.gradient_checkpointing=true # override a YAML field
#   SAVE_TRAINING_STATE=true bash run_lingbot_vla_v2_finetune.sh --save-format dcp
#
# The pinned upstream ``lingbotvla`` package is installed on demand:
#   INSTALL_UPSTREAM=1 bash run_lingbot_vla_v2_finetune.sh   # pip install the pin
#   SKIP_UPSTREAM_CHECK=1 bash run_lingbot_vla_v2_finetune.sh # already on PYTHONPATH
#
# Note: the gradient-ready order is measured inside the run -- the first step
# buckets by reverse parameter-registration order and a later step rebuilds the
# plan from the measured order. No order file and no warm-up run is involved.
# ============================================================================
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd -- "$SCRIPT_DIR/../../.." && pwd)"
export LOONGFORGE_PATH=${LOONGFORGE_PATH:-"$PROJECT_ROOT"}

# -- Paths --------------------------------------------------------------------
# Data, checkpoints and assets live under $LOCAL_VLA_ARTIFACTS_ROOT/lingbot_vla_v2
# (synced from BOS by tests/embodied/config/prepare.sh):
#   models/{lingbot-vla-v2-6b,Qwen3-VL-4B-Instruct,moge-2-vitb-normal}
#   datasets/lerobot_robotwin_eef_aug_500/<task>-aloha-agilex_randomized_500-1000
#   assets/norm_stats/robotwin.json
#   configs/robot_configs/{robotwin,agilex_cobot_magic}.yaml
export LOCAL_VLA_ARTIFACTS_ROOT=${LOCAL_VLA_ARTIFACTS_ROOT:-"/ssd2/loongforge_embodied_ci/vla_artifacts"}
LINGBOT_ROOT=${LINGBOT_ROOT:-"$LOCAL_VLA_ARTIFACTS_ROOT/lingbot_vla_v2"}

CKPT_PATH=${CKPT_PATH:-"$LINGBOT_ROOT/models/lingbot-vla-v2-6b"}
TOKENIZER_PATH=${TOKENIZER_PATH:-"$LINGBOT_ROOT/models/Qwen3-VL-4B-Instruct"}
MOGE_PATH=${MOGE_PATH:-"$LINGBOT_ROOT/models/moge-2-vitb-normal/model.pt"}
DATASET_DIR=${DATASET_DIR:-"$LINGBOT_ROOT/datasets/lerobot_robotwin_eef_aug_500"}
TASK_NAME=${TASK_NAME:-"click_bell"}
OUTPUT_DIR=${OUTPUT_DIR:-"$PROJECT_ROOT/outputs/lingbot_vla_v2"}
mkdir -p "$OUTPUT_DIR"

# The dataset list is "<robot_config_name> <dataset_dir>" per line. Unless
# DATA_PATH points at a prepared list, write one for TASK_NAME so no absolute
# path is baked into a checked-in file.
if [[ -z "${DATA_PATH:-}" ]]; then
    DATA_PATH="$OUTPUT_DIR/dataset_list.txt"
    echo "robotwin $DATASET_DIR/${TASK_NAME}-aloha-agilex_randomized_500-1000" > "$DATA_PATH"
fi

# Robot config root, and the asset root the run launches from. The robot config
# declares its normalization stats as a path relative to the data-repo root
# (e.g. ``norm_stats: assets/norm_stats/robotwin.json``); the data loader opens
# it against the process CWD, so the launch must happen from the directory that
# contains ``assets/``. That root is the data repo two levels above
# ``configs/robot_configs``; derive it so overriding ROBOT_CONFIG_ROOT keeps the
# asset lookup consistent. Both are overridable.
ROBOT_CONFIG_ROOT=${ROBOT_CONFIG_ROOT:-"$LINGBOT_ROOT/configs/robot_configs"}
ASSET_ROOT=${ASSET_ROOT:-"$(cd -- "$ROBOT_CONFIG_ROOT/../.." 2>/dev/null && pwd || true)"}
# The model YAML leaves these paths null; fail early with a pointer to prepare.sh
# rather than deep inside teacher loading.
for _p in "$CKPT_PATH" "$TOKENIZER_PATH" "$MOGE_PATH" "$ROBOT_CONFIG_ROOT" \
          "$CKPT_PATH/depth/model.pt" "$CKPT_PATH/dino_video/config.yaml"; do
    if [[ ! -e "$_p" ]]; then
        echo "ERROR: missing dependency: $_p" >&2
        echo "       sync artifacts with:  bash tests/embodied/config/prepare.sh" >&2
        exit 1
    fi
done
if [[ -z "$ASSET_ROOT" ]]; then
    echo "ERROR: cannot resolve ASSET_ROOT from ROBOT_CONFIG_ROOT=$ROBOT_CONFIG_ROOT" >&2
    echo "       set ASSET_ROOT to the data repo that contains assets/norm_stats/." >&2
    exit 1
fi

export PYTORCH_CUDA_ALLOC_CONF=${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}
# albumentations checks its own version over the network on import.
export NO_ALBUMENTATIONS_UPDATE=${NO_ALBUMENTATIONS_UPDATE:-1}

# -- Upstream dependency ------------------------------------------------------
# The network is consumed from the pinned third-party ``lingbotvla`` package
# (see loongforge/embodied/model/lingbot_vla_v2/upstream_adapter.py). It is not
# an installable extra of this project, so the pin lives here. Install it once
# per environment with INSTALL_UPSTREAM=1; otherwise a missing package fails
# fast here instead of deep inside the first import.
LINGBOTVLA_UPSTREAM_PIN=${LINGBOTVLA_UPSTREAM_PIN:-"lingbotvla @ git+https://github.com/Robbyant/lingbot-vla-v2.git@69729b4ef24c63ec25e750915491635f4753be1d"}
if [[ ${INSTALL_UPSTREAM:-0} == 1 ]]; then
    python3 -m pip install "$LINGBOTVLA_UPSTREAM_PIN"
elif [[ ${SKIP_UPSTREAM_CHECK:-0} != 1 ]] \
    && ! python3 -c "import importlib.util, sys; sys.exit(0 if importlib.util.find_spec('lingbotvla') else 1)"; then
    echo "ERROR: the pinned upstream 'lingbotvla' package is not installed." >&2
    echo "       install it with:  INSTALL_UPSTREAM=1 bash $0" >&2
    echo "       or:  pip install \"$LINGBOTVLA_UPSTREAM_PIN\"" >&2
    echo "       (if it is provided via PYTHONPATH instead, set SKIP_UPSTREAM_CHECK=1)" >&2
    exit 1
fi

# -- Distributed --------------------------------------------------------------
GPUS_PER_NODE=${GPUS_PER_NODE:-8}
MASTER_ADDR=${MASTER_ADDR:-"localhost"}
MASTER_PORT=${MASTER_PORT:-"29611"}
NNODES=${NNODES:-"1"}
NODE_RANK=${RANK:-"0"}

DISTRIBUTED_ARGS=(
    --nproc_per_node $GPUS_PER_NODE
    --nnodes $NNODES
    --node_rank $NODE_RANK
    --master_addr $MASTER_ADDR
    --master_port $MASTER_PORT
)

# -- custom_dp knobs ----------------------------------------------------------------
# Collective precision and overlap live in the model YAML (model.grad_reduce_dtype,
# model.param_sync_precision, model.grad_overlap, model.param_overlap,
# model.comm_bucket_mb, model.grad_inflight_mb). Override per run on the command
# line:
#   bash run_lingbot_vla_v2_finetune.sh model.grad_reduce_dtype=fp32
# The precision settings exempt the parameters the model policy marks
# precision-critical (MoE router/gate weights, 1-D norms/biases): expert selection
# is decided by fp32 gate logits, so a rounded router weight flips top-k rather
# than perturbing it.

# -- Model config -------------------------------------------------------------
MODEL_NAME=${MODEL_NAME:-"lingbot_vla_v2"}
MODEL_CONFIG_ARGS=(
    --model-name $MODEL_NAME
)

NUM_WORKERS=${NUM_WORKERS:-16}
DATA_ARGS=(
    --dataset-format lerobot_datasets
    --dataset-strategy lingbot_vla_v2
    --dataset-path "$DATA_PATH"
    --tokenizer-path "$TOKENIZER_PATH"
    --num-workers $NUM_WORKERS
)
if (( NUM_WORKERS > 0 )); then
    DATA_ARGS+=(--dataloader-multiprocessing-context spawn)
fi

# -- Training params ----------------------------------------------------------
# micro batch 10 x 8 GPUs = GBS 80 (micro batch 16 => GBS 128, the GBS ceiling).
MICRO_BS=${MICRO_BS:-10}
GRAD_ACCUM=${GRAD_ACCUM:-1}
TRAINING_ARGS=(
    --trainer-type LingbotVlaV2CustomDpTrainer
    --train-iters ${TRAIN_ITERS:-20}
    --per-device-batch-size $MICRO_BS
    --gradient-accumulation-steps $GRAD_ACCUM
    --seed 42
    --output-dir "$OUTPUT_DIR"
    --manual-gc
    --manual-gc-interval 0
    # Learning rate
    --lr-base 1e-4
    --lr-decay-style constant
    # Optimizer
    --optimizer muon
    --clip-grad 1
    --weight-decay 0
    --save-interval ${SAVE_INTERVAL:-1000}
)
# Training state (optimizer + scheduler + RNG) is needed to resume. Off by
# default because the default run is a short throughput check; set
# SAVE_TRAINING_STATE=true together with --save-format dcp to exercise resume.
if [[ ${SAVE_TRAINING_STATE:-false} == true ]]; then
    TRAINING_ARGS+=(--save-training-state)
else
    TRAINING_ARGS+=(--no-save-training-state)
fi
if [[ ${USE_NSYS_PROFILE:-0} == 1 ]]; then
    export NSYS_GPU_METRICS_DEVICE=${NSYS_GPU_METRICS_DEVICE:-all}
    TRAINING_ARGS+=(
        --use-nsys-profiler
        --profile-step-start "${PROFILE_STEP_START:-5}"
        --profile-step-end "${PROFILE_STEP_END:-7}"
        --profile-ranks 0
    )
fi

DISTRIBUTED_TRAINING_ARGS=(
    # The trainer wraps the model itself, so neither DDP nor FSDP applies.
    --distributed-strategy custom_dp
    # custom_dp requires ZeRO-1: it owns the fp32 masters and the optimizer state.
    --zero-optimizer
    --dtype float32
)

# Only fields this run has to pin go here; everything else comes from the model
# YAML, which stays the single source of the model defaults.
MODEL_OVERRIDE_ARGS=(
    data.robot_config_root="$ROBOT_CONFIG_ROOT"
    model.model_path="$CKPT_PATH"
    model.tokenizer_path="$TOKENIZER_PATH"
    model.align_params.depth.moge_path="$MOGE_PATH"
    model.align_params.depth.morgbd_path="$CKPT_PATH/depth/model.pt"
    model.align_params.video.ckpt_path="$CKPT_PATH/dino_video/teacher_step_10000.pth"
    model.align_params.video.config_path="$CKPT_PATH/dino_video/config.yaml"
)
if [[ -n "${USE_COMPILE:-}" ]]; then
    MODEL_OVERRIDE_ARGS+=(model.use_compile=$USE_COMPILE)
fi
if [[ -n "${GRADIENT_CHECKPOINTING:-}" ]]; then
    MODEL_OVERRIDE_ARGS+=(model.gradient_checkpointing=$GRADIENT_CHECKPOINTING)
fi

# -- Logging params -----------------------------------------------------------
LOGGING_ARGS=(
    --log-interval 1
)

# -- Launch -------------------------------------------------------------------
echo "========================================================================"
echo "  LoongForge lingbot_vla_v2 Training (custom_dp)"
echo "  Model:        $MODEL_NAME"
echo "  GPUs:         $GPUS_PER_NODE x $NNODES nodes"
echo "  Micro batch:  $MICRO_BS"
echo "  Grad accum:   $GRAD_ACCUM  (global batch $(( MICRO_BS * GRAD_ACCUM * GPUS_PER_NODE * NNODES )))"
echo "  Data:         $DATA_PATH"
echo "  Output:       $OUTPUT_DIR"
echo "  Comm:         precision and overlap come from the model YAML"
echo "========================================================================"

# Launch from the asset root so the robot config's relative ``norm_stats`` path
# resolves; train.py is referenced by absolute path and imports resolve through
# PYTHONPATH, so the working directory only anchors data-asset lookups.
cd "$ASSET_ROOT"
PYTHONPATH=$LOONGFORGE_PATH:${PYTHONPATH:-} \
    torchrun "${DISTRIBUTED_ARGS[@]}" \
    "$LOONGFORGE_PATH/loongforge/embodied/train.py" \
    "${MODEL_CONFIG_ARGS[@]}" \
    "${DATA_ARGS[@]}" \
    "${TRAINING_ARGS[@]}" \
    "${DISTRIBUTED_TRAINING_ARGS[@]}" \
    "${LOGGING_ARGS[@]}" \
    "${MODEL_OVERRIDE_ARGS[@]}" \
    "$@"
