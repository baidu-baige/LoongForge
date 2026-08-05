#!/usr/bin/env bash
# Copyright 2026 The LoongForge Authors.
# SPDX-License-Identifier: Apache-2.0
#
# Modified from https://github.com/thu-ml/Motus under the Apache-2.0 License.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ═══════════════════════════════════════════════════════════════
# run_motus_precompute_latent_cache.sh - Offline VAE-latent precompute
#   (WAN Wan2.2 frozen VAE encode, one fp32 latent per anchor)
#
# Precomputes `clean_full_latent` for every valid (episode, condition_frame)
# anchor of the Motus dataset and writes it to $CACHE_DIR keyed by flat frame
# index ({flat_idx:08d}.pt). Online training then skips the ~500ms/step frozen
# VAE encode by loading the cached latent (enable with --latent-cache-dir).
#
# Parity contract (proven bit-identical to the online encode, 2026-07-29):
#   1. same input pixels     -> reuse the SAME dataset decode+transform per anchor
#   2. same VAE weights+prec  -> Wan2_2_VAE(vae_pth=model_cfg.vae_path) fp32/TF32
#   3. same GPU arch+versions -> RUN THIS ON THE SAME A800 + torch/cuDNN as training
#   4. same batch SIZE = 8    -> encode in EXACT groups of 8 (tail padded, sliced)
#
# Single process (NO torchrun): the script enumerates the full anchor set itself.
# Pass the SAME model/data flags as the training launch so the dataset geometry,
# decode/transform, and VAE checkpoint all match.
#
# Usage:
#   bash run_motus_precompute_latent_cache.sh
#   CACHE_DIR=/path/to/cache bash run_motus_precompute_latent_cache.sh
#   bash run_motus_precompute_latent_cache.sh --limit-anchors 16     # smoke test
# ═══════════════════════════════════════════════════════════════
set -euo pipefail

export LOONGFORGE_PATH=${LOONGFORGE_PATH:-"/workspace/AIAK-Training-Omni"}

# ── Paths ─────────────────────────────────────────────────────
DATA_PATH=${DATA_PATH:-"/workspace/motus/data/aloha_mobile_cabinet"}
CACHE_DIR=${CACHE_DIR:-"/workspace/motus/data/latent_cache/aloha_mobile_cabinet"}
CACHE_DTYPE=${CACHE_DTYPE:-"fp32"}   # fp32 = bit-identical (default); bf16 = half size, NOT bit-identical
DEVICE=${DEVICE:-"cuda:0"}
mkdir -p "$CACHE_DIR"

# ── Env ───────────────────────────────────────────────────────
# Skip albumentations' online version check (no network on the training node).
export NO_ALBUMENTATIONS_UPDATE=1
# Deterministic cuDNN is also forced inside the script; set here for consistency.
export CUBLAS_WORKSPACE_CONFIG=${CUBLAS_WORKSPACE_CONFIG:-":4096:8"}

# ── Model config ──────────────────────────────────────────────
MODEL_NAME=${MODEL_NAME:-"motus"}
MODEL_CONFIG_ARGS=(
    --model-name $MODEL_NAME
)

# ── Data params (must match the training launch's dataset geometry) ──
DATA_ARGS=(
    --dataset-format lerobot_datasets
    --dataset-strategy motus
    --dataset-path $DATA_PATH
    --video-backend torchcodec
)

# ── Training params (only the batch size drives cuDNN algo -> parity req #4) ──
TRAINING_ARGS=(
    --per-device-batch-size 8
)

# ── Precompute-specific flags ─────────────────────────────────
# ── Precompute-specific flags (shared; --device / shard flags added per launch) ─
PRECOMPUTE_ARGS=(
    --cache-dir "$CACHE_DIR"
    --cache-dtype "$CACHE_DTYPE"
)

# NUM_SHARDS>1 -> launch one process per shard on cuda:0..N-1 for multi-GPU
# concurrency (near-linear speedup; decode is the bottleneck). Resume is ON by
# default in the python script (skips batches whose .pt already exist), so a
# re-launch continues where a previous/interrupted run left off. Set
# OVERWRITE=1 to force re-encode. Batch=8 is preserved within each shard, so
# outputs are bit-identical regardless of shard count.
NUM_SHARDS=${NUM_SHARDS:-1}
OVERWRITE_ARG=()
if [[ "${OVERWRITE:-0}" == "1" ]]; then
    OVERWRITE_ARG=(--overwrite)
fi

# ── Launch ────────────────────────────────────────────────────
echo "════════════════════════════════════════════════════════════"
echo "  Motus offline VAE-latent precompute"
echo "  Model:      $MODEL_NAME"
echo "  Data:       $DATA_PATH"
echo "  Cache dir:  $CACHE_DIR  (dtype=$CACHE_DTYPE)"
echo "  Shards:     $NUM_SHARDS   Overwrite: ${OVERWRITE:-0}"
echo "════════════════════════════════════════════════════════════"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

if [[ "$NUM_SHARDS" -gt 1 ]]; then
    # Multi-GPU: one background process per shard, shard i pinned to cuda:i,
    # each with its own log; wait for all and fail if any shard fails.
    pids=()
    for ((i = 0; i < NUM_SHARDS; i++)); do
        shard_log="/tmp/precompute_shard${i}.log"
        echo "  -> shard $i on cuda:$i  (log: $shard_log)"
        PYTHONPATH=$LOONGFORGE_PATH:${PYTHONPATH:-} \
            python3 "$SCRIPT_DIR/precompute_latent_cache.py" \
            "${PRECOMPUTE_ARGS[@]}" \
            --device "cuda:$i" \
            --num-shards "$NUM_SHARDS" \
            --shard-id "$i" \
            "${OVERWRITE_ARG[@]}" \
            "${MODEL_CONFIG_ARGS[@]}" \
            "${DATA_ARGS[@]}" \
            "${TRAINING_ARGS[@]}" \
            "$@" > "$shard_log" 2>&1 &
        pids+=($!)
    done
    rc=0
    for p in "${pids[@]}"; do
        wait "$p" || rc=1
    done
    echo "[ALL SHARDS DONE] rc=$rc  (per-shard logs: /tmp/precompute_shard*.log)"
    exit $rc
fi

# Single process (default): one shard on $DEVICE.
PYTHONPATH=$LOONGFORGE_PATH:${PYTHONPATH:-} \
    python3 "$SCRIPT_DIR/precompute_latent_cache.py" \
    "${PRECOMPUTE_ARGS[@]}" \
    --device "$DEVICE" \
    "${OVERWRITE_ARG[@]}" \
    "${MODEL_CONFIG_ARGS[@]}" \
    "${DATA_ARGS[@]}" \
    "${TRAINING_ARGS[@]}" \
    "$@"
