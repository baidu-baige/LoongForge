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
# run_verify_offline_latents.sh - Offline VAE-latent parity (COMPUTE link)
#
# Wraps verify_offline_latents.py: loads the batches dumped by an ONLINE
# training run (LATENT_DUMP=1 hook in MotusTrainer._prefetch_batch), rebuilds
# ONLY the frozen Wan2.2 VAE in a FRESH process, replays encode_video_latents at
# the SAME batch size, and bit-compares recomputed vs dumped latent. A PASS
# proves "same input bytes -> same latent bytes, in a different process"
# (the risky COMPUTE link of the offline latent cache).
#
# PREREQUISITE - produce the dumps first (online training run):
#   LATENT_DUMP=1 LATENT_DUMP_DIR=/tmp/latent_dump PARITY_DATA_SEED=42 \
#     bash examples/embodied/motus/run_motus_deepcompile_finetune.sh
#   (stop after step >= LATENT_DUMP_STEPS, default 3)
#
# Run this on the SAME GPU type (A800) + SAME torch/cuDNN version as training.
#
# Usage:
#   bash run_verify_offline_latents.sh
#   DUMP_DIR=/tmp/latent_dump bash run_verify_offline_latents.sh
#   bash run_verify_offline_latents.sh --rank 0
# ═══════════════════════════════════════════════════════════════
set -euo pipefail

export LOONGFORGE_PATH=${LOONGFORGE_PATH:-"/workspace/AIAK-Training-Omni"}

# ── Paths ─────────────────────────────────────────────────────
DUMP_DIR=${DUMP_DIR:-"/tmp/latent_dump"}
VAE_PATH=${VAE_PATH:-"/workspace/motus/models/hf/Wan2.2-TI2V-5B/Wan2.2_VAE.pth"}
DEVICE=${DEVICE:-"cuda:0"}

# ── Env ───────────────────────────────────────────────────────
export NO_ALBUMENTATIONS_UPDATE=1

echo "════════════════════════════════════════════════════════════"
echo "  Motus offline-latent COMPUTE-link parity check"
echo "  Dump dir:   $DUMP_DIR"
echo "  VAE path:   $VAE_PATH"
echo "  Device:     $DEVICE"
echo "════════════════════════════════════════════════════════════"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

PYTHONPATH=$LOONGFORGE_PATH:${PYTHONPATH:-} \
    python3 "$SCRIPT_DIR/verify_offline_latents.py" \
    --dump-dir "$DUMP_DIR" \
    --vae-path "$VAE_PATH" \
    --device "$DEVICE" \
    "$@"
