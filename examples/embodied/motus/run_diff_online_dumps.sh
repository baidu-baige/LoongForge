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
# run_diff_online_dumps.sh - Online-vs-online VAE-input parity (DATA link)
#
# Wraps diff_online_dumps.py: compares the LATENT_DUMP dumps of TWO separate
# ONLINE training runs (both launched with the SAME PARITY_DATA_SEED so the
# sampler visits identical anchors in identical order). For each matching
# rankR_stepN.pt it bit-compares first_frame / video_frames / clean_full_latent.
# A PASS on first_frame/video_frames proves the data pipeline (torchcodec decode
# + transform + collate) is bit-deterministic ACROSS PROCESSES -> the offline
# precompute will re-fetch identical input pixels per anchor (the DATA link).
#
# PREREQUISITE - produce TWO dump dirs (two online runs, same seed):
#   LATENT_DUMP=1 LATENT_DUMP_DIR=/tmp/latent_dump_run1 PARITY_DATA_SEED=42 \
#     bash examples/embodied/motus/run_motus_deepcompile_finetune.sh
#   LATENT_DUMP=1 LATENT_DUMP_DIR=/tmp/latent_dump_run2 PARITY_DATA_SEED=42 \
#     bash examples/embodied/motus/run_motus_deepcompile_finetune.sh
#   (stop each after step >= LATENT_DUMP_STEPS, default 3)
#
# Usage:
#   bash run_diff_online_dumps.sh
#   DIR_A=/tmp/latent_dump_run1 DIR_B=/tmp/latent_dump_run2 bash run_diff_online_dumps.sh
# ═══════════════════════════════════════════════════════════════
set -euo pipefail

export LOONGFORGE_PATH=${LOONGFORGE_PATH:-"/workspace/AIAK-Training-Omni"}

# ── Paths ─────────────────────────────────────────────────────
DIR_A=${DIR_A:-"/tmp/latent_dump_run1"}
DIR_B=${DIR_B:-"/tmp/latent_dump_run2"}

# ── Env ───────────────────────────────────────────────────────
export NO_ALBUMENTATIONS_UPDATE=1

echo "════════════════════════════════════════════════════════════"
echo "  Motus online-vs-online DATA-link parity check"
echo "  Dir A:      $DIR_A"
echo "  Dir B:      $DIR_B"
echo "════════════════════════════════════════════════════════════"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

PYTHONPATH=$LOONGFORGE_PATH:${PYTHONPATH:-} \
    python3 "$SCRIPT_DIR/diff_online_dumps.py" \
    --dir-a "$DIR_A" \
    --dir-b "$DIR_B" \
    "$@"
