#!/usr/bin/env bash
# Copyright 2026 The LoongForge Authors.
# SPDX-License-Identifier: Apache-2.0
# Retarget the bundled case_01 sample onto Panda + one dexterous hand.
#
#   bash examples/run_case_01.sh <hand> [output_dir]
#
# <hand>: allegro leap wuji inspire shadow svh ability dexhand021 (sharpa needs
# EGO2ROBOT_SHARPA_UPSTREAM, see README). Run tools/fetch_hand_assets.py first.
set -euo pipefail
HAND=${1:?usage: run_case_01.sh <hand> [output_dir]}
REPO=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
EX=$REPO/examples/case_01
OUT=${2:-$REPO/examples/output/case_01_$HAND}
PY=${PYTHON:-python}

export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 LP_NUM_THREADS=1
export MUJOCO_GL=${MUJOCO_GL:-osmesa}
if [ "$MUJOCO_GL" = osmesa ]; then export PYOPENGL_PLATFORM=${PYOPENGL_PLATFORM:-osmesa}; fi
# Arm trajectory from the accepted Sharpa run of this sample; it makes the
# result reproducible. Set it to an empty string to start the arm solve from
# the initial pose.
export EGO2ROBOT_ARM_INITIALIZER=${EGO2ROBOT_ARM_INITIALIZER-$EX/arm_seed}

EXTRA=()
if [ "$HAND" = sharpa ]; then
  EXTRA=(--acceptance_profile mild --hand_initializer v28-v29)
  export PYTHONPATH=${EGO2ROBOT_SHARPA_UPSTREAM:?set EGO2ROBOT_SHARPA_UPSTREAM}/retargeting${PYTHONPATH:+:$PYTHONPATH}
fi

cd "$REPO"
"$PY" cli.py retarget --robot_type panda --end_effector "$HAND" \
  --orientation_cache "$EX/cache/output" --hand_reference_dir "$EX/references" \
  --zarr_dir "$EX/source_aux/pick_place" --bg_dir "$EX/cache/output/04_inpaint" \
  --state_dir "$EX/cache/output/02_align" --output_dir "$OUT" \
  --episodes pick_place__2026-04-14-04-29-46-000000 \
  --hand_rotation_policy stabilize --hand_depth_mode alpha --motion_continuity --joint_pose \
  "${EXTRA[@]}"
echo "result: $OUT"
