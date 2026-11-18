#!/usr/bin/env bash
# Copyright 2026 The LoongForge Authors.
# SPDX-License-Identifier: Apache-2.0
#
# GigaBrain-0.7 × predict_action server smoke (standalone, no benchmark env).
#
# Mirrors examples/embodied/xvla/eval/run_*_eval.sh (env wiring + a single
# CONFIG YAML), but GigaBrain-0.7 has no wired simulation benchmark yet, so
# instead of launching the orchestrator (server + benchmark env loop) this
# script runs the verified link smoke: start the LoongForge policy server from
# configs/smoke.yaml, wait for health, send one predict_action RPC through the
# real GigaBrain07PayloadBuilder, and assert a [B=1, 50, 32] action chunk. The
# server is stopped on exit.
#
# Open weight: open-gigaai/GigaBrain-0.7-3.5B-Base (arch from its config.json:
# proprio_anchor / 8 embodiments / 32-dim action). Fill any /path overrides via
# env or edit configs/smoke.yaml.

set -euo pipefail

# ── Repo / example layout (override via env) ────────────────────────────────
REPO_ROOT=${REPO_ROOT:-/workspace/LoongForge}
EXAMPLE_EVAL_ROOT=${EXAMPLE_EVAL_ROOT:-${REPO_ROOT}/examples/embodied/giga_brain07/eval}
CONFIG=${CONFIG:-${EXAMPLE_EVAL_ROOT}/configs/smoke.yaml}
if [[ "${CONFIG}" != /* ]]; then
  CONFIG=${REPO_ROOT}/${CONFIG}
fi

# giga_models lives under the model dir; the dataset transforms under the data
# dir. Both must precede REPO_ROOT on PYTHONPATH (same order as the training
# entry run_giga_brain07_torchrun.sh) so ``import giga_models`` resolves.
GB_MODEL_DIR=${GB_MODEL_DIR:-${REPO_ROOT}/loongforge/embodied/model/giga_brain07}
GB_DATA_DIR=${GB_DATA_DIR:-${REPO_ROOT}/loongforge/embodied/data/datasets/giga_brain07}
export PYTHONPATH=${GB_MODEL_DIR}:${GB_DATA_DIR}:${REPO_ROOT}:${PYTHONPATH:-}

# ── Runtime env ─────────────────────────────────────────────────────────────
export LOONGFORGE_PATH=${LOONGFORGE_PATH:-${REPO_ROOT}}
export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0}
export HF_HUB_OFFLINE=${HF_HUB_OFFLINE:-1}
export TRANSFORMERS_OFFLINE=${TRANSFORMERS_OFFLINE:-1}
export TOKENIZERS_PARALLELISM=${TOKENIZERS_PARALLELISM:-false}
export PYTORCH_CUDA_ALLOC_CONF=${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}

# Host / ports are read from the YAML server: block; mirror the defaults here so
# the client knows where to connect and so health polling targets the right port.
HOST=${HOST:-127.0.0.1}
PORT=${PORT:-13391}
HEALTH_PORT=${HEALTH_PORT:-13392}
START_TIMEOUT_SEC=${START_TIMEOUT_SEC:-600}
SERVER_LOG=${SERVER_LOG:-/tmp/gb07_server.log}

PYTHON=${PYTHON:-python3}

cd "${REPO_ROOT}"

# ── Machine paths (hardcoded here; this launcher is the single source of the
#    absolute paths and writes them into the server config below) ────────────
GB07_CKPT=/workspace/ckpt/open-gigaai/GigaBrain-0.7-3.5B-Base
GB07_TOKENIZER=/workspace/ckpt/paligemma-3b-pt-224
GB07_NORM_STATS=/workspace/data/agibot-world/AgiBotWorldLerobot/agibot_a2d_norm_stats.json

# ── Generate the server config from the hardcoded values above ──────────────
mkdir -p "$(dirname "${CONFIG}")"
cat > "${CONFIG}" <<EOF
model:
  backend: loongforge
  model_type: giga_brain07
  pretrained: ${GB07_CKPT}
  state_encoding: ee6d
  action_encoding: ee6d
  action_horizon: 50
  max_action_dim: 32
  emb_id: 0

server:
  host: ${HOST}
  port: ${PORT}
  health_port: ${HEALTH_PORT}
  random_init: false
  ckpt_path: ${GB07_CKPT}
  tokenizer_path: ${GB07_TOKENIZER}
  dataset_statistics_path: ${GB07_NORM_STATS}
  use_bf16: true
  loongforge_root: ${REPO_ROOT}
EOF

echo "════════════════════════════════════════════════════════════"
echo "  GigaBrain-0.7 predict_action server smoke"
echo "  config=${CONFIG}"
echo "  host=${HOST} port=${PORT} health=${HEALTH_PORT} gpu=${CUDA_VISIBLE_DEVICES}"
echo "  server_log=${SERVER_LOG}"
echo "════════════════════════════════════════════════════════════"

# ── 1. Start the policy server in the background ────────────────────────────
"${PYTHON}" loongforge/embodied/eval/servers/loongforge_server.py \
  --config "${CONFIG}" > "${SERVER_LOG}" 2>&1 &
SERVER_PID=$!
echo "[run] server started pid=${SERVER_PID}; waiting for health on :${HEALTH_PORT} (<=${START_TIMEOUT_SEC}s)"

cleanup() {
  echo "[run] stopping server pid=${SERVER_PID}"
  kill "${SERVER_PID}" 2>/dev/null || true
  wait "${SERVER_PID}" 2>/dev/null || true
}
trap cleanup EXIT

# ── 2. Wait for warmup/health (fail fast if the server process dies) ────────
ready=0
for ((i = 0; i < START_TIMEOUT_SEC; i += 5)); do
  if ! kill -0 "${SERVER_PID}" 2>/dev/null; then
    echo "[run] ERROR: server process exited early; tail of ${SERVER_LOG}:"
    tail -30 "${SERVER_LOG}" || true
    exit 1
  fi
  if curl -s -m 2 "http://${HOST}:${HEALTH_PORT}/" >/dev/null 2>&1 \
     && grep -q "Model warmup complete" "${SERVER_LOG}" 2>/dev/null; then
    ready=1
    echo "[run] server ready after ~${i}s"
    break
  fi
  sleep 5
done
if [[ "${ready}" -ne 1 ]]; then
  echo "[run] ERROR: server not ready within ${START_TIMEOUT_SEC}s; tail of ${SERVER_LOG}:"
  tail -30 "${SERVER_LOG}" || true
  exit 1
fi

# ── 3. Run the predict_action client smoke ([B=1, 50, 32]) ──────────────────
set +e
"${PYTHON}" "${EXAMPLE_EVAL_ROOT}/smoke_predict_action.py" \
  --host "${HOST}" --port "${PORT}" --action-horizon 50 --action-dim 32
rc=$?
set -e

echo "[run] smoke client exit code=${rc}"
exit "${rc}"
