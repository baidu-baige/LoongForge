#!/usr/bin/env bash
# Copyright 2026 The LoongForge Authors.
# SPDX-License-Identifier: Apache-2.0
#
# LIBERO-Pro 5-dimension parallel evaluation entry point.
#
# LIBERO-Pro is a drop-in replacement for the `libero` package. It registers
# 20 perturbation suites on top of the stock four:
#
#   libero_<base>_<dim>,  base in {spatial, object, goal, 10}
#                         dim  in {object (Obj), swap (Pos), lan (Sem),
#                                  task (Task), env (Env)}
#
# Parallel mode: the coordinator starts one inference service per GPU listed
# under `servers[].gpus`, N CPU-sim workers per GPU
# (`pools.cpu_sim.n_sims_per_server`), and runs every cell from the YAML's
# `benchmark.suites` list in one coordinated plan. Each worker's EGL device is
# pinned to its slot GPU, and `benchmark.tasks_per_unit` tasks share a process.
#
# Prerequisites:
#   * BENCHMARK_PYTHON: env with LIBERO-Pro installed in place of stock LIBERO
#   * YAML `env.libero_config_path` points at the LIBERO-Pro config.yaml
#   * YAML `server.python`: env that can import LoongForge
#   * `libero_*_env` bddl/init are NOT shipped upstream; generate them once with
#     GEN_ENV=1 (official perturbation.py + generate_init_states.py pipeline)
#
# Usage:
#   BENCHMARK_PYTHON=/path/to/libero_pro/bin/python \
#     bash run_libero_pro_eval.sh [CONFIG]
#
# Overrides:
#   CONFIG=<yaml>                 default configs/libero/libero_pro.yaml
#   LIBERO_PRO_ROOT=<dir>         checkout root, for the preflight check
#   GEN_ENV=1                     generate missing libero_*_env suites first
#   LOONGFORGE_COORD_PORT_BASE=... base port for the per-GPU services
#   DRY_RUN=1                     print env/command only
#
# NOTE: do not pin CUDA_VISIBLE_DEVICES / MUJOCO_EGL_DEVICE_ID here. The
# coordinator owns GPU selection and per-worker EGL pinning; exporting a single
# device would collapse every service and renderer onto one card.

set -euo pipefail

SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
REPO_ROOT=${REPO_ROOT:-$(cd "${SCRIPT_DIR}/../../../.." && pwd)}
CONFIG=${1:-${CONFIG:-${SCRIPT_DIR}/configs/libero/libero_pro.yaml}}
BENCHMARK_PYTHON=${BENCHMARK_PYTHON:-/path/to/libero_pro/bin/python}
LIBERO_PRO_ROOT=${LIBERO_PRO_ROOT:-}

if [[ ! -x "${BENCHMARK_PYTHON}" ]]; then
  echo "ERROR: BENCHMARK_PYTHON is not executable: ${BENCHMARK_PYTHON}" >&2
  echo "Set it to the python of the env that has LIBERO-Pro installed." >&2
  exit 1
fi
if [[ ! -f "${CONFIG}" ]]; then
  echo "ERROR: config not found: ${CONFIG}" >&2
  exit 1
fi

# Client-side runtime. The policy services are started by the coordinator from
# the YAML's server.python with their own environment.
export PYTHONPATH="${REPO_ROOT}:${PYTHONPATH:-}"
# Offscreen rendering for the GEN_ENV subprocess (generate_init_states.py
# creates envs; headless needs EGL/OSMesa explicitly). Per-worker rendering in
# the coordinated run comes from the YAML's env.mujoco_gl.
export MUJOCO_GL=${MUJOCO_GL:-egl}
export PYOPENGL_PLATFORM=${PYOPENGL_PLATFORM:-egl}
export TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD=1   # .pruned_init pickles under torch>=2.6
export TOKENIZERS_PARALLELISM=${TOKENIZERS_PARALLELISM:-false}
export HF_HUB_OFFLINE=${HF_HUB_OFFLINE:-1}
export TRANSFORMERS_OFFLINE=${TRANSFORMERS_OFFLINE:-1}

# ---------------------------------------------------------------------------
# Optional: generate the Env-perturbation suites (not shipped upstream).
# ---------------------------------------------------------------------------
if [[ "${GEN_ENV:-0}" == "1" ]]; then
  if [[ -z "${LIBERO_PRO_ROOT}" ]]; then
    echo "ERROR: GEN_ENV=1 requires LIBERO_PRO_ROOT=/path/to/LIBERO-PRO" >&2
    exit 1
  fi
  echo "[gen-env] generating libero_*_env under ${LIBERO_PRO_ROOT}/libero/libero ..."
  PYTHONPATH="${LIBERO_PRO_ROOT}:${PYTHONPATH}" "${BENCHMARK_PYTHON}" - <<'PY'
import os, shutil, sys, yaml
root = os.environ["LIBERO_PRO_ROOT"]
base = os.path.join(root, "libero", "libero")
sys.path.insert(0, root)
import perturbation

cfg = yaml.safe_load(open(os.path.join(root, "evaluation_config.yaml"), encoding="utf-8"))
ood = {k: os.path.join(root, v.lstrip("./")) for k, v in cfg["ood_task_configs"].items()}
for suite in ("libero_goal", "libero_spatial", "libero_10", "libero_object"):
    bddl_env, init_env = f"{base}/bddl_files/{suite}_env", f"{base}/init_files/{suite}_env"
    # Generation is expensive (~10 tasks x 50 envs per suite); skip when the
    # suite was already generated once.
    if os.path.isdir(bddl_env) and os.listdir(bddl_env) and os.path.isdir(init_env) and os.listdir(init_env):
        print(f"[gen-env] {suite}_env already exists, skip")
        continue
    for path in (f"{base}/bddl_files/{suite}_temp", f"{base}/init_files/{suite}_temp"):
        shutil.rmtree(path, ignore_errors=True)
    tmp = perturbation.process_bddl_file_mixed(
        input_dir=f"{base}/bddl_files/{suite}", task_suite_name=suite,
        flags=perturbation.PerturbFlags(use_environment=True), configs=ood, seed=42,
    )
    perturbation.EvalEnvCreator(
        input_dir=tmp, script_path=os.path.join(root, "notebooks", "generate_init_states.py"),
        base_output_dir=f"{base}/init_files",
    ).create_env()
    for kind in ("bddl_files", "init_files"):
        dst = f"{base}/{kind}/{suite}_env"
        shutil.rmtree(dst, ignore_errors=True)
        os.rename(f"{base}/{kind}/{suite}_temp", dst)
    print(f"[gen-env] {suite}_env ready")
PY
fi

# ---------------------------------------------------------------------------
# Preflight: every suite in the YAML `benchmark.suites` needs
# bddl_files/<suite> and init_files/<suite>.
# ---------------------------------------------------------------------------
"${BENCHMARK_PYTHON}" - "${CONFIG}" <<'PY'
import os, sys, yaml
from pathlib import Path

cfg = yaml.safe_load(open(sys.argv[1], encoding="utf-8"))
bench = cfg.get("benchmark") or {}
suite_list = [str(s) for s in (bench.get("suites") or [bench.get("suite")]) if s]

root = os.environ.get("LIBERO_PRO_ROOT", "")
if root:
    base = Path(root) / "libero" / "libero"
    bddl_dir, init_dir = base / "bddl_files", base / "init_files"
else:
    conf_dir = Path(os.environ.get("LIBERO_CONFIG_PATH", ""))
    conf_file = conf_dir / "config.yaml"
    if not conf_file.exists():
        sys.exit("ERROR: set LIBERO_PRO_ROOT or LIBERO_CONFIG_PATH (with config.yaml) for the preflight check")
    conf = yaml.safe_load(open(conf_file, encoding="utf-8"))
    bddl_dir, init_dir = Path(conf["bddl_files"]), Path(conf["init_states"])

missing = []
for suite in suite_list:
    if not (bddl_dir / suite).is_dir():
        missing.append(f"bddl_files/{suite}")
    if not (init_dir / suite).is_dir():
        missing.append(f"init_files/{suite}")
if missing:
    print("ERROR: missing LIBERO-Pro suite data:", ", ".join(missing), file=sys.stderr)
    if any(m.endswith("_env") for m in missing):
        print("  *_env is not shipped upstream: rerun with GEN_ENV=1 "
              "(official perturbation.py + generate_init_states.py).", file=sys.stderr)
    sys.exit(1)
print(f"[preflight] {len(suite_list)} suites OK (5 dims = object/swap/lan/task/env)")
PY

# ---------------------------------------------------------------------------
# Run
# ---------------------------------------------------------------------------
CMD=("${BENCHMARK_PYTHON}" -m loongforge.embodied.eval.orchestrator.run --config "${CONFIG}")

if [[ "${DRY_RUN:-0}" == "1" ]]; then
  echo "REPO_ROOT=${REPO_ROOT}"
  echo "CONFIG=${CONFIG}"
  echo "PYTHONPATH=${PYTHONPATH}"
  echo "LOONGFORGE_COORD_PORT_BASE=${LOONGFORGE_COORD_PORT_BASE:-<default>}"
  printf 'CMD:'
  printf ' %q' "${CMD[@]}"
  echo
  exit 0
fi

exec "${CMD[@]}"
