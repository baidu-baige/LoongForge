# LIBERO Evaluation

LIBERO is a tabletop robot manipulation benchmark with 4 task suites (Spatial, Object, Goal, Long Horizon), totaling 40 tasks, evaluated on a Franka arm. This page is a step-by-step reproduction guide for running LIBERO evaluation through the LoongForge eval module.

Three models are verified on LIBERO: **pi05**, **xvla**, and **GR00T-N1.6** (pi05/xvla weights are public; GR00T-N1.6 uses a local LIBERO fine-tune).

## Step 0: Download weights

| Model | Weights |
|---|---|
| pi05 | [lerobot/pi05_libero_finetuned_v044](https://huggingface.co/lerobot/pi05_libero_finetuned_v044) (`model.safetensors` + `dataset_statistics.json`) |
| xvla | [2toINF/X-VLA-LIBERO](https://huggingface.co/2toINF/X-VLA-LIBERO) |
| GR00T-N1.6 | local LIBERO fine-tune (weight dir + Eagle3 processor + `libero_panda` stats) |

## Step 1: Environment setup

### Standard environment

Install the base LIBERO environment following the official [LIBERO repository](https://github.com/Lifelong-Robot-Learning/LIBERO) instructions, then install the additional dependencies the eval client needs:

```bash
pip install websockets msgpack pyyaml
pip install numpy==1.24.4   # downgrade numpy for simulator compatibility
```

⚠️ Common issues:

- LIBERO historically defaults to Python 3.8; the syntax differences up to 3.10 are substantial. Python 3.10 avoids many issues.
- `numpy==1.24.4` is pinned for compatibility with the simulation environment — do not upgrade.
- The LIBERO runner uses MuJoCo with offscreen GL. Keep `MUJOCO_GL=osmesa` / `PYOPENGL_PLATFORM=osmesa` (set by the shipped run scripts).

The benchmark dataset is downloaded following the official LIBERO data instructions; the eval module does not repackage it.

## Step 2: Run evaluation

Run from inside the **benchmark** environment. The run scripts and eval YAMLs ship with `/path/to/...` placeholders — fill them in before running:

```bash
cd /path/to/LoongForge-VLA

# pi05
examples/embodied/pi05/eval/run_libero_eval.sh

# xvla
examples/embodied/xvla/eval/run_libero_eval.sh

# GR00T-N1.6
examples/embodied/groot_n1_6/eval/run_libero_eval.sh
```

Environment variables:

| Variable | Meaning | Default |
|---|---|---|
| `CONFIG` | eval YAML config path | `<model>/eval/configs/libero/object_smoke.yaml` (xvla: `libero_weight_object_smoke.yaml`) |
| `REPO_ROOT` | LoongForge repo root | `/path/to/LoongForge-VLA` |
| `BENCHMARK_PYTHON` | LIBERO env interpreter | `/path/to/libero/bin/python` |
| `CUDA_VISIBLE_DEVICES` | GPU id for the policy server | `0` |
| `MUJOCO_GL` / `PYOPENGL_PLATFORM` | MuJoCo offscreen GL backend | `osmesa` |
| `LD_LIBRARY_PATH` | NVIDIA libs (xvla script) | `/path/to/nvidia_lib:/usr/lib64` |

Key config fields (see `examples/embodied/<model>/eval/configs/libero/*.yaml`):

- `benchmark.suite` — `libero_object` | `libero_spatial` | `libero_goal` | `libero_10`
- `benchmark.max_tasks` / `benchmark.episodes_per_task` — raise for full-suite / multi-episode eval (e.g. `max_tasks: 0`, `episodes_per_task: 10`)
- `benchmark.continuous_gripper` — `true` for pi05 continuous gripper
- `benchmark.control_mode` — LIBERO OSC: `auto` | `absolute` | `delta`; xvla uses `absolute`
- pi05: `model.action_dim: 7`, `model.action_horizon: 50`, plus a matching `server.dataset_statistics_path`
- xvla: `model.domain_id: 3` (or omit to auto-resolve), `server.chunk_execute_steps: 10`, recommended `benchmark.max_steps: 800`
- GR00T-N1.6: `model.model_type: Gr00tN1d6`, `benchmark.control_mode: delta`, `server.embodiment_tag: libero_panda`, plus `model.base_model_path` / `model.model_name` (Eagle3 processor)

## Verification

Task-success status (2026-07-21):

| Model | LIBERO status | Notes |
|---|---|---|
| pi05 | ✅ task success | pi05 LIBERO fine-tune |
| xvla | ✅ task success | ~94% on libero_object full suite |
| GR00T-N1.6 | ✅ task success | libero_object 5/5 |

Report rates over multiple episodes; for xvla, `examples/embodied/xvla/eval/` historical full runs are under `reports/xvla/libero/libero_weight_object_full/`.
