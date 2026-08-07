# ManiSkill Evaluation

ManiSkill is a SAPIEN-based robot manipulation benchmark. The LoongForge eval module's default task is `PickCube-v1` with `pd_ee_delta_pose` control (7D action).

Current status: **no released ManiSkill-domain weights for pi05/xvla**. Shipped configs run as connectivity checks (`server.random_init: true`), not scores.

Note: xvla cannot run with real ManiSkill weights yet — it emits absolute EE poses while ManiSkill only ships the delta controller (`pd_ee_delta_pose`); the required absolute-EE control mode is not implemented, so the runner raises `NotImplementedError` for a non-`random_init` xvla run (see the [model integration guide §4.2](../model_integration.md#42-control-mode)).

## Step 0: Download weights

| Model | Weights |
|---|---|
| pi05 | none released (connectivity only) |
| xvla | none released (connectivity only) |

## Step 1: Environment setup

### Standard environment

Install ManiSkill following the official repository instructions (SAPIEN-based), then install the additional dependencies:

```bash
pip install websockets msgpack pyyaml
```

⚠️ Common issues:

- **Vulkan / SAPIEN.** Verify a real NVIDIA Vulkan ICD with `vulkaninfo` (expect `deviceName = NVIDIA ...`); set `LD_LIBRARY_PATH` / `VK_ICD_FILENAMES` before ManiSkill/SAPIEN is imported.

## Step 2: Run evaluation

Run from inside the **benchmark** environment. The run scripts and eval YAMLs ship with `/path/to/...` placeholders — fill them in before running:

```bash
cd /path/to/LoongForge-VLA
examples/embodied/pi05/eval/run_maniskill_eval.sh    # pi05 (connectivity only)
examples/embodied/xvla/eval/run_maniskill_eval.sh    # xvla (connectivity only)
```

Environment variables: `CONFIG`, `BENCHMARK_PYTHON` (ManiSkill env interpreter), `CUDA_VISIBLE_DEVICES`, plus the SAPIEN Vulkan variables.

Key config fields (see `examples/embodied/<model>/eval/configs/maniskill/pick_cube_smoke.yaml`):

- `benchmark.task_name` — `PickCube-v1` (default)
- `benchmark.control_mode` — `pd_ee_delta_pose` (7D action)
- `server.random_init` — `true` in shipped configs. With ManiSkill-domain weights set `false`, fill `ckpt_path` / `dataset_statistics_path` (state dim is typically 8: 7 arm joints + gripper width)

## Verification

| Model | Status | Notes |
|---|---|---|
| pi05 | connectivity only | needs ManiSkill-domain weights + `dataset_statistics.json` |
| xvla | connectivity only | needs ManiSkill-domain weights + absolute-EE control mode (not implemented, see above) |
