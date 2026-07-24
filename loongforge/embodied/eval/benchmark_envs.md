# Benchmark Environments

This document records the benchmark runtime environments used by the LoongForge-VLA eval module under `loongforge/embodied/eval`.

## Scope

The benchmark side uses conda environments for isolation. The training/inference base image is outside the scope of this document, so this file only lists benchmark client environments and their key dependency versions.

User-editable benchmark configs live under `examples/embodied/pi05/eval/configs`.

Current benchmark environments:

```text
LIBERO      /path/to/miniconda3/envs/libero
CALVIN      /path/to/miniconda3/envs/calvin
SimplerEnv  /path/to/miniconda3/envs/simplerenv
RoboTwin    /path/to/miniconda3/envs/robotwin
ManiSkill   /path/to/miniconda3/envs/maniskill
```

No benchmark currently uses uv for environment isolation.

## Tool Version

```text
conda 26.3.2
```

## LIBERO

Runtime Python:

```text
/path/to/miniconda3/envs/libero/bin/python
```

Selected dependency versions:

```text
python          3.8.13
torch           2.1.2
numpy           1.24.4
libero          0.1.0 dev
robosuite       1.4.0
mujoco          3.2.3
gym             0.25.2
imageio         2.35.1
imageio-ffmpeg  0.5.1
opencv-python   4.6.0.66
websockets      13.1
msgpack         1.1.1
pyyaml          6.0.3
```

Used by:

```text
examples/embodied/pi05/eval/configs/libero/*.yaml
```

## CALVIN

Runtime Python:

```text
/path/to/miniconda3/envs/calvin/bin/python
```

CALVIN evaluation uses the original-format CALVIN validation dataset and config assets, including `validation/.hydra/merged_config.yaml`, `calvin_models/conf`, and `eval_sequences.json`. LeRobot-format CALVIN datasets are useful for training/statistics but are not sufficient by themselves for official online long-horizon rollout.

Current CALVIN status:

```text
benchmark env:       /path/to/miniconda3/envs/calvin
repo/config assets:  /path/to/calvin
validation dataset:  /path/to/calvin_debug_dataset
pi05 configs:        only smoke.yaml
                     server.random_init: true (link smoke, not task-success)
smoke status:        pass; 1 sequence, first subtask capped at 30 steps
model score status:  need CALVIN-domain ckpt + dataset_statistics.json
```

Used by:

```text
examples/embodied/pi05/eval/configs/calvin/smoke.yaml
```

## SAPIEN Vulkan Runtime

SAPIEN-based benchmarks must use a real Vulkan GPU runtime for visual observations, replay rendering, and headless camera pipelines. This has come up repeatedly while adapting SimplerEnv, RoboTwin, and ManiSkill: `nvidia-smi` seeing GPUs is not enough. Verify the Vulkan ICD separately with `vulkaninfo`, and make sure it lists the NVIDIA device rather than only Mesa `llvmpipe`/`lavapipe` CPU devices.

Required environment variables for SAPIEN visual runners:

```text
LD_LIBRARY_PATH=/path/to/nvidia_lib:/usr/lib64:$LD_LIBRARY_PATH
VK_ICD_FILENAMES=/path/to/nvidia_icd.json
XDG_RUNTIME_DIR=/tmp/runtime-<uid>
```

Example host values:

```text
LD_LIBRARY_PATH=/path/to/nvidia_lib:/usr/lib64:$LD_LIBRARY_PATH
VK_ICD_FILENAMES=/path/to/nvidia_lib/10_nvidia.json
```

These variables must be effective before Python imports SAPIEN, svulkan2, ManiSkill, or the benchmark renderer. Runners that depend on SAPIEN should prepare the variables and re-exec the benchmark Python process once before constructing the environment. Setting `LD_LIBRARY_PATH` only after Python has already started is not sufficient for this renderer stack.

Use this quick check before debugging benchmark adapter code:

```bash
LD_LIBRARY_PATH=/path/to/nvidia_lib:/usr/lib64:${LD_LIBRARY_PATH:-} \
VK_ICD_FILENAMES=/path/to/nvidia_lib/10_nvidia.json \
vulkaninfo
```

Expected signal: `deviceName = NVIDIA ...` and `driverName = NVIDIA`. If the output only shows `llvmpipe` or `lavapipe`, visual SAPIEN tasks may segfault or fail even if state-only rollout works.

## SimplerEnv

Runtime Python:

```text
/path/to/miniconda3/envs/simplerenv/bin/python
```

Selected dependency versions:

```text
python                 3.10.20
numpy                  1.24.4
mani-skill2-real2sim   0.5.3
sapien                 2.2.2
gymnasium              0.29.1
imageio                2.37.3
imageio-ffmpeg         0.6.0
opencv-python          4.13.0.92
websockets             16.0
msgpack                1.1.2
pyyaml                 6.0.3
```

Runtime variables are covered in the shared SAPIEN Vulkan Runtime section above. The SimplerEnv runner prepares `LD_LIBRARY_PATH`, `VK_ICD_FILENAMES`, and `XDG_RUNTIME_DIR`, then re-execs the benchmark Python process once before constructing the environment.

Current SimplerEnv status:

```text
X-VLA WidowX status:   task success after absolute EE controller patch
                       (see examples/embodied/xvla/eval/SIMPLERENV_PATCH_en.md)
                       configs: examples/embodied/xvla/eval/configs/simplerenv/*
pi05 configs:          only widowx_stack_cube_smoke.yaml
                       server.random_init: true (link smoke, not task-success)
                       other Bridge tasks: edit task_name in-file comments
```

Upstream SimplerEnv without the 255isWhite-style absolute EE registration will mis-execute absolute pose actions as deltas. Prefer cloning the fork documented in `SIMPLERENV_PATCH_en.md` or applying the two manual patches.

Used by:

```text
examples/embodied/pi05/eval/configs/simplerenv/*.yaml
examples/embodied/xvla/eval/configs/simplerenv/*.yaml
```

## RoboTwin

Runtime Python:

```text
/path/to/miniconda3/envs/robotwin/bin/python
```

Selected dependency versions:

```text
python            3.10.20
torch             2.4.1
numpy             1.26.4
sapien            3.0.0b1
imageio           2.34.2
imageio-ffmpeg    0.6.0
opencv-python     4.11.0.86
websockets        16.0
msgpack           1.1.2
msgpack-numpy     0.4.8
pyyaml            6.0.3
```

Video logging dependency:

```text
/path/to/miniconda3/envs/robotwin/bin/ffmpeg
ffmpeg version 7.0.2-static
```

The `ffmpeg` executable is provided by the installed `imageio-ffmpeg` package and linked into the `robotwin` env `bin` directory so RoboTwin official video logging can launch `ffmpeg` directly.

Current RoboTwin status (2026-07-21):

```text
Official evaluator:    script/eval_policy.py via robotwin_runner + bridges/robotwin_policy.py
action_bridge modes:   strict_14d | duplicate_7d | pi05_aloha_14d | ee6d_dual
pi05 RoboTwin2:        task success (adjust_bottle demo_clean)
                       action_bridge=pi05_aloha_14d, action_dim=14, action_horizon=32
                       checkpoint example: /path/to/pi0.5_robotwin2
                       stats: examples/embodied/pi05/eval/assets/pi05_robotwin2_dataset_stats.json
                       (from ckpt assets/.../norm_stats.json: state→observation.state, actions→action)
xvla RoboTwin2:        task success (adjust_bottle demo_clean)
                       action_bridge=ee6d_dual, domain_id=6
                       checkpoint example: /path/to/X-VLA-RoboTwin2
Smoke-only:            edit adjust_bottle_smoke*.yaml (random_init / duplicate_7d); no separate YAML
```

Used by:

```text
examples/embodied/pi05/eval/configs/robotwin/*.yaml
examples/embodied/xvla/eval/configs/robotwin/*.yaml
```

## ManiSkill

Runtime Python:

```text
/path/to/miniconda3/envs/maniskill/bin/python
```

Current ManiSkill status:

```text
pi05 configs:          only pick_cube_smoke.yaml
                       server.random_init: true (link smoke, not task-success)
Task:                  PickCube-v1, 7D, pd_ee_delta_pose
Model score status:    need ManiSkill-domain ckpt + dataset_statistics.json
```

Visual smoke uses the shared SAPIEN Vulkan runtime documented above. The ManiSkill runner prepares the NVIDIA ICD and library path before importing ManiSkill/SAPIEN.

Used by:

```text
examples/embodied/pi05/eval/configs/maniskill/*.yaml
```

## Config-to-Environment Mapping

```text
examples/embodied/pi05/eval/configs/libero/*.yaml
examples/embodied/xvla/eval/configs/libero/*.yaml
  benchmark env: /path/to/miniconda3/envs/libero
  runtime:       /path/to/miniconda3/envs/libero/bin/python

examples/embodied/pi05/eval/configs/calvin/*.yaml
examples/embodied/xvla/eval/configs/calvin/*.yaml
  benchmark env: /path/to/miniconda3/envs/calvin
  runtime:       /path/to/miniconda3/envs/calvin/bin/python

examples/embodied/pi05/eval/configs/simplerenv/*.yaml
examples/embodied/xvla/eval/configs/simplerenv/*.yaml
  benchmark env: /path/to/miniconda3/envs/simplerenv
  runtime:       /path/to/miniconda3/envs/simplerenv/bin/python

examples/embodied/pi05/eval/configs/robotwin/*.yaml
examples/embodied/xvla/eval/configs/robotwin/*.yaml
  benchmark env: /path/to/miniconda3/envs/robotwin
  runtime:       /path/to/miniconda3/envs/robotwin/bin/python

examples/embodied/pi05/eval/configs/maniskill/*.yaml
examples/embodied/xvla/eval/configs/maniskill/*.yaml
  benchmark env: /path/to/miniconda3/envs/maniskill
  runtime:       /path/to/miniconda3/envs/maniskill/bin/python
```

## Re-check Commands

Use these commands to refresh version information after environment changes:

```bash
/path/to/miniconda3/bin/conda list -n libero
/path/to/miniconda3/bin/conda list -n calvin
/path/to/miniconda3/bin/conda list -n simplerenv
/path/to/miniconda3/bin/conda list -n robotwin
/path/to/miniconda3/bin/conda list -n maniskill
```
