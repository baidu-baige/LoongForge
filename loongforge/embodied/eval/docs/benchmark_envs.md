# Benchmark Environments

This document records the benchmark runtime environments used by the LoongForge-VLA eval module under `loongforge/embodied/eval`.

**For installation, follow the official benchmark homepages** — each benchmark page links to its official repository. This document is not an install guide; it records the dependency versions of the environments currently used by the eval module, so they can be compared / reproduced when debugging.

**Environment isolation.** The benchmark client and the policy server run as separate processes with different dependencies (see [Architecture](../README.md#architecture)). We use conda environments for this isolation; other approaches — e.g. running the two sides in separate pods — work too, but then you are responsible for the inter-process communication yourself (the two sides must reach each other over the configured host/port).

The version lists below are the verified combinations of the internal envs.

## Tool Version

```text
conda 26.3.2
```

## LIBERO

Runtime Python:

```text
/path/to/conda/envs/libero/bin/python
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
/path/to/conda/envs/calvin/bin/python
```

CALVIN evaluation uses the original-format CALVIN validation dataset and config assets, including `validation/.hydra/merged_config.yaml`, `calvin_models/conf`, and `eval_sequences.json`. LeRobot-format CALVIN datasets are useful for training/statistics but are not sufficient by themselves for official online long-horizon rollout.

Current CALVIN status:

```text
benchmark env:       /path/to/conda/envs/calvin
repo/config assets:  /path/to/calvin
validation dataset:  /path/to/calvin_debug_dataset
pi05 configs:        only smoke.yaml
                     server.random_init: true (connectivity only, not task success)
connectivity status:  pass; 1 sequence, first subtask capped at 30 steps
model score status:  need CALVIN-domain weights + dataset_statistics.json
```

Used by:

```text
examples/embodied/pi05/eval/configs/calvin/smoke.yaml
```

## SimplerEnv

Runtime Python:

```text
/path/to/conda/envs/simplerenv/bin/python
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

The SimplerEnv runner prepares `LD_LIBRARY_PATH`, `VK_ICD_FILENAMES`, and `XDG_RUNTIME_DIR` (SAPIEN Vulkan requirements, see the [SimplerEnv page](benchmarks/simplerenv.md) ⚠️ common issues), then re-execs the benchmark Python process once before constructing the environment.

Current SimplerEnv status:

```text
X-VLA WidowX status:   task success after absolute EE controller patch
                       (see patches/simplerenv/xvla.md)
                       configs: examples/embodied/xvla/eval/configs/simplerenv/*
GR00T-N1.6 status:     task success (eggplant 20/20, official prepackaged_config)
                       uses the stock upstream delta controller, no env change
                       configs: examples/embodied/groot_n1_6/eval/configs/simplerenv/*
pi05 configs:          only widowx_stack_cube_smoke.yaml
                       server.random_init: true (connectivity only, not task success)
                       other Bridge tasks: edit task_name in-file comments
```

Upstream SimplerEnv without the 255isWhite-style absolute EE registration will mis-execute absolute pose actions as deltas. Prefer cloning the fork documented in `patches/simplerenv/xvla.md` or applying the two manual patches.

Used by:

```text
examples/embodied/pi05/eval/configs/simplerenv/*.yaml
examples/embodied/xvla/eval/configs/simplerenv/*.yaml
```

## RoboTwin

Runtime Python:

```text
/path/to/conda/envs/robotwin/bin/python
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
/path/to/conda/envs/robotwin/bin/ffmpeg
ffmpeg version 7.0.2-static
```

The `ffmpeg` executable is provided by the installed `imageio-ffmpeg` package and linked into the `robotwin` env `bin` directory so RoboTwin official video logging can launch `ffmpeg` directly.

Current RoboTwin status (2026-07-21):

```text
Official evaluator:    script/eval_policy.py via robotwin_runner + bridges/robotwin_policy.py
action_bridge modes:   pi05_aloha_14d | ee6d_dual
pi05 RoboTwin2:        task success (adjust_bottle demo_clean)
                       action_bridge=pi05_aloha_14d, action_dim=14, action_horizon=32
                       weight example: /path/to/pi0.5_robotwin2
                       stats: examples/embodied/pi05/eval/assets/pi05_robotwin2_dataset_stats.json
                       (from the weight's assets/.../norm_stats.json: state→observation.state, actions→action)
xvla RoboTwin2:        task success (adjust_bottle demo_clean)
                       action_bridge=ee6d_dual, domain_id=6
                       weight example: /path/to/X-VLA-RoboTwin2
Connectivity only:      edit adjust_bottle_smoke*.yaml (random_init); no separate YAML
```

Used by:

```text
examples/embodied/pi05/eval/configs/robotwin/*.yaml
examples/embodied/xvla/eval/configs/robotwin/*.yaml
```

## ManiSkill

Runtime Python:

```text
/path/to/conda/envs/maniskill/bin/python
```

Current ManiSkill status:

```text
pi05 configs:          only pick_cube_smoke.yaml
                       server.random_init: true (connectivity only, not task success)
Task:                  PickCube-v1, 7D, pd_ee_delta_pose
Model score status:    need ManiSkill-domain weights + dataset_statistics.json
```

Visual smoke needs the SAPIEN Vulkan runtime (see the [SimplerEnv page](benchmarks/simplerenv.md) ⚠️ common issues). The ManiSkill runner prepares the NVIDIA ICD and library path before importing ManiSkill/SAPIEN.

Used by:

```text
examples/embodied/pi05/eval/configs/maniskill/*.yaml
```

