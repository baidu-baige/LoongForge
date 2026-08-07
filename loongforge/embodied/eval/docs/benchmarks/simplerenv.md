# SimplerEnv WidowX Evaluation

SimplerEnv is a real-to-sim robot manipulation evaluation benchmark. This page covers the WidowX (Bridge) setups used by the LoongForge eval module: 4 standard Bridge tasks (`widowx_spoon_on_towel`, `widowx_carrot_on_plate`, `widowx_stack_cube`, `widowx_put_eggplant_in_basket`) plus 2 drawer tasks (`widowx_open_drawer` / `widowx_close_drawer`, GR00T fork only).

Weight coverage: **xvla** reaches task success with absolute EE control; **GR00T-N1.6** (bridge) reaches task success with delta control on the stock env; **pi05** has no Bridge weights (connectivity only).

## Step 0: Download weights

| Model | Weights |
|---|---|
| xvla | [2toINF/X-VLA-WidowX](https://huggingface.co/2toINF/X-VLA-WidowX) |
| GR00T-N1.6 | [nvidia/GR00T-N1.6-bridge](https://huggingface.co/nvidia/GR00T-N1.6-bridge) (safetensors + `statistics.json`) |
| pi05 | none released (connectivity only) |

## Step 1: Environment setup

### Standard environment

Install SimplerEnv following the official [SimplerEnv repository](https://github.com/simpler-env/SimplerEnv) instructions (with the `ManiSkill2_real2sim` submodule), then install the additional dependencies:

```bash
pip install websockets msgpack msgpack-numpy pyyaml
pip install numpy==1.24.4   # downgrade numpy for simulator compatibility
```

⚠️ Common issues:

- **Vulkan / SAPIEN.** SimplerEnv renders with SAPIEN and needs a real NVIDIA Vulkan ICD. `nvidia-smi` is not enough — verify with `vulkaninfo` and expect `deviceName = NVIDIA ...`. If only `llvmpipe`/`lavapipe` appears, camera images and replays are unreliable.
- Set `LD_LIBRARY_PATH`, `VK_ICD_FILENAMES`, and `XDG_RUNTIME_DIR` **before** SAPIEN is imported; the runner re-execs the process so these take effect.
- `numpy==1.24.4` is pinned for compatibility — do not upgrade.

### Model-specific environment changes

#### 1. xvla — absolute EE control

X-VLA emits absolute EE poses, but upstream SimplerEnv WidowX only ships delta control. Use the official fork [255isWhite/SimplerEnv](https://github.com/255isWhite/SimplerEnv) (recommended) or apply the two manual patches: register controller `arm_pd_ee_target_base_pose` in `WidowXDefaultConfig`, and make `PDEEPoseController.compute_target_pose` parse the non-delta branch's rotation as Euler xyz. Delta control modes (`arm_pd_ee_target_delta_pose_align2_*`) are unaffected. See the [SimplerEnv patch guide](../patches/simplerenv/xvla.md).

#### 2. GR00T-N1.6 (drawer tasks) — drawer task port

The 2 drawer tasks (`widowx_open_drawer` / `widowx_close_drawer`) use fork-specific env files: ports 3 files from the NVIDIA fork (drawer env class, `small_drawer.urdf`, `bridge_small_drawer.png`), registers the two drawer envs, and exposes `tcp_pose` in the drawer env obs for proprio. Port steps in the [GR00T-N1.6 SimplerEnv guide](../patches/simplerenv/groot_n1_6.md).

The benchmark dataset/assets come from the SimplerEnv repo itself; no separate dataset download is needed beyond the environment install.

## Step 2: Run evaluation

Two conda environments: the **SimplerEnv client** env and the **model server** env. The run scripts and eval YAMLs ship with `/path/to/...` placeholders — fill them in before running:

```bash
cd /path/to/LoongForge-VLA

# xvla (task success template)
examples/embodied/xvla/eval/run_simplerenv_eval.sh

# GR00T-N1.6 (public template)
bash examples/embodied/groot_n1_6/eval/run_simplerenv_eval.sh
```

Environment variables:

| Variable | Meaning | Default |
|---|---|---|
| `CONFIG` | eval YAML config path | `<model>/eval/configs/simplerenv/widowx_stack_cube_smoke.yaml` |
| `BENCHMARK_PYTHON` | SimplerEnv env interpreter | `/path/to/simplerenv/bin/python` |
| `CUDA_VISIBLE_DEVICES` | GPU id for the policy server | `0` |
| `LD_LIBRARY_PATH` | NVIDIA libs (must include `/usr/lib64`) | `/path/to/nvidia_lib:/usr/lib64` |
| `VK_ICD_FILENAMES` | NVIDIA Vulkan ICD | `/path/to/nvidia_icd.json` |
| `XDG_RUNTIME_DIR` | runtime dir for Vulkan | `/tmp/runtime-<uid>` |

Key config fields (xvla, see `examples/embodied/xvla/eval/configs/simplerenv/widowx_stack_cube_smoke.yaml`):

- `benchmark.control_mode` — `arm_pd_ee_target_base_pose_gripper_pd_joint_pos` (needs the absolute EE env, see [model-specific environment changes](#model-specific-environment-changes))
- `benchmark.max_steps` — `1200` (official horizon)
- `benchmark.task_name` / `robot_setup` / `scene_name` / `rgb_overlay_path` — switch among the 4 standard Bridge tasks
- `model.domain_id` — `0` (WidowX)
- GR00T-N1.6: `benchmark.prepackaged_config: true` (official visual-matching config), `server.chunk_execute_steps: 4`, `model.use_flash_attention: true`. GR00T runs on the stock upstream delta controller — no env change for the 4 standard tasks (see the [GR00T-N1.6 SimplerEnv guide](../patches/simplerenv/groot_n1_6.md)).

## Verification

| Model | Status | Notes |
|---|---|---|
| xvla | ✅ task success | WidowX + absolute EE control (needs [model-specific environment changes](#model-specific-environment-changes)) |
| GR00T-N1.6 | ✅ task success | eggplant 20/20 (100%) with `prepackaged_config`; official N1.6 = 89% |
| pi05 | connectivity only | `server.random_init: true` |

Report rates over ≥20 episodes — GR00T flow-matching sampling is stochastic.
