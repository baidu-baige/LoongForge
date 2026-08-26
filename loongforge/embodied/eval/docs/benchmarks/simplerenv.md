# SimplerEnv WidowX Evaluation

SimplerEnv is a real-to-sim robot manipulation evaluation benchmark. This page covers the WidowX (Bridge) setups used by the LoongForge eval module: 4 standard Bridge tasks (`widowx_spoon_on_towel`, `widowx_carrot_on_plate`, `widowx_stack_cube`, `widowx_put_eggplant_in_basket`) plus 2 drawer tasks (`widowx_open_drawer` / `widowx_close_drawer`, GR00T fork only).

Weight coverage: **xvla** reaches task success with absolute EE control; **GR00T-N1.6** (bridge) reaches task success with delta control on the stock env; **pi05** has no Bridge weights (connectivity only).

## Step 0: Download weights

| Model | Weights |
|---|---|
| xvla | [2toINF/X-VLA-WidowX](https://huggingface.co/2toINF/X-VLA-WidowX) |
| GR00T-N1.6 | [nvidia/GR00T-N1.6-bridge](https://huggingface.co/nvidia/GR00T-N1.6-bridge) (safetensors + `statistics.json`) plus an Eagle processor dir (`config.json` = Eagle3VL + tokenizer), e.g. [nvidia/Eagle-Block2A-2B-v2](https://huggingface.co/nvidia/Eagle-Block2A-2B-v2) |
| GR00T-N1.7 | [nvidia/GR00T-N1.7-SimplerEnv-Bridge](https://huggingface.co/nvidia/GR00T-N1.7-SimplerEnv-Bridge) (safetensors + `statistics.json` + `processor_config.json`) plus the Cosmos-Reason2-2B backbone dir |
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

#### 2. GR00T (N1.6 / N1.7) — use NVIDIA's pinned fork

Both GR00T models run against the SimplerEnv fork that NVIDIA/Isaac-GR00T pins as its `external_dependencies/SimplerEnv` submodule ([squarefk/SimplerEnv](https://github.com/squarefk/SimplerEnv)). No env-side patch is needed there. Upstream is not a substitute: it registers neither drawer task (`widowx_open_drawer` / `widowx_close_drawer`) and does not expose `agent.eef_pos`, which is what the official `WidowXBridgeEnv` builds proprio from. Details in the [GR00T SimplerEnv guide](../patches/simplerenv/groot_n1_6.md). Note X-VLA needs the *other* fork (above), so the two checkouts coexist.

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
- GR00T-N1.6: `benchmark.prepackaged_config: true` (official visual-matching config), `server.chunk_execute_steps: 4` (official `n_action_steps`; replanning every step is unstable for grasping), `model.use_flash_attention: true` (or `false` for sdpa if `flash_attn` is missing). The run script also exports `CUDA_GRAPH_IMPL=local` so the Eagle backbone loads via the repo-local builder (offline, from the processor dir's `config.json`) instead of HF remote code. GR00T runs on the stock delta controller — no env change at all (see the [GR00T SimplerEnv guide](../patches/simplerenv/groot_n1_6.md)).

## Verification

| Model | Status | Notes |
|---|---|---|
| xvla | ✅ task success | WidowX + absolute EE control (needs [model-specific environment changes](#model-specific-environment-changes)) |
| GR00T-N1.6 | ✅ task success | per-task rates below |
| GR00T-N1.7 | ✅ task success | 85/120 over six WidowX tasks (20 eps each). Requires **transformers 4.57.3** on the policy side — on 5.x the same run scores 35/120. See the [GR00T-N1.7 SimplerEnv guide](../patches/simplerenv/groot_n1_7.md) |
| pi05 | connectivity only | `server.random_init: true` |

GR00T-N1.6 (bridge weights) under the official protocol — `prepackaged_config: true`,
`max_steps: 300`, `chunk_execute_steps: 4`, one process per `run.episode_idx`,
10 episodes per task:

| Task | Measured | Official N1.6 (as published) |
|---|---|---|
| `widowx_open_drawer` | 10/10 | 95% |
| `widowx_close_drawer` | 10/10 | 73% |
| `widowx_spoon_on_towel` | 7/10 | 64.5% |
| `widowx_put_eggplant_in_basket` | 9/10 | 89% |
| `widowx_carrot_on_plate` | 5/10 | 65.5% |
| `widowx_stack_cube` | 1/10 | — |
| total | 42/60 | — |

xvla official reference (WidowX Visual Matching, as published): Spoon 100 /
Carrot 91.7 / Blocks 95.8 / Eggplant 95.8, Avg 95.8.

Report rates as successes/episodes over ≥30 episodes — GR00T flow-matching sampling
is unseeded, so the same config over the same episodes does not reproduce: eggplant
`episode_idx` 0-9 scored 8/10, 9/10 and 6/10 across three runs (the same task over
0-19 scored 17/20), and carrot's 0-9 subset moved 2/10 → 6/10 between sweeps. Never
compare two configs at different sample sizes, and do not quote single-episode smoke
runs as rates.

Do not hand-set a single `scene_name` / `rgb_overlay_path` / `robot_init_x/y` in
place of `prepackaged_config: true`: the env then applies its official
visual-matching configuration and randomizes overlay + robot/object init per
episode from `run.seed + run.episode_idx`, which is what `simpler_env.make(task)`
does officially. A hand-fixed scene is off-distribution — measured on eggplant over
20 episodes, 7/20 with a hand-fixed scene vs 20/20 with the official config.
