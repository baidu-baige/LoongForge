# GR00T-N1.6 SimplerEnv Guide

How to reproduce GR00T-N1.6 on SimplerEnv WidowX (Bridge) in the LoongForge eval
harness, including the fork reference and the two model-side fixes that actually
decide success. Companion to X-VLA's [xvla.md](xvla.md).

## Background

The official GR00T-N1.6 SimplerEnv eval uses NVIDIA's pinned fork
[`squarefk/SimplerEnv`](https://github.com/NVIDIA/Isaac-GR00T/blob/main/examples/SimplerEnv/README.md),
whose `ManiSkill2_real2sim` submodule points at
[youliangtan/ManiSkill2_real2sim@c2a9e87](https://github.com/youliangtan/ManiSkill2_real2sim/tree/c2a9e87c186300b694da6f2497dd68d2c347a4b7).

Unlike X-VLA (which needs an *absolute* EE controller patch), **GR00T bridge
emits delta EE actions**, so it runs on the stock upstream delta controller
`arm_pd_ee_target_delta_pose_align2_gripper_pd_joint_pos` — **no controller
patch required**. The only fork-specific pieces are the WidowX **drawer** tasks
(`widowx_open_drawer` / `widowx_close_drawer`), which live in that fork and are
ported below.

The 4 standard WidowX tasks (`widowx_spoon_on_towel`, `widowx_carrot_on_plate`,
`widowx_stack_cube`, `widowx_put_eggplant_in_basket`) already exist in upstream
[simpler-env/SimplerEnv](https://github.com/simpler-env/SimplerEnv), so they need no env changes.

## Patch requirement check

The 4 standard WidowX tasks need no env changes — GR00T-N1.6 emits delta EE actions and runs on the stock upstream delta controller. Only the drawer tasks (`widowx_open_drawer` / `widowx_close_drawer`) need the env port in [Env-side drawer task port](#env-side-drawer-task-port).

## Model-side adaptation

GR00T-N1.6 reuses the shared `predict_action` path (no bespoke policy). Eval
files: `loongforge/embodied/eval/factories/groot_n1_6_factory.py`,
`.../payload_builders/groot_n1_6.py`, and the `oxe_widowx` embodiment added to
`loongforge/embodied/data/datasets/groot_n1_6/transforms/utils.py`.

Two fixes were decisive (both must match the checkpoint's own config files — do
NOT hand-derive from `statistics.json` ranges):

1. **Action normalization = mean-std for pos+rotation** (THE bug). The bridge
   checkpoint's `processor_config.json` declares for `oxe_widowx` action:
   `mean_std_embedding_keys = [x, y, z, roll, pitch, yaw]` (gripper stays
   min-max). If you min-max the euler action over its ±2π range instead, the
   model's output blows up to ~radian-scale rotations and the arm never
   converges — and LIBERO's small action ranges hide the bug. Set
   `mean_std_embedding_keys` on the `oxe_widowx` action `ModalityConfig`.

2. **Proprio built the official way, action rotation passed straight through.**
   The official `WidowXBridgeEnv` (Isaac-GR00T
   `gr00t/eval/sim/SimplerEnv/simpler_env.py`) builds state as
   `[x, y, z, roll, pitch, yaw, pad=0, gripper]` where
   `rpy = mat2euler(quat2mat(eef_quat) @ default_rot.T)` with
   `default_rot = [[0,0,1],[0,1,0],[-1,0,0]]`; the ManiSkill2 obs here lacks
   `agent.eef_pos`, so we reconstruct the ee pose as `base_pose⁻¹ · tcp_pose`
   (verified to reproduce the small in-range euler). Action rotation is the
   model's rpy delta passed **directly** to the delta controller (no
   euler→axis-angle), i.e. eval `rotation_mode: axis_angle`.

Other required runtime bits (see `run_simplerenv_eval.sh`):
- `CUDA_GRAPH_IMPL=local` so the Eagle backbone loads via the repo-local builder
  (offline, from the processor dir's `config.json`) instead of HF remote code.
- `flash_attn` installed in the loongforge env (or set
  `model.use_flash_attention: false` for sdpa).
- `server.chunk_execute_steps: 4` → open-loop n_action_steps=4 (official WidowX
  value; stabilizes the gripper vs replanning every step).

## Env-side drawer task port

Our base env already supports the `dummy_drawer` scene and the `widowx` robot,
and the drawer URDF is primitive geometry (no meshes), so the port is 3 files +
2 registrations (no `base_env` changes):

```bash
SHA=c2a9e87c186300b694da6f2497dd68d2c347a4b7
R=https://raw.githubusercontent.com/youliangtan/ManiSkill2_real2sim/$SHA
MS=/path/to/SimplerEnv/ManiSkill2_real2sim

# 1) env class (registers OpenSmallDrawerCustomInScene-v0 / CloseSmallDrawerCustomInScene-v0)
curl -sL "$R/mani_skill2_real2sim/envs/custom_scenes/open_small_drawer_in_scene.py" \
  -o "$MS/mani_skill2_real2sim/envs/custom_scenes/open_small_drawer_in_scene.py"
# 2) drawer articulation (box/cylinder primitives, no external meshes)
curl -sL "$R/data/custom/small_drawer.urdf" -o "$MS/data/custom/small_drawer.urdf"
# 3) background overlay
curl -sL "$R/data/real_inpainting/bridge_small_drawer.png" \
  -o "$MS/data/real_inpainting/bridge_small_drawer.png"
```

Then register + map:
- Add `from . import open_small_drawer_in_scene` to
  `$MS/mani_skill2_real2sim/envs/custom_scenes/__init__.py`.
- Add to `loongforge/embodied/eval/adapters/simplerenv.py` `TASK_TO_ENV_NAME`:
  ```python
  "widowx_open_drawer":  "OpenSmallDrawerCustomInScene-v0",
  "widowx_close_drawer": "CloseSmallDrawerCustomInScene-v0",
  ```
- **Expose `tcp_pose` in the drawer env's obs (REQUIRED for proprio).** The
  ported drawer env does not override `_get_obs_extra`, so `obs.extra` is empty
  and the adapter's proprio silently falls back to a zero state (fine for the
  state-dropout model on a forgiving task, but it tanks the precision drawer:
  the arm barely moves, gripper never grips). Add to `OpenSmallDrawerInSceneEnv`
  (mirroring `grasp_single`/`move_near`):
  ```python
  from mani_skill2_real2sim.utils.sapien_utils import vectorize_pose
  def _get_obs_extra(self):
      return OrderedDict(tcp_pose=vectorize_pose(self.tcp.pose))
  ```
  Verify after: `env.reset()` → `obs["extra"]["tcp_pose"]` exists and the
  reconstructed proprio is in the `oxe_widowx` training ranges (small euler,
  in-range xyz), not zeros.

Drawer env facts: `robot=widowx`, `scene_name=dummy_drawer`,
`control_mode=arm_pd_ee_target_delta_pose_align2_gripper_pd_joint_pos`,
overlay `bridge_small_drawer.png`, env registers `max_episode_steps=120` but the
official client passes `--max-episode-steps 300` (use 300, not the registration
default), success = drawer joint qpos ≥0.10 (open) / ≤0.04 (close).

## Assets

- Weights: [nvidia/GR00T-N1.6-bridge](https://huggingface.co/nvidia/GR00T-N1.6-bridge) (2-shard safetensors + `statistics.json`
  carrying the `oxe_widowx` state/action stats).
- Eagle processor dir (`config.json` = Eagle3VL + tokenizer), e.g.
  [nvidia/Eagle-Block2A-2B-v2](https://huggingface.co/nvidia/Eagle-Block2A-2B-v2) / [aravindhs-NV/eagle3-processor-groot-n1d6](https://huggingface.co/aravindhs-NV/eagle3-processor-groot-n1d6).

## Eval config

Set `benchmark.prepackaged_config: true`. This makes the runner build the env
via `build_maniskill2_env(env_name, prepackaged_config=True, ...)` and reset
with only a per-episode seed — the env then applies its **official
visual-matching configuration** (scene, overlay set, robot, control_mode,
lighting) and randomizes overlay + robot/object init per episode via its
`episode_rng`. This is exactly what `simpler_env.make(task)` does officially
(`simpler_env/__init__.py` sets `prepackaged_config=True`).

Do NOT hand-set a single `scene_name` / `rgb_overlay_path` / `robot_init_x/y`:
those are off-distribution and tank the success rate. Measured on eggplant:
- hand-set fixed scene/overlay/init → **36%** (20 eps)
- official `prepackaged_config: true` → **100%** (20 eps; official N1.6 = 89%)

Because GR00T flow-matching sampling is stochastic (unseeded `torch.randn`),
always report the rate over ≥20 episodes, not a single run.

## Run

```bash
# eggplant (standard bridge task)
bash examples/embodied/groot_n1_6/eval/run_simplerenv_eval.sh

# drawer tasks: set benchmark.task_name: widowx_open_drawer | widowx_close_drawer,
# benchmark.scene_name: dummy_drawer,
# rgb_overlay_path: .../real_inpainting/bridge_small_drawer.png,
# max_steps: 300
```

Matching eval YAML fields:
```yaml
benchmark:
  control_mode: arm_pd_ee_target_delta_pose_align2_gripper_pd_joint_pos  # stock upstream, no patch
  rotation_mode: axis_angle          # pass model rpy delta straight to controller
model:
  model_type: Gr00tN1d6
  state_encoding: simpler_widowx     # base^-1*tcp + default_rot -> euler; gripper from qpos
  action_encoding: simpler_abs_euler # == adapter action_space -> IdentityDecoder
  use_flash_attention: true
server:
  embodiment_tag: oxe_widowx
  dataset_statistics_path: /path/to/GR00T-N1.6-bridge/statistics.json
  chunk_execute_steps: 4             # official n_action_steps=4
```

## Results

Official `prepackaged_config: true` + all model-side fixes, 20 episodes each:
- `widowx_put_eggplant_in_basket`: **20/20 (100%)** (official N1.6 = 89%)
- `widowx_open_drawer` / `widowx_close_drawer`: run with the ported drawer env
  (official N1.6 = 95% / 73%).

Pitfalls that cost time here (don't repeat):
1. Hand-deriving the `oxe_widowx` modality/normalization from `statistics.json`
   ranges (missed `mean_std_embedding_keys`) → catastrophic ±2π action blowup.
   Read the checkpoint's `processor_config.json`.
2. LIBERO (small action ranges) passed while SimplerEnv (±2π) failed — a
   passing small-range benchmark does NOT prove the normalization is correct.
3. Hand-set scene/overlay/init instead of `prepackaged_config: true` → 36% vs
   100%.
4. Our ManiSkill2 obs has no `agent.eef_pos`; reconstruct proprio from
   `base_pose⁻¹ · tcp_pose` (+ `default_rot`), not the raw tcp euler.
5. A factory `predict_action` wrapper that declares `**kwargs` must swallow and
   NOT forward them (else `cfg_scale` reaches the model and crashes).
6. Replan-every-step is unstable for grasping; use `chunk_execute_steps: 4`.
7. The ported drawer env didn't expose `tcp_pose` in `obs.extra` → proprio was
   silently zero → the precision drawer failed (arm barely moves, no grip),
   while zero-proprio-tolerant eggplant still hit 100%. Add `_get_obs_extra`
   returning `tcp_pose` to the drawer env; verify `obs.extra.tcp_pose` exists.
8. Don't use the env's registered `max_episode_steps` (drawer=120) — match the
   official client's `--max-episode-steps 300`.

## References

- NVIDIA GR00T SimplerEnv README + benchmark table:
  [https://github.com/NVIDIA/Isaac-GR00T/blob/main/examples/SimplerEnv/README.md](https://github.com/NVIDIA/Isaac-GR00T/blob/main/examples/SimplerEnv/README.md)
- Official WidowX obs/action wrapper: `gr00t/eval/sim/SimplerEnv/simpler_env.py`
- Pinned fork: [squarefk/SimplerEnv](https://github.com/squarefk/SimplerEnv) → [youliangtan/ManiSkill2_real2sim@c2a9e87](https://github.com/youliangtan/ManiSkill2_real2sim/tree/c2a9e87c186300b694da6f2497dd68d2c347a4b7)
- Bridge action normalization: the checkpoint's `processor_config.json`
  (`oxe_widowx` → `mean_std_embedding_keys`).
