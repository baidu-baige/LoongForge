# X-VLA SimplerEnv (WidowX) Eval: SimplerEnv Version Requirements and Patches

## Background

The official X-VLA SIMPLER evaluation ([evaluation/simpler](https://github.com/2toinf/X-VLA/tree/main/evaluation/simpler)) does **not** use upstream
[simpler-env/SimplerEnv](https://github.com/simpler-env/SimplerEnv). It uses the official fork
[255isWhite/SimplerEnv](https://github.com/255isWhite/SimplerEnv) (including the `ManiSkill2_real2sim`
submodule pinned at commit `54ae2e0e`). From the official README:

> The evaluation follows [SimplerEnv](https://github.com/255isWhite/SimplerEnv), with **minor
> environment modifications** to support **absolute end-effector (EE) control**.

Reason: X-VLA emits **absolute EE poses** (target position + Euler orientation in the base frame), while upstream
SimplerEnv WidowX only provides **delta** control modes. Mismatch leads to two failure modes:

| Environment state | Symptom |
| --- | --- |
| Upstream + config `arm_pd_ee_target_base_pose_*` | `KeyError` at env create — control mode missing; eval never starts |
| Upstream + fallback delta mode (e.g. `arm_pd_ee_target_delta_pose_align2_*`) | Runs, but absolute actions are applied as deltas; arm flies out on the first step; success rate stays 0 |

Measured on 2026-07-21: before the patch, stack_cube failed for all steps through 120/300; after the patch and switching to absolute control, episode 0 succeeded in 55 steps.

## How to check whether your SimplerEnv needs the patch

Run (replace python and paths):

```bash
python -c "
import sys
sys.path.insert(0, '/path/to/SimplerEnv')
sys.path.insert(0, '/path/to/SimplerEnv/ManiSkill2_real2sim')
from mani_skill2_real2sim.agents.configs.widowx.defaults import WidowXDefaultConfig
print('arm_pd_ee_target_base_pose_gripper_pd_joint_pos' in WidowXDefaultConfig().controllers)
"
```

- Prints `True`: nothing to do (official fork or already patched).
- Prints `False`: apply one of the two options below.

## Option 1 (recommended): use the official fork

```bash
git clone https://github.com/255isWhite/SimplerEnv.git --recurse-submodules
cd SimplerEnv/ManiSkill2_real2sim && pip install -e .
cd .. && pip install -e .
```

Point `env.simplerenv_root` in the eval YAML at that directory.

## Option 2: patch upstream (two edits, both under the `ManiSkill2_real2sim` submodule)

These changes strictly follow the official fork (`simpler-env/ManiSkill2_real2sim@54ae2e0e`).
Delta control modes are unchanged (pi0.5 and other models on the delta path are unaffected).

### Patch 1: register absolute EE pose controller

File: `mani_skill2_real2sim/agents/configs/widowx/defaults.py`

Add a controller config in `WidowXDefaultConfig.controllers` and insert it into `_C["arm"]`:

```python
# 255isWhite/SimplerEnv fork (X-VLA): absolute EE pose control in base frame
arm_pd_ee_target_base_pose = PDEEPoseControllerConfig(
    *arm_common_args, frame="base", use_target=True, use_delta=False, **arm_common_kwargs
)

_C["arm"] = dict(
    ...,  # keep existing entries
    arm_pd_ee_target_base_pose=arm_pd_ee_target_base_pose,
)
```

The combined control mode used by the eval YAML is `arm_pd_ee_target_base_pose_gripper_pd_joint_pos`.

### Patch 2: parse rotation as Euler xyz on the non-delta branch

File: `mani_skill2_real2sim/agents/controllers/pd_ee_pose.py`

In `PDEEPoseController.compute_target_pose`, on the `use_delta=False` branch, parse rotation as Euler xyz instead of rotvec (axis-angle):

```python
else:
    assert self.config.frame == "base", self.config.frame
    target_pos, target_rot = action[0:3], action[3:6]
    # 255isWhite/SimplerEnv fork (X-VLA): absolute action rotation is Euler xyz
    target_quat = Rotation.from_euler("xyz", target_rot).as_quat()[[3, 0, 1, 2]]
    target_pose = sapien.Pose(target_pos, target_quat)
```

Why this is required: the X-VLA client protocol outputs Euler xyz (rot6d → euler, then add bias `[0, π/2, 0]`; see post-processor `ee6d_to_simpler_abs_euler`). Upstream dead code used `from_rotvec` on the same three numbers, which yields a totally different orientation — position may look roughly right while the gripper is twisted, IK fails often, and grasps fail.

> Note: on stock upstream, no control mode reaches this non-delta branch (dead code), so the edit does not change any existing upstream behavior.

## Matching eval YAML fields

See `configs/simplerenv/widowx_stack_cube_smoke_internal.yaml`:

```yaml
benchmark:
  control_mode: arm_pd_ee_target_base_pose_gripper_pd_joint_pos  # needs patch 1
  max_steps: 1200          # official horizon
  domain_id: 0             # WidowX (Bridge)
  action_postprocess: ee6d_to_simpler_abs_euler  # rot6d->euler + [0, pi/2, 0], grip<0.91
  robot_init_x: 0.147
  robot_init_y: 0.028
```

## Rollback

Patches are not committed to git; inspect or restore with:

```bash
git -C /path/to/SimplerEnv/ManiSkill2_real2sim diff      # inspect
git -C /path/to/SimplerEnv/ManiSkill2_real2sim checkout -- .  # restore
```

## References

- Official eval client: [evaluation/simpler/WidowX/client_blocks.py](https://github.com/2toinf/X-VLA/blob/main/evaluation/simpler/WidowX/client_blocks.py)
- Official fork controller: [ManiSkill2_real2sim@54ae2e0e defaults.py](https://github.com/simpler-env/ManiSkill2_real2sim/blob/54ae2e0e9422807d060aa15ff7c04970d38d3cf8/mani_skill2_real2sim/agents/configs/widowx/defaults.py)
- Official numbers (WidowX Visual Matching): Spoon 100 / Carrot 91.7 / Blocks 95.8 / Eggplant 95.8, Avg 95.8
