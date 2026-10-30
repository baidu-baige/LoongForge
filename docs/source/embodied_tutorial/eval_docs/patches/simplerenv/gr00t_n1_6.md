# GR00T-N1.6 SimplerEnv Guide

What GR00T-N1.6 needs from the **SimplerEnv environment itself** on WidowX
(Bridge). Model-side adaptation (proprio / normalization / action encoding) is in
[model_integration.md](../../model_integration.md); measured rates are in
[benchmarks/simplerenv.md](../../benchmarks/simplerenv.md). Companion to X-VLA's
[xvla.md](xvla.md).

## Impact boundary

- **No controller patch.** GR00T bridge emits delta EE actions and runs on the
  stock upstream delta controller
  `arm_pd_ee_target_delta_pose_align2_gripper_pd_joint_pos`, so the 4 standard
  WidowX tasks (`widowx_spoon_on_towel`, `widowx_carrot_on_plate`,
  `widowx_stack_cube`, `widowx_put_eggplant_in_basket`) need **zero** env change
  on upstream [simpler-env/SimplerEnv](https://github.com/simpler-env/SimplerEnv).
- The only env work is porting the 2 **drawer** tasks (`widowx_open_drawer` /
  `widowx_close_drawer`), which exist solely in NVIDIA's pinned fork.
- The port is additive: 3 new files, 1 import line, and an `_get_obs_extra` added
  to the new drawer env class. It touches no shared controller or `base_env`
  code, so pi05 / X-VLA and the 4 standard tasks are unaffected.
- Written against [youliangtan/ManiSkill2_real2sim@c2a9e87](https://github.com/youliangtan/ManiSkill2_real2sim/tree/c2a9e87c186300b694da6f2497dd68d2c347a4b7)
  (the submodule pin of [`squarefk/SimplerEnv`](https://github.com/NVIDIA/Isaac-GR00T/blob/main/examples/SimplerEnv/README.md)).

## Patch requirement check

```bash
python -c "
import sys
sys.path.insert(0, '/path/to/SimplerEnv')
sys.path.insert(0, '/path/to/SimplerEnv/ManiSkill2_real2sim')
import mani_skill2_real2sim.envs  # noqa
from mani_skill2_real2sim.utils.registration import REGISTERED_ENVS
print('OpenSmallDrawerCustomInScene-v0' in REGISTERED_ENVS)
"
```

- Prints `True`: the drawer envs are present (NVIDIA fork, or already ported).
- Prints `False`: only needed for the drawer tasks — apply the port below.

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
  and the adapter's proprio silently falls back to a zero state (tolerable for a
  forgiving task, but it tanks the precision drawer: the arm barely moves, the
  gripper never grips). Add to `OpenSmallDrawerInSceneEnv` (mirroring
  `grasp_single` / `move_near`):
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
overlay `bridge_small_drawer.png`, success = drawer joint qpos ≥0.10 (open) /
≤0.04 (close). The env registers `max_episode_steps=120`, but the official client
passes `--max-episode-steps 300` — set `benchmark.max_steps: 300`, not the
registration default.

## Eval config

Drawer-specific fields (everything else is shared with the standard tasks):

```yaml
benchmark:
  task_name: widowx_open_drawer        # | widowx_close_drawer
  scene_name: dummy_drawer
  rgb_overlay_path: /path/to/SimplerEnv/ManiSkill2_real2sim/data/real_inpainting/bridge_small_drawer.png
  max_steps: 300                       # not the env's registered 120
```

## Rollback

The port is not committed to the SimplerEnv checkout; inspect or restore with:

```bash
git -C /path/to/SimplerEnv/ManiSkill2_real2sim status      # the 3 new files
git -C /path/to/SimplerEnv/ManiSkill2_real2sim checkout -- .  # revert tracked edits
```

The 3 downloaded files are untracked — remove them by hand to fully revert.

## References

- NVIDIA GR00T SimplerEnv README + benchmark table:
  [https://github.com/NVIDIA/Isaac-GR00T/blob/main/examples/SimplerEnv/README.md](https://github.com/NVIDIA/Isaac-GR00T/blob/main/examples/SimplerEnv/README.md)
- Official WidowX obs/action wrapper: `gr00t/eval/sim/SimplerEnv/simpler_env.py`
- Pinned fork: [squarefk/SimplerEnv](https://github.com/squarefk/SimplerEnv) → [youliangtan/ManiSkill2_real2sim@c2a9e87](https://github.com/youliangtan/ManiSkill2_real2sim/tree/c2a9e87c186300b694da6f2497dd68d2c347a4b7)
