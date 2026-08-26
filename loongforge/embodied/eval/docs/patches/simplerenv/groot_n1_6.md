# GR00T SimplerEnv Guide (WidowX / Bridge)

What the GR00T models need from the **SimplerEnv checkout** on WidowX (Bridge).
Both GR00T-N1.6 and GR00T-N1.7 share one checkout, so this is the env-side
reference for both; N1.7 additionally has a runtime dependency pin, see
[groot_n1_7.md](groot_n1_7.md). Model-side adaptation (proprio / normalization /
action encoding) is in [model_integration.md](../../model_integration.md);
measured rates are in [benchmarks/simplerenv.md](../../benchmarks/simplerenv.md).
Companion to X-VLA's [xvla.md](xvla.md).

## Which checkout

Use the fork that NVIDIA/Isaac-GR00T pins as its `external_dependencies/SimplerEnv`
submodule: [squarefk/SimplerEnv](https://github.com/squarefk/SimplerEnv), whose
`ManiSkill2_real2sim` in turn pins
[youliangtan/ManiSkill2_real2sim@c2a9e87](https://github.com/youliangtan/ManiSkill2_real2sim/tree/c2a9e87c186300b694da6f2497dd68d2c347a4b7).
Upstream [simpler-env/SimplerEnv](https://github.com/simpler-env/SimplerEnv) is
not a substitute:

- it registers neither WidowX drawer task
  (`OpenSmallDrawerCustomInScene-v0` / `CloseSmallDrawerCustomInScene-v0`);
- it does not expose `agent.eef_pos` / `agent.ee_pose`.

Verify the checkout in `env.simplerenv_root`:

```bash
python -c "
import sys
sys.path.insert(0, '$SIMPLERENV_ROOT')
sys.path.insert(0, '$SIMPLERENV_ROOT/ManiSkill2_real2sim')
import mani_skill2_real2sim.envs  # noqa
from mani_skill2_real2sim.utils.registration import REGISTERED_ENVS
print('OpenSmallDrawerCustomInScene-v0' in REGISTERED_ENVS)  # True on the fork
"
```

## Why `agent.eef_pos` matters

The official `WidowXBridgeEnv` builds proprio from `obs.agent.eef_pos`
(`[x,y,z, quat_wxyz(4), gripper]`). On the fork our payload builders read that
same attribute — see `_encode_simpler_widowx` in
`loongforge/embodied/eval/payload_builders/groot_n1_6.py`, which prefers
`state_raw["eef_pos"]` and only then falls back to reconstructing
`inv(base_pose) @ tcp_pose` from `obs.extra.tcp_pose`.

The fallback is not equivalent in coverage. It needs the env to implement
`_get_obs_extra`, and the drawer envs do not, so `obs.extra` comes back empty and
proprio silently degrades to a zero state — tolerable on a forgiving task, fatal
on the precision drawer (the arm barely moves, the gripper never grips). Where
both paths are available they agree to `8.9e-08` (per-channel diff all 0), so on
the four standard tasks preferring `eef_pos` is hygiene rather than a fix.

## Impact boundary

- **No env-side patch and no controller patch.** GR00T bridge emits delta EE
  actions and runs on the stock
  `arm_pd_ee_target_delta_pose_align2_gripper_pd_joint_pos`. Nothing in the
  SimplerEnv checkout needs editing, so there is nothing to roll back either.
- **pi05 / X-VLA do not share this checkout.** They need
  `arm_pd_ee_target_base_pose`, which the fork's `WidowXDefaultConfig` does not
  define, so they run against a separate upstream checkout with their own
  controller edits. Two SimplerEnv checkouts coexisting on one machine is
  expected — point `env.simplerenv_root` deliberately and do not "unify" them.
- Measured on GR00T-N1.6, six WidowX tasks x 10 episodes, only
  `env.simplerenv_root` differing: **42/60 on the fork** against 40/60 on an
  upstream checkout carrying a hand-applied drawer port. Per task the two are
  within +-3 episodes, i.e. inside the run-to-run noise of an unseeded policy;
  `widowx_open_drawer` is 10/10 both ways, which is the case that exercises the
  `eef_pos` path.

## Eval config

Both models mirror the official protocol, so the env owns the scene:

```yaml
benchmark:
  prepackaged_config: true   # official simpler_env.make() forces it; the env then
                             # owns scene / rgb_overlay / robot init and randomizes
                             # per episode from `run.seed + run.episode_idx`
  max_steps: 300             # official --max-episode-steps; NOT the drawer envs'
                             # registered max_episode_steps=120
  control_freq: 3            # drawer tasks; the four standard tasks use 5
```

Drawer env facts: `robot=widowx`, `scene_name=dummy_drawer`,
`control_mode=arm_pd_ee_target_delta_pose_align2_gripper_pd_joint_pos`,
success = drawer joint qpos >=0.10 (open) / <=0.04 (close).

`widowx_put_eggplant_in_sink` is registered but upstream marks it incomplete
(`TODO` in both `simpler_env/__init__.py` and `put_on_in_scene.py`): the target is
an invisible dummy plane and `xy_configs` has a single grid point, so object init
is effectively fixed across episodes. Keep it out of reported totals.

## References

- NVIDIA GR00T SimplerEnv README + benchmark table:
  [https://github.com/NVIDIA/Isaac-GR00T/blob/main/examples/SimplerEnv/README.md](https://github.com/NVIDIA/Isaac-GR00T/blob/main/examples/SimplerEnv/README.md)
  (the fork is pinned in that repo's `.gitmodules`, not named in the README)
- Official WidowX obs/action wrapper: `gr00t/eval/sim/SimplerEnv/simpler_env.py`
