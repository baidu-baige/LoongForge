# GR00T-N1.7 SimplerEnv Guide

The SimplerEnv checkout is NVIDIA's fork, same as N1.6 — see
[groot_n1_6.md](groot_n1_6.md). What N1.7 additionally needs is one hard runtime
requirement: **transformers 4.57.3**. Model-side adaptation is in
[model_integration.md](../../model_integration.md); measured rates are in
[benchmarks/simplerenv.md](../../benchmarks/simplerenv.md).

## Requirement: transformers 4.57.3

LoongForge's own dependency list installs **transformers 5.3.0**, and on 5.3.0
this port scored far below NVIDIA's published numbers: the six-task WidowX total
came out at 35/120. Nothing in the run warns about it — the model loads and every
episode completes, the score is simply lower.

The cause is a numerical divergence in the **Qwen3-VL backbone**, localised by a
per-module tensor diff and written up in
[patches/libero/groot_n1_7.md](../libero/groot_n1_7.md). It is backbone-level, so
it applies to every benchmark this model runs, and it affects the whole 5.x
series. Isaac-GR00T's `pyproject.toml` declares `transformers==4.57.3`; matching
it restores 85/120. So point `server.python` at an env whose transformers is
4.57.3 — a correctness requirement, not an optimization. **GR00T-N1.6 is
unaffected and needs no switch**, for the reason given in the same document.

Cost in success rate, 20 episodes per task (seed 0, `prepackaged_config`,
`max_steps=300`, `chunk_execute_steps=4`, same checkpoint, only the server env
differs):

| task | 4.57.3 | 5.3.0 |
|---|---|---|
| `widowx_open_drawer` | 20/20 | 20/20 |
| `widowx_close_drawer` | 20/20 | 9/20 |
| `widowx_spoon_on_towel` | 17/20 | 3/20 |
| `widowx_put_eggplant_in_basket` | 13/20 | 0/20 |
| `widowx_carrot_on_plate` | 11/20 | 2/20 |
| `widowx_stack_cube` | 4/20 | 1/20 |
| total | **85/120 (70.8%)** | **35/120 (29.2%)** |

`widowx_put_eggplant_in_sink` is excluded: upstream marks it incomplete
(invisible dummy target plane, a single `xy_configs` grid point), so its 2/20
measures the task definition rather than the policy. `open_drawer` being
unaffected is consistent with it having the loosest success threshold —
successful episodes finish in ~20 steps.

## Impact boundary

- **No env-side patch, and no controller patch.** The SimplerEnv checkout itself
  needs no edits. Which checkout to use and why — the fork pinned by
  NVIDIA/Isaac-GR00T, shared by both GR00T models — is in
  [groot_n1_6.md](groot_n1_6.md).
- Model-side changes are almost entirely additive: one modality block per
  embodiment in
  `loongforge/embodied/data/datasets/groot_n1_7/transforms/groot_transform.py`
  and one branch in
  `loongforge/embodied/eval/payload_builders/groot_n1_7.py`. The one
  non-additive edit is in the same transform file: `libero_sim`'s `gripper` state
  slot becomes `{"start": 6, "end": 8}`, because `robot0_gripper_qpos` is the 2D
  finger qpos and the released checkpoint statistics carry two values for that
  group. `libero_sim` behaviour is unchanged in practice — verified by re-running
  `libero_spatial` task 0 x10 (10/10, no change).

## Why the fork

See [groot_n1_6.md](groot_n1_6.md) — the drawer task registrations and
`agent.eef_pos` are the reasons, and they apply to both GR00T models.

## Config parity with NVIDIA official

Audited field-by-field against Isaac-GR00T n1.7 (`examples/SimplerEnv/README.md`
+ `gr00t/eval/sim/`); all 18 `processor_kwargs` keys of
`nvidia/GR00T-N1.7-SimplerEnv-Bridge` are accounted for.

- `prepackaged_config: true` — official `simpler_env.make()` calls
  `gym.make(env_name, obs_mode="rgbd", prepackaged_config=True)`;
  `ENVIRONMENT_MAP` extra kwargs are empty for all 7 widowx tasks.
- `max_steps: 300` — counts **inner env** steps (`MultiStepWrapper` measures
  `len(self.reward)`), matching `--max-episode-steps 300`.
- `chunk_execute_steps: 4` <-> `--n-action-steps 4`. Verified in a trace: fresh
  inference at steps 0, 4, 8, ... (75 inferences / 300 steps, 0 consecutive
  identical raw actions). Official slices the *first* `action_horizon` steps of
  the chunk, same as ours.
- `action_horizon` — **do not set it in the eval YAML.** The key lands in
  `GrootN1d7Config.action_horizon`, which is the DiT flow-matching sequence
  length fixed by the weights (40 in both released checkpoints, and the dataclass
  default). Nothing validates an override: a shorter sequence still runs, because
  the position embedding table is `max_seq_len`-sized, and it silently changes
  every DiT output. The decoded chunk length is a separate quantity and comes from
  `len(delta_indices)` = 8 (`_split_action_chunk`).
- `rotation_mode: axis_angle` is **pass-through**. The official
  `WidowXBridgeEnv.step` does no euler->axis-angle conversion; setting `euler`
  here would insert a transform official does not have.
- Single view `image_0`, resized to `256x256` (cv2 `INTER_LINEAR`) in the payload
  builder — the letterbox-pad + shortest-edge + center-crop stage runs inside
  `predict_action`, not client-side.
- Gripper: command binarized in the adapter as `2*(a>0.5)-1`, matching
  `_postprocess_gripper`.
- **One deliberate divergence:** we call `env.reset(seed=run.seed +
  run.episode_idx)`; official passes no seed. Ours is reproducible for the
  *layout*; see the determinism caveat below.

## Determinism caveat

Rollouts are **not** reproducible even at a fixed env seed: flow-matching noise
comes from a global unseeded `torch.randn`
(`loongforge/embodied/model/groot_n1_7/modeling_groot_n1_7.py:718`). Confirmed
in practice — one LIBERO episode with an identical seed flipped fail@720 ->
success@273.

This matches official (`gr00t/model/gr00t_n1d7/gr00t_n1d7.py:335` is also
unseeded, and `gr00t/eval/` performs no seeding at all), so it is not a port
defect. Consequence: any 10-episode rate carries unquantified variance — do not
read small deltas as signal.

## Parity with the official stack (verified)

On transformers 4.57.3 the port is aligned with NVIDIA's stack end-to-end; the
following were checked and need no further work:

- **Tensor-level single-step parity.** Both stacks run on a byte-identical
  observation with the flow-matching noise pinned to a shared `(1, 40, 132)`
  tensor. Every stage agrees to bf16 rounding — `state_feat` max-abs 2.0e-3,
  `vl_embeds` rel 4.1e-3, per-denoising-step `dit_out` rel <= 1.2e-2, and the
  decoded 8x7 action chunk differs by 6.3e-4. `modules/dit.py` has no functional
  delta against official `gr00t/model/modules/dit.py`.
- **First frame and instruction.** Image byte-identical (md5
  `498b1dab3ccc2b10d2dc89592d0d2e5a`, 256x256, mean 111.1676), state to 8
  decimals, instruction from `env.get_language_instruction()`.
- **Replan cadence.** 75 inferences over 300 steps (fresh inference at steps
  0, 4, 8, ...), zero consecutive-identical actions: the server-side
  `chunk_execute_steps` truncation plus policy cache behaves as open-loop 4,
  matching `MultiStepWrapper` with `n_action_steps=4`.
- **Gripper proprio.** Matches official `get_gripper_closedness()` to 5 decimals
  at three poses; `robot.get_qlimits()[-2:] == [[0.015, 0.037], [0.015, 0.037]]`
  confirms the hardcoded `WIDOWX_FINGER_QMIN/QMAX`.
- **EE pose.** The `eef_pos` branch and the `inv(base_pose) @ tcp_pose` fallback
  differ by 8.9e-08.
- **`formalize_language`.** `re.sub(r"[^\w\s]", "", s.lower())`,
  character-identical to official and applied unconditionally in both.
- **Env / step semantics.** Passing a `seed` into the inner `env.reset()` and
  setting the inner `_max_episode_steps` to 300 (official uses 10000 and lets the
  wrapper truncate) both leave the official harness at 2/2, so the shared
  `simplerenv_runner._reset_env` / `_build_env` need no change.

Two things that look like problems but are not: an episode burning all 300 steps
is not a failure signal (official's successful episodes also run the full 300),
and gripper-command wobble is not a decode bug (unseeded flow-matching noise
makes any two rollouts diverge from step 0).

Remaining gap versus the official numbers is `widowx_stack_cube` (4/20), which is
a policy-strength question rather than a port defect.

## References

- Official WidowX obs/action wrapper: `gr00t/eval/sim/SimplerEnv/simpler_env.py`
  (`WidowXBridgeEnv`, `_process_observation`, `_postprocess_gripper`)
- Official step semantics: `gr00t/eval/sim/wrapper/multistep_wrapper.py`
- NVIDIA SimplerEnv README + benchmark table:
  [https://github.com/NVIDIA/Isaac-GR00T/blob/main/examples/SimplerEnv/README.md](https://github.com/NVIDIA/Isaac-GR00T/blob/main/examples/SimplerEnv/README.md)
- Pinned fork: [squarefk/SimplerEnv](https://github.com/squarefk/SimplerEnv)
