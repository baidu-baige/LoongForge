# GR00T-N1.7 LIBERO Guide

What GR00T-N1.7 needs from the **LIBERO checkout** and its dependency stack.
Model-side adaptation is in [model_integration.md](../../model_integration.md);
the SimplerEnv counterpart is
[patches/simplerenv/groot_n1_7.md](../simplerenv/groot_n1_7.md).

## No benchmark source patch

The LIBERO checkout is pinned at `8f1084e` (matches the official gitlink) with a
clean worktree. Nothing in LIBERO or robosuite is modified, so unlike
`patches/TransformerEngine_v2.9` there is no diff to apply.

## Requirement: transformers 4.57.3 on the policy side

Point `server.python` at an env with **transformers 4.57.3** — the version the
official GR00T-N1.7 release targets. This is a correctness requirement, not an
optimization, and it is the single largest factor in LIBERO success rate. Nothing
warns you when it is wrong: the model loads, every episode completes, and the
score is simply lower.

### Per-task comparison on `libero_10`

5 episodes per task, same weights, same harness, same benchmark env; only the
policy-side transformers version differs. `task_id` as recorded in
`results.jsonl` (0-indexed):

| transformers | t0 | t1 | t2 | t3 | t4 | t5 | t6 | t7 | t8 | t9 | suite |
|---|---|---|---|---|---|---|---|---|---|---|---|
| **4.57.3** | 3/5 | 5/5 | 4/5 | 5/5 | 5/5 | 5/5 | 5/5 | 4/5 | 5/5 | 5/5 | **46/50 (92%)** |
| 5.3.0 | 1/5 | 2/5 | 4/5 | 0/5 | 0/5 | 4/5 | 0/5 | 0/5 | 0/5 | 0/5 | 11/50 (22%) |

5.3.0 is the transformers version LoongForge's own dependency list installs, so it
is the version this document reports throughout. An earlier 5.x release was
measured too and reproduces the same per-task signature, so the divergence is
**5.x-wide** rather than one bad release.

The damage is **task-specific, not uniform**: six of ten tasks collapse to
exactly 0/5 under either 5.x release while t2/t5 stay near their 4.57.3 level.
That shape is why the ~20% suite average read for months like a weak checkpoint on
a hard suite — a suite mean cannot distinguish "slightly worse everywhere" from
"a subset is completely broken". The other three suites degrade too, less
dramatically.

### Where the divergence comes from

The Qwen3-VL backbone forward, not the eval harness — which is why it shows up on
every benchmark this model runs. The evidence is a per-module tensor diff, not a
success-rate inference: one fixed observation, the flow-matching noise replaced by
a shape-keyed deterministic tensor so both runs sample identically, bf16, and only
the policy env's transformers version changed. Relative error is
`max|a-b| / max|b|` against the 4.57.3 run:

| stage | 4.57.3 vs 5.3.0 | note |
|---|---|---|
| `state_encoder` output | 0.000e+00 | bit-identical — input and noise are pinned |
| `action_encoder`, step 1 | 0.000e+00 | bit-identical |
| `vl_embeds` (backbone → DiT) | 82.1% | where the divergence enters |
| DiT output, step 1 | 30.9% | |
| DiT output, step 4 | 74.5% | amplified along the sampling loop |
| decoded action chunk | 342.9% | mean/std -0.0105/0.0859 → +0.1591/0.3462 |

The two tensors that precede the backbone are bit-identical, so the 82% at
`vl_embeds` cannot be an input or seeding artifact. Downstream, the DiT
self-attends over the whole action sequence conditioned on those embeddings, so
the perturbation is amplified along the sampling loop rather than averaged out,
and the resulting trajectories sit close enough to the success threshold that some
tasks fail every time.

A second 5.x release run through the same probe lands within 0.2%–1.1% of these
tensors — bf16 magnitude (machine epsilon ≈ 3.9e-3) — which is why the whole
series behaves alike. That residual is reproducible rather than jitter: repeating
a run in the same env reproduces every tensor exactly (noise floor `0.000e+00`),
so the two releases are in the same regime without being the same code path.
The 4.57.3 path was separately confirmed against the official Isaac-GR00T stack;
[patches/simplerenv/groot_n1_7.md](../simplerenv/groot_n1_7.md) has that
comparison.

**GR00T-N1.6 is not affected.** The same probe run on N1.6 (Bridge checkpoint,
`oxe_widowx`) is **bit-identical across 4.57.3 and 5.3.0** — every stage,
including the decoded action chunk, at `0.000e+00`. N1.6 uses the Eagle3 backbone,
whose forward never enters the transformers Qwen3-VL implementation. So this pin
is specific to N1.7: do not apply it to N1.6 configs, and do not read an N1.6
score difference across transformers versions as a version effect.

### Our own dependency list installs the degraded version

LoongForge's `pyproject.toml` installs `transformers==5.3.0`, so an env built
straight from it lands in the 11/50 regime. Isaac-GR00T's `pyproject.toml`
declares `transformers==4.57.3`; matching it restores 46/50. Keep a separate
aligned env for this eval rather than changing the repo-wide version, which other
models depend on.

## The benchmark-side interpreter is not in the YAML

`server.python` in the eval YAML selects the **policy** env only. The benchmark
process is whatever `BENCHMARK_PYTHON` points at, so the two sides are chosen in
different places — the policy env in the config, the benchmark env on the command
line:

```bash
BENCHMARK_PYTHON=<path-to-libero-env>/bin/python \
CONFIG=examples/embodied/groot_n1_7/eval/configs/libero/libero_goal.yaml \
bash examples/embodied/groot_n1_7/eval/run_libero_eval.sh
```

## Policy side

`server.python` must name an env whose transformers is 4.57.3 (see the
requirement above). The env name itself is not special — `loongforge` in the
public configs — only its transformers version is.

## Other declared deviations

- **Rendering**: `MUJOCO_GL=osmesa` / `PYOPENGL_PLATFORM=osmesa` (CPU rendering;
  the GPU is left to the policy server).
- **`max_steps: 720`** in all four suite configs, above the adapter's per-suite
  defaults (`libero.py:22` — spatial 220, object 280, goal 300, libero_10 520).
  `benchmark.max_steps` overrides them (`orchestrator/config.py:170`).
- **`num_steps_wait: 0`**, `control_mode: delta`, `continuous_gripper: false`,
  `chunk_execute_steps: 8`.
- **`env.libero_config_path`** points at an explicit config directory instead of
  the default `~/.libero`.
- **`action_horizon` is never set in the YAML.** It is the DiT flow-matching
  sequence length fixed by the weights (40, and the dataclass default); an
  override is not validated anywhere and silently changes every DiT output. Same
  rule as SimplerEnv.

## Reproducibility

Rollouts are not reproducible at a fixed env seed: the flow-matching noise comes
from an unseeded global `torch.randn`
(`loongforge/embodied/model/groot_n1_7/modeling_groot_n1_7.py:718`), matching
official. One LIBERO episode with an identical seed has flipped fail@720 ->
success@273. Small deltas between runs are not signal.

## Measured rates

5 episodes per task, all 10 tasks per suite (50 per suite), policy on
**transformers 4.57.3**. Per task, `task_id` as reported in `results.jsonl`:

| suite | t0 | t1 | t2 | t3 | t4 | t5 | t6 | t7 | t8 | t9 | suite |
|---|---|---|---|---|---|---|---|---|---|---|---|
| `libero_goal` | 5/5 | 5/5 | 5/5 | 5/5 | 5/5 | 5/5 | 5/5 | 5/5 | 4/5 | 5/5 | 49/50 (98%) |
| `libero_spatial` | 5/5 | 5/5 | 5/5 | 5/5 | 4/5 | 5/5 | 4/5 | 5/5 | 5/5 | 5/5 | 48/50 (96%) |
| `libero_object` | 5/5 | 5/5 | 5/5 | 4/5 | 4/5 | 5/5 | 5/5 | 5/5 | 5/5 | 5/5 | 48/50 (96%) |
| `libero_10` | 3/5 | 5/5 | 4/5 | 5/5 | 5/5 | 5/5 | 5/5 | 4/5 | 5/5 | 5/5 | 46/50 (92%) |
