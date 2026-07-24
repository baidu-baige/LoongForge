# Evaluation Module

> **Note:** This module is still under active development and may see changes or adjustments in the future.

This directory contains the LoongForge-Embodied offline evaluation module. It runs benchmark clients and model policy servers as separate processes connected by a WebSocket/msgpack-numpy RPC protocol.

## Docs

| Doc | Content |
|---|---|
| [user_guide_en.md](user_guide_en.md) | User guide (YAML, benchmarks, bridges, verified runs) |
| [model_integration_guide.md](model_integration_guide.md) | New-model integration checklist (pi05 vs xvla) |
| [predict_action_interface.md](predict_action_interface.md) | **For model owners:** `predict_action` contract, unnorm ownership, postprocess vs model |
| [benchmark_envs.md](benchmark_envs.md) | Per-benchmark conda envs and dependencies |

## Scope

- `loongforge.embodied.eval.protocol`: canonical observation/action/result schema.
- `loongforge.embodied.eval.transport`: WebSocket RPC client/server utilities.
- `loongforge.embodied.eval.adapters`: benchmark-side adapters.
- `loongforge.embodied.eval.bridges`: RoboTwin official evaluator policy bridge (`action_bridge`).
- `loongforge.embodied.eval.orchestrator`: unified YAML entry (`run.py`), server manager, runners.
- `loongforge.embodied.eval.servers.loongforge_server`: LoongForge policy server entrypoint.
- `loongforge.embodied.eval.servers.loongforge_policy`: generic `predict_action` eval policy (`GenericPredictActionPolicy`) and shared data types.
- `loongforge.embodied.eval.servers.eval_server_config`: `EvalServerArgs` dataclass and `parse_eval_server_config` YAML parser.
- `loongforge.embodied.eval.servers.predict_action_interface`: shared model interface checks, action shape normalization, `ACTION_POSTPROCESS_REGISTRY`.
- `loongforge.embodied.eval.servers.mock_policy` / `mock_server`: lightweight protocol mock for tests.
- `loongforge.embodied.eval.factories.registry`: model factory registry (`MODEL_FACTORY_REGISTRY`, `register_factory`, `build_model_spec`).
- `loongforge.embodied.eval.factories.pi05_factory`: PI05 model factory (`PI05ModelFactory`).
- `loongforge.embodied.eval.factories.xvla_factory`: XVLA model factory (`XVLAModelFactory`).

LoongForge training source under the repo root is not patched by this eval module. Model-specific load/build logic lives in `eval/factories`; generic RPC and interface helpers live in `eval/servers`.

## Quick Start: pi05 on LIBERO / CALVIN / SimplerEnv / RoboTwin / ManiSkill

Use YAML as the only user-facing entrypoint:

```bash
cd /path/to/LoongForge
examples/embodied/pi05/eval/run_libero_eval.sh
```

The LIBERO simulator runs in the benchmark environment. The policy server is launched from the YAML `server.python` field, typically a Python 3.12 LoongForge environment with `lerobot==0.5.0`.

## Configs

Environment details for each benchmark are recorded in `benchmark_envs.md`. User-editable eval configs and launch scripts live under `examples/embodied/<model>/eval`. Scripts use public configs with `/path/to/...` placeholders.

Each benchmark ships one public YAML (optional knobs in file-header comments; no extra full-suite files).

### pi05 (`examples/embodied/pi05/eval/`)

- `configs/libero/object_smoke.yaml`: **task-success** (`libero_object`); other suites via YAML comments.
- `configs/robotwin/adjust_bottle_smoke.yaml`: **task-success** (`pi05_aloha_14d`); stats: `assets/pi05_robotwin2_dataset_stats.json`.
- `configs/calvin/smoke.yaml`: **link smoke only**, `server.random_init: true` (no CALVIN-domain pi05 weight).
- `configs/simplerenv/widowx_stack_cube_smoke.yaml`: **link smoke only**, `random_init: true` (no Bridge pi05 weight; task-success is xvla + X-VLA-WidowX).
- `configs/maniskill/pick_cube_smoke.yaml`: **link smoke only**, `random_init: true` (no ManiSkill-domain pi05 weight).

### xvla (`examples/embodied/xvla/eval/`)

- `configs/libero/libero_weight_object_smoke.yaml`: **task-success** (full suite: edit `max_tasks` / `episodes_per_task` in the same YAML; historical full run under `reports/xvla/libero/libero_weight_object_full/`).
- `configs/robotwin/adjust_bottle_smoke.yaml`: **task-success** (`ee6d_dual`).
- `configs/simplerenv/widowx_stack_cube_smoke.yaml`: **task-success** (requires patch in `SIMPLERENV_PATCH_en.md`).
- `configs/calvin/smoke.yaml`: **link smoke** (needs official ABC_D weight for formal scores).
- `configs/maniskill/pick_cube_smoke.yaml`: **link smoke** (no matching open weight).

### Task-success status (2026-07-21)

| Model | LIBERO | SimplerEnv | RoboTwin2 | CALVIN / ManiSkill |
|---|---|---|---|---|
| pi05 | success (finetuned) | no Bridge ckpt | success (`pi05_aloha_14d`) | smoke only |
| xvla | success (~94% object full) | success (WidowX + patch) | success (`ee6d_dual`) | need domain ckpt |

See `user_guide_en.md` §2.1–2.3 / §9 / §11 for examples, open weights, bridges, and verified runs.

The pi05 configs use:

- `model.backend: loongforge`
- `model.model_type: pi05`
- `model.action_dim`, `model.action_horizon`, etc.: Pi05ModelConfig structure fields
- `server.ckpt_path`: checkpoint directory or `model.safetensors`, unless `server.random_init: true` is set for a smoke run
- `server.dataset_statistics_path`: dataset stats passed through to model `predict_action()` for model-owned normalization or unnormalization
- `server.python`: LoongForge Python environment
- `run.output_dir`: runtime output directory under `eval/reports/<model>/<benchmark>/<run_name>/`; `reports/` is generated locally and is not committed
- `run.timestamped_output`: default `true` in the unified entry creates `<yyyymmdd_hhmmss>_<run_tag>` under the parent of `output_dir`; set `false` to reuse a fixed directory

For protocol/debug smoke testing, copy an existing YAML and set `model.backend: mock`; mock backend configs are not shipped as pi05 examples because they are not model-specific.

## Model Interface

LoongForge model servers prefer a shared `predict_action` interface instead of a separate full policy adapter for every model. A reusable `GenericPredictActionPolicy` handles eval RPC behavior: canonical image view selection, `predict_action` invocation, action shape validation, action-dim truncation, chunk caching, latency reporting, metadata, and dataset statistics loading. Model-specific normalization and unnormalization should happen inside the model's `predict_action()` implementation.

Model-specific logic should be kept in a thin factory under `eval/factories/`. The current pi05 path is:

```text
loongforge_server.py
  -> parse_eval_server_config(yaml)      # EvalServerArgs + raw_model_dict
  -> build_model_spec(server_args, raw_model_dict)
       -> build_model_config("pi05", raw_model_dict)  # OmegaConf → Pi05ModelConfig
       -> PI05ModelFactory.build(model_cfg, server_args)
            -> build_model(model_cfg)                 # training-side registry
            -> model.model.load_pretrained(ckpt_path) # training-side weight loader
            -> model.to(device).eval()
  -> _warmup_model(model_spec)           # dummy predict_action() to resolve lazy imports
  -> GenericPredictActionPolicy(...)
  -> PI05Policy.predict_action(images, instructions, state=None, dataset_stats=None)
```

To add a new model, create `eval/factories/<model>_factory.py`, implement a factory class with `model_config_cls` and a `build(model_cfg, server_args)` classmethod, decorate it with `@register_factory("<model_type>")`, and ensure the factory module is imported so registration runs (see existing factories and `eval/factories/registry.py`). No changes needed in `loongforge_server.py`.

Beyond the factory, a new model usually needs model-semantic decisions (action space, absolute vs delta control, action post-processing, proprio state format, normalization ownership, model-specific request fields such as `domain_id`, chunk length, eval horizon). See `model_integration_guide.md` for the full checklist derived from the pi05 and xvla integrations.

RoboTwin-specific `benchmark.action_bridge` values (implemented in `bridges/robotwin_policy.py`; set in YAML only):

- `strict_14d` — default 14D joint actions
- `duplicate_7d` — 7D→14D smoke only
- `pi05_aloha_14d` — openpi Aloha for pi0.5 RoboTwin (adapt_to_pi + delta→abs)
- `ee6d_dual` — official X-VLA RoboTwin EE control

The xvla path additionally uses `benchmark.domain_id`, `benchmark.action_postprocess` (e.g. `ee6d_to_axis_angle`, `ee6d_to_simpler_abs_euler`, `ee6d_to_calvin_abs`) and `server.state_format: ee6d`; configs live under `examples/embodied/xvla/eval/configs/`.

## Outputs

A benchmark run writes:

- `results.jsonl`: one JSON record per episode
- `summary.csv`: task-level summary
- `suite_summary.csv`: suite-level summary
- `artifacts/.../replay_*.gif`: optional replay for direct rollout runners such as LIBERO and SimplerEnv
- `artifacts/.../trace_*.json` or `artifacts/.../trace.json`: optional per-step action trace
- `artifacts/.../videos/*.mp4`: optional RoboTwin official evaluator videos when video logging is enabled
- `policy_server.log`: model server stdout/stderr

## Development Notes

- Keep model framework logic in `eval/factories/<model>_factory.py`. Do not add model-specific code to `loongforge_server.py`.
- Do not modify training-tree LoongForge source for eval-specific compatibility.
- Prefer adding YAML configs over command-line parameter sprawl.
- Use `model.backend` in YAML for backend selection and `benchmark.name` for benchmark selection.
- New model factories should ensure their `predict_action()` tolerates a warmup call with a zero-filled dummy image and an empty instruction string (`_warmup_model` in `loongforge_server.py` runs this before serving).
- After changing shared runner / adapter / bridge / generic policy code, regression-test at least pi05 × LIBERO.
