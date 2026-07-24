---
name: vla-model-eval-adapter
description: Use this skill when adapting a VLA/model backend into the LoongForge embodied eval system after benchmark runners already exist. Trigger on requests to add or refactor model eval integration, validate a model predict_action interface, create model factory/loader code, wire model.backend/server routing, write eval YAML, handle action normalization or dataset_statistics, align eval config with a model's official inference configuration, or reproduce pi05/xvla-style integration. At run start, collect required inputs from the user (target model + benchmark, model package location, checkpoint/tokenizer/dataset-statistics paths). Before writing configs, search the web for the model's official inference config (repo, paper, HuggingFace card), diff it against the existing eval YAML/ModelConfig, and ask the user before adding missing fields; then iterate the eval code to follow the official config. Prefer the shared predict_action contract plus GenericPredictActionPolicy before creating a bespoke policy adapter. Prefer eval-only changes; put protocol logic in action_bridge/action_postprocess. By default, ship one public + one _internal YAML pair (and matching run scripts) per supported benchmark (LIBERO, CALVIN, SimplerEnv, RoboTwin, ManiSkill), using random_init link smoke when no domain weight exists, unless the user narrows scope.
---

# VLA Model Eval Adapter

Use this skill to connect a model backend to the existing LoongForge embodied evaluation stack. The benchmark runners and benchmark adapters are assumed to already exist. The preferred model-side architecture is now:

```text
model factory/loader
  -> model instance exposing predict_action(images, instructions, state=None, dataset_stats=None)
  -> GenericPredictActionPolicy
  -> PolicyServer RPC
```

Create a bespoke policy adapter only when the model cannot reasonably expose the shared `predict_action` interface or needs custom RPC behavior that `GenericPredictActionPolicy` cannot cover.

## Mental model

Keep four boundaries separate:

```text
benchmark adapter:
  benchmark obs/action <-> canonical eval schema
  owns benchmark-native state, action conversion, and debug/trace metadata
  examples: adapters/libero.py, adapters/maniskill.py, adapters/simplerenv.py

model factory/loader:
  model config/import/checkpoint/tokenizer/device/dtype/random-init/metadata
  returns a model object implementing predict_action(...)
  example: PI05ModelFactory in factories/pi05_factory.py

generic eval policy:
  RPC payload handling, image view selection, predict_action invocation,
  action chunk cache, action shape validation, action dim truncation,
  dataset statistics loading, latency, metadata.
  View packing (see `_build_image_input`): primary/head required; if both
  left and right exist → 3 views [head, left, right] (RoboTwin); else at most
  one wrist/right/left as the second view. Models must accept dynamic view
  count (e.g. `num_images = len(images[0])`), never hardcode 2 or 3.
  Action unnormalization is NOT done here; it is the model's responsibility
  inside predict_action().
  example: GenericPredictActionPolicy in servers/loongforge_policy.py

eval bridges / action postprocess (protocol-specific, not in the model package):
  RoboTwin: bridges/robotwin_policy.py via benchmark.action_bridge
    (pi05_aloha_14d | ee6d_dual | strict_14d | duplicate_7d).
  Runner-side action conversion: benchmark.action_postprocess keys from
    ACTION_POSTPROCESS_REGISTRY in servers/predict_action_interface.py
    (ee6d_to_axis_angle, ee6d_to_simpler_abs_euler, ee6d_to_calvin_abs, ...).
  Proprio layout for models that need ee6d state: server.state_format: ee6d.
```

The adapter state boundary matters. Benchmark-native structured state stays in `canonical_obs["state"]` for eval/debug/trace. Only model-ready state goes into `canonical_obs["model_state"]`, which runners forward as RPC payload `state`, and which eventually reaches `predict_action(state=...)`.

```text
adapter.model_state -> RPC payload.state -> predict_action(state=...)
```

Do not clean, drop, or reinterpret benchmark-native dict state inside a model factory. That belongs in the benchmark adapter or payload boundary.

## Expected deliverables

Produce concrete files whenever implementation is requested. A complete model integration usually includes:

- An official inference-config research note: which official sources were consulted (official repo, paper, HuggingFace model card, official deployment/eval scripts), which official inference parameters were found, and a field-by-field comparison against the existing eval YAML. Missing fields must be explicitly confirmed or rejected by the user before being added, and confirmed fields land in eval-side code only (factory, eval policy, eval YAML) — never in the training-side `ModelConfig`.
- A model factory/loader that returns a model instance and metadata, preferably via `PredictActionModelSpec` or an equivalent local pattern.
- A model object exposing `predict_action(images, instructions, state=None, dataset_stats=None)`. This method is implemented by the training team in the model package's `xxxx_modeling.py`; this skill consumes it as-is and does not reimplement inference. Thin eval-side wrappers are allowed only for type/coercion plumbing (e.g. xvla `domain_id` int → LongTensor). If the method is missing or does not match the contract, report that to the user/training team as a blocker.
- Interface validation using `validate_predict_action_model()` and `call_predict_action()` from `loongforge/embodied/eval/servers/predict_action_interface.py`.
- Reuse of `GenericPredictActionPolicy` when the shared interface is sufficient.
- A bespoke `servers/<model>_policy.py` only if shared `predict_action` is not a good fit.
- `loongforge/embodied/eval/servers/<model>_server.py` or existing server entrypoint reuse if applicable.
- `loongforge/embodied/eval/orchestrator/server_manager.py` routing only when adding a new `model.backend` that cannot reuse existing LoongForge routing.
- Demo YAML configs for every already-supported benchmark under `examples/embodied/<model>/eval/configs/<benchmark>/`, unless the user narrows scope. Prefer the **one public + one `_internal` pair per benchmark** layout used by pi05/xvla (see step 6); do not ship extra full/regression YAMLs unless the user asks.
- Matching run scripts under `examples/embodied/<model>/eval/` (`run_<benchmark>_eval.sh` + `run_<benchmark>_eval_internal.sh`) when following the pi05/xvla pattern.
- A smoke-test matrix covering every generated benchmark demo, with one command and expected artifact path per benchmark.
- Executed smoke tests for every generated benchmark demo that can run in the current environment; do not stop at generating YAML/matrix files.
- Documentation updates in the relevant eval docs, especially `README.md`, `user_guide_en.md`, `benchmark_envs.md`, `loongforge_eval_summary.md`, or `predict_action_interface.md`.

If the user asks for a dry-run, generation test, or re-application test, write generated artifacts to a temp directory such as `/tmp/<model>_eval_adapter_*` and do not overwrite repo files except the skill itself when explicitly requested.

## Required inputs before starting

At the start of a run, collect the following from the user before doing any implementation work. If any required item is missing and cannot be unambiguously discovered from the repo, stop and ask the user for it instead of guessing.

Required (blockers if missing):

- Target model + target benchmark(s), e.g. "xvla on LIBERO". Also whether this is a new backend, a variant of an existing backend, or a refactor. If the user does not narrow benchmarks, confirm whether to generate demo YAMLs for all supported benchmarks.
- Model package location: `loongforge/embodied/model/<model>/` containing `xxxx_modeling.py` (training-team `predict_action`) and `model_configuration_<model>.py` (read-only `ModelConfig`). If the model package is not in the repo yet, report it as a blocker.
- Asset paths: `ckpt_path`, `tokenizer_path`, `dataset_statistics_path`. Without checkpoint or `dataset_statistics.json`, validation stops at random-init smoke; state this to the user and confirm whether to proceed anyway.

Strongly recommended (ask for them; proceed with discovery only if the user defers):

- Official source links for the model + benchmark: official GitHub repo, HuggingFace model card, paper, ideally the official eval script/config for that benchmark. These become mandatory if web access is unavailable.
- Runtime environments: the conda env name for each target benchmark client, and the model server Python path for YAML `server.python`.
- The existing eval YAML to use as the diff baseline under `examples/embodied/<model>/eval/configs/`, or confirmation that configs are generated from scratch.

Optional (ask once; if unanswered, apply the defaults stated below):

- Pre-authorized decisions for the human-intervention gates: e.g. "add missing fields at official recommended values without asking". Default: no pre-authorization — every gate stops and asks.
- Port range constraints (default: pick unique unused ports), `benchmark.domain_id` if the model requires one (e.g. xvla; default: the known per-benchmark IDs listed in step 6 — note SimplerEnv WidowX/Bridge is **0**, not 1), dry-run vs writing to the repo (default: write to the repo), and GPU/Vulkan readiness for SAPIEN-based benchmarks (default: probe with `vulkaninfo` per step 8).

## Workflow

1. Collect the required inputs above from the user, then inspect the target model and benchmark context.
   - Find existing model configs, server entries, policy adapters, and model inference APIs.
   - Identify whether this is a new backend, a variant of an existing backend, or a refactor of an existing integration.
   - Confirm target benchmarks are already wired in the orchestrator.
   - Discover the already-supported benchmark set from runner/config directories rather than assuming only one benchmark.

2. Research the official inference configuration online and reconcile it with the existing eval config.
   - Search the web for the model's official sources: the official GitHub repo (inference/eval scripts, `config.json`, deployment docs), the paper, and the HuggingFace model card. Prefer official first-party sources over blog posts or third-party reproductions.
   - If web access is unavailable, stop and ask the user to intervene: either restore web access or provide the official inference config manually. Do not silently substitute guesses or unverified local sources for the official config.
   - The research target is the specific model plus the benchmark to be evaluated (e.g. "xvla on LIBERO"); official inference configs often differ per benchmark, so look for the official setup used for that benchmark.
   - Extract the official inference-time parameters, typically: action horizon / chunk size, action dim, state dim, image resolution and preprocessing, expected camera views, prompt/instruction format, normalization scheme (q01/q99, mean/std, none) and its statistics source, denoising/sampling steps, dtype, and any control-frequency or replan-interval assumptions.
   - Compare field-by-field against the existing eval YAML (`model:` and `server:` sections). Read the training-side `XxxModelConfig` dataclass only to learn which `model:` fields exist and take effect; it is training-side code and must NOT be modified by this skill. Produce a three-column diff: official value, current eval value, and status (match / mismatch / missing in eval config).
   - For fields present in the official config but missing from the eval YAML, stop and ask the user before adding them. Present each missing field with its official value, its likely impact on eval correctness, and a recommendation. Only add fields the user confirms; record rejected fields and the reason in the research note.
   - Route each confirmed missing field to an eval-side home: if the training-side `ModelConfig` already declares it, set it via the YAML `model:` section; otherwise put it in the `server:` section (extending `EvalServerArgs` if needed) or handle it inside the eval-side factory / eval policy. Never add fields to the training-side `ModelConfig` to make a YAML knob work.
   - For fields whose eval value mismatches the official value, flag them the same way; do not silently overwrite user-tuned values.
   - Also compare against eval-pipeline capabilities, not just YAML fields: e.g. camera-view packing (1 primary + optional wrist, or 3-view head/left/right for RoboTwin), `action_postprocess` / `action_bridge` coverage, or a state format the runner does not forward. When the official config exceeds what the eval pipeline can do, stop and ask the user to decide whether to add the corresponding capability to the eval-side code; do not extend the pipeline or accept the deviation on your own.
   - If web research was possible but genuinely no official inference config exists for this model + benchmark, say so explicitly and proceed with the existing eval config, noting the gap in the research note and the final report. This clause does not apply when web access is unavailable — that case requires user intervention as above.

3. Iterate the eval code to follow the official inference configuration.
   - Prefer **eval-only** edits: `loongforge/embodied/eval/**` (factories, servers, eval policy, adapters, bridges, orchestrator, `EvalServerArgs`) and `examples/embodied/<model>/eval/**` YAMLs/scripts. Training-side code — the model package, its `ModelConfig`, and `predict_action()` in `xxxx_modeling.py` — is out of scope by default; if a required change can only be made there, stop and report it to the user/training team as a blocker instead of editing it. If the user later authorizes a minimal model-side fix (e.g. dynamic `num_images`), keep it multi-benchmark safe and re-smoke a known-good combo (typically pi05×LIBERO) after shared changes.
   - Put protocol-specific logic in eval bridges / `action_postprocess` named modes (e.g. `pi05_aloha_14d` vs `ee6d_dual`), not as hard-coded model defaults.
   - Apply the user-confirmed field additions to `EvalServerArgs` / the eval-side model factory / the eval YAMLs together, so every YAML field maps to a field that actually takes effect.
   - Drive official-config alignment through eval-side knobs only: pass officially-documented values through existing `ModelConfig` fields via the YAML `model:` section, and adjust the eval-side factory (loading, device/dtype, paths) and eval policy/adapter where the change is eval-owned. If the official behavior depends on model-internal logic (normalization scheme, preprocessing, prompt format, denoising steps) that `predict_action()` does not expose a knob for, report it as a training-side blocker.
   - Keep boundaries intact while iterating: normalization stays inside the training-side `predict_action()`, view selection and chunk caching stay in `GenericPredictActionPolicy`, and benchmark state conversion stays in the adapter.
   - After each iteration, re-run local interface validation and at least one benchmark smoke to confirm the change did not break the RPC contract or action shapes.

4. Decide whether the shared `predict_action` path applies.
   - `predict_action` is implemented by the training team in the model package's `xxxx_modeling.py`; this skill only consumes it. Check that the existing method matches this shape:

     ```python
     def predict_action(images, instructions, state=None, dataset_stats=None):
         ...
     ```

   - Use `PredictActionModel`, `validate_predict_action_model()`, and `call_predict_action()` to define and test the contract.
   - Accept these output shapes from the model and normalize to `[H, action_dim]`: `[D]`, `[H, D]`, or `[B, H, D]`.
   - Make the model factory handle model-private setup: imports, config registration, checkpoint loading, tokenizer paths, device/dtype, compile flags, random-init, and metadata.
   - Let `GenericPredictActionPolicy` handle eval-private behavior: RPC payloads, image view packing (`primary`/`head` required; both `left`+`right` → 3 views for RoboTwin; else at most one wrist view), chunk caching, action shape validation, latency, request IDs, and response format. Action unnormalization is the model's responsibility and must happen inside `predict_action()`. Models must tolerate dynamic view counts.
   - Extra model keywords (e.g. xvla `domain_id`) are passed via RPC payload from `benchmark.domain_id`; the factory may wrap `predict_action` for type coercion only, not for reimplementing inference.

5. Define state and action semantics explicitly.
   - Keep benchmark-native `canonical_obs["state"]` out of the model server unless it is already model-ready.
   - Add or preserve `canonical_obs["model_state"]`; runners should forward `canonical_obs.get("model_state")` as RPC payload `state`.
   - If a model consumes state, verify that `model_state` ordering, units, frame, shape, and `dataset_stats["observation.state"]` match training. For ee6d models set `server.state_format: ee6d` so adapters build rot6d proprio.
   - Record dims/horizon fields that the ModelConfig actually declares (pi05: `action_dim`/`state_dim`/`action_horizon`/…; xvla: `real_action_dim`/`action_mode`/`num_actions`/…).
   - Check whether the model emits raw actions or normalized actions.
   - If normalized actions require inverse transform (e.g. pi05 q01/q99, or LeRobot mean/std), it belongs inside the training-side `predict_action()`, not in the generic eval policy. Verify the training-side implementation handles it; if it does not, report it to the user/training team as a blocker instead of implementing it on the eval side.
   - Do not reuse pi05 q01/q99 unnormalization for another model unless the model uses that exact convention.
   - Map model output → env action with eval-side protocol knobs only: `benchmark.action_postprocess` and/or RoboTwin `benchmark.action_bridge`. Document abs vs delta control in the YAML header.

6. Write YAML configs for all supported benchmarks.
   - By default, generate demos for each supported benchmark: LIBERO, CALVIN, SimplerEnv, RoboTwin, ManiSkill.
   - **Layout (pi05/xvla convention):** for each benchmark ship **exactly one public + one `_internal` pair** (e.g. `object_smoke.yaml` + `object_smoke_internal.yaml`). Public templates use `/path/to/...` placeholders; `_internal` holds machine absolute paths and is the one-click default for internal scripts. Do **not** leave extra full-suite / alternate smoke YAMLs in-tree unless the user asks (optional knobs go in comments on the single pair).
   - **Task-success vs link smoke:** only mark a config as task-success when a matching domain checkpoint was verified. If there is no domain weight, set `server.random_init: true`, empty `ckpt_path`, keep steps short, and comment clearly that it is **not** task-success (see pi05/xvla CALVIN + ManiSkill; also pi05 SimplerEnv).
   - Keep success smokes bounded by default: one task, one episode where knobs allow. Full-suite sizes belong in comments (`max_tasks: 0`, `episodes_per_task: 10`, …), not as a second YAML file.
   - Use local runner knobs where available: LIBERO `max_tasks: 1` and `episodes_per_task: 1`; CALVIN one sequence and low max steps for link smoke (raise toward official EP_LEN only with domain weights); SimplerEnv one task/episode (X-VLA WidowX task-success uses official `max_steps: 1200`); RoboTwin one task with bounded `max_steps`; ManiSkill one episode with low `max_steps` for link smoke.
   - The `model:` section fields must correspond 1:1 to the model's `XxxModelConfig` dataclass (e.g. `Pi05ModelConfig`, `XvlaModelConfig`). Before writing the YAML, inspect `loongforge/embodied/model/<model>/model_configuration_<model>.py` to find the declared fields. Only include fields that exist in the ModelConfig; unknown fields are silently filtered out by OmegaConf merge and will not take effect. Do NOT include eval-only or runner-level fields (e.g. `name`, `action_dim` as benchmark-target dim, `num_image_views`) in `model:`. Exception: some ModelConfigs legitimately declare `action_dim`/`state_dim` (pi05); xvla uses `real_action_dim` / `action_mode` instead — follow the dataclass.
   - Include `model.backend` and `model.model_type` in the `model:` section.
   - Include infrastructure fields (`ckpt_path`, `tokenizer_path` / `processor_path` when needed, `dataset_statistics_path`, `use_bf16`, `loongforge_root`, `random_init`) in the `server:` section, not `model:`. Put proprio layout knobs such as `state_format: ee6d` in `server:` as well.
   - Include `server.python`, `server.host`, `server.port`, `server.health_port`, `server.start_timeout_sec`, and `server.log`.
   - Use unique ports per smoke config to avoid health-port collisions during repeated runs.
   - Include `run.output_dir`, `run.seed`, `run.save_trace`, and replay flags when supported.
   - Pair each public YAML with `run_<benchmark>_eval.sh` and each `_internal` YAML with `run_<benchmark>_eval_internal.sh` under `examples/embodied/<model>/eval/`.
   - For models that require a domain identifier (e.g. xvla), put `benchmark.domain_id` in the `benchmark:` section rather than `model:`. Domain ID is a benchmark/dataset-level concept, not a model-structure field. Known xvla domain IDs used in-repo: LIBERO=3, CALVIN=2, **SimplerEnv WidowX/Bridge=0**, RoboTwin2=6 (VLABench=8 if ever wired). Do not invent IDs; prefer official eval scripts.
   - Protocol fields that are **not** ModelConfig:
     - `benchmark.action_postprocess` — runner-side conversion (registry in `predict_action_interface.py`).
     - `benchmark.action_bridge` — RoboTwin bridge modes in `bridges/robotwin_policy.py` (`pi05_aloha_14d`, `ee6d_dual`, …).
     - Absolute vs delta control may be implied by postprocess/bridge (e.g. LIBERO abs EE when postprocess is set); document it in the YAML header.
   - Document open weight URLs and verified local paths in the YAML header comments (public vs `_internal`).

7. Wire server startup only as needed.
   - Reuse existing LoongForge server routing for pi05-style integrations when possible.
   - Add server-manager routing for a new `model.backend` only when an existing server entrypoint cannot serve it.
   - Ensure health readiness means the model factory has completed, the warmup `predict_action()` call has run (see below), and action RPC can start.
   - Use reusable health server binding for short repeated smoke runs when local patterns support it.
   - After model factory build and before health server startup, run a warmup `predict_action()` with a zero-filled dummy image and an empty instruction. This forces all lazy imports (including potential circular-import paths) to complete before the first real episode arrives. The call is wrapped in a try/except so a warmup exception does not abort startup, but the model must not enter a corrupted state from a warmup call.
   - Before running the smoke matrix, check for leftover orchestrator/policy-server processes and occupied server ports.
   - Keep benchmark client Python env and model server Python env explicit. Run the top-level orchestrator with the benchmark/simulator conda environment, while YAML `server.python` starts the model server environment.

8. Check runtime-specific traps.
   - For SAPIEN-based benchmarks such as SimplerEnv, RoboTwin, and ManiSkill, verify NVIDIA Vulkan with `vulkaninfo`, not just `nvidia-smi`, when visual rollout correctness matters.
   - Expected Vulkan signal is `deviceName = NVIDIA ...` and `driverName = NVIDIA`; `llvmpipe`/`lavapipe` means visual rollout is not trustworthy.
   - SAPIEN runners may need `LD_LIBRARY_PATH`, `VK_ICD_FILENAMES`, and `XDG_RUNTIME_DIR` set before importing SAPIEN/svulkan2/ManiSkill.
   - For MuJoCo/LIBERO/CALVIN, preserve existing `MUJOCO_GL`, `PYOPENGL_PLATFORM`, and benchmark config-path patterns.

9. Validate in layers.
   - First run local interface validation without a benchmark when possible:

     ```bash
     PYTHONPATH=/workspace/LoongForge-VLA python - <<'PY'
     import numpy as np
     from loongforge.embodied.eval.servers.predict_action_interface import call_predict_action, validate_predict_action_model

     class MyModel:
         def predict_action(self, images, instructions, state=None, dataset_stats=None):
             return np.zeros((len(instructions), 4, 7), dtype=np.float32)

     model = MyModel()
     validate_predict_action_model(model)
     print(call_predict_action(model, images=[[]], instructions=["task"], state=None, dataset_stats=None, action_dim=7).shape)
     PY
     ```

   - Then execute every generated benchmark demo that can run in the current environment.
   - Use the benchmark client's conda environment for the top-level orchestrator command: LIBERO with the LIBERO env, CALVIN with the CALVIN env, SimplerEnv with the SimplerEnv env, RoboTwin with the RoboTwin env, ManiSkill with the ManiSkill env.
   - Mark a benchmark `passed` only when the command exits successfully and the expected outputs prove at least one policy call or official runner completion.
   - Mark a benchmark `blocked` when required runtime, simulator assets, checkpoint, stats, or environment support is missing; include the concrete error or missing path.
   - Mark a benchmark `skipped` only when the user explicitly narrows scope or asks not to run it.
   - Protocol or mock smoke proves runner/server/RPC/action shape.
   - Random-init smoke proves the real model class can initialize and answer RPC, but is **not** a benchmark score and must not be reported as task-success.
   - Real-checkpoint smoke proves the real checkpoint can run one short episode.
   - Credible / task-success requires matching domain checkpoint, matching stats when the model needs them, correct action semantics (`action_postprocess` / `action_bridge` / abs vs delta), and enough episodes to support the claim.
   - After shared eval or model-inference changes (especially view packing / `num_images`), re-run a small known-good regression (e.g. pi05×LIBERO smoke) before claiming no impact.

10. Update docs with precise status.
    - Separate `mock`, `random-init`, `real checkpoint`, and `credible score` statuses.
    - Put user-facing usage in README/user guide; avoid filling README with internal validation logs.
    - Put detailed interface contract and local interface-validation examples in `predict_action_interface.md`.
    - Mention missing assets directly, especially checkpoint and `dataset_statistics.json`.
    - Record runtime requirements that future users must set before import.

## Pi05 reference mapping

Use pi05 as the canonical example of the shared `predict_action` architecture:

- Model interface: `PI05Policy.predict_action(images, instructions, state=None, dataset_stats=None)` in the model package.
- Interface helpers: `loongforge/embodied/eval/servers/predict_action_interface.py`.
- Generic eval policy: `GenericPredictActionPolicy` in `loongforge/embodied/eval/servers/loongforge_policy.py`.
- Model factory: `PI05ModelFactory` in `loongforge/embodied/eval/factories/pi05_factory.py`.
- Factory registry: `register_factory`, `build_model_spec` in `loongforge/embodied/eval/factories/registry.py`. New models register with `@register_factory("<model_type>")` and declare `model_config_cls = <ModelConfig>`.
- Server config: `EvalServerArgs` dataclass and `parse_eval_server_config` in `loongforge/embodied/eval/servers/eval_server_config.py`. The YAML `server:` section is merged directly into `EvalServerArgs` via OmegaConf; the YAML `model:` section is merged into the registered `ModelConfig` (e.g. `Pi05ModelConfig`) via OmegaConf.
- Backward-compatible wrapper: `LoongForgePI05Policy` in `loongforge/embodied/eval/factories/pi05_factory.py`, which should not be the preferred pattern for new integrations.
- Server entrypoint: `loongforge/embodied/eval/servers/loongforge_server.py` calls `parse_eval_server_config` to get `EvalServerArgs` + `raw_model_dict`, then `build_model_spec` to load the model, then `_warmup_model()` to resolve lazy imports, then wraps in `GenericPredictActionPolicy`.
- Routing: `loongforge/embodied/eval/orchestrator/server_manager.py` maps `loongforge`, `pi05`, and `loongforge_pi05` to the LoongForge server.
- Adapter state boundary: benchmark adapters provide `canonical_obs["state"]` for native structured state and `canonical_obs["model_state"]` for model-ready state; runners forward only `model_state` as RPC payload `state`.
- YAML + scripts layout: `examples/embodied/pi05/eval/configs/<benchmark>/{smoke,object_smoke,adjust_bottle_smoke,...}.yaml` + matching `_internal.yaml`; `run_<benchmark>_eval.sh` / `_internal.sh`. Same shape under `examples/embodied/xvla/eval/`.
- Infrastructure fields (`ckpt_path`, `tokenizer_path`, `dataset_statistics_path`, `use_bf16`, `loongforge_root`, `random_init`, `state_format`) live in `server:`; model-structure fields live in `model:` (pi05: `action_dim`, `action_horizon`, …; xvla: `action_mode`, `real_action_dim`, `num_actions`, …).
- RoboTwin bridge: `bridges/robotwin_policy.py` — pi05 uses `action_bridge: pi05_aloha_14d` (+ stats asset under `examples/embodied/pi05/eval/assets/`); xvla uses `ee6d_dual` + `domain_id: 6`.
- xvla extras: factory wrap for `domain_id` tensor; `benchmark.action_postprocess` for LIBERO/SimplerEnv/CALVIN; SimplerEnv abs EE may need env patch (`examples/embodied/xvla/eval/SIMPLERENV_PATCH_en.md`).

For new models, create `loongforge/embodied/eval/factories/<model>_factory.py` with a `@register_factory("<model_type>")` class that declares `model_config_cls` and implements `build(model_cfg, server_args) -> PredictActionModelSpec`. Add the module path to `_FACTORY_MODULES` in `factories/registry.py`. No changes to `loongforge_server.py` are needed.

## Required final response

When finished, report:

- Files created or modified.
- The official inference-config sources consulted, the field-by-field diff result (match / mismatch / missing), which missing or mismatched fields the user confirmed or rejected, and which eval-code changes were driven by the official config. If no official config was found, state that explicitly.
- Whether the integration used the shared `predict_action` path or a bespoke policy adapter, and why.
- The discovered supported benchmark set and whether each benchmark received a demo YAML, unless the user narrowed scope.
- A per-benchmark smoke matrix with status: passed, skipped, or blocked.
- Which smoke layer passed for each benchmark: local interface validation, mock, random-init, real checkpoint, or credible score.
- Exact command used for each validation that ran.
- Output artifact path such as `results.jsonl` or `policy_server.log` for each completed smoke, or note if temp artifacts were deleted after validation.
- Any remaining blocker, especially missing checkpoint, missing `dataset_statistics.json`, action mismatch, runtime driver issue, missing simulator env, or user-narrowed scope.
