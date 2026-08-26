---
name: vla-model-eval-adapter
description: Use this skill when adapting a VLA/model backend into the LoongForge embodied eval system after benchmark runners already exist. Trigger on requests to add or refactor model eval integration, validate a model predict_action interface, create model factory/loader code, wire model.backend/server routing, write eval YAML, handle action normalization or dataset_statistics, align eval config with a model's official inference configuration, or reproduce pi05/xvla-style integration. At run start, collect required inputs from the user (target model + benchmark, model package location, checkpoint/tokenizer/dataset-statistics paths). Before writing configs, search the web for the model's official inference config (repo, paper, HuggingFace card), diff it against the eval YAML you draft from the pi05/xvla template (there is no pre-existing model eval code in a real adaptation — you create the factory, PayloadBuilder, YAMLs, and run scripts) and against the training-side ModelConfig, and ask the user before values that deviate from the official config; then iterate the eval code to follow the official config. Prefer the shared predict_action contract plus GenericPredictActionPolicy before creating a bespoke policy adapter. Prefer eval-only changes; put per-model protocol logic in the model's PayloadBuilder (canonical -> predict_action kwargs) plus the ActionDecoder registry (raw action chunk -> env action space), and use RoboTwin's action_bridge only to select that wiring. By default, ship one public + one _internal YAML pair (and matching run scripts) per supported benchmark (LIBERO, CALVIN, SimplerEnv, RoboTwin, ManiSkill), using random_init link smoke when no domain weight exists, unless the user narrows scope. Adaptation is only complete after the smoke tests are actually executed (not just generated): author a run script per benchmark that activates the matching conda env and sets the correct environment variables (PYTHONPATH, CUDA_VISIBLE_DEVICES, LD_LIBRARY_PATH so libcuda resolves, plus simulator vars), run each benchmark through that generated script, and do not report a GPU/driver blocker before re-checking in that environment.
---

# VLA Model Eval Adapter

Use this skill to connect a model backend to the existing LoongForge embodied evaluation stack. The benchmark runners and benchmark adapters are assumed to already exist. The model-integration architecture is now a **3-part component split** on the eval side:

```text
benchmark adapter  (emits canonical obs incl. state_raw)
  -> PayloadBuilder (client-side): canonical dict -> predict_action(**kwargs)
  -> GenericPredictActionPolicy -> PolicyServer RPC
  -> model factory/loader -> model.predict_action(images, instructions, state=None, dataset_stats=None, ...)
  -> raw action chunk
  -> ActionDecoder (client-side): raw chunk -> benchmark env action space
```

Each supported model contributes three registered pieces: a **ModelFactory** (`factories/<model>_factory.py`, `@register_factory`, server-side), a **PayloadBuilder** (`payload_builders/<model>.py`, `@register_payload_builder`, client-side), and — only when its action encoding is not already covered — an **ActionDecoder** (`action_decoders/`). The ActionDecoder key is auto-composed at orchestrator startup from `payload_builder.action_encoding` and `adapter.action_space`, so users no longer hand-write it in YAML.

Create a bespoke policy adapter only when the model cannot reasonably expose the shared `predict_action` interface or needs custom RPC behavior that `GenericPredictActionPolicy` cannot cover.

## What already exists vs what you create

This skill adapts a **new model**. The benchmark side and the shared eval infrastructure already exist, but **none of the model's own eval glue does** — do not assume the model's factory, PayloadBuilder, YAMLs, or run scripts are present. You author them from scratch, using pi05/xvla as templates.

Already exists — reuse, do not recreate:
- Benchmark adapters (`adapters/*.py`) and their runners; each adapter's `action_space` / `cameras` / `default_fps`.
- Shared eval infra: `GenericPredictActionPolicy`, `servers/predict_action_interface.py`, the factory and payload-builder registries, `action_decoders/base.py` + `rotation.py` and the already-registered `ee6d_*` / `joint` decoders, `bridges/robotwin_policy.py`, `EvalServerArgs`, `loongforge_server.py`, and `server_manager.py` routing.
- The training-side model package: `xxxx_modeling.py` (its `predict_action`) and `model_configuration_<model>.py` (`ModelConfig`) — read-only.
- The pi05/xvla integrations — as **templates to copy**, not code your model reuses at runtime.

You create for the new model, from scratch (pi05/xvla is the structural template):
- `factories/<model>_factory.py`, `payload_builders/<model>.py`, the per-benchmark YAML pairs, the `run_*.sh` scripts, and — only if the model's action encoding is not already covered — a new ActionDecoder. Register the new factory and payload-builder module paths in the two registries.

Because the model's eval YAML does not exist yet, the official-config reconciliation in step 2 diffs the **official inference config** against the YAML you are drafting from the pi05/xvla template — not against a pre-existing model YAML. The human-approval gate then applies to values that **deviate from the official config** or to non-obvious choices, not to every field of a from-scratch YAML (a from-scratch YAML has no "existing value" to preserve).

## Mental model

Keep these boundaries separate:

```text
benchmark adapter (adapters/):
  benchmark obs/action <-> canonical eval schema
  OUTPUTS raw obs fields in canonical_obs["state_raw"] (NOT an encoded model state)
  declares capability class attrs: action_space / default_fps / cameras
  owns benchmark-native state and debug/trace metadata
  examples: adapters/libero.py, adapters/maniskill.py, adapters/simplerenv.py

model factory/loader (factories/):
  model config/import/checkpoint/tokenizer/device/dtype/random-init/metadata
  server-side (needs torch); returns a PredictActionModelSpec whose model
  implements predict_action(...). Role UNCHANGED by the refactor.
  example: PI05ModelFactory in factories/pi05_factory.py, XVLAModelFactory in
  factories/xvla_factory.py (still wraps predict_action for domain_id int->tensor).

payload builder (payload_builders/, client-side):
  Converts the adapter's canonical dict -> the kwargs for
  model.predict_action(**kwargs). This is where per-model payload assembly
  now lives (previously scattered across runner `_canonical_to_*_payload`,
  the factory's `_predict_action_wrapper`, the policy's image packing, and
  adapters' state building). Declares model capabilities as typed class
  attributes (state_encoding / action_encoding / action_dim / action_horizon /
  domain_id / unnorm_key) — the annotated names are the whitelist for YAML
  `model:` overrides. Base class payload_builders/base.py: PayloadBuilder with
  build(canonical, ctx) / reset(episode_id) / update_from_response(response) /
  note_env_action(env_action). Registry: payload_builders/registry.py
  (build_payload_builder). Image view packing is `_pack_images` in
  payload_builders/pi05.py (reused by xvla): primary/head required; if both
  left and right exist -> 3 views [head, left, right] (RoboTwin); else at most
  one wrist/right/left as the second view. Models must accept dynamic view
  count (e.g. `num_images = len(images[0])`), never hardcode 2 or 3.
  Proprio encoding lives here via the state_encoding logic.
  examples: Pi05PayloadBuilder, XVLAPayloadBuilder.

generic eval policy (servers/loongforge_policy.py):
  RPC payload handling, chunk caching, predict_action invocation, action shape
  validation, action dim truncation, dataset statistics loading, latency,
  metadata. It NO LONGER packs images (that moved to _pack_images in the
  PayloadBuilder). RPC payload is v2 only: `images` = list of view arrays,
  `instructions` = list[str]. Action unnormalization is NOT done here; it is
  the model's responsibility inside predict_action().
  example: GenericPredictActionPolicy in servers/loongforge_policy.py

action decoders (action_decoders/, client-side):
  Convert the raw model action chunk -> the benchmark env action space.
  base.py = ActionDecoder base (__call__(actions[H,D], ctx) -> env_actions,
  optional reset()), IdentityDecoder, and the registry (register_action_decoder
  / build_action_decoder). rotation.py = pure rot6d math. ee6d.py = decoders
  for ee6d-output models (ee6d_to_axis_angle, ee6d_to_euler, ee6d_to_quat,
  ee6d_to_calvin_abs, ee6d_to_simpler_abs_euler, ee6d_robotwin_ee_dual).
  joint.py = decoders for joint(-delta)-output models (pi05_aloha_robotwin,
  stateful). The decoder key is AUTO-COMPOSED at orchestrator startup by
  orchestrator/config.py: resolve_action_decoder_key(payload_builder, adapter)
  = "{payload_builder.action_encoding}_to_{adapter.action_space}"; identity
  (source == target) -> empty key -> IdentityDecoder (no-op). Users do not
  hand-write a decoder key in YAML.

RoboTwin form-B bridge (bridges/robotwin_policy.py, protocol-specific):
  Thinned to the 4-component chain (adapter -> PayloadBuilder -> PolicyClient
  -> ActionDecoder). `_BRIDGE_WIRING` maps benchmark.action_bridge ->
  (model_type, payload-builder state_encoding, decoder key):
    ee6d_dual      -> (xvla, ee6d_dual, ee6d_robotwin_ee_dual)
    pi05_aloha_14d -> (pi05, aloha_pi, pi05_aloha_robotwin)
```

The state boundary is now: the adapter emits raw obs fields under
`canonical_obs["state_raw"]`; the PayloadBuilder encodes them per its
`state_encoding` into the `predict_action` `state` kwarg; RPC forwards that to
`predict_action(state=...)`. Benchmark-native structured state stays in
`canonical_obs["state"]` for eval/debug/trace. There is no longer an
adapter-provided `model_state`; proprio encoding is the PayloadBuilder's job.

```text
adapter.state_raw -> PayloadBuilder encodes per state_encoding -> RPC payload.state -> predict_action(state=...)
```

Do not clean, drop, or reinterpret benchmark-native dict state inside a model factory. That belongs in the benchmark adapter (raw fields) or the PayloadBuilder (encoding).

## Expected deliverables

Produce concrete files whenever implementation is requested. A complete model integration usually includes:

- An official inference-config research note: which official sources were consulted (official repo, paper, HuggingFace model card, official deployment/eval scripts), which official inference parameters were found, and a field-by-field comparison against the existing eval YAML. Missing fields must be explicitly confirmed or rejected by the user before being added, and confirmed fields land in eval-side code only (factory, eval policy, eval YAML) — never in the training-side `ModelConfig`.
- A model factory/loader that returns a model instance and metadata, preferably via `PredictActionModelSpec` or an equivalent local pattern.
- A model object exposing `predict_action(images, instructions, state=None, dataset_stats=None)`. This method is implemented by the training team in the model package's `xxxx_modeling.py`; this skill consumes it as-is and does not reimplement inference. Thin eval-side wrappers are allowed only for type/coercion plumbing (e.g. xvla `domain_id` int → LongTensor). If the method is missing or does not match the contract, report that to the user/training team as a blocker.
- A PayloadBuilder registered with `@register_payload_builder("<model_type>")` under `payload_builders/<model>.py`. It converts the adapter's canonical dict into `predict_action(**kwargs)`, declares model capabilities as typed class attributes (`state_encoding` / `action_encoding` / `action_dim` / `action_horizon` / `domain_id` / `unnorm_key` — the annotated names are the whitelist for YAML `model:` overrides), and encodes proprio per `state_encoding` from `canonical_obs["state_raw"]`. The orchestrator asserts `set(MODEL_FACTORY_REGISTRY) == set(PAYLOAD_BUILDER_REGISTRY)` at startup, so a factory without its paired payload builder (or vice-versa) fails fast — always ship the factory and payload builder together.
- An ActionDecoder under `action_decoders/` only if the model's action encoding is not already covered. Otherwise it is auto-selected: identity source==target action space → `IdentityDecoder` (no-op), or an existing `ee6d_*` decoder when `{action_encoding}_to_{action_space}` already maps to one. New decoders register with `@register_action_decoder("<key>")` (grouped by source encoding in `ee6d.py` / `joint.py`); the key is auto-composed at startup, not written in YAML.
- Interface validation using `validate_predict_action_model()` and `call_predict_action()` from `loongforge/embodied/eval/servers/predict_action_interface.py`.
- Reuse of `GenericPredictActionPolicy` when the shared interface is sufficient.
- A bespoke `servers/<model>_policy.py` only if shared `predict_action` is not a good fit.
- `loongforge/embodied/eval/servers/<model>_server.py` or existing server entrypoint reuse if applicable.
- `loongforge/embodied/eval/orchestrator/server_manager.py` routing only when adding a new `model.backend` that cannot reuse existing LoongForge routing.
- Demo YAML configs for every already-supported benchmark under `examples/embodied/<model>/eval/configs/<benchmark>/`, unless the user narrows scope. Prefer the **one public + one `_internal` pair per benchmark** layout used by pi05/xvla (see step 6); do not ship extra full/regression YAMLs unless the user asks.
- Matching run scripts under `examples/embodied/<model>/eval/` (`run_<benchmark>_eval.sh` + `run_<benchmark>_eval_internal.sh`) when following the pi05/xvla pattern. These run scripts do not exist yet during a real adaptation — you must **author them as part of the deliverables**, and each one must bake in (a) the correct benchmark conda env for the orchestrator (`BENCHMARK_PYTHON` / the env used to launch `-m loongforge.embodied.eval.orchestrator.run`) and (b) all required env exports before launch: `PYTHONPATH=<repo_root>`, `CUDA_VISIBLE_DEVICES`, `LD_LIBRARY_PATH` including the machine's `nvidia_lib` dir so `libcuda.so.<driver-version>` resolves, and the simulator vars for that benchmark (`MUJOCO_GL`/`PYOPENGL_PLATFORM` for LIBERO/CALVIN; `VK_ICD_FILENAMES`/`XDG_RUNTIME_DIR` for SAPIEN). Copy the env-export block from the pi05/xvla run scripts as the template.
- A smoke-test matrix covering every generated benchmark demo, with one command and expected artifact path per benchmark.
- Executed smoke tests for every generated benchmark demo that can run in the current environment; do not stop at generating YAML/matrix files.
- Documentation updates in the relevant eval docs, especially `README.md`, `user_guide_en.md`, `benchmark_envs.md`, `loongforge_eval_summary.md`, or `model_integration.md`.

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
- The diff baseline for configs: for a new model there is normally none, so configs are drafted from scratch using the pi05/xvla template. Only when refactoring an existing integration is there an existing eval YAML under `examples/embodied/<model>/eval/configs/` to diff against.

Optional (ask once; if unanswered, apply the defaults stated below):

- Pre-authorized decisions for the human-intervention gates: e.g. "add missing fields at official recommended values without asking". Default: no pre-authorization — every gate stops and asks.
- Port range constraints (default: pick unique unused ports), `model.domain_id` if the model requires one (e.g. xvla; default: the known per-benchmark IDs listed in step 6 — note SimplerEnv WidowX/Bridge is **0**, not 1), dry-run vs writing to the repo (default: write to the repo), and GPU/Vulkan readiness for SAPIEN-based benchmarks (default: probe with `vulkaninfo` per step 8).

## Workflow

1. Collect the required inputs above from the user, then inspect the target model and benchmark context.
   - Find existing model configs, server entries, policy adapters, and model inference APIs.
   - Identify whether this is a new backend, a variant of an existing backend, or a refactor of an existing integration.
   - Confirm target benchmarks are already wired in the orchestrator.
   - Discover the already-supported benchmark set from runner/config directories rather than assuming only one benchmark.

2. Research the official inference configuration online and reconcile it with the eval config you are drafting.
   - **MANDATORY — official-config parity gate (blocking).** You MUST reproduce the model+benchmark's exact official configuration before running for a success rate, and you MUST NOT attribute a low/zero success rate to "model behavior", "inference numerics", or "hard task" until EVERY inference-relevant knob has been proven equal to (a) the checkpoint's own bundled files and (b) the official eval client. Build an explicit parity checklist and verify each item = official; treat any unverified item as a suspected bug, not the model's fault. The checklist MUST cover, at minimum: per-embodiment state/action layout; normalization method per dim (min-max vs mean-std / `mean_std_embedding_keys`, sin/cos, q01/q99); relative-vs-absolute action (`action_configs`/`use_relative_action`); **action horizon / `delta_indices`**; **open-loop execution length `n_action_steps` / `chunk_execute_steps`**; `max_episode_steps`; `control_freq`/`sim_freq`; embodiment→projector id; image size + preprocessing + camera view(s); prompt/instruction format; denoising/sampling steps; dtype; and the benchmark's episode-init protocol (e.g. `prepackaged_config`). Numbers come from the checkpoint's `processor_config.json`/`experiment_cfg`/`config.json` and the official eval CLIENT command — NEVER from library/registration defaults or from eyeballing `statistics.json` ranges. When a knob cannot be confirmed against an official source, stop and ask the user rather than guessing. Only after this checklist is 100% green may you investigate model/inference-level causes.
   - **The checkpoint's own bundled config files are authoritative — read them, do not hand-derive.** When adapting a real checkpoint, its directory ships the exact modality + normalization spec (e.g. `config.json`, `processor_config.json`, `experiment_cfg/`, `meta/modality.json`, `statistics.json`, `embodiment_id.json`). Read these to get the per-embodiment state/action **layout AND the normalization method**: min-max vs mean-std (`mean_std_embedding_keys`), sin/cos encoding, relative vs absolute (`action_configs` / `use_relative_action`), embodiment→projector id, action `delta_indices`/horizon, image size/preprocessing. Do NOT reconstruct the embodiment/modality config by eyeballing `statistics.json` min/max ranges — the ranges do not tell you the normalization scheme, and guessing it (e.g. assuming min-max when the checkpoint uses mean-std over a wide ±2π euler range) silently produces catastrophically wrong actions that only show up on wide-range action dims (a small-range benchmark like LIBERO will hide the bug). If the checkpoint bundles a processor/modality config, mirror it verbatim into the eval-side embodiment config.
   - Search the web for the model's official sources: the official GitHub repo (inference/eval scripts, `config.json`, deployment docs), the paper, and the HuggingFace model card. Prefer official first-party sources over blog posts or third-party reproductions.
   - If web access is unavailable, stop and ask the user to intervene: either restore web access or provide the official inference config manually. Do not silently substitute guesses or unverified local sources for the official config.
   - The research target is the specific model plus the benchmark to be evaluated (e.g. "xvla on LIBERO"); official inference configs often differ per benchmark, so look for the official setup used for that benchmark.
   - Extract the official inference-time parameters, typically: action horizon / chunk size, action dim, state dim, image resolution and preprocessing, expected camera views, prompt/instruction format, normalization scheme (q01/q99, mean/std, none) and its statistics source, denoising/sampling steps, dtype, and any control-frequency or replan-interval assumptions.
   - **Also align the benchmark/client-side runtime knobs, and take them from the official eval CLIENT command — NOT the env registration default.** These include `max_episode_steps` (episode horizon), `control_freq`, open-loop execution horizon / `n_action_steps` (→ `chunk_execute_steps`), replan interval, and episode count. The official client often overrides the gym env's registered default (e.g. a drawer env registers `max_episode_steps=120` but the official rollout passes `--max-episode-steps 300`; using the 120 registration default silently starves a slow precision task and tanks the rate). Copy these from the published eval command, and include them in the same field-by-field diff. In short: **every knob that affects the rollout — model, server, AND benchmark — must match the official value; do not fall back to library/registration defaults.**
   - Compare field-by-field against the eval YAML you are drafting (`model:` and `server:` sections), bootstrapped from the pi05/xvla template — for a new model there is no pre-existing model YAML, so the template defaults are the starting point. Read the training-side `XxxModelConfig` dataclass only to learn which `model:` fields exist and take effect; it is training-side code and must NOT be modified by this skill. Produce a three-column diff: official value, drafted eval value (or template default), and status (match / mismatch / missing in eval config).
   - For fields present in the official config but not yet in your drafted YAML, add them at the official value when they are unambiguous and clearly required (this is the normal from-scratch case — do not stop on every field). Stop and ask the user only for fields where the official value is ambiguous, contested across sources, or where you must deviate from it. Present each such field with its official value, its likely impact on eval correctness, and a recommendation; record rejected fields and the reason in the research note.
   - Route each confirmed field to an eval-side home: if the training-side `ModelConfig` already declares it, set it via the YAML `model:` section; otherwise put it in the `server:` section (extending `EvalServerArgs` if needed) or handle it inside the eval-side factory / eval policy. Never add fields to the training-side `ModelConfig` to make a YAML knob work.
   - For fields whose drafted value mismatches the official value, flag them the same way; do not silently overwrite a value the user has already tuned in an earlier iteration.
   - Also compare against eval-pipeline capabilities, not just YAML fields: e.g. camera-view packing (1 primary + optional wrist, or 3-view head/left/right for RoboTwin), PayloadBuilder `state_encoding` / `action_encoding` coverage and ActionDecoder coverage (auto-composed `{action_encoding}_to_{action_space}` key), or a state field the adapter does not emit under `state_raw`. When the official config exceeds what the eval pipeline can do, stop and ask the user to decide whether to add the corresponding capability to the eval-side code; do not extend the pipeline or accept the deviation on your own.
   - If web research was possible but genuinely no official inference config exists for this model + benchmark, say so explicitly and proceed with the existing eval config, noting the gap in the research note and the final report. This clause does not apply when web access is unavailable — that case requires user intervention as above.

3. Iterate the eval code to follow the official inference configuration.
   - Prefer **eval-only** edits: `loongforge/embodied/eval/**` (factories, payload builders, action decoders, servers, adapters, bridges, orchestrator, `EvalServerArgs`) and `examples/embodied/<model>/eval/**` YAMLs/scripts. Training-side code — the model package, its `ModelConfig`, and `predict_action()` in `xxxx_modeling.py` — is out of scope by default; if a required change can only be made there, stop and report it to the user/training team as a blocker instead of editing it. If the user later authorizes a minimal model-side fix (e.g. dynamic `num_images`), keep it multi-benchmark safe and re-smoke a known-good combo (typically pi05×LIBERO) after shared changes.
   - Put per-model protocol logic in the PayloadBuilder (`state_encoding` / `action_encoding` capability attrs) plus the ActionDecoder registry, not as hard-coded model defaults. RoboTwin form-B uses `benchmark.action_bridge` (`pi05_aloha_14d` vs `ee6d_dual`) only to select the (model_type, state_encoding, decoder key) wiring in `bridges/robotwin_policy.py::_BRIDGE_WIRING`.
   - Apply the user-confirmed field additions to `EvalServerArgs` / the eval-side model factory / the eval YAMLs together, so every YAML field maps to a field that actually takes effect.
   - Drive official-config alignment through eval-side knobs only: pass officially-documented values through existing `ModelConfig` fields via the YAML `model:` section, and adjust the eval-side factory (loading, device/dtype, paths) and eval policy/adapter where the change is eval-owned. If the official behavior depends on model-internal logic (normalization scheme, preprocessing, prompt format, denoising steps) that `predict_action()` does not expose a knob for, report it as a training-side blocker.
   - Keep boundaries intact while iterating: normalization stays inside the training-side `predict_action()`, image view packing and proprio `state_encoding` stay in the PayloadBuilder, chunk caching stays in `GenericPredictActionPolicy`, action decoding stays in the ActionDecoder, and benchmark raw-state extraction (`state_raw`) stays in the adapter.
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
   - Let the PayloadBuilder handle client-side payload assembly: image view packing (`primary`/`head` required; both `left`+`right` → 3 views for RoboTwin; else at most one wrist view, via `_pack_images` in `payload_builders/pi05.py`), proprio `state_encoding`, and model-extra kwargs. Let `GenericPredictActionPolicy` handle eval-private behavior: RPC payloads, chunk caching, action shape validation, latency, request IDs, and response format. Action unnormalization is the model's responsibility and must happen inside `predict_action()`. Models must tolerate dynamic view counts.
   - Extra model keywords (e.g. xvla `domain_id`) are emitted by the PayloadBuilder into the `predict_action` kwargs (`domain_id` resolved from the `model.domain_id` YAML override or the per-benchmark default map); the factory may wrap `predict_action` for type coercion only (int → LongTensor), not for reimplementing inference.

5. Define state and action semantics explicitly.
   - Keep benchmark-native `canonical_obs["state"]` out of the model server unless it is already model-ready; it exists for eval/debug/trace.
   - The adapter emits raw obs fields under `canonical_obs["state_raw"]`; the PayloadBuilder encodes them per its `state_encoding` into the `predict_action` `state` kwarg. There is no adapter `model_state` anymore.
   - If a model consumes state, verify that the PayloadBuilder's encoded proprio ordering, units, frame, shape, and `dataset_stats["observation.state"]` match training. For ee6d models set the PayloadBuilder `state_encoding` (e.g. `ee6d` / `ee6d_calvin` / `ee6d_widowx` / `ee6d_dual`) via the YAML `model:` section so it builds rot6d proprio from `state_raw`.
   - Record dims/horizon fields that the ModelConfig actually declares (pi05: `action_dim`/`state_dim`/`action_horizon`/…; xvla: `real_action_dim`/`action_mode`/`num_actions`/…).
   - Check whether the model emits raw actions or normalized actions.
   - **Get the per-dim normalization method from the checkpoint's bundled modality/processor config, not from the stats file.** For models with a per-embodiment modality spec (e.g. GR00T `processor_config.json`), the action dims can mix schemes (e.g. pos+rotation use mean-std via `mean_std_embedding_keys`, gripper uses min-max); mirror it exactly into the eval-side embodiment config. Assuming min-max when training used mean-std silently blows up wide-range dims (e.g. a ±2π euler action) while a narrow-range benchmark (LIBERO) hides it.
   - If normalized actions require inverse transform (e.g. pi05 q01/q99, or LeRobot mean/std), it belongs inside the training-side `predict_action()`, not in the generic eval policy. Verify the training-side implementation handles it; if it does not, report it to the user/training team as a blocker instead of implementing it on the eval side.
   - Do not reuse pi05 q01/q99 unnormalization for another model unless the model uses that exact convention.
   - Map the model's raw action chunk → env action with the ActionDecoder. The decoder key is auto-composed at startup from `payload_builder.action_encoding` × `adapter.action_space`; source == target yields `IdentityDecoder`. RoboTwin form-B selects its decoder via `benchmark.action_bridge`. Document abs vs delta control in the YAML header.

6. Write YAML configs for all supported benchmarks.
   - By default, generate demos for each supported benchmark: LIBERO, CALVIN, SimplerEnv, RoboTwin, ManiSkill.
   - **Never invent `benchmark:` / per-benchmark `env:` / `run:` fields.** These belong to the benchmark runner, not the model, and every benchmark has its own required keys. Source them by copying the `benchmark:`/`env:`/`run:` blocks from an existing model's example YAML for that same benchmark (pi05/xvla under `examples/embodied/<model>/eval/configs/<benchmark>/`) and/or by reading the runner (`orchestrator/runners/<benchmark>_runner.py` and `orchestrator/run.py::_run_<benchmark>_once`) for the keys it reads. Change only the model-specific values; keep the benchmark keys verbatim. A wrong or missing benchmark key fails the runner immediately (e.g. CALVIN raises `dataset path must be set` when `benchmark.dataset_path` is absent). Per-benchmark required keys seen in-repo (verify against the runner, do not treat as exhaustive):
     - LIBERO: `suite`, `max_tasks`, `episodes_per_task`, `max_steps`, `num_steps_wait`, `control_mode`.
     - CALVIN: `suite` (e.g. `task_D_D`), `dataset_path`, `calvin_config_path`, `eval_sequences_path`, `num_sequences`, `max_steps_per_subtask`, `control_hz`.
     - SimplerEnv: `task_name`, `robot_setup`, `scene_name`, `rgb_overlay_path`, `sim_freq`, `control_freq`, `control_mode`, `rotation_mode`, `max_steps`, `robot_init_x/y`; `env.simplerenv_root` + `env.nvidia_lib_dir` + `env.nvidia_icd_json`.
     - ManiSkill: `task_name`, `robot_uid`, `instruction`, `obs_mode`, `control_mode`, `control_freq`, `render_mode`, `sim_backend`, `render_backend`, `camera_name`, `action_scale`, `max_steps`.
     - RoboTwin (**form-B**): put `domain_id` **and** `action_bridge` under `benchmark:`, and do **not** set `model.state_encoding` (the bridge's `_BRIDGE_WIRING` selects state_encoding + decoder). Also `task_name`, `task_config`, `start_seed`, `episodes_per_task`, `max_steps`; `env.robotwin_root` + `env.robotwin_python`.
   - **`random_init: true` still needs a real `processor_path`/`tokenizer_path`.** Models that load an HF processor/tokenizer at init (e.g. X-VLA) require a valid processor dir even with no checkpoint weights — point `processor_path`/`tokenizer_path` at any full checkpoint dir of that model. Leaving them empty under `random_init` crashes at processor load, not at inference.
   - **Layout (pi05/xvla convention):** for each benchmark ship **exactly one public + one `_internal` pair** (e.g. `object_smoke.yaml` + `object_smoke_internal.yaml`). Public templates use `/path/to/...` placeholders; `_internal` holds machine absolute paths and is the one-click default for internal scripts. Do **not** leave extra full-suite / alternate smoke YAMLs in-tree unless the user asks (optional knobs go in comments on the single pair).
   - **Task-success vs link smoke:** only mark a config as task-success when a matching domain checkpoint was verified. If there is no domain weight, set `server.random_init: true`, empty `ckpt_path`, keep steps short, and comment clearly that it is **not** task-success (see pi05/xvla CALVIN + ManiSkill; also pi05 SimplerEnv).
   - Keep success smokes bounded by default: one task, one episode where knobs allow. Full-suite sizes belong in comments (`max_tasks: 0`, `episodes_per_task: 10`, …), not as a second YAML file.
   - Use local runner knobs where available: LIBERO `max_tasks: 1` and `episodes_per_task: 1`; CALVIN one sequence and low max steps for link smoke (raise toward official EP_LEN only with domain weights); SimplerEnv one task/episode (X-VLA WidowX task-success uses official `max_steps: 1200`); RoboTwin one task with bounded `max_steps`; ManiSkill one episode with low `max_steps` for link smoke.
   - The `model:` section carries two kinds of fields: (a) fields that correspond 1:1 to the model's `XxxModelConfig` dataclass (e.g. `Pi05ModelConfig`, `XvlaModelConfig`), consumed server-side by the factory; and (b) the per-model PayloadBuilder capability fields `state_encoding` / `action_encoding` / `domain_id` / `unnorm_key`, consumed client-side by the PayloadBuilder whitelist. Before writing the YAML, inspect `loongforge/embodied/model/<model>/model_configuration_<model>.py` for the ModelConfig fields and `payload_builders/<model>.py` for the annotated capability attrs. ModelConfig fields not declared in the dataclass are silently filtered out by OmegaConf merge and will not take effect. Do NOT include eval-only or runner-level fields (e.g. `name`, `action_dim` as benchmark-target dim, `num_image_views`) in `model:` beyond the two categories above. Exception: some ModelConfigs legitimately declare `action_dim`/`state_dim` (pi05); xvla uses `real_action_dim` / `action_mode` instead — follow the dataclass.
    - Include `model.backend` and `model.model_type` in the `model:` section. `model.model_type` is **required** — `EvalServerArgs.model_type` has no default and `parse_eval_server_config` raises if the `model:` section omits it.
   - Include infrastructure fields (`ckpt_path`, `tokenizer_path` / `processor_path` when needed, `dataset_statistics_path`, `use_bf16`, `loongforge_root`, `random_init`) in the `server:` section, not `model:`. Proprio layout is now a PayloadBuilder capability: set `model.state_encoding` (e.g. `ee6d`) rather than a `server.state_format` field (that field was removed).
   - Include `server.python`, `server.host`, `server.port`, `server.health_port`, `server.start_timeout_sec`, and `server.log`.
   - Use unique ports per smoke config to avoid health-port collisions during repeated runs.
   - Include `run.output_dir`, `run.seed`, `run.save_trace`, and replay flags when supported.
   - Pair each public YAML with `run_<benchmark>_eval.sh` and each `_internal` YAML with `run_<benchmark>_eval_internal.sh` under `examples/embodied/<model>/eval/`.
   - For models that require a domain identifier (e.g. xvla), put `domain_id` under the `model:` section (it is a PayloadBuilder capability attr, resolved from the YAML override or the per-benchmark default map). Known xvla domain IDs used in-repo: LIBERO=3, CALVIN=2, **SimplerEnv WidowX/Bridge=0**, ManiSkill=5, RoboTwin2=6 (VLABench=8 if ever wired). Do not invent IDs; prefer official eval scripts.
   - Protocol fields that are **not** ModelConfig:
     - `model.state_encoding` / `model.action_encoding` — PayloadBuilder capability attrs; the ActionDecoder key is auto-composed `{action_encoding}_to_{action_space}` at startup, so there is no `benchmark.action_postprocess` field anymore.
     - `benchmark.action_bridge` — RoboTwin form-B wiring selector in `bridges/robotwin_policy.py::_BRIDGE_WIRING` (`pi05_aloha_14d`, `ee6d_dual`).
     - Absolute vs delta control may be implied by the composed decoder / bridge (e.g. LIBERO abs EE when the `ee6d_to_axis_angle` decoder is composed); document it in the YAML header.
   - Document open weight URLs and verified local paths in the YAML header comments (public vs `_internal`).

7. Wire server startup only as needed.
   - Reuse existing LoongForge server routing for pi05-style integrations when possible.
   - Add server-manager routing for a new `model.backend` only when an existing server entrypoint cannot serve it.
   - Ensure health readiness means the model factory has completed, the warmup `predict_action()` call has run (see below), and action RPC can start.
   - Use reusable health server binding for short repeated smoke runs when local patterns support it.
   - After model factory build and before health server startup, run a warmup `predict_action()` with a zero-filled dummy image and an empty instruction. This forces all lazy imports (including potential circular-import paths) to complete before the first real episode arrives. The call is wrapped in a try/except so a warmup exception does not abort startup, but the model must not enter a corrupted state from a warmup call.
   - Before running the smoke matrix, check for leftover orchestrator/policy-server processes and occupied server ports.
   - Keep benchmark client Python env and model server Python env explicit. Run the top-level orchestrator with the benchmark/simulator conda environment, while YAML `server.python` starts the model server environment. **The run script that wires this up does not exist yet in a real adaptation — you author it (step 6 / deliverables) and it must bake in the env activation and all env exports (`PYTHONPATH`, `CUDA_VISIBLE_DEVICES`, `LD_LIBRARY_PATH` so `libcuda` resolves, and simulator vars). Then validate by launching through that generated run script — never a bare `python -m ...`. Running with the wrong conda env or a missing library path is the most common cause of a false failure.**

8. Check runtime-specific traps.
   - For SAPIEN-based benchmarks such as SimplerEnv, RoboTwin, and ManiSkill, verify NVIDIA Vulkan with `vulkaninfo`, not just `nvidia-smi`, when visual rollout correctness matters.
   - Expected Vulkan signal is `deviceName = NVIDIA ...` and `driverName = NVIDIA`; `llvmpipe`/`lavapipe` means visual rollout is not trustworthy.
   - SAPIEN runners may need `LD_LIBRARY_PATH`, `VK_ICD_FILENAMES`, and `XDG_RUNTIME_DIR` set before importing SAPIEN/svulkan2/ManiSkill.
   - **Point `VK_ICD_FILENAMES` at the machine's actual NVIDIA ICD json** (e.g. `/path/to/nvidia_lib/10_nvidia.json`), and add its dir to `LD_LIBRARY_PATH` — copy the exact path from an existing SAPIEN run script rather than guessing `/usr/share/vulkan/icd.d/nvidia_icd.json`. A wrong or missing ICD path makes SAPIEN silently fall back to `llvmpipe` or fail with `vk::createInstanceUnique: ErrorInitializationFailed`; that is a path mistake, not missing hardware. Verify with `VK_ICD_FILENAMES=<icd.json> vulkaninfo | grep -E 'deviceName|driverName'` → expect `NVIDIA`, and only report a Vulkan blocker after the correct ICD still fails.
   - For MuJoCo/LIBERO/CALVIN, preserve existing `MUJOCO_GL`, `PYOPENGL_PLATFORM`, and benchmark config-path patterns.
   - **Offline VLM-backbone loading:** if the model builds a VLM/vision backbone (e.g. GR00T's Eagle) that defaults to a HuggingFace repo id + `trust_remote_code`, it will fail offline unless the backbone code/config is local. Prefer a repo-local build path (for GR00T, env `CUDA_GRAPH_IMPL=local` routes to the bundled Eagle builder) and point `model_name`/tokenizer at the local processor dir. Install the attention backend the checkpoint expects (`flash_attn`) or set the model's `use_flash_attention: false` for sdpa.
   - **Factory `predict_action` wrappers must swallow-not-forward extra kwargs.** The runner passes generic control knobs (`cfg_scale`, `unnorm_key`, ...) that the interface filters by the callable's signature. If your factory wraps `predict_action` and declares `**kwargs`, do NOT forward them to the underlying model (whose signature has none) — absorb and drop them, else you get `unexpected keyword argument 'cfg_scale'`. Mirror the wrapped signature and pass only the known args.
   - **CUDA false-negatives are an environment problem, not a driver rollback.** If `torch.cuda.is_available()` is `False` or you see `Found no NVIDIA driver on your system` / `Error 803: system has unsupported display driver / cuda driver combination` while `nvidia-smi -L` clearly lists GPUs, the cause is almost always that `libcuda.so` is not on `LD_LIBRARY_PATH` for the conda env being used, or you are running the wrong Python (bare system `python` instead of the run script's env). Before concluding a benchmark is `blocked` on GPU/driver grounds, re-check with the run script's exact `LD_LIBRARY_PATH` (e.g. the machine's `nvidia_lib` directory that holds `libcuda.so.<driver-version>`) and the correct conda env; confirm with a real allocation (`torch.zeros(1, device="cuda")`), not just `is_available()`. Only report a GPU/driver blocker after that check still fails.
   - **Verify proprio is actually populated — a missing obs field silently becomes a zero state.** If the PayloadBuilder builds proprio from an adapter `state_raw` field (e.g. `tcp_pose`/`eef_pos`) and the env doesn't provide it, the encoder returns `None` and the model's `validation_zero_state` fallback feeds an all-zero proprio. This is invisible on state-dropout-trained models / forgiving tasks (they still work) but silently cripples precision tasks. Two traps seen: (a) a benchmark env variant omits the field other envs expose (a SimplerEnv drawer env does not override `_get_obs_extra` to emit `tcp_pose`, while grasp/pick envs do) — prefer a checkout that exposes an equivalent field over patching the env; (b) always assert the reconstructed proprio is non-zero and within the checkpoint's `statistics.json` state ranges at reset, rather than trusting the run to "not crash".
   - **Don't kill your own shell.** Never use `pkill -f <pattern>` where the pattern also matches the command line of the run you just launched (e.g. `pkill -f loongforge_server` while your command string contains that string) — it SIGTERMs the launching shell (exit 143 at ~launch). Clean up by port or specific PID instead.

9. Validate in layers. **Adaptation is not complete until tests are actually executed — generating files, YAMLs, and a smoke matrix is not enough. Always finish by running the tests in the correct environment.**
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
   - Run each benchmark **via the run script you generated for it** (author it to activate the right conda env and export `PYTHONPATH` / `CUDA_VISIBLE_DEVICES` / `LD_LIBRARY_PATH` / simulator vars — see step 6 / deliverables; there is no pre-existing script in a real adaptation). Use the benchmark client's conda environment for the top-level orchestrator command: LIBERO with the LIBERO env, CALVIN with the CALVIN env, SimplerEnv with the SimplerEnv env, RoboTwin with the RoboTwin env, ManiSkill with the ManiSkill env; the model server uses the env in YAML `server.python`. If you invoke the orchestrator directly instead of through the script, replicate the script's env exports exactly.
   - **Force a real run when validating — do not be fooled by cached results.** With `run.timestamped_output: false`, an orchestrator run whose `output_dir` already contains a `results.jsonl` will resume/reuse the cached record (`new_records: 0`, `elapsed_sec` ~0) instead of executing a new episode. To prove the integration actually runs now, point `output_dir` at a fresh path (or set `timestamped_output: true`) and confirm `new_records >= 1` with a non-trivial `elapsed_sec` / `avg_inference_latency_ms`.
   - Mark a benchmark `passed` only when the command exits successfully and the expected outputs prove at least one policy call or official runner completion.
   - Mark a benchmark `blocked` when required runtime, simulator assets, checkpoint, stats, or environment support is missing; include the concrete error or missing path. Before marking `blocked` for a GPU/driver/import reason, first re-run in the correct conda env with the run script's full env exports (see step 8 on CUDA false-negatives) — a wrong env or missing `LD_LIBRARY_PATH` must not be reported as a driver/hardware blocker.
   - Mark a benchmark `skipped` only when the user explicitly narrows scope or asks not to run it.
   - Protocol or mock smoke proves runner/server/RPC/action shape.
   - Random-init smoke proves the real model class can initialize and answer RPC, but is **not** a benchmark score and must not be reported as task-success.
   - Real-checkpoint smoke proves the real checkpoint can run one short episode.
   - Credible / task-success requires matching domain checkpoint, matching stats when the model needs them, correct action semantics (PayloadBuilder `state_encoding` / `action_encoding`, the composed ActionDecoder / RoboTwin `action_bridge`, abs vs delta), and enough episodes to support the claim.
   - **To reproduce a published success rate, run the benchmark's official/env-driven eval protocol — not hand-set init.** Many sims (SimplerEnv `prepackaged_config=True`, i.e. `simpler_env.make(task)`) apply an official visual-matching config and randomize overlay + robot/object init per episode via the env's own rng. Hand-setting a single `scene_name`/`rgb_overlay_path`/`robot_init_x/y` is off-distribution and can crater the rate (measured: eggplant 36% hand-set vs 100% prepackaged). Add/flip the runner flag to the official path before comparing to published numbers.
   - **Report the rate over enough episodes; sampling may be stochastic.** Diffusion/flow-matching policies sample from unseeded noise, so a single episode is noisy — run ≥20 and report the fraction.
   - **A passing narrow-action-range benchmark does NOT prove correctness on a wide-range one.** A normalization/scale bug (e.g. min-max vs mean-std over a ±2π euler action) is invisible when the action range is small (LIBERO) but blows up on wide ranges (SimplerEnv widowx). Validate on the actual target benchmark, and treat wide-range/degenerate-action symptoms as a normalization mismatch to reconcile against the checkpoint's own `processor_config.json`.
   - **Transcribe the official eval entrypoint parameter by parameter, from CLI down to the env constructor — before you write the YAML.** Reading the README command line is not enough; follow each flag into the wrapper and the env factory and record where it lands. GR00T-N1.6 × SimplerEnv cost two full eval batches to this: `--max_episode_steps=300` is enforced by `MultiStepWrapper` (`multistep_wrapper.py:271-273`) and counts **inner env steps** while the inner env's own limit is disabled (`_max_episode_steps = 10000`); `simpler_env.make()` **forces** `prepackaged_config=True` (`simpler_env/__init__.py:83`), so there is no official non-prepackaged path; under prepackaged the per-episode variable is the reset seed (`run.seed + run.episode_idx`), not `obj_episode_id`. Measured impact of the step cap alone: successful trajectories take 23~279 steps, so a 120-step cap scored viable rollouts as failures (spoon 10.5% → 60%, carrot 15.8% → 43% after aligning).
   - **Diff proprio value by value, per dimension — not "did we implement the same reconstruction".** An item-by-item static review can pass a channel that is simply wrong. In GR00T-N1.6 × widowx the gripper dim was encoded as the sum of the two finger joint positions (0.074 m when open) while the official wrapper feeds `agent.eef_pos[7] = 1 - get_gripper_closedness()` (1.0 when open); after min-max normalization that is **-0.949 vs +0.789**, i.e. the model was told "gripper fully closed" while it was physically fully open. The wrong value sat inside the training range, so no assertion or range check could catch it — only dumping and comparing each normalized dimension against the official run did.
   - **Do not trust a code comment that declares a channel low-impact.** The same gripper channel carried a comment arguing it was low-impact because the checkpoint was trained with `state_dropout_prob=0.8`. The remaining 20% of steps that do carry state were enough to dominate the gripper decision: the policy closed the gripper at step 3 and retreated (x-velocity mean -0.0056 vs official +0.0078).
   - **Verify a config knob is actually consumed; trace it to its reader.** `server.chunk_execute_steps` is applied in each model factory by truncating the predicted chunk (`groot_n1_6_factory.py:112-135`, `xvla_factory.py:66-95`), not in `GenericPredictActionPolicy` — the generic chunk cache just walks whatever length it receives. A config that omits the field therefore runs the **full** `action_horizon` open-loop with no error and no warning (GR00T LIBERO configs: 16 steps open-loop where official uses `--n_action_steps 8`). Check the recorded `server_metadata.chunk_execute_steps` in `results.jsonl` to confirm what actually took effect.
   - **Run the official stack as an oracle, not just as a reference to read.** It is the only way to learn whether a gap lives in your implementation or in the model, and it supplies per-step reference values for value-by-value comparison. GR00T-N1.6 × `widowx_open_drawer` went 0/5 locally and 5/5 on the official stack with the same checkpoint, which is what redirected the investigation from "config/model" to "our inference path". Later, on a carrot episode with bit-identical object poses, official succeeded in 58 steps while ours ran out at 300 with 27% of the motion magnitude — a signal no amount of code reading had produced.
   - **Static diff cannot catch "implemented but not wired".** The missing `eval_image_transform` already existed in this repo and the training path called it; only the separate eval `predict_action` path skipped it. An item-by-item "do we have this?" review passes. Compare behaviour/values, not the presence of code.
   - **Beware episode-index segment bias when reporting.** Under prepackaged randomization the object layout is a function of `episode_idx`, so a fixed low range is a fixed (possibly hard) sample. Measured on carrot: `episode_idx` 0-9 gave 2/10 = 20% while 20-39 gave 11/20 = 55%. Report over more episodes or over disjoint segments; do not treat one 10-episode block as the rate.
   - **Construct the processor from the checkpoint's `processor_kwargs` as a whole; never hand-transcribe the argument list.** Official loaders do `cls(**processor_kwargs)` (`processing_gr00t_n1d7.py:767`), so a key you forget is a `TypeError`. A hand-written `pk.get("<key>")` per argument fails silently three different ways, and GR00T-N1.7 × LIBERO hit all three: **dropped** (`use_percentiles: true` never read → `StateActionProcessor`'s `use_percentiles=False` default silently switched state *and* action normalization from q01/q99 to min/max), **re-derived** (`crop_fraction` computed as `image_crop_size[0]/image_target_size[0]` = 0.898 instead of reading the stored 0.95), and **over-read** (`letter_box_transform` is in the config but official marks it "stored but not actively used" and always applies `LetterBoxPad`, so following the config's `false` diverges *from* official). Measured: min/max vs q01/q99 shifted normalized proprio by 23% and dropped libero_object to 0/10.
   - **"The checkpoint declares X" is not "our code does X".** Confirming `use_percentiles: true` in `processor_config.json` and then reporting normalization as verified is a false green — the claim needs the reader traced (`grep` the flag from config to the constructor call) or, better, the value dumped. Enumerate every `processor_kwargs` key against the keys your code actually consumes and justify each one you ignore; on this integration 12 of 19 keys were unread, and 11 of those happened to match hardcoded values while one did not.
   - **Golden-reference at tensor level beats reading code, and localizes the fault in one shot.** Run official and ours on a byte-identical observation with the initial flow-matching noise pinned to a shared fixed tensor (monkeypatch `torch.randn` for the sampler's shape), hook the key intermediates (`state_encoder` input, DiT `encoder_hidden_states`), and diff element-wise. On GR00T-N1.7 this returned `vl_embeds` maxabs 0.256 / rel 0.27% (bf16 noise → image pipeline, tokenization, RoPE, and a large transformers-5.x compat layer all correct) against `state_in` rel 23.5% — pointing straight at normalization after days of unproductive suspicion of the backbone. Post-fix the same probe confirmed rel 0.15%.
   - **The official stack may not import under your environment's dependency versions; shim, don't port.** Official n1.7 targets transformers 4.57 while the eval env was on the 5.x series. Getting the oracle running needed: pure-Python stubs for missing deps on `PYTHONPATH` (`tyro.conf.subcommand`, `dm-tree`'s `map_structure`/`items`), direct `Gr00tN1d7(cfg)` construction plus manual `safetensors` load because `AutoModel.from_pretrained` now uses a meta-device init context that collides with the backbone calling `from_pretrained` inside `__init__`, a `language_model`/`visual` property shim for submodules that moved onto `.model`, and a `nvidia/Cosmos-Reason2-2B` symlink under the cwd because `get_backbone_cls` dispatches on that literal substring. All of it lives outside the repo (`/tmp`), mutates no env, and is deleted afterwards. Watch for `tree.map_structure` on `BatchFeature`: it is a `UserDict`, so an `isinstance(x, dict)` stub treats it as a leaf — traverse `collections.abc.Mapping`.
   - **When a shared component is reused across model versions, re-check every default.** N1.7's `predict_action` imports N1.6's `StateActionProcessor`; the class is compatible but its defaults encode N1.6's conventions (`use_percentiles=False`). Reuse hides version-specific defaults behind an interface that still type-checks.
   - **Do not edit training-side shared modules to fix an eval-side mismatch.** The right lever for a train/eval divergence is the eval call site (pass the flag explicitly), not the shared transform. Changing `data/datasets/.../image_augmentations.py` silently changes training behaviour for everyone; report the divergence and let the owner decide.
   - **A policy-side dependency version can outrank every config knob. "It imports and runs" is not "it is numerically equivalent".** If the model's official stack targets a specific transformers (or similar) version, treat that version as a first-class experimental variable, not an environment detail: build a matching env and measure both. GR00T-N1.7 × LIBERO, everything else held fixed, 5 ep/task: `libero_10` 11/50 on transformers 5.3.0 vs **46/50 on 4.57.3**; `libero_object` 28/50 → 48/50. The whole 5.x series diverges, not one release. Nothing raised a warning under 5.x — the shim layer imported cleanly and every episode completed. Point `server.python` at the aligned env, and note that the benchmark-side interpreter is a separate choice (`BENCHMARK_PYTHON`), so the two can and often must differ.
   - **A divergence in the shared backbone applies to every benchmark, not just the one where you found it.** The N1.7 transformers gap was first localized on SimplerEnv WidowX (35/120 → 85/120) and was assumed benchmark-specific for weeks; the same switch then moved LIBERO by 36/50 on one suite. Once a fault is localized below the harness (backbone forward, processor, normalization), re-run *every* benchmark you have already reported rather than scoping the fix to one.
   - **Re-test existing environment pins after every root-cause fix — a pin that "helps" may only be compensating for a bug.** LIBERO/mujoco 2.3.7 looked worth pinning (+14pp on `goal`/`spatial` over 3.2.3) while the backbone was perturbed; after transformers was aligned the mujoco effect vanished (max column difference 3/50, i.e. within noise at n=50 where SE ≈ 3pp). A numerically wrong policy sits near the success threshold, so contact-dynamics differences flip outcomes and manufacture a spurious version effect. Carry the fewest pins you can defend, and do not read a ≤3/50 column difference as an effect.
   - **Read the per-task breakdown, not the suite average — a systematic bug has a signature the average hides.** Under 5.x, `libero_10` was 0/5 on six of ten tasks and near-perfect on the rest; the 20% suite figure is indistinguishable from a weak checkpoint. A cluster of *exactly* 0/N tasks alongside healthy ones is evidence of a systematic fault (normalization, version, decode), not of capability. Verify the breakdown against `results.jsonl` rather than trusting a summary you did not derive, and state whether the task index is 0-based.
   - **The repo's own dependency list is not automatically the eval-correct set.** Our `pyproject.toml` installs `transformers==5.3.0` while Isaac-GR00T declares `4.57.3`, so an env built straight from ours lands N1.7 in the degraded regime (`libero_10` 11/50). Do not "fix" this by editing the shared pin — other models already depend on 5.x behaviour (`configs/models/embodied/xvla.yaml` drops `num_beams`, removed in 5.x). Instead audit the eval env against the extra, keep a separate aligned env for this model, and declare the delta in `loongforge/embodied/eval/docs/patches/<benchmark>/<model>.md`. Expect coupled pins to move together: 4.57.3 forces `huggingface_hub` back to the 0.x series.
   - **Distinguish the model's internal action-sequence length from the number of actions you execute.** For DiT flow-matching heads, `action_horizon` (N1.7: 40) is the sequence length baked into the weights — setting it in eval YAML silently reshapes the head and corrupts inference. The number of predicted steps you actually run open-loop is `server.chunk_execute_steps` (N1.7 LIBERO: 8, from the embodiment's `delta_indices`). Two different knobs with similar names; only the second belongs in an eval config.
   - **Use the benchmark checkout the model's official repo pins, before you consider porting anything into another one.** Which SimplerEnv checkout is in `env.simplerenv_root` decided how much work the GR00T integration looked like. Against upstream `simpler-env/SimplerEnv` the two WidowX drawer tasks are unregistered and `agent.eef_pos` is absent, so it took a 3-file task port plus a hand-added `_get_obs_extra` — all of which became unnecessary once the checkout was switched to the fork Isaac-GR00T pins as its `external_dependencies/SimplerEnv` submodule. Check the official repo's `.gitmodules` and setup scripts, not just its README: the README had stopped naming the fork while the submodule url still pointed at it. Also expect one checkout per policy family rather than one per machine — X-VLA needs a controller registration the GR00T fork lacks, so the two coexist.
   - **Do not add a guard for a mismatch that cannot occur; measure the three numbers first.** A checkpoint-vs-config `action_horizon` validator was written for both GR00T factories after the `8` vs `40` incident. It was dead code in both: the dataclass default, every released checkpoint, and every shipped YAML agreed (N1.6 50/50/unset, N1.7 40/40/unset), so neither the correction branch nor the raise could ever fire. Before hardening a field, print the default, the checkpoint value, and what the configs actually set. If they agree, a comment stating the constraint is the whole fix.
   - **State reproducibility explicitly: these rollouts are not deterministic.** The flow-matching noise comes from unseeded global `torch.randn`, so a fixed env seed does not pin the trajectory and the same config rerun gives a different number. Say so in the patch doc, and never present a small delta between two runs as a change in behaviour.
   - After shared eval or model-inference changes (especially image view packing in the PayloadBuilder / `num_images`), re-run a small known-good regression (e.g. pi05×LIBERO smoke) before claiming no impact.

10. Update docs with precise status.
    - Separate `mock`, `random-init`, `real checkpoint`, and `credible score` statuses.
    - Put user-facing usage in README/user guide; avoid filling README with internal validation logs.
    - Put detailed interface contract and local interface-validation examples in `model_integration.md`.
    - Mention missing assets directly, especially checkpoint and `dataset_statistics.json`.
    - Record runtime requirements that future users must set before import.
    - **Declare every deviation from the benchmark's official versions/dependencies in `loongforge/embodied/eval/docs/patches/<benchmark>/<model>.md`.** The default assumption is that we run the benchmark's official version untouched, so anything model-specific (a pinned dependency on the policy side, a source patch, an overridden step cap, a render backend) must be stated there with the evidence that motivated it. This document declares *deviations*, not troubleshooting history: no config mistakes, no debugging narrative. It is public — no absolute or personal paths, and no `sweep:` (an internal-only config mechanism; explain the episode variable in a comment instead).

## Pi05 reference mapping

Use pi05 as the canonical example of the shared `predict_action` architecture:

- Model interface: `PI05Policy.predict_action(images, instructions, state=None, dataset_stats=None)` in the model package.
- Interface helpers: `loongforge/embodied/eval/servers/predict_action_interface.py` — the model-author contract only (`PredictActionModel`, `validate_predict_action_model`, `_filter_supported_kwargs`, `call_predict_action`). It contains no action-decode logic.
- PayloadBuilder: `Pi05PayloadBuilder` in `loongforge/embodied/eval/payload_builders/pi05.py` (image packing `_pack_images`, `state_encoding` proprio, `unnorm_key`); registry `build_payload_builder` in `payload_builders/registry.py`; base class in `payload_builders/base.py`. xvla: `XVLAPayloadBuilder` in `payload_builders/xvla.py` (reuses `_pack_images`, ee6d proprio, `domain_id`).
- ActionDecoder: `loongforge/embodied/eval/action_decoders/` — `base.py` (`ActionDecoder`, `IdentityDecoder`, `register_action_decoder`, `build_action_decoder`), `rotation.py` (rot6d math), `ee6d.py` (`ee6d_to_axis_angle` / `ee6d_to_euler` / `ee6d_to_quat` / `ee6d_to_calvin_abs` / `ee6d_to_simpler_abs_euler` / `ee6d_robotwin_ee_dual`), `joint.py` (`pi05_aloha_robotwin`). Key auto-composed by `resolve_action_decoder_key(payload_builder, adapter)` in `orchestrator/config.py`.
- Generic eval policy: `GenericPredictActionPolicy` in `loongforge/embodied/eval/servers/loongforge_policy.py` (chunk caching, predict_action invocation, output shaping; no image packing — RPC payload v2: `images` list of views + `instructions` list[str]).
- Model factory: `PI05ModelFactory` in `loongforge/embodied/eval/factories/pi05_factory.py`; `XVLAModelFactory` in `factories/xvla_factory.py` (still wraps `predict_action` for `domain_id` int → LongTensor).
- Factory registry: `register_factory`, `build_model_spec` in `loongforge/embodied/eval/factories/registry.py`. New models register with `@register_factory("<model_type>")` and declare `model_config_cls = <ModelConfig>`.
- Startup consistency check: the orchestrator asserts `set(MODEL_FACTORY_REGISTRY) == set(PAYLOAD_BUILDER_REGISTRY)`, so a factory without its paired PayloadBuilder (or vice-versa) fails fast.
- Server config: `EvalServerArgs` dataclass and `parse_eval_server_config` in `loongforge/embodied/eval/servers/eval_server_config.py` (the `state_format` field was removed; `model_type` has no default and is **required** — the parser raises when `model.model_type` is missing). The YAML `server:` section is merged directly into `EvalServerArgs` via OmegaConf; the YAML `model:` section is merged into the registered `ModelConfig` (e.g. `Pi05ModelConfig`) via OmegaConf, and its PayloadBuilder capability fields (`state_encoding` / `action_encoding` / `domain_id` / `unnorm_key`) are read by the PayloadBuilder whitelist.
- Backward-compatible wrappers: `LoongForgePI05Policy` / `LoongForgeXVLAPolicy` in the respective factory modules, which should not be the preferred pattern for new integrations (still exist).
- Server entrypoint: `loongforge/embodied/eval/servers/loongforge_server.py` calls `parse_eval_server_config` to get `EvalServerArgs` + `raw_model_dict`, then `build_model_spec` to load the model, then `_warmup_model()` to resolve lazy imports, then wraps in `GenericPredictActionPolicy`.
- Routing: `loongforge/embodied/eval/orchestrator/server_manager.py` maps `loongforge`, `pi05`, and `loongforge_pi05` to the LoongForge server.
- Adapter state boundary: benchmark adapters provide `canonical_obs["state"]` for native structured state and `canonical_obs["state_raw"]` for raw obs fields; the PayloadBuilder encodes `state_raw` per its `state_encoding` into RPC payload `state` (no adapter `model_state`). Adapters also declare `action_space` / `default_fps` / `cameras` capability class attrs.
- YAML + scripts layout: `examples/embodied/pi05/eval/configs/<benchmark>/{smoke,object_smoke,adjust_bottle_smoke,...}.yaml` + matching `_internal.yaml`; `run_<benchmark>_eval.sh` / `_internal.sh`. Same shape under `examples/embodied/xvla/eval/`.
- Infrastructure fields (`ckpt_path`, `tokenizer_path`, `dataset_statistics_path`, `use_bf16`, `loongforge_root`, `random_init`) live in `server:`; ModelConfig fields and PayloadBuilder capability fields live in `model:` (pi05: `action_dim`, `action_horizon`, `state_encoding`, …; xvla: `action_mode`, `real_action_dim`, `num_actions`, `state_encoding`, `domain_id`, …).
- RoboTwin bridge: `bridges/robotwin_policy.py::_BRIDGE_WIRING` — pi05 uses `action_bridge: pi05_aloha_14d` → (pi05, `aloha_pi`, `pi05_aloha_robotwin`) (+ stats asset under `examples/embodied/pi05/eval/assets/`); xvla uses `ee6d_dual` → (xvla, `ee6d_dual`, `ee6d_robotwin_ee_dual`) + `domain_id: 6`.
- xvla extras: factory wrap for `domain_id` tensor; `action_encoding: ee6d` composes decoders for LIBERO/SimplerEnv/CALVIN automatically; SimplerEnv abs EE may need env patch (`examples/embodied/xvla/eval/SIMPLERENV_PATCH_en.md`).

For new models, create `loongforge/embodied/eval/factories/<model>_factory.py` with a `@register_factory("<model_type>")` class that declares `model_config_cls` and implements `build(model_cfg, server_args) -> PredictActionModelSpec`, and pair it with `loongforge/embodied/eval/payload_builders/<model>.py` registered via `@register_payload_builder("<model_type>")` (same key). Add an ActionDecoder under `action_decoders/` only if the model's action encoding is not already covered. Add the module paths to `_FACTORY_MODULES` / the payload-builder auto-import list in the respective registries. No changes to `loongforge_server.py` are needed.

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
