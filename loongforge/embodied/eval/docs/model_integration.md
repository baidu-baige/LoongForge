# Guide to Integrating a New Model into the Eval System

This document summarizes the complete workflow, key configuration items, and the `predict_action` interface contract that the model side must implement in order to integrate a new VLA model into the `loongforge/embodied/eval` eval system.
Beyond implementing a Factory + PayloadBuilder (+ an optional ActionDecoder) and writing the YAML config, integration also involves a series of model-semantics-level configuration points that must be confirmed one by one.
pi05 and xvla differ almost entirely on these configuration points, so this document uses the two as examples to walk through each item and provides a configuration comparison table.

**Scope note:** The semantic checklist below covers all benchmarks currently integrated (LIBERO / RoboTwin / SimplerEnv / CALVIN / ManiSkill).
Early integration used LIBERO as the primary acceptance path, so some items are illustrated with LIBERO; cross-benchmark protocol differences (such as RoboTwin `action_bridge`) are called out separately.
For the config layout conventions see [§1](#1-overview); for task success status see `../README.md` / `user_guide_en.md`.

Overall architecture (a four-stage chain that decouples the model from the benchmark):

```text
Adapter (obs -> canonical, exposes state_raw)
  -> PayloadBuilder (canonical -> predict_action kwargs, per model, capabilities declared via class attributes)
  -> model.predict_action (policy server, server side)
  -> ActionDecoder (raw chunk -> env action; key auto-assembled as {action_encoding}_to_{action_space}, identity -> IdentityDecoder)
```

---

## 1. Overview

Integrating a new model = **Factory + PayloadBuilder + optional ActionDecoder**, with the three separated by responsibility (component details in [§3](#3-component-details)).
Add new files, and **do not modify existing files**.

**Components** — responsibilities:

- **ModelFactory** (server side, requires torch) — `factories/<model>_factory.py`, registered via `@register_factory("<model_type>")`; `build()` returns `PredictActionModelSpec(model, metadata)`. Responsibilities:
  - load the model: import, config, weight loading / `random_init`, device/dtype
  - wrap it behind the unified `predict_action(images, instructions, state=None, dataset_stats=None, **kwargs)` interface (see [§2](#2-the-predict_action-contract))
- **PayloadBuilder** (client side) — `payload_builders/<model>.py`, registered via `@register_payload_builder("<model_type>")`, inherits from `PayloadBuilder`. Responsibilities:
  - declare capabilities via type-annotated class attributes (`state_encoding` / `action_encoding` / `action_dim` / `action_horizon` / `domain_id` / `unnorm_key`, ...; YAML overrides same-name fields)
  - `build(canonical, ctx)`: canonical dict → `predict_action` kwargs
  - optionally `reset` / `update_from_response` / `note_env_action` for closed-loop feedback

- **ActionDecoder** (optional, eval side) — `action_decoders/`, only when the model's `action_encoding` has no decoder yet. Decoder selection:
  - key auto-assembled as `{action_encoding}_to_{action_space}` (see [§4](#4-state--action-semantics))
  - source == target space → `IdentityDecoder` passthrough, no new code
  - hits an already-registered key (e.g. `ee6d_*`) → selected automatically, no new code

**Steps** — integration flow:

1. **Implement the model components** — ModelFactory + PayloadBuilder (+ optional ActionDecoder), per the responsibilities above and the details in [§3](#3-component-details).
2. **Implement the `predict_action` interface** — the model must expose `predict_action(images, instructions, state=None, dataset_stats=None)`; model-specific preprocessing and normalization live inside it, and the eval server calls it over RPC (see [§2](#2-the-predict_action-contract)).
3. **Write the eval config and scripts** (`examples/embodied/<model>/eval/`) — model-specific YAML fields (`state_encoding` / `action_encoding` / `domain_id`, ...) go under `model:`; `model.model_type` is **required** (no default). Verify each config point in [§4](#4-state--action-semantics) and [§5](#5-engineering-parameters).
4. **Run a smoke** (1 task × 1 episode, or a chain smoke) to validate the RPC / action semantics, then — with domain weights — task-success and full-scale eval.
5. **Regression-test** already-succeeding combinations (at least pi05 × LIBERO) after changing **shared** code (runner / adapter / decoder / bridge / generic policy).

> **Startup fail-fast:** the Factory and the PayloadBuilder are registered
> independently, and a missing one fails fast on its own side — the eval server
> exits in `build_model_spec` when `model.model_type` has no registered factory,
> and the runner exits in `build_payload_builder` when it has no registered
> PayloadBuilder. Both must be registered for a run to start.

**Runtime path** — once a benchmark runner exists, do not write a per-model policy class:
the server side is uniformly handled by `GenericPredictActionPolicy` for RPC / caching / statistics path / shape checking,
and the model only implements `predict_action` in its factory:

```text
loongforge_server.py
  -> <Model>ModelFactory.build(...)              # factories/<model>_factory.py
  -> GenericPredictActionPolicy(...)
  -> model.predict_action(images, instructions, state=None, dataset_stats=None, **kwargs)
```

---

## 2. The `predict_action` contract

This section is the interface contract between the **model author** and the eval stack, aimed at the model owner who implements `predict_action()`.
The **single source of truth** for the helper functions is `loongforge/embodied/eval/servers/predict_action_interface.py`:
after refactoring, this file **only retains the model-author contract** — the `PredictActionModel` protocol, `validate_predict_action_model`,
`_filter_supported_kwargs`, and `call_predict_action`; it **does not contain** any action-space decoding logic,
which has been moved to `loongforge/embodied/eval/action_decoders/` (see [§4](#4-state--action-semantics)).

### 2.1 Required signature

```python
def predict_action(images, instructions, state=None, dataset_stats=None, **kwargs):
    """Return decoder-ready actions as a float array."""
    ...
```

Parameter meanings:

| Parameter | Typical type | Meaning |
|---|---|---|
| `images` | a batch-of-1 list-of-list (each element is a view array) | e.g. `[[view_primary, view_wrist]]`. The PayloadBuilder packs the per-camera dict into a view list (prefer `primary`/`head`; if `left` and `right` both exist → `[primary, left, right]`; otherwise append at most one more `wrist` → `[primary, wrist]`; single view → `[primary]`); the server then wraps it into batch-of-1. |
| `instructions` | `list[str]` | Batched language instructions (batch=1). |
| `state` | `None` or a numeric vector / array | The **model-usable** proprio the PayloadBuilder produces from `canonical["state_raw"]` according to `state_encoding` (may be `None`). |
| `dataset_stats` | `dict` or `None` | Loaded by the eval side from `server.dataset_statistics_path` and **passed through**; used internally by the model as needed for state normalization / action denormalization. |
| Extra kwargs | e.g. `domain_id`, `unnorm_key` | Emitted by the PayloadBuilder. Passed through only when the signature accepts them (explicitly named or `**kwargs`); unknown ones are dropped with a WARNING. |

Images are a batch-of-1 list-of-list (each inner element is a view array), and instructions are `list[str]`.
They are delivered to the model via the RPC v2 payload.

### 2.2 kwargs filtering

`call_predict_action` inspects the signature before calling:

- Signature contains `**kwargs` → all extra keywords are passed through.
- Fixed signature → only the declared parameters are kept. Any field the PayloadBuilder emits but the model does not declare (such as a mistakenly emitted
  `domain_id` / `unnorm_key`) is dropped and a **WARNING is logged**, making a misconfigured PayloadBuilder
  visible in the logs rather than producing a "no error but wrong result" run.

If a model needs to consume a certain field, it must declare that parameter in the signature (or use `**kwargs`).

### 2.3 Where `state` comes from

The adapter **no longer encodes** proprio; it only exposes the raw fields in `canonical["state_raw"]`
(`eef_pos` / `eef_quat` / `ee_ori_mat` / `joint` / `endpose` / `robot_obs`, etc.);
**each model's PayloadBuilder** encodes them into the `state` kwarg according to its `state_encoding`
(`""` → no state; `ee6d` / `aloha_pi` / `passthrough`, etc.).

The PayloadBuilder's `build(canonical, ctx)` returns the model kwargs; the runner adds RPC control fields around them:

```python
model_kwargs = payload_builder.build(canonical_obs, ctx)  # {"images": [...], "instructions": [...], "state": ...}
rpc = {"episode_id": ..., "episode_step": ..., "disable_action_cache": ..., "return_action_chunk": ...}
rpc.update(model_kwargs)
```

```text
canonical.state_raw  ->  PayloadBuilder(state_encoding)  ->  RPC payload.state  ->  predict_action(state=...)
```

### 2.4 Validation helpers

```text
loongforge/embodied/eval/servers/predict_action_interface.py
```

| API | Purpose |
|---|---|
| `PredictActionModel` | Protocol |
| `validate_predict_action_model(model)` | Pre-call signature check |
| `_filter_supported_kwargs(func, kwargs)` | Drops kwargs the signature does not accept (one WARNING logged per dropped item) |
| `call_predict_action(model, images, instructions, state, dataset_stats, action_dim, **kwargs)` | validate → filter kwargs → call → reshape / truncate to `[H, action_dim]` |

`GenericPredictActionPolicy` always calls through `call_predict_action`. Action-space decoding is **not here**; it is in `action_decoders/`.

`validate_predict_action_model` checks: `predict_action` exists and is callable; the required parameters `images` / `instructions` are present;
the optional parameters `state` / `dataset_stats` can be accepted (as named parameters or via `**kwargs`).

```python
# invalid: missing instructions, and cannot accept state/dataset_stats
def predict_action(self, images):
    ...

# valid
def predict_action(self, images, instructions, state=None, dataset_stats=None):
    ...

def predict_action(self, images, instructions, **kwargs):
    state = kwargs.get("state")
    dataset_stats = kwargs.get("dataset_stats")
    ...
```

### 2.5 Action output contract

The model may return `[D]` / `[H, D]` / `[B, H, D]`; `call_predict_action` uniformly normalizes to `[H, action_dim]`:

| Input shape | Behavior |
|---|---|
| `[D]` | → `[1, D]` |
| `[H, D]` | Kept as a chunk |
| `[B, H, D]` | → `[-1, D]` (single-request path) |
| Other ndim | `ValueError` |
| Last dim `< action_dim` | `ValueError` |
| Last dim `> action_dim` | **Truncated** to the first `action_dim` columns |

`action_dim` comes from the model / YAML config (e.g. 7 for single-arm, 14 for RoboTwin joints, 20 for xvla ee6d). **Truncation cannot replace correct action semantics.**

**Normalization / denormalization belongs to the model:** the eval does **not** perform q01/q99, mean/std, or min/max outside the model.
If the network outputs normalized actions, they must be denormalized **inside** `predict_action` using `dataset_stats` (and the training normalization mode).
For example, pi05's ACTION quantile normalization uses `dataset_stats["action"].q01` / `.q99`; other LeRobot-family models may use mean/std or min/max.
The return value should be in the **model action space** (i.e. the encoding declared by the PayloadBuilder's `action_encoding`), and either match the environment after truncation
(e.g. pi05 LIBERO 7D `axis_angle`), or be converted by an eval `ActionDecoder` / RoboTwin `action_bridge` (e.g. xvla 20D ee6d).

Action decoding on the eval side does **not** belong to `predict_action`: after the server returns the chunk, the runner applies the decoders in `action_decoders/`:

```python
from loongforge.embodied.eval.action_decoders import build_action_decoder
from loongforge.embodied.eval.orchestrator.config import resolve_action_decoder_key

key = resolve_action_decoder_key(payload_builder, adapter)  # {action_encoding}_to_{action_space}
decoder = build_action_decoder(key)                         # empty key -> IdentityDecoder
env_actions = decoder(raw_chunk, ctx)                       # __call__(actions[H, D], ctx) -> env_actions
```

### 2.6 Local interface validation

No GPU / weights needed; use this to validate the eval helpers before fully loading the model:

```bash
cd /workspace/LoongForge-VLA
PYTHONPATH=/workspace/LoongForge-VLA python - <<'PY'
import numpy as np
from loongforge.embodied.eval.servers.predict_action_interface import (
    call_predict_action,
    validate_predict_action_model,
)

class FixedSigModel:
    """pi05-style: extra kwargs are filtered out."""

    def predict_action(self, images, instructions, state=None, dataset_stats=None):
        return np.zeros((len(instructions), 4, 7), dtype=np.float32)

class KwargsModel:
    """xvla-style: domain_id is forwarded."""

    def predict_action(self, images, instructions, state=None, dataset_stats=None, **kwargs):
        assert kwargs.get("domain_id") == 6
        return np.zeros((4, 20), dtype=np.float32)

images = [[np.zeros((224, 224, 3), dtype=np.uint8)]]
common = dict(instructions=["pick up the cube"], state=None, dataset_stats=None)

m = FixedSigModel()
validate_predict_action_model(m)
print(call_predict_action(m, images=images, action_dim=7, do_sample=False, cfg_scale=1.5, **common).shape)

m = KwargsModel()
validate_predict_action_model(m)
print(call_predict_action(m, images=images, action_dim=20, domain_id=6, **common).shape)
PY
```

Expected output:

```text
(4, 7)
(4, 20)
```

Comparison of the `predict_action` output of the real pi05 / xvla after loading via factory (`random_init` is enough for a contract check,
but full correctness (unnorm, abs/delta, decode) still requires a YAML smoke / task-success run, see `user_guide_en.md`):

| Model | Raw `predict_action` | After `call_predict_action` |
|---|---|---|
| pi05 | `[B, action_horizon, max_action_dim]`, e.g. `(1, 50, 32)` | truncate last dim → `(50, 7)` |
| xvla | `[B, num_actions, real_action_dim]`, e.g. `(1, 30, 20)` | reshape + keep dim → `(30, 20)` |

- pi05 **requires** `tokenizer_path` (PaliGemma tokenize) even with `random_init`.
- xvla **requires** a valid Florence processor/tokenizer directory (`tokenizer_path`); an empty path causes HF loading to fail.
- The xvla factory converts the `domain_id` int → a `LongTensor` on the device; `call_predict_action(..., domain_id=3)` can just pass a YAML-style int.

### 2.7 Warmup and common errors

Before the health endpoint is ready, the server may make one call first (`images=[[np.zeros((224,224,3), uint8)]]`, `instructions=["warmup"]`, `state=None`, `dataset_stats=None`).
A failure only logs a warning, but this call must not corrupt weights or render the process unusable; prefer a lazy import that can complete safely.

| Error | Cause |
|---|---|
| `TypeError: model must expose a callable predict_action(...)` | Missing method |
| `TypeError: ... missing required parameters: ['instructions']` | Wrong signature |
| `TypeError: ... cannot accept eval keyword parameters: ['state']` | No `state` and no `**kwargs` |
| `ValueError: ... unsupported action shape` | Not `[D]` / `[H,D]` / `[B,H,D]` |
| `ValueError: ... action dim X, expected at least Y` | Output dimension narrower than `action_dim` |
| The env has steps but the success rate is always 0 | Usually the control mode (abs vs delta), a wrong ActionDecoder / bridge, or wrong unnorm — **not** a missing `predict_action` |

Self-check for the model owner before delivery: `predict_action(images, instructions, state=None, dataset_stats=None)` is callable;
`validate_predict_action_model` passes; returns `[D]`/`[H,D]`/`[B,H,D]` with last dim ≥ `action_dim`;
denormalization (if any) happens **inside** `predict_action`, and the eval only passes through `dataset_stats`; warmup is safe;
the factory is responsible for loading weights / tokenizer / processor and does not rewrite the benchmark dict observation.

---

## 3. Component details

Factory / PayloadBuilder / ActionDecoder are summarized in [§1](#1-overview); this section walks through each component in detail and covers the RoboTwin `action_bridge`.

### 3.1 ModelFactory

Location: `factories/<model>_factory.py`; register with `@register_factory("<model_type>")`. Declare `model_config_cls` and implement `build(model_cfg, server_args) -> PredictActionModelSpec`:

- `model_config_cls`: the typed config dataclass resolved from the YAML `model:` section (e.g. `Pi05ModelConfig`).
- `build(...)`: load the model — import, config, weights / `random_init`, device/dtype — and return `PredictActionModelSpec(model=..., metadata={...})`, where `model` implements `predict_action` (see [§2](#2-the-predict_action-contract)).
- The registry key must pair with a PayloadBuilder of the same `model_type` (both must be registered for the run to start — see the startup fail-fast in [§1](#1-overview)).

References: `factories/pi05_factory.py`, `factories/xvla_factory.py`, `factories/groot_n1_6_factory.py`, `factories/groot_n1_7_factory.py`.

### 3.2 PayloadBuilder

Location: `payload_builders/<model>.py`; register with `@register_payload_builder("<model_type>")` and inherit from `PayloadBuilder`:

- Capability class attributes: `state_encoding` / `action_encoding` / `action_dim` / `action_horizon` / `domain_id` / `unnorm_key`, ... — YAML `model:` fields of the same name override them.
- `build(canonical, ctx) -> dict`: canonical observation (from the adapter) → `predict_action` kwargs — image packing per the view policy, state per `state_encoding`, model-specific fields.
- Optional closed-loop hooks: `reset()` / `update_from_response()` / `note_env_action()`.

References: `payload_builders/pi05.py`, `payload_builders/xvla.py`, `payload_builders/groot_n1_6.py`, `payload_builders/groot_n1_7.py`.

### 3.3 ActionDecoder

Components: `action_decoders/` (`base.py` defines the `ActionDecoder` base class + `IdentityDecoder` +
`ACTION_DECODER_REGISTRY`; `ee6d.py` holds the ee6d source-encoding decoders; `joint.py` holds the joint source-encoding decoders;
`rotation.py` stores the rotation math). The orchestrator **automatically assembles** the decoder key from
`{payload_builder.action_encoding}_to_{adapter.action_space}` (`resolve_action_decoder_key`), and when the source encoding == target space it returns an empty key → `IdentityDecoder` passthrough.

Registered keys (`ACTION_DECODER_REGISTRY`) auto-assembled as `{action_encoding}_to_{action_space}`:

| key | Use |
|---|---|
| `ee6d_to_axis_angle` | xvla × LIBERO / ManiSkill: 20D EE6D → 7D (pos + axis-angle + grip) |
| `ee6d_to_simpler_abs_euler` | xvla × SimplerEnv WidowX: rot6d→euler + offset + grip mapping |
| `ee6d_to_calvin_abs` | xvla × CALVIN official absolute-pose protocol |
| `ee6d_to_euler` / `ee6d_to_quat` | Other EE variants |
| `pi05_aloha_robotwin` | pi05 × RoboTwin joint decoder (stateful; via bridge) |
| `ee6d_robotwin_ee_dual` | xvla × RoboTwin dual-arm ee decoder (via bridge) |

- pi05: `action_encoding == adapter.action_space` (e.g. `axis_angle`) → empty key → passthrough.
- xvla: `action_encoding: ee6d` × each benchmark's `action_space` → auto-select the key from the table above.

If the new model's output is inconsistent with the environment's native action space, register the corresponding decoder under `action_decoders/`
(`@register_action_decoder("<encoding>_to_<space>")`);
**do not** write environment special-casing into the training-side `predict_action`.

### 3.4 `benchmark.action_bridge`

Most benchmarks run through the standard runner chain: the eval runner drives the env, calls `predict_action` over RPC, and applies the decoded action. RoboTwin is different: the official protocol only exposes a policy-plugin interface — the official evaluator (`script/eval_policy.py`) owns the env (observation collection, stepping, success judgment) and reverse-calls the policy plugin. We therefore run the official evaluator as a subprocess, and `action_bridge` selects the `(model_type, PayloadBuilder state_encoding, decoder key)` wiring hosted inside the plugin (`bridges/robotwin_policy.py`), so the official evaluator can drive a LoongForge policy without touching the model's default behavior.

Implementation: `bridges/robotwin_policy.py` (`_BRIDGE_WIRING` maps a bridge name to
`(model_type, payload-builder state_encoding, decoder key)`, assembling the shared
adapter → PayloadBuilder → PolicyClient → ActionDecoder four-component chain).

| bridge | Use |
|---|---|
| `pi05_aloha_14d` | **pi05 RoboTwin official protocol** (`Pi05PayloadBuilder(state_encoding="aloha_pi")` + `pi05_aloha_robotwin` decoder: adapt_to_pi + delta→abs, stateful) |
| `ee6d_dual` | **xvla RoboTwin official protocol** (`XVLAPayloadBuilder(state_encoding="ee6d_dual")` + `ee6d_robotwin_ee_dual` decoder; 20D EE, three views, `action_type='ee'`) |

Protocol logic is placed in named bridge modes, avoiding changes to the model's default behavior that would affect other benchmarks.

---

## 4. state / action semantics

This chapter lists **what to confirm** when integrating a model. The per-model
values are not duplicated here — they live with the model itself: the
PayloadBuilder class attributes and comments in `payload_builders/<model>.py`
(`state_encoding` / `action_encoding` / `action_dim` / `action_horizon`, and the
supported encoding values), the factory, and that model's eval YAMLs under
`examples/embodied/<model>/eval/configs/`.

### 4.1 Action space and dimensions

The action dimension **changes with the benchmark protocol** and cannot simply be hard-coded to LIBERO
(a dual-arm benchmark such as RoboTwin needs a different protocol from single-arm LIBERO,
which is what `benchmark.action_bridge` selects).

If the model's `action_encoding` equals the env `action_space`, the decoder key is
empty and an `IdentityDecoder` passes the action straight through; otherwise the
decoder key is auto-assembled as `{action_encoding}_to_{action_space}`.

Points to confirm: the total number of model output dimensions, the semantic layout of each dimension (position / rotation / gripper), the rotation representation
(axis-angle, 6D rotation, quaternion), and the target environment's control interface (joint vs EE).

### 4.2 Control mode

Confirm whether the model emits **delta** or **absolute** targets and whether the env
is configured to match: `benchmark.control_mode` (`delta` / `absolute` / `auto`,
where `auto` infers from the decoder key) plus, for some envs, a controller that
must exist on the env side (e.g. SimplerEnv WidowX absolute EE, see
[patches/simplerenv/xvla.md](patches/simplerenv/xvla.md)).

It should be emphasized that an absolute pose cannot be crudely turned into a delta by "linearly subtracting" the current pose:
axis-angle rotation does not satisfy linear subtraction, and the delta mode often has action scaling. A wrong control mode is
the primary reason xvla initially had a 0 success rate on LIBERO.

### 4.3 Proprioceptive input

Proprio (proprioception) is the robot's own state — joint positions, end-effector pose, gripper state — as opposed to external perception (cameras). In this framework it is what the model receives as the `state` argument of `predict_action`.

The adapter **no longer encodes** proprio: it only exposes the raw EE / joint fields in `canonical["state_raw"]`
(`eef_pos` / `eef_quat` / `ee_ori_mat` / `joint` / `endpose` / `robot_obs`, etc.);
the encoding is done by **each model's PayloadBuilder** according to `model.state_encoding`
(the supported values are listed on each PayloadBuilder class).

Points to confirm:

- Every slot must reproduce the **training-time definition**, value by value, from the
  official wrapper — not a physically similar quantity. A normalized gripper openness
  and a finger width in metres both "look like a gripper state" but land in completely
  different places after normalization, and nothing errors out.
- Rotation layout (e.g. column-major vs interleaved 6D) — a wrong layout silently
  shifts the input distribution; this was the second reason xvla scored 0 on LIBERO.
- Whether proprio should be the env's measured state or a closed-loop feedback of the
  previous predicted action (the official client decides; the stateful encodings
  implement feedback via `update_from_response` / `note_env_action`).

Boundary: `canonical["state_raw"]` holds the raw fields (for the PayloadBuilder to encode + for trace/debug);
the `state` kwarg produced by the PayloadBuilder's `build()` enters `predict_action(state=...)` via RPC.
Do not pass a nested dict directly as `state` to `predict_action` (unless the model explicitly declares that layout);
prefer passing a flat `float32` vector aligned with the training `observation.state`.

### 4.4 Normalization approach

Normalization/denormalization stays **inside the model** (`predict_action`); the generic
policy never unnorms (see the ownership convention in [§2.5](#25-action-output-contract)).
What differs per model is only where the statistics come from — an external
`dataset_statistics.json` via `server.dataset_statistics_path`, the model's own action
space, or config files shipped with the weights.

Points to confirm: the stats source and the normalization mode per key are read from the
**weights' own config**, not hand-derived from observed value ranges; and denormalization
happens in exactly one place. A benchmark whose action ranges are small can pass while a
wide-range benchmark fails — a passing benchmark does not prove the normalization is right.

### 4.5 Model-specific request fields

Model-specific fields are declared by the **PayloadBuilder** and injected into the `predict_action` kwargs; the YAML writes them under the `model:` section. A multi-domain model, for instance, needs a domain/task id chosen per benchmark (the PayloadBuilder may default it from the benchmark name), and the factory converts it to whatever tensor type the model expects. A misconfiguration usually produces **no error**, but the action distribution is wrong — so treat these fields as part of the protocol and take their values from the official eval / production config, never invent them.

The same applies to a new model's task embedding / domain embedding / special prompt, etc.:
`model:` YAML → PayloadBuilder → RPC payload → factory → model, end to end.
If the PayloadBuilder emits a kwarg the model signature does not accept, `_filter_supported_kwargs` drops it with a
WARNING (see [§2.2](#22-kwargs-filtering)); use this to diagnose misconfigurations.

### 4.6 Number and order of image views

View packing follows these rules (**not** a fixed 2 views):

1. There must be a `primary` or `head`;
2. If `left` and `right` both exist → **3 views** `[primary, left, right]` (RoboTwin / X-VLA official);
3. Otherwise append at most one more `wrist` (or a standalone left/right) as the 2nd view.

(Implemented in `payload_builders/pi05.py::_pack_images`, which the xvla PayloadBuilder reuses.)

The adapter declares its camera set with the `cameras` class attribute (e.g. LIBERO `("primary", "wrist")`),
and the PayloadBuilder packs the model's expected view list from `canonical["images"]` (a per-camera dict) per the rules above.
The view count therefore follows the adapter, not a per-model setting: the model side must support a **dynamic**
`num_images = len(images[0])` and **must not** hardcode `num_images=2` or `3` (otherwise it breaks other benchmarks).

Points to confirm: the number, order, and resolution of cameras at training time, and whether they are flipped
(LIBERO agentview vertical flip is handled uniformly by the adapter).

Views are dynamically packed by the PayloadBuilder from the obs; do not rely on fabricated YAML fields to control the number of views.

---

## 5. Engineering parameters

- **Load timeout (`start_timeout_sec`):** xvla cold start can be >900s, and task-success configs commonly use 2400;
  900 is usually enough for pi05; a `random_init` chain smoke can be shorter.
- **processor / tokenizer:** pi05 needs an external paligemma (`tokenizer_path`);
  xvla's processor is in the weight directory (`processor_path` / `tokenizer_path` are often the same directory).
- **`server.random_init`:** chain smoke when there are no weights; paired with an empty `ckpt_path`.
- **Ports:** each run uses an independent `port` / `health_port` to avoid chunk-cache cross-contamination.
- **GPU:** usually one card per policy server; running eval tasks serially is more reliable.
- **Environment separation:** the orchestrator uses the **benchmark conda**; `server.python` uses the **model server** environment.
- **SAPIEN (SimplerEnv / RoboTwin / ManiSkill):** besides `nvidia-smi`, use `vulkaninfo`
  to confirm `deviceName=NVIDIA` (not llvmpipe); set `LD_LIBRARY_PATH` and `VK_ICD_FILENAMES`.

Eval parameters (max_steps / num_steps_wait, etc.) must be configured according to the **original eval protocol + the current smoke intent**, not mixed:

- **pi05 × LIBERO:** smoke commonly uses `max_steps: 300`; full-scale long-horizon suites such as libero_10
  are recommended to be higher (e.g. 520). A `max_steps` that is too small will judge in-progress episodes as failures.
- **xvla × LIBERO:** original horizon 800, `num_steps_wait: 10`; smoke can be 1 task × 1 ep,
  but keeping 800 is still recommended so long tasks are not truncated.
- **xvla × SimplerEnv WidowX:** official `max_steps: 1200` (task-success config).
- **Chain smoke (random_init):** deliberately short step counts (e.g. 20–30), only to prove RPC, not counted as a result.

runner semantics: when `benchmark.max_steps > 0`, it takes precedence over the suite default (not bounded by its cap).

---

## 6. Development notes

- Keep model framework logic in `eval/factories/<model>_factory.py`. Do not add model-specific code to `loongforge_server.py`.
- Do not modify training-tree LoongForge source for eval-specific compatibility.
- Prefer adding YAML configs over command-line parameter sprawl.
- Use `model.backend` in YAML for backend selection and `benchmark.name` for benchmark selection.
- New model factories should ensure their `predict_action()` tolerates a warmup call with a zero-filled dummy image and an empty instruction string (`_warmup_model` in `loongforge_server.py` runs this before serving).
- After changing shared runner / adapter / payload builder / action decoder / bridge / generic policy code, regression-test at least pi05 × LIBERO.
