# Predict Action Interface

**Audience:** model owners implementing `predict_action()` for LoongForge embodied offline eval.

This document is the **contract** between a model and the eval stack. For YAML / runners / verified scores, see [`user_guide_en.md`](user_guide_en.md). For action-space / absolute-vs-delta / bridge semantics (pi05 vs xvla), see [`model_integration_guide.md`](model_integration_guide.md).

**Source of truth for helpers:** `loongforge/embodied/eval/servers/predict_action_interface.py`.

---

## 1. What you implement vs what eval owns

| Layer | Owner | Responsibility |
|-------|--------|----------------|
| `predict_action(...)` | **Model** | Inference; optional **state norm** / **action unnorm** using `dataset_stats`; return a numeric action chunk |
| Model factory | Eval `factories/` | Import, config, ckpt / `random_init`, device/dtype, wrap into a `predict_action` callable |
| `GenericPredictActionPolicy` | Eval server | RPC, image packing, action-chunk cache, latency, shape check, **dim truncate** to `action_dim` |
| `action_postprocess` / RoboTwin `action_bridge` | Eval runner / bridge | Convert model space → **env** action (e.g. 20D ee6d → 7D LIBERO); **not** dataset q99 unnorm |

Do **not** put LIBERO / RoboTwin / SimplerEnv environment special-cases inside training-tree `predict_action`. Prefer a named eval postprocess / bridge when the model action space differs from the env.

---

## 2. Recommended integration path

After benchmark runners exist, do **not** fork a full `LoongForgeXXXPolicy`. Prefer:

1. Model exposes a unified `predict_action()`.
2. Eval adds a thin factory (`@register_factory("<model_type>")`) that loads the model and returns a `PredictActionModelSpec`.
3. Server uses `GenericPredictActionPolicy` for RPC / cache / stats path / shape checks.

Reference paths already in-tree:

```text
loongforge_server.py
  -> PI05ModelFactory.build(...)
  -> GenericPredictActionPolicy(...)
  -> PI05Policy.predict_action(images, instructions, state=None, dataset_stats=None)

loongforge_server.py
  -> XVLAModelFactory.build(...)
  -> GenericPredictActionPolicy(...)
  -> wrapper.predict_action(images, instructions, state=None, dataset_stats=None, domain_id=..., **kwargs)
```

Two signature styles are supported:

- **Fixed signature** (pi05-style): no `**kwargs`. Extra runner keywords are **silently dropped**.
- **`**kwargs` / extra params** (xvla-style): model-specific fields such as `domain_id` are passed through.

---

## 3. Required signature

```python
def predict_action(images, instructions, state=None, dataset_stats=None, **kwargs):
    """Return env-ready or postprocess-ready actions as float array."""
    ...
```

### 3.1 Parameters

| Arg | Type (typical) | Meaning |
|-----|----------------|---------|
| `images` | nested list of `uint8` HWC arrays | Batched views. Generic policy packs from canonical views: prefer `primary`/`head`; if both `left` and `right` exist (e.g. RoboTwin) → `[primary, left, right]`; else optional single wrist → `[primary, wrist]`; single-view → `[primary]`. |
| `instructions` | `list[str]` | Batched language instructions. |
| `state` | `None` or numeric vector / array | **Model-ready** proprio from adapter `model_state` (not the structured `state` dict). |
| `dataset_stats` | `dict` or `None` | Loaded by eval from `server.dataset_statistics_path` and **passed through**. Use inside the model for state norm / action unnorm if your training pipeline requires it. |
| extra kwargs | e.g. `domain_id` | Only if the signature accepts them (explicit name or `**kwargs`). YAML `benchmark.domain_id` is injected by the runner into the RPC payload. |

### 3.2 Kwarg filtering (`_filter_supported_kwargs`)

`call_predict_action` inspects the signature before calling you:

- Signature has `**kwargs` → all extra keywords are forwarded.
- Fixed signature → only declared parameters are kept. Runner fields such as `do_sample`, `use_ddim`, `num_ddim_steps`, `cfg_scale`, or bridge-only `meta` are dropped without error.

Declare a parameter (or use `**kwargs`) if the model must consume it.

### 3.3 Name mapping: `model_state` → `state`

Adapters keep two fields:

- `canonical_obs["state"]` — structured, for trace / debug.
- `canonical_obs["model_state"]` — numeric (or `None`), for the model.

Runners send:

```python
payload = {
    "images": canonical_obs["images"],
    "instruction": canonical_obs["instruction"],
    "state": canonical_obs.get("model_state"),  # may be None
}
```

```text
adapter.model_state  ->  RPC payload.state  ->  predict_action(state=...)
```

---

## 4. Validation helpers (eval code)

```text
loongforge/embodied/eval/servers/predict_action_interface.py
```

| API | Role |
|-----|------|
| `PredictActionModel` | Protocol |
| `validate_predict_action_model(model)` | Signature check before call |
| `call_predict_action(model, images, instructions, state, dataset_stats, action_dim, **kwargs)` | Validate, call, reshape / truncate |
| `postprocess_actions(actions, key)` | Optional **eval-side** env conversion after the server returns actions |

`GenericPredictActionPolicy` always goes through `call_predict_action`.

### 4.1 What `validate_predict_action_model` checks

- `predict_action` exists and is callable.
- Required parameters: `images`, `instructions`.
- Optional parameters `state` and `dataset_stats` must be acceptable (named args or `**kwargs`).

**Invalid:**

```python
def predict_action(self, images):  # missing instructions; cannot take state/dataset_stats
    ...
```

**Valid:**

```python
def predict_action(self, images, instructions, state=None, dataset_stats=None):
    ...

def predict_action(self, images, instructions, **kwargs):
    state = kwargs.get("state")
    dataset_stats = kwargs.get("dataset_stats")
    ...
```

---

## 5. Action output contract

### 5.1 Allowed shapes

Model may return:

```text
[D]
[H, D]
[B, H, D]
```

`call_predict_action` normalizes to:

```text
[H, action_dim]
```

Rules:

| Input shape | Behavior |
|-------------|----------|
| `[D]` | → `[1, D]` |
| `[H, D]` | kept as chunk |
| `[B, H, D]` | → `[-1, D]` (single-request path) |
| other ndim | `ValueError` |
| last dim `< action_dim` | `ValueError` |
| last dim `> action_dim` | **truncate** to first `action_dim` columns |

`action_dim` comes from model / YAML config (e.g. 7 single-arm, 14 RoboTwin joints, 20 xvla ee6d). Truncation is **not** a substitute for correct action semantics.

### 5.2 Normalization / unnormalization (model-owned)

Eval does **not** apply q01/q99, mean/std, or min/max outside the model.

- If the network emits **normalized** actions, unnormalize **inside** `predict_action` using `dataset_stats` (and your training norm mode).
- Example: pi05 ACTION quantile norm → use `dataset_stats["action"].q01` / `.q99` (or the project’s equivalent structure).
- Other LeRobot-style models may use mean/std or min/max.

Return values should be in the **model action space** that either:

1. Matches the env after dim truncate (e.g. pi05 LIBERO 7D), or  
2. Is converted by eval via `benchmark.action_postprocess` / RoboTwin `action_bridge` (e.g. xvla 20D ee6d).

### 5.3 Eval-side postprocess (not part of `predict_action`)

After the server returns a chunk, runners may call:

```python
postprocess_actions(raw_chunk, key)  # key from YAML benchmark.action_postprocess
```

Registered keys (see `ACTION_POSTPROCESS_REGISTRY`):

| Key | Typical use |
|-----|-------------|
| `ee6d_to_axis_angle` | xvla × LIBERO (20D → 7D pos + axis-angle + grip) |
| `ee6d_to_simpler_abs_euler` | xvla × SimplerEnv WidowX |
| `ee6d_to_calvin_abs` | xvla × CALVIN absolute pose |
| `ee6d_to_euler` / `ee6d_to_quat` | other EE variants |

RoboTwin does **not** use this registry for formal protocols; it uses `bridges/robotwin_policy.py` modes (`pi05_aloha_14d`, `ee6d_dual`, …) selected by YAML `benchmark.action_bridge`.

---

## 6. `state` / `model_state` expectations

Structured adapter state is for logging. Only `model_state` reaches the model.

**Do not** pass a nested dict as `state` into `predict_action` unless the model explicitly documents that layout. Prefer a flat `float32` vector aligned with training `observation.state`.

### 6.1 Current practice (pi05 / xvla)

| Setup | Typical `model_state` |
|-------|------------------------|
| pi05 × LIBERO / CALVIN / SimplerEnv | Often `None` unless YAML / adapter builds a training-aligned vector |
| pi05 / xvla × ManiSkill (PickCube) | Numeric from Panda qpos: **8D** when raw qpos is 9D (7 arm + mean of 2 finger joints); full joint list stays in structured `state` |
| pi05 × RoboTwin (`pi05_aloha_14d`) | 14D joint vector from adapter; **bridge** applies openpi `adapt_to_pi` before the RPC state is finalized for the model path |
| xvla + `server.state_format: ee6d` | 20D ee6d proprio (per arm: pos3 + rot6d6 + grip1; single-arm padded). Layout must match the official client (column packing / interleaved rot6d as documented for that benchmark). |

If you need proprio:

1. Align dim, order, units, frame, gripper convention with training data and `dataset_stats["observation.state"]` when used.
2. Have the **adapter** emit numeric `model_state`; do not “clean” benchmark-native dicts inside the factory.

Example of a correct numeric handoff:

```python
canonical_obs = {
    "state": {"eef_pos": [...], "gripper": ..., "frame": "base", ...},  # debug only
    "model_state": np.asarray([...], dtype=np.float32),  # what the model sees
}
# -> predict_action(..., state=model_state)
```

---

## 7. Warmup

Before the health endpoint is marked ready, the server may call once:

```python
predict_action(
    images=[[np.zeros((224, 224, 3), dtype=np.uint8)]],
    instructions=["warmup"],
    state=None,
    dataset_stats=None,
)
```

Failures are logged as warnings, but the call must not corrupt model weights or leave the process unusable. Prefer lazy imports that complete safely on first call.

---

## 8. Local interface check (no benchmark)

### 8.1 Mock (signature + `call_predict_action` only)

No GPU / weights required. Use this to verify the eval helpers before a full model load.

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

Expected:

```text
(4, 7)
(4, 20)
```

### 8.2 Real pi05 / xvla (`predict_action` via factory)

Use the model factory (same path as the policy server). Prefer `server.random_init: true` for a **contract** check without domain weights; point `tokenizer_path` / Florence processor path at any complete checkpoint tree the model can load from.

**Images layout** matches `GenericPredictActionPolicy`: batch outer list, per-sample view list (LIBERO-style two views shown).

```bash
cd /workspace/LoongForge-VLA
CUDA_VISIBLE_DEVICES=0 PYTHONPATH=/workspace/LoongForge-VLA \
  /path/to/loongforge/bin/python - <<'PY'
import numpy as np
from loongforge.embodied.eval.servers.predict_action_interface import (
    call_predict_action,
    validate_predict_action_model,
)
from loongforge.embodied.eval.servers.eval_server_config import EvalServerArgs
from loongforge.embodied.eval.factories.pi05_factory import PI05ModelFactory
from loongforge.embodied.eval.factories.xvla_factory import XVLAModelFactory
from loongforge.embodied.model.pi05.model_configuration_pi05 import Pi05ModelConfig
from loongforge.embodied.model.xvla.model_configuration_xvla import XvlaModelConfig

img = np.zeros((224, 224, 3), dtype=np.uint8)
images = [[img, img]]  # one batch item, two views
instructions = ["pick up the cube"]

# --- pi05 (fixed signature; extra kwargs dropped) ---
pi05 = PI05ModelFactory.build(
    Pi05ModelConfig(action_dim=7, state_dim=8, action_horizon=50, max_action_dim=32, max_state_dim=32),
    EvalServerArgs(
        random_init=True,
        tokenizer_path="/path/to/paligemma-3b-pt-224",  # required even for random_init
        device="cuda",
        loongforge_root="/workspace/LoongForge-VLA",
    ),
).model
validate_predict_action_model(pi05)
out = call_predict_action(
    pi05, images=images, instructions=instructions, state=None, dataset_stats=None,
    action_dim=7, do_sample=False, cfg_scale=1.5, domain_id=3,  # last three ignored
)
print("pi05", out.shape)  # e.g. (50, 7) for action_horizon=50

# --- xvla (domain_id coerced by factory wrapper) ---
# tokenizer_path must be a full X-VLA tree (Florence processor + tokenizer files).
xvla = XVLAModelFactory.build(
    XvlaModelConfig(action_mode="ee6d", num_actions=30, action_horizon=30, max_action_dim=20, real_action_dim=20),
    EvalServerArgs(
        random_init=True,
        tokenizer_path="/path/to/xvla-libero",  # or any full X-VLA dir
        device="cuda",
        loongforge_root="/workspace/LoongForge-VLA",
        state_format="ee6d",
    ),
).model
validate_predict_action_model(xvla)
out = call_predict_action(
    xvla, images=images, instructions=instructions,
    state=np.zeros(20, dtype=np.float32), dataset_stats=None,
    action_dim=20, domain_id=3,
)
print("xvla", out.shape)  # e.g. (30, 20); raw model is often [1, H, 20]
PY
```

Notes from real runs:

| Model | Raw `predict_action` | After `call_predict_action` |
|-------|----------------------|-----------------------------|
| pi05 | `[B, action_horizon, max_action_dim]` e.g. `(1, 50, 32)` | truncate last dim → `(50, 7)` |
| xvla | `[B, num_actions, real_action_dim]` e.g. `(1, 30, 20)` | reshape + keep dim → `(30, 20)` |

- pi05 **needs** `tokenizer_path` even under `random_init` (PaliGemma tokenize).
- xvla **needs** a valid Florence processor/tokenizer dir (`tokenizer_path`); empty path fails HF load.
- xvla factory wraps `domain_id` int → `LongTensor` on device; pass YAML-style int in `call_predict_action(..., domain_id=3)`.
- `EvalServerArgs` has **no** `processor_path` field — use `tokenizer_path` only (factory sets `model._processor_path` from it).
- Full correctness (unnorm stats, abs/delta, postprocess) still needs a YAML smoke / task-success run (`user_guide_en.md`).

---

## 9. Checklist for model owners

Interface layer:

- [ ] Callable `predict_action(images, instructions, state=None, dataset_stats=None)` (optional extra kwargs if needed).
- [ ] `validate_predict_action_model(model)` passes.
- [ ] Returns `[D]`, `[H, D]`, or `[B, H, D]` with last dim ≥ configured `action_dim`.
- [ ] Unnormalization (if any) lives **inside** `predict_action`; eval only passes `dataset_stats`.
- [ ] Warmup call is safe.
- [ ] Factory loads weights / tokenizer / processor; does not rewrite benchmark dict observations.

Semantics (see `model_integration_guide.md`):

- [ ] `action_dim` / horizon match the target benchmark protocol (7 / 14 / 20, …).
- [ ] Absolute vs delta control matches training; do not invent env-side “delta = abs − current” for axis-angle without a real protocol.
- [ ] If output space ≠ env space, coordinate an eval `action_postprocess` or RoboTwin `action_bridge` key—do not hardcode env names in the model.
- [ ] `domain_id` / `state_format` / stats paths match the official inference config for that domain weight.

---

## 10. Common errors

| Error | Cause |
|-------|--------|
| `TypeError: model must expose a callable predict_action(...)` | Missing method |
| `TypeError: ... missing required parameters: ['instructions']` | Bad signature |
| `TypeError: ... cannot accept eval keyword parameters: ['state']` | No `state` and no `**kwargs` |
| `ValueError: ... unsupported action shape` | Not `[D]` / `[H,D]` / `[B,H,D]` |
| `ValueError: ... action dim X, expected at least Y` | Output narrower than `action_dim` |
| Env steps but success stays 0 | Often control mode (abs vs delta), wrong postprocess/bridge, or wrong unnorm—not a missing `predict_action` |

---

## 11. Related docs

| Doc | Use |
|-----|-----|
| [`user_guide_en.md`](user_guide_en.md) | Run eval, YAML, verified scores (canonical) |
| [`model_integration_guide.md`](model_integration_guide.md) | Semantic checklist (pi05 vs xvla) |
| [`benchmark_envs.md`](benchmark_envs.md) | Per-benchmark observation / action notes |
| `servers/predict_action_interface.py` | Implementation of this contract + postprocess registry |
