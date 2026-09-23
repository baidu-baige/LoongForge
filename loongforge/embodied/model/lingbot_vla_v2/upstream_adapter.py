# Copyright 2026 The LoongForge Authors.
# SPDX-License-Identifier: Apache-2.0

"""Adapter (seam) between LoongForge and the upstream lingbot-vla-v2 stack.

Background
----------
The lingbot-vla-v2 network is consumed from a pinned third-party ``lingbotvla``
package (delivered as an editable / source install of a fixed commit). This
module is the single seam that imports the upstream symbols and layers
LoongForge's own additions on top of them, without editing upstream source.

Terminology: *upstream* is that pinned package; *the reference copy* is the
LingBot-VLA training source this port was aligned against (``lingbot_vla/``).
Several shims below reproduce a fix the reference copy made in-source, so they
are described as "matching the reference copy".

Every public ``resolve_*`` function returns one upstream symbol for a category:

* model body / config / loader / HF patches;
* frozen depth / video vision teachers;
* dataset builder and data collator.

LoongForge-specific behaviour is added at two layers, never by patching upstream
source: runtime seams installed here (see the ``_apply_lingbot_*`` shims) and
components under :mod:`loongforge.embodied.model.lingbot_vla_v2`.

Lazy-import contract
--------------------
The external ``lingbotvla`` package and ``torch`` / ``transformers`` are imported
**inside** functions, not at module top level, so importing this module is cheap
and does not require the heavy stack to be present until a symbol is resolved.

Transformers 5.x compatibility
------------------------------
The upstream ``models_common`` loader references symbols that Transformers 5.x
removed / relocated:

* ``transformers.AutoModelForVision2Seq`` -> replaced by
  ``transformers.AutoModelForImageTextToText``;
* ``transformers.modeling_utils.no_init_weights`` -> moved to
  ``transformers.initialization.no_init_weights``.

:func:`_ensure_transformers5_compat` re-exposes the old names on the installed
``transformers`` before any upstream import runs, so the compat shim lives here
in the adapter rather than being scattered across call sites. It is a no-op on
Transformers 4.x.
"""

from __future__ import annotations

__all__ = [
    # model body / config / loader
    "resolve_config_cls",
    "resolve_build_foundation_model",
    "resolve_qwen2_patch",
    "resolve_qwen3_vl_patch",
    # teachers
    "resolve_build_depth_model",
    "resolve_build_video_model",
    "resolve_get_depth_target",
    "resolve_get_video_target",
    # data
    "resolve_build_vla_dataset",
    "resolve_vla_data_collator",
]



def _register_lazy_symbol(mod, name, value) -> None:
    """Expose ``name`` on ``mod`` so ``from mod import name`` resolves to ``value``.

    Transformers 5.x wraps sub-packages in a ``_LazyModule`` whose ``__getattr__``
    resolves from a persistent ``_objects`` dict *before* the instance ``__dict__``.
    A plain ``setattr`` on such a module lands in ``__dict__`` and is wiped the next
    time an unrelated lazy attribute resolves, so we inject into ``_objects`` when it
    exists and additionally ``setattr`` for ordinary (non-lazy) modules.
    """
    objects = getattr(mod, "_objects", None)
    if isinstance(objects, dict):
        objects[name] = value
    try:
        setattr(mod, name, value)
    except Exception:
        pass


def _require_upstream_package() -> None:
    """Fail fast with an actionable message when the upstream package is absent.

    ``lingbotvla`` is a pinned third-party dependency delivered as a source /
    editable install; surface how to install it instead of a bare
    ``ModuleNotFoundError`` from deep inside an import.
    """
    import importlib.util

    if importlib.util.find_spec("lingbotvla") is None:
        raise ImportError(
            "The upstream 'lingbotvla' package is required but not installed. "
            "Install the pinned commit with:\n"
            '    pip install -e ".[lingbotvla-upstream]"\n'
            "(defined in pyproject.toml; git + network access required)."
        )


def _ensure_transformers5_compat() -> None:
    """Re-expose Transformers-4.x names removed/relocated in Transformers 5.x.

    Called only from the upstream branch, before importing upstream modules that
    reference these symbols at import time. All shims live here in the adapter; no
    upstream source file is modified. No-op on 4.x (the names still resolve).

    Symbols handled (verified against transformers 5.3.0):

    * ``transformers.AutoModelForVision2Seq`` -> ``AutoModelForImageTextToText``
      (renamed). This lives on the top-level ``_LazyModule``, which transformers 5.x
      *re-creates and swaps into ``sys.modules`` during its first heavy warm-up*
      (triggered by ``AutoProcessor`` / ``AutoTokenizer``). We therefore force that
      warm-up first and only then inject, so the alias lands on the module object
      that stays live when upstream's ``loader`` imports from it.
    * ``transformers.modeling_utils.no_init_weights`` -> moved to
      ``transformers.initialization``.
    * ``transformers.utils(.import_utils).is_safetensors_available`` -> removed.
    * ``transformers.models.qwen2_5_vl...Qwen2RMSNorm`` -> moved to the ``qwen2``
      module (upstream's Qwen2.5-VL patch still imports the old path).
    * ``transformers.cache_utils.{SlidingWindowCache,HybridCache}`` -> removed;
      aliased to ``DynamicCache`` (upstream imports but does not use them).
    """
    _require_upstream_package()

    import importlib
    import sys

    import transformers

    # Force the one-time lazy-module swap before injecting top-level aliases.
    from transformers import (  # noqa: F401
        AutoConfig,
        AutoProcessor,
        AutoTokenizer,
        PreTrainedModel,
    )
    transformers = sys.modules["transformers"]

    # AutoModelForVision2Seq -> AutoModelForImageTextToText (renamed in 5.x).
    if not hasattr(transformers, "AutoModelForVision2Seq"):
        try:
            from transformers import AutoModelForImageTextToText
        except ImportError:
            AutoModelForImageTextToText = None
        if AutoModelForImageTextToText is not None:
            _register_lazy_symbol(
                transformers, "AutoModelForVision2Seq", AutoModelForImageTextToText
            )

    # is_safetensors_available removed from transformers.utils(.import_utils) in 5.x.
    def _safetensors_available() -> bool:
        try:
            import safetensors  # noqa: F401
            return True
        except ImportError:
            return False

    for _mod_name in ("transformers.utils.import_utils", "transformers.utils"):
        try:
            _mod = importlib.import_module(_mod_name)
        except Exception:
            _mod = None
        if _mod is not None and not hasattr(_mod, "is_safetensors_available"):
            _register_lazy_symbol(_mod, "is_safetensors_available", _safetensors_available)

    # no_init_weights moved to transformers.initialization in 5.x.
    try:
        import transformers.modeling_utils as _modeling_utils
    except Exception:
        _modeling_utils = None
    if _modeling_utils is not None and not hasattr(_modeling_utils, "no_init_weights"):
        try:
            from transformers.initialization import no_init_weights as _no_init_weights
        except ImportError:
            _no_init_weights = None
        if _no_init_weights is not None:
            _register_lazy_symbol(_modeling_utils, "no_init_weights", _no_init_weights)

    # Qwen2RMSNorm moved from qwen2_5_vl to qwen2 in 5.x (Qwen2.5-VL patch path).
    try:
        import transformers.models.qwen2_5_vl.modeling_qwen2_5_vl as _q25
    except Exception:
        _q25 = None
    if _q25 is not None and not hasattr(_q25, "Qwen2RMSNorm"):
        try:
            from transformers.models.qwen2.modeling_qwen2 import Qwen2RMSNorm as _rms
        except ImportError:
            _rms = None
        if _rms is not None:
            _register_lazy_symbol(_q25, "Qwen2RMSNorm", _rms)

    # SlidingWindowCache / HybridCache removed in 5.x; upstream imports but does not
    # use them. Alias to DynamicCache so the import statements succeed.
    try:
        import transformers.cache_utils as _cache_utils
    except Exception:
        _cache_utils = None
    if _cache_utils is not None:
        _dynamic = getattr(_cache_utils, "DynamicCache", None)
        if _dynamic is not None:
            for _name in ("SlidingWindowCache", "HybridCache"):
                if not hasattr(_cache_utils, _name):
                    _register_lazy_symbol(_cache_utils, _name, _dynamic)


def _apply_qwen3_vl_rope_index_tf4_compat() -> None:
    """Let upstream's TF4-style ``get_rope_index`` call succeed on Transformers 5.x.

    In Transformers 5.x, ``Qwen3VLModel.get_rope_index`` gained a **required**
    ``mm_token_type_ids`` positional argument; upstream's ``build_prefix_position_ids``
    (written for 4.57.3) still calls it without that argument, raising ``TypeError``
    at forward time. We wrap the method to synthesise ``mm_token_type_ids`` from the
    image / video token ids (exactly as the reference copy does) when it is omitted,
    then delegate to the original 5.x implementation. No-op on 4.x; idempotent.
    """
    import inspect

    import torch

    try:
        from transformers.models.qwen3_vl.modeling_qwen3_vl import Qwen3VLModel
    except Exception:
        return

    original = Qwen3VLModel.get_rope_index
    if getattr(original, "_lingbot_tf4_rope_compat", False):
        return
    try:
        params = inspect.signature(original).parameters
    except (TypeError, ValueError):
        return
    if "mm_token_type_ids" not in params:
        return  # Transformers 4.x — call site already matches.

    def get_rope_index(self, input_ids=None, mm_token_type_ids=None,
                       image_grid_thw=None, video_grid_thw=None,
                       attention_mask=None, **kwargs):
        if mm_token_type_ids is None and input_ids is not None:
            mm_token_type_ids = torch.zeros_like(input_ids, dtype=torch.int)
            image_token_id = getattr(self.config, "image_token_id", None)
            video_token_id = getattr(self.config, "video_token_id", None)
            if image_token_id is not None:
                mm_token_type_ids[input_ids == image_token_id] = 1
            if video_token_id is not None:
                mm_token_type_ids[input_ids == video_token_id] = 2
        return original(self, input_ids=input_ids, mm_token_type_ids=mm_token_type_ids,
                        image_grid_thw=image_grid_thw, video_grid_thw=video_grid_thw,
                        attention_mask=attention_mask, **kwargs)

    get_rope_index._lingbot_tf4_rope_compat = True
    Qwen3VLModel.get_rope_index = get_rope_index


def _apply_qwen3_vl_structure_fixups() -> None:
    """Re-align upstream's Qwen3-VL with Transformers 5.x expectations at runtime.

    Upstream (written for 4.57.3) diverges from 5.x in three ways that the reference
    copy fixed in-source; we reproduce those fixes on the upstream class object at
    runtime (no source edit), all idempotent:

    * ``.visual`` / ``.language_model`` forwarders — 5.x nests them inside the inner
      ``model`` submodule and dropped the convenience forwarders; upstream accesses
      ``self.qwenvl.visual`` / ``.language_model`` directly.
    * ``_tied_weights_keys`` — 5.x ``get_expanded_tied_weights_keys`` calls ``.keys()``
      on it (dict form); upstream still uses the 4.x list form, so ``post_init``
      raises ``AttributeError: 'list' object has no attribute 'keys'``. Match the
      native 5.x / reference dict mapping.
    * ``get_rope_index`` TF4 call convention (see the rope helper).
    """
    from lingbotvla.models.vla.lingbot_vla.qwen3vl_in_vla import (
        Qwen3VLForConditionalGeneration as _CG,
    )

    if "visual" not in _CG.__dict__:
        _CG.visual = property(lambda self: self.model.visual)
    if "language_model" not in _CG.__dict__:
        _CG.language_model = property(lambda self: self.model.language_model)

    if isinstance(getattr(_CG, "_tied_weights_keys", None), list):
        _CG._tied_weights_keys = {
            "lm_head.weight": "model.language_model.embed_tokens.weight"
        }

    _apply_qwen3_vl_rope_index_tf4_compat()


def _apply_qwen2_action_expert_fixups() -> None:
    """Convert the Qwen2 action expert's ``_tied_weights_keys`` to 5.x dict form.

    Same 4.x-list vs 5.x-dict incompatibility as Qwen3-VL: upstream's
    ``qwen2_action_expert.Qwen2ForCausalLM`` declares ``_tied_weights_keys`` as a
    list, which 5.x ``get_expanded_tied_weights_keys`` cannot handle. Match the
    reference dict mapping. Idempotent; no upstream source edit.
    """
    from lingbotvla.models.vla.lingbot_vla.qwen2_action_expert import Qwen2ForCausalLM

    if isinstance(getattr(Qwen2ForCausalLM, "_tied_weights_keys", None), list):
        Qwen2ForCausalLM._tied_weights_keys = {
            "lm_head.weight": "model.embed_tokens.weight"
        }


def _apply_lingbot_async_teacher_shim() -> None:
    """Give the upstream policy LoongForge's async-teacher handle protocol.

    LoongForge computes the frozen depth / video teacher targets on a side stream
    (``model.async_teacher=True``, the default) and hands the *in-flight* result to
    the model before the forward: ``recipe.prepare_batch`` calls
    ``policy.set_pending_teacher(handle)``, and the first auxiliary head that reads
    its targets joins the side stream. The reference copy carries both halves
    (``LingbotVlaV2Policy.set_pending_teacher`` + a ``_pending_teacher`` join inside
    ``FlowMatchingV2.forward``); upstream has **neither** and supports only
    *synchronous* teacher targets passed as ``forward`` kwargs. On the LoongForge
    harness with async teachers enabled, ``policy.set_pending_teacher(...)`` would
    therefore raise ``AttributeError``.

    We add the missing seam on the upstream ``LingbotVlaV2Policy`` class at runtime
    (no upstream source edit): ``set_pending_teacher`` stashes the handle, and the
    ``forward`` wrapper passes *unresolved* targets down to the flow-matching model.
    That is the part that matters for throughput: joining the side stream at the
    policy's forward entry would serialize the teacher ahead of the backbone and
    throw the overlap away. Each target is instead a lazy placeholder, and the only
    consumers of those kwargs are the auxiliary heads (``*_emb_forward``), which run
    *after* the backbone -- so the join lands where the reference copy has it and the
    teacher stays hidden behind the student forward.

    Numerically this is identical to the synchronous path: the teachers are frozen
    and run under ``no_grad``, so *where* ``handle.result()`` is joined does not
    change the targets, only when the side-stream is awaited. Idempotent.
    """
    from lingbotvla.models.vla.lingbot_vla.modeling_lingbot_vla_v2 import (
        FlowMatchingV2,
        LingbotVlaV2Policy,
    )

    if getattr(LingbotVlaV2Policy, "_lingbot_async_teacher_shim", False):
        return

    class _PendingTargets:
        """The teacher outputs for one forward, joined on first use.

        Created per forward and shared by every placeholder of that forward, so a
        single join serves all the auxiliary heads and nothing can leak across
        steps (a stale join would silently train on the previous step's targets).
        """

        __slots__ = ("_handle", "_resolved")

        def __init__(self, handle):
            self._handle = handle
            self._resolved = None

        def resolve(self):
            """Block for the teacher output, once, and return the target mapping."""
            if self._resolved is None:
                self._resolved = self._handle.result()
            return self._resolved

    class _DeferredTarget:
        """One teacher target, resolved on first use.

        A placeholder rather than a resolved tensor so the policy forward can pass
        the targets down without blocking; the auxiliary head that consumes the
        target is what joins the side stream.
        """

        __slots__ = ("_pending", "_key")

        def __init__(self, pending, key):
            self._pending = pending
            self._key = key

        def resolve(self):
            """Block for the teacher output and return this target's value."""
            return self._pending.resolve().get(self._key)

    def _substitute(value):
        """Replace every placeholder in ``value`` with its resolved target."""
        if isinstance(value, _DeferredTarget):
            return value.resolve()
        if isinstance(value, dict):
            return {key: _substitute(item) for key, item in value.items()}
        if isinstance(value, list):
            return [_substitute(item) for item in value]
        if isinstance(value, tuple):
            return tuple(_substitute(item) for item in value)
        return value

    def _has_placeholder(value):
        """True when ``value`` holds a placeholder anywhere inside it."""
        if isinstance(value, _DeferredTarget):
            return True
        if isinstance(value, dict):
            value = value.values()
        if isinstance(value, (list, tuple)):
            return any(_has_placeholder(item) for item in value)
        return False

    def _wrap_aux_head(original):
        """Join the teacher stream before an auxiliary head reads its targets."""

        def head(self, *args, **kwargs):
            if not _has_placeholder(args) and not _has_placeholder(kwargs):
                return original(self, *args, **kwargs)
            return original(self, *_substitute(args), **_substitute(kwargs))

        return head

    def set_pending_teacher(self, handle):
        # Mirror the reference contract: fail loudly rather than silently dropping
        # the handle, which would leave the depth head with no targets. The handle
        # stays on the policy, which is what the forward wrapper reads it from.
        if handle is None:
            raise RuntimeError("set_pending_teacher called with a None handle")
        self._pending_teacher = handle

    def forward(self, *args, **kwargs):
        pending = getattr(self, "_pending_teacher", None)
        if pending is not None:
            # Hand the targets down unresolved; the auxiliary heads join the stream
            # when they read them. Clear the handle first so a failed forward cannot
            # reuse a stale one next step.
            self._pending_teacher = None
            targets = _PendingTargets(pending)
            kwargs["depth_targets"] = _DeferredTarget(targets, "depth_targets")
            kwargs["future_depth_targets"] = _DeferredTarget(
                targets, "future_depth_targets"
            )
            kwargs["future_video_targets"] = _DeferredTarget(
                targets, "future_video_targets"
            )
            kwargs["future_video_cls_targets"] = _DeferredTarget(
                targets, "future_video_cls_targets"
            )
            # ``future_video_current_patch`` is legitimately absent when the video
            # teacher is off; resolving to None keeps the head's own guard skipping it.
            kwargs["future_video_current_patch"] = _DeferredTarget(
                targets, "future_video_current_patch"
            )
        return _orig_forward(self, *args, **kwargs)


    _orig_forward = LingbotVlaV2Policy.forward
    LingbotVlaV2Policy.set_pending_teacher = set_pending_teacher
    LingbotVlaV2Policy.forward = forward
    LingbotVlaV2Policy._lingbot_async_teacher_shim = True

    for name in [item for item in dir(FlowMatchingV2) if item.endswith("_emb_forward")]:
        setattr(FlowMatchingV2, name, _wrap_aux_head(getattr(FlowMatchingV2, name)))


def _apply_lingbot_fm_seed_shim() -> None:
    """Make upstream draw the flow-matching noise/time from the *seeded* per-step
    generator, so the FM draw is identical to the reference copy for the same seed.

    The flow-matching loss depends on a per-step Gaussian ``noise`` and a Beta
    ``time`` draw. The reference copy makes these a pure function of
    ``(seed, step, rank)``: LoongForge's trainer sets a process-global step seed
    (``recipe._set_fm_step_seed`` -> reference ``utils.set_fm_step_seed``) and the
    reference ``FlowMatchingV2.forward`` draws both from a fresh ``torch.Generator``
    seeded with it (``_fm_generator``), so the draw is immune to how much *global*
    CUDA RNG the surrounding setup consumed (teachers, parameter manager, compile).

    Upstream has **no** ``_fm_generator`` / ``get_fm_step_seed`` and draws
    ``torch.randn(...)`` / ``sample_time(...)`` straight off the **global** CUDA RNG
    (``modeling_lingbot_vla_v2.py``: ``noise = torch.randn(actions.shape, ...)``;
    ``time = self.sample_time(...)``). Under LoongForge that global RNG is *not*
    reseeded per step, so upstream would draw a *different* noise/time than reference
    even at the same seed — the ``vla`` loss would then diverge far beyond FP jitter,
    invalidating the parity comparison.

    We wrap upstream ``FlowMatchingV2.forward`` to pre-fill ``noise`` / ``time`` from
    the same seeded generator, reusing LoongForge's own ``get_fm_step_seed`` /
    ``sample_beta`` (``fm_seed``) so the values are bit-identical: same seed,
    same draw order (noise first, then the two Beta uniforms), same formula
    (``sample_beta(1.5, 1.0) * 0.999 + 0.001``). No upstream source edit; idempotent;
    no-op when the trainer has not set a step seed (falls back to upstream's own draw).
    """
    import torch

    from lingbotvla.models.vla.lingbot_vla.modeling_lingbot_vla_v2 import FlowMatchingV2

    if getattr(FlowMatchingV2, "_lingbot_fm_seed_shim", False):
        return

    from loongforge.embodied.model.lingbot_vla_v2.fm_seed import (
        get_fm_step_seed,
        sample_beta,
    )

    def _fm_generator(device):
        step_seed = get_fm_step_seed()
        if step_seed is None:
            return None
        g = torch.Generator(device=device)
        g.manual_seed(int(step_seed) & 0x7FFFFFFFFFFFFFFF)
        return g

    _orig_forward = FlowMatchingV2.forward

    def forward(self, images, img_masks, lang_tokens, lang_masks, state, actions,
                noise=None, time=None, *args, **kwargs):
        if noise is None or time is None:
            gen = _fm_generator(state.device)
            if gen is not None:
                device = state.device
                dtype = state.dtype
                # Match the reference draw order exactly: noise (randn) first, then
                # the Beta time (two uniforms inside sample_beta).
                if noise is None:
                    noise = torch.randn(
                        actions.shape, device=device, dtype=dtype, generator=gen
                    )
                if time is None:
                    time_beta = sample_beta(1.5, 1.0, actions.size(0), device, generator=gen)
                    time = (time_beta * 0.999 + 0.001).to(
                        dtype=torch.float32, device=device
                    ).to(dtype)
        return _orig_forward(
            self, images, img_masks, lang_tokens, lang_masks, state, actions,
            noise, time, *args, **kwargs
        )

    FlowMatchingV2.forward = forward
    FlowMatchingV2._lingbot_fm_seed_shim = True


def _apply_lingbot_parallel_state_shim() -> None:
    """Initialise the upstream package's own parallel-state singleton for DDP.

    Upstream's fused-MoE (``lingbotvla.ops.fused_moe.fused_moe_forward``) reads
    ``lingbotvla.distributed.parallel_state.get_parallel_state()``; when that global
    was never initialised it lazily constructs ``ParallelState()`` with all sizes = 1,
    whose ``__post_init__`` asserts ``pp*dp*cp*ulysses*tp == world_size`` and raises
    ``ValueError: The product of parallel sizes should be equal to the world size`` on
    a >1-GPU run. The upstream *trainer* normally calls ``init_parallel_state(...)``,
    but under LoongForge that trainer never runs — LoongForge's own parallel state
    is managed by the trainer, which leaves this upstream singleton untouched. So
    we must initialise it here, matching LoongForge's plain replicated DDP: ``dp_size =
    world_size``, ``dp_replicate = world_size``, ``dp_shard = 1``, ``ep = tp = pp = 1``
    (so ``ep_enabled`` is False and the MoE takes its non-expert-parallel path).

    Idempotent. No-op without a >1-rank distributed group, where the lazy default
    (world_size == 1) is already valid. No upstream source edit.

    We construct ``ParallelState`` **directly** rather than calling
    ``init_parallel_state``: the latter builds a ``DeviceMesh`` (via
    ``init_device_mesh``), which creates *new* NCCL sub-communicators that LoongForge
    does not own and never tears down. Those orphaned groups make the workers spin in
    an NCCL barrier at process teardown (GPUs pinned at 100 %, memory not released
    after the last step) — the manual-cleanup symptom. With ``ep = tp = pp = cp =
    ulysses = 1`` the non-expert-parallel MoE path never touches the mesh (only
    ``ep_enabled`` is read, which is False), and ``sp_enabled`` is False, so a mesh is
    unnecessary: ``device_mesh=None`` is valid and leaves NCCL state entirely to
    LoongForge, so the process exits cleanly.
    """
    import torch.distributed as dist

    from lingbotvla.distributed import parallel_state as _ups_ps

    if getattr(_ups_ps, "_PARALLEL_STATE", None) is not None:
        return
    if not (dist.is_available() and dist.is_initialized()):
        return
    world_size = dist.get_world_size()
    if world_size <= 1:
        return
    # Direct construction (no init_device_mesh -> no orphaned NCCL groups).
    _ups_ps._PARALLEL_STATE = _ups_ps.ParallelState(
        dp_size=world_size,
        dp_replicate_size=world_size,
        dp_shard_size=1,
        dp_mode="ddp",
        device_mesh=None,
    )


def _apply_lingbot_flex_compile_shim(model_cfg=None) -> None:
    """Compile upstream's flex-attention op so its throughput matches reference.

    Vendored ``lingbot_vla/flex_attention.py`` compiles the flex-attention op once
    at module level (``_flex_attention_compiled = torch.compile(flex_attention,
    dynamic=False)``, selected via ``_FLEX_COMPILE`` / ``set_flex_compile``), because
    the *eager* op materializes the ``[B, H, Q, KV]`` scores and its host time
    scales with the sequence length. Upstream ships the same file with the ``@torch.compile``
    decorator **commented out** and calls the eager op directly from
    ``flex_attention_with_block_mask`` / ``flex_attention_forward``, so the upstream
    model runs the slow eager attention path and is markedly slower per step than the
    reference copy.

    We rebind the upstream module's ``flex_attention`` global to the compiled op,
    mirroring reference exactly, gated by the same ``model.flex_compile`` flag. Both
    call sites read the module global, so this covers both paths. torch.compile is
    numerically equivalent to the eager op (reference already relies on this for
    parity), so this narrows the reference/upstream gap rather than widening it. No
    upstream source edit; idempotent.
    """
    enabled = True if model_cfg is None else bool(getattr(model_cfg, "flex_compile", True))
    if not enabled:
        return

    import torch

    from lingbotvla.models.vla.lingbot_vla import flex_attention as _ufa

    if getattr(_ufa, "_lingbot_flex_compile_shim", False):
        return
    if getattr(_ufa, "flex_attention", None) is None:
        return
    _ufa.flex_attention = torch.compile(_ufa.flex_attention, dynamic=False)
    _ufa._lingbot_flex_compile_shim = True


def _import_upstream_data_guarded(import_fn):
    """Import an upstream ``data`` module while neutralising its datasets<4 patch.

    Upstream ``lingbotvla.data.dataset`` unconditionally monkeypatches
    ``datasets.features.features.generate_from_dict`` to rewrite ``List`` -> ``Sequence``
    at import time. That is only correct for ``datasets < 4.0``; under datasets 4.x
    (which has a native ``List`` feature) it corrupts schema resolution. We snapshot
    the real ``generate_from_dict`` before the import and, on datasets>=4, restore it
    afterwards — without editing upstream source.
    """
    try:
        import datasets.features.features as _features
    except Exception:
        return import_fn()

    has_list = hasattr(_features, "List")
    original = getattr(_features, "generate_from_dict", None)
    result = import_fn()
    if has_list and original is not None:
        _features.generate_from_dict = original
    return result


def _import_symbol(module_name, symbol_name):
    """Import one symbol lazily from a module."""
    import importlib

    return getattr(importlib.import_module(module_name), symbol_name)


def _resolve_symbol(symbol_name, upstream_module):
    """Import a symbol from the upstream ``lingbotvla`` package.

    Runs the Transformers 5.x compat shim and the upstream ``datasets`` import
    guard first, so every upstream lookup goes through the same path.
    """
    _ensure_transformers5_compat()
    return _import_upstream_data_guarded(
        lambda: _import_symbol(upstream_module, symbol_name)
    )


# --------------------------------------------------------------------------- #
# Model body / config / loader
# --------------------------------------------------------------------------- #
def resolve_config_cls(model_cfg=None):
    """Return the upstream ``LingbotVLAV2Config`` class."""
    return _resolve_symbol(
        "LingbotVLAV2Config",
        "lingbotvla.models.vla.lingbot_vla.configuration_lingbot_vla",
    )


def resolve_build_foundation_model(model_cfg=None):
    """Return ``build_foundation_model`` from the upstream ``models_common`` loader.

    The returned callable first installs LoongForge's runtime seams on the
    upstream policy: the async-teacher handle protocol
    (:func:`_apply_lingbot_async_teacher_shim`) so the model honours LoongForge's
    default ``model.async_teacher=True``; the seeded flow-matching draw
    (:func:`_apply_lingbot_fm_seed_shim`) so its per-step ``noise`` / ``time`` are
    a pure function of the seed; the parallel-state singleton init
    (:func:`_apply_lingbot_parallel_state_shim`); and the flex-attention compile
    (:func:`_apply_lingbot_flex_compile_shim`). None edits upstream source.
    """
    build_foundation_model = _resolve_symbol(
        "build_foundation_model", "lingbotvla.models.auto"
    )

    def _build_foundation_model_with_shims(*args, **kwargs):
        _apply_lingbot_async_teacher_shim()
        _apply_lingbot_fm_seed_shim()
        _apply_lingbot_parallel_state_shim()
        _apply_lingbot_flex_compile_shim(model_cfg)
        return build_foundation_model(*args, **kwargs)

    return _build_foundation_model_with_shims


def resolve_qwen2_patch(model_cfg=None):
    """Return ``apply_lingbot_qwen2_patch`` with LoongForge's 5.x fixups."""
    apply_lingbot_qwen2_patch = _resolve_symbol(
        "apply_lingbot_qwen2_patch",
        "lingbotvla.models.vla.lingbot_vla.qwen2_action_expert",
    )

    def _apply_lingbot_qwen2_patch_with_fixups():
        apply_lingbot_qwen2_patch()
        _apply_qwen2_action_expert_fixups()

    return _apply_lingbot_qwen2_patch_with_fixups


def resolve_qwen3_vl_patch(model_cfg=None):
    """Return ``apply_lingbot_qwen3_vl_patch`` with LoongForge's 5.x fixups.

    The returned callable also re-adds the ``.visual`` / ``.language_model``
    forwarders that Transformers 5.x dropped, so the upstream model body (which
    accesses them directly) works unmodified.
    """
    apply_lingbot_qwen3_vl_patch = _resolve_symbol(
        "apply_lingbot_qwen3_vl_patch",
        "lingbotvla.models.vla.lingbot_vla.qwen3vl_in_vla",
    )

    def _apply_lingbot_qwen3_vl_patch_with_fixups():
        apply_lingbot_qwen3_vl_patch()
        _apply_qwen3_vl_structure_fixups()

    return _apply_lingbot_qwen3_vl_patch_with_fixups


# --------------------------------------------------------------------------- #
# Teachers
# --------------------------------------------------------------------------- #
def resolve_build_depth_model(model_cfg=None):
    """Return ``build_depth_model`` from the vision teachers module."""
    return _resolve_symbol(
        "build_depth_model",
        "lingbotvla.models.vla.vision_models.module_utils",
    )


def resolve_build_video_model(model_cfg=None):
    """Return ``build_video_model`` from the vision teachers module."""
    return _resolve_symbol(
        "build_video_model",
        "lingbotvla.models.vla.vision_models.module_utils",
    )


def resolve_get_depth_target(model_cfg=None):
    """Return ``get_depth_target`` from the vision teachers module."""
    return _resolve_symbol(
        "get_depth_target",
        "lingbotvla.models.vla.vision_models.module_utils",
    )


def resolve_get_video_target(model_cfg=None):
    """Return ``get_video_target`` from the vision teachers module."""
    return _resolve_symbol(
        "get_video_target",
        "lingbotvla.models.vla.vision_models.module_utils",
    )


# --------------------------------------------------------------------------- #
# Data
# --------------------------------------------------------------------------- #
def resolve_build_vla_dataset():
    """Return ``build_vla_dataset``."""
    return _resolve_symbol(
        "build_vla_dataset",
        "lingbotvla.data.dataset",
    )


def resolve_vla_data_collator():
    """Return the ``VLADataCollatorWithPacking`` class."""
    return _resolve_symbol(
        "VLADataCollatorWithPacking",
        "lingbotvla.data.multimodal.data_collator",
    )
