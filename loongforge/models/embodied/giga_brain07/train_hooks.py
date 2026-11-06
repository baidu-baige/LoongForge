# Copyright 2026 The LoongForge Authors.
# SPDX-License-Identifier: Apache-2.0

"""Opt-in giga_brain07 training-time patches, applied after distributed wrapping.

Both are driven by the typed model config and run from
``GigaBrain07Policy.on_train_begin`` (FinetuneTrainer calls that hook once
after AC + FSDP2/DDP wrapping and optimizer construction). Neither changes
parameters, so running after the optimizer is built is safe.

* ``apply_decoder_kv_slice`` -- ``model.decoder_kv_slice``
* ``apply_regional_compile`` -- ``model.compile_model`` / ``compile_scope`` ...

Moved verbatim from the former ``GigaBrain07Trainer._maybe_apply_kv_slice`` /
``_maybe_apply_compile``; logic unchanged.
"""

from __future__ import annotations

import logging
import re
from typing import Dict

import torch
import torch.nn as nn

logger = logging.getLogger(__name__)


def apply_decoder_kv_slice(model: nn.Module, model_cfg, is_main: bool = True) -> None:
    """Hand each decoder layer only its own past_key_values entry.

    Registered as a forward-pre-hook with prepend=True AFTER FSDP2 wrapping,
    so it runs before FSDP2's ``_pre_forward`` input cast. The layer only
    ever reads ``past_key_values[self.layer_idx]``, so the sliced dict is
    semantically identical; it just stops FSDP from casting (and AC from
    retaining) fp32 copies of all 26 layers' KV at every layer.
    """
    if not bool(getattr(model_cfg, "decoder_kv_slice", False)):
        return
    from loongforge.models.embodied.giga_brain07.paligemma2_with_expert import (
        Gemma2DecoderLayerWithExpert,
    )

    pat = re.compile(r"(^|\.)paligemma_with_expert\.layers\.(\d+)$")
    n = 0
    for name, mod in model.named_modules():
        if not pat.search(name):
            continue
        inner = mod
        while not isinstance(inner, Gemma2DecoderLayerWithExpert):
            inner = getattr(inner, "_checkpoint_wrapped_module", None)
            if inner is None:
                break
        if inner is None:
            continue
        idx = int(inner.self_attn.layer_idx)

        def _hook(module, args, kwargs, idx=idx):
            pkv = kwargs.get("past_key_values")
            if isinstance(pkv, dict) and idx in pkv and len(pkv) > 1:
                kwargs = dict(kwargs)
                kwargs["past_key_values"] = {idx: pkv[idx]}
                return args, kwargs
            return None

        mod.register_forward_pre_hook(_hook, prepend=True, with_kwargs=True)
        n += 1
    if is_main:
        logger.info("GigaBrain07 decoder_kv_slice ENABLED on %d layers", n)
    if n == 0:
        raise RuntimeError("model.decoder_kv_slice=true but no decoder layer matched")


def apply_regional_compile(model: nn.Module, model_cfg, is_main: bool = True) -> None:
    """Regionally torch.compile inner compute modules (default OFF).

    Runs after AC + FSDP2 wrapping. Only the *inner* module's instance
    ``forward`` is replaced by a compiled callable, so the FSDP2 pre/post
    forward hooks and the non-reentrant AC CheckpointWrapper stay eager
    around each compiled region (the AC recompute re-enters the same
    compiled graph). All layers of one class share one code object, so
    Dynamo compiles one graph per region class (params are graph inputs
    with inline_inbuilt_nn_modules) instead of one per layer.
    """
    m = model_cfg
    if not bool(getattr(m, "compile_model", False)):
        return
    from loongforge.models.embodied.giga_brain07.paligemma2_with_expert import (
        Gemma2AttentionWithExpert,
        Gemma2DecoderLayerWithExpert,
        Gemma2MLP,
        SiglipEncoderLayer,
    )

    scope = {s.strip() for s in str(m.compile_scope).split(",") if s.strip()}
    known = {"decoder", "vision", "decoder_mlp", "decoder_attn"}
    if not scope or scope - known:
        raise ValueError(f"model.compile_scope={m.compile_scope!r}; allowed {sorted(known)}")
    mode = str(m.compile_mode) if m.compile_mode not in (None, "", "default") else None
    kw = dict(
        backend="inductor",
        mode=mode,
        dynamic=getattr(m, "compile_dynamic", None),
        fullgraph=bool(getattr(m, "compile_fullgraph", False)),
    )

    def _in_decoder(name: str) -> bool:
        return "paligemma_with_expert.layers." in name

    def _decoder_kv_shim(layer, compiled):
        # The 2-stage train forward (prefix fill_kv_cache -> suffix reading
        # past_key_values[self.layer_idx]) makes Dynamo guard on the int
        # ``self_attn.layer_idx`` -> one recompile per layer, and after
        # recompile_limit(8) the remaining layers silently run EAGER. For
        # the cache-read call we hand the layer a 1-entry view {0: kv} and
        # present layer_idx=0 for the duration of the call, so all 26
        # layers share ONE graph. Same tensors -> numerics unchanged; the
        # AC recompute re-enters this shim with the same args.
        attn = layer.self_attn if hasattr(layer, "self_attn") else layer
        real_idx = int(attn.layer_idx)
        import inspect
        sig = inspect.signature(layer.forward)

        def fwd(*args, **kwargs):
            ba = sig.bind(*args, **kwargs)
            ba.apply_defaults()
            a = ba.arguments
            pkv = a.get("past_key_values")
            if (a.get("use_cache") and not a.get("fill_kv_cache")
                    and isinstance(pkv, dict) and real_idx in pkv):
                a["past_key_values"] = {0: pkv[real_idx]}
                attn.layer_idx = 0
                try:
                    return compiled(*ba.args, **ba.kwargs)
                finally:
                    attn.layer_idx = real_idx
            return compiled(*ba.args, **ba.kwargs)

        return fwd

    counts: Dict[str, int] = {}
    for name, mod in model.named_modules():
        tag = None
        if "decoder" in scope and isinstance(mod, Gemma2DecoderLayerWithExpert):
            tag = "decoder"
        elif "vision" in scope and isinstance(mod, SiglipEncoderLayer):
            tag = "vision"
        elif ("decoder_mlp" in scope and "decoder" not in scope
              and isinstance(mod, Gemma2MLP) and _in_decoder(name)):
            tag = "decoder_mlp"
        elif ("decoder_attn" in scope and "decoder" not in scope
              and isinstance(mod, Gemma2AttentionWithExpert) and _in_decoder(name)):
            tag = "decoder_attn"
        if tag is None or getattr(mod, "_gb0_compiled", False):
            continue
        compiled = torch.compile(mod.forward, **kw)
        if tag in ("decoder", "decoder_attn"):
            compiled = _decoder_kv_shim(mod, compiled)
        mod.forward = compiled
        mod._gb0_compiled = True
        counts[tag] = counts.get(tag, 0) + 1
    if is_main:
        logger.info(
            "GigaBrain07 torch.compile ENABLED: scope=%s compiled=%s kwargs=%s",
            sorted(scope), counts, kw,
        )
    if not counts:
        raise RuntimeError("model.compile_model=true but no module matched compile_scope")
    if is_main:
        import atexit

        def _report_dynamo():
            try:
                from torch._dynamo.utils import counters
                gb = dict(counters.get("graph_break", {}))
                st = dict(counters.get("stats", {}))
                fr = dict(counters.get("frames", {}))
                print("[gb0-compile-health] frames=%s stats=%s graph_breaks=%d %s"
                      % (fr, st, sum(gb.values()), list(gb.items())[:10]), flush=True)
            except Exception as e:  # pragma: no cover
                print("[gb0-compile-health] unavailable: %r" % (e,), flush=True)

        atexit.register(_report_dynamo)


__all__ = ["apply_decoder_kv_slice", "apply_regional_compile"]
