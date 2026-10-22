# Copyright 2026 The LoongForge Authors.
# SPDX-License-Identifier: Apache-2.0

"""FSDP2 + DMuon wrapping for Wall-OSS-0.5.

The wrap is model-specific: DMuon claims Muon-route matrices before any FSDP2
group exists, and the group layout follows Wall-OSS's Qwen2.5-VL structure
(decoder layers, vision blocks, fp32 norm leaves, fp32 ``action_preprocessor``
leaves). The engine owns the model-agnostic part (storage dtype, mesh, mixed
precision policy) and calls :func:`shard_wall_oss_dmuon` through
``wrap_model(..., dmuon_shard=...)``.
"""

from __future__ import annotations

import logging

import torch
import torch.nn as nn

from loongforge.models.embodied.wall_oss_0_5.qwen2_5.modeling_qwen2_5_vl import (
    Qwen25VLPatchMerger,
    Qwen25VLVisionBlock,
)
from loongforge.models.embodied.wall_oss_0_5.qwen2_5.modeling_qwen2_5_vl_act import (
    Qwen25VLDecoderLayerWithMoE,
    is_qwen25_dmuon_target_param,
)

logger = logging.getLogger(__name__)


def shard_wall_oss_dmuon(model, *, mesh, mp_policy, reshard_after_forward: bool):
    """Shard ``model`` (``WallOss05Model``) with FSDP2 after DMuon claims its params.

    DMuon keeps fp32 master weights for the params it dedicates; the remaining
    params are sharded in place by FSDP2. The dedicated-communication context
    is copied onto the outer wrapper so the engine can detect a DMuon model.
    """
    core = model.model
    ordered_norms = _shard_core(core, mesh=mesh, mp_policy=mp_policy, reshard_after_forward=reshard_after_forward)
    _configure_norm_forward_prefetch(ordered_norms, model.norm_forward_prefetch_distance)
    if hasattr(core, "_dedicated_comm_ctx"):
        model._dedicated_comm_ctx = core._dedicated_comm_ctx
    return model


def _shard_core(core, *, mesh, mp_policy, reshard_after_forward: bool):
    """Wrap ``Qwen25VLMoEForAction`` and return the ordered fp32 norm leaves."""
    import dmuon
    from torch.distributed.fsdp import fully_shard, MixedPrecisionPolicy

    # DMuon keeps fp32 master weights for dedicated params; symmetric
    # FSDP2 params still use MixedPrecisionPolicy for bf16 compute.
    for param in core.parameters():
        if param.requires_grad and param.dtype != torch.float32:
            param.data = param.data.float()

    action_preprocessor_linear_ids = {
        id(module) for module in core.action_preprocessor.modules() if isinstance(module, nn.Linear)
    }

    def hook_boundary(module: nn.Module) -> bool:
        # ActionProcessor helper methods call internal Linear layers
        # directly, bypassing ActionProcessor.forward(). Those internal
        # Linear layers need their own hooks.
        if id(module) in action_preprocessor_linear_ids:
            return True
        return isinstance(
            module,
            (
                Qwen25VLDecoderLayerWithMoE,
                Qwen25VLVisionBlock,
                Qwen25VLPatchMerger,
                type(core.action_preprocessor),
            ),
        )

    assignment = dmuon.dedicate_params(
        core,
        mesh,
        predicate=is_qwen25_dmuon_target_param,
        hook_boundary_predicate=hook_boundary,
        hook_boundary_strict=True,
        compute_dtype=torch.bfloat16,
        reshard_after_forward=reshard_after_forward,
    )
    logger.info(
        "[Qwen2.5 FSDP2 + DMuon] dedicate_params: %d params assigned",
        len(assignment),
    )

    # DMuon links its forward-prefetch chain by __init__ assignment
    # order, in which ``action_preprocessor`` (declared after
    # ``self.model``) lands behind every decoder layer even though it
    # runs right after the vision encoder. Relink in execution order so
    # its unshard is prefetched instead of queueing behind the
    # already-dispatched layer broadcasts.
    _relink_dmuon_forward_prefetch(core, (core.visual, core.action_preprocessor))

    ordered_norms = []
    for layer in core.model.layers:
        for norm_name in ("input_layernorms", "post_attention_layernorms"):
            norm_container = getattr(layer, norm_name, None)
            if norm_container is not None:
                ordered_norms.extend(list(norm_container))

    fp32_mp = MixedPrecisionPolicy(
        param_dtype=torch.float32,
        reduce_dtype=torch.float32,
    )
    # Same fp32 params and fp32 gradient reduction, but FSDP no longer
    # materializes an fp32 copy of the activation. The RMSNorm kernel widens
    # bf16 -> fp32 per element instead, which is lossless, so this removes a
    # cast without touching how the norm parameters are sharded or reduced.
    fp32_mp_nocast = MixedPrecisionPolicy(
        param_dtype=torch.float32,
        reduce_dtype=torch.float32,
        cast_forward_inputs=False,
    )
    shard_kwargs = dict(
        mesh=mesh,
        mp_policy=mp_policy,
        reshard_after_forward=reshard_after_forward,
    )
    fp32_shard_kwargs = dict(shard_kwargs, mp_policy=fp32_mp)
    fp32_nocast_shard_kwargs = dict(shard_kwargs, mp_policy=fp32_mp_nocast)
    nocast_leaves = 0

    def fp32_kwargs_for(leaf: nn.Module) -> dict:
        """Skip the forward-input cast for plain (non-adaptive) RMSNorm leaves.

        An AdaRMSNorm leaf keeps the cast: it returns a ``gate`` whose dtype
        follows the input, and downstream ``_gated_residual`` consumes that
        gate in fp32.
        """
        nonlocal nocast_leaves
        if getattr(leaf, "dense", None) is not None:
            return fp32_shard_kwargs
        nocast_leaves += 1
        return fp32_nocast_shard_kwargs

    def fully_shard_fp32_leaf_or_container(name: str, module: nn.Module) -> None:
        # FSDP2 cannot shard containers without forward(), e.g. ModuleList.
        # When a norm match resolves to a container, shard only norm leaves.
        def is_norm_leaf(leaf_name: str, leaf: nn.Module) -> bool:
            return "norm" in leaf_name.lower() or "norm" in type(leaf).__name__.lower()

        if isinstance(module, (nn.ModuleList, nn.ModuleDict)):
            for leaf_name, leaf in module.named_modules():
                if not leaf_name:
                    continue
                if any(True for _ in leaf.children()):
                    continue
                if not is_norm_leaf(leaf_name, leaf):
                    continue
                logger.info(
                    "[FSDP2] fully_shard fp32 layernorm: %s.%s",
                    name,
                    leaf_name,
                )
                fully_shard(leaf, **fp32_kwargs_for(leaf))
            return

        logger.info("[FSDP2] fully_shard fp32 layernorm: %s", name)
        fully_shard(module, **fp32_kwargs_for(module))

    # Layernorm leaves inside decoder layers / vision blocks / model.norm
    # - wrap with fp32 mp_policy BEFORE wrapping their parents so the
    # nested policy survives. Match-by-FQN mirrors the FSDP1 logic.
    for module_name, module in list(core.named_modules()):
        for child_name, child in list(module.named_children()):
            if any(
                k in child_name.lower() for k in ("input_layernorm", "post_attention_layernorm")
            ) or ("norm" in child_name.lower() and module_name.endswith("model")):
                fully_shard_fp32_leaf_or_container(f"{module_name}.{child_name}", child)

    logger.info(
        "[FSDP2] fp32 norm leaves without forward-input cast: %d",
        nocast_leaves,
    )

    # action_preprocessor leaves: same fp32 treatment as layernorms.
    for name, module in list(core.named_modules()):
        if "action_preprocessor" not in name.lower():
            continue
        if any(True for _ in module.children()):
            continue  # only leaves
        logger.info(
            "[FSDP2] fully_shard fp32 action_preprocessor leaf: %s",
            name,
        )
        fully_shard(module, **fp32_shard_kwargs)

    # Each transformer decoder block: bf16 mp_policy from the engine.
    for idx, layer in enumerate(core.model.layers):
        fully_shard(layer, **shard_kwargs)
        if idx == 0:
            logger.info(
                "[FSDP2] fully_shard model.layers.* (bf16, reshard_after_forward=%s)",
                reshard_after_forward,
            )

    # Each vision tower block: same.
    if hasattr(core, "visual") and hasattr(core.visual, "blocks"):
        for idx, block in enumerate(core.visual.blocks):
            fully_shard(block, **shard_kwargs)
            if idx == 0:
                logger.info("[FSDP2] fully_shard visual.blocks.* (bf16)")

    # Root: covers lm_head, embed_tokens, and any params not in a
    # nested fully_shard. Inner fp32 wraps maintain their own policy.
    logger.info("[FSDP2] fully_shard root (bf16)")
    fully_shard(core, **shard_kwargs)
    return ordered_norms


def _relink_dmuon_forward_prefetch(root, subtree_order) -> int:
    """Rebuild DMuon's forward-prefetch chain in true execution order.

    ``dmuon.api._link_forward_prefetch_states`` orders states by
    ``model.modules()``, i.e. by the order submodules were *assigned* in
    ``__init__``. Wall-OSS declares ``action_preprocessor`` after ``model``
    but runs it before ``model.layers``, so every ``visual.*`` group prefetches
    ``model.layers.0..N`` and ``action_preprocessor.*`` never gets prefetched at
    all. Its demand unshard then queues behind the already-dispatched layer
    broadcasts on the single ``broadcast_stream``, and the compute stream stalls
    for the full head-of-line latency.

    ``subtree_order`` lists the roots of each subtree in the order they actually
    run; modules under an earlier subtree sort before modules under a later one,
    and declaration order is preserved inside each subtree. ``root`` itself
    always sorts first (it owns the embedding/head groups and is unsharded
    before the first forward). Modules matching no entry keep their declaration
    order and sort last.
    """
    states = []
    for module in root.modules():
        states.extend(getattr(module, "_dedicated_states", ()))
    if not states:
        return 0
    comm_ctx = states[0].comm_ctx
    # ``comm_ctx.all_states`` is the list DMuon itself iterates; reorder that
    # exact list rather than a re-derived one.
    states = list(comm_ctx.all_states)
    if len(states) <= 1:
        return 0

    decl_index = {id(module): idx for idx, module in enumerate(root.modules())}
    phase_of: dict[int, int] = {id(root): -1}
    for phase, subtree_root in enumerate(subtree_order):
        if subtree_root is None:
            continue
        for module in subtree_root.modules():
            phase_of.setdefault(id(module), phase)
    fallback_phase = len(subtree_order)

    def sort_key(state):
        module_id = id(state.module)
        return (
            phase_of.get(module_id, fallback_phase),
            decl_index.get(module_id, len(decl_index)),
        )

    ordered = sorted(states, key=sort_key)
    for state in ordered:
        state._next_group = None
        state._next_groups = []
    for index in range(len(ordered) - 1):
        ordered[index]._next_group = ordered[index + 1].group
        ordered[index]._next_groups = [later.group for later in ordered[index + 1 :]]
    ordered[0].comm_ctx.all_states[:] = ordered

    logger.info(
        "[DMuon] forward prefetch relinked in execution order: states=%d head=%s",
        len(ordered),
        [type(state.module).__name__ for state in ordered[:4]],
    )
    return len(ordered)


def _configure_norm_forward_prefetch(ordered_norms, distance: int) -> int:
    """Prefetch future norm leaves without changing their ownership."""
    from torch.distributed.fsdp import FSDPModule

    if distance <= 0:
        return 0

    modules = [module for module in ordered_norms if isinstance(module, FSDPModule)]
    if len(modules) <= 1:
        return 0

    edge_count = 0
    for index, module in enumerate(modules):
        targets = modules[index + 1 : index + 1 + distance]
        if targets:
            module.set_modules_to_forward_prefetch(targets)
            edge_count += len(targets)

    logger.info(
        "[FSDP2] norm forward prefetch: modules=%d distance=%d edges=%d",
        len(modules),
        distance,
        edge_count,
    )
    return edge_count
