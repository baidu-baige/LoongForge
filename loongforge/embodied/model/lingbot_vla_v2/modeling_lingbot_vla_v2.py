# Copyright 2026 The LoongForge Authors.
# SPDX-License-Identifier: Apache-2.0

"""LoongForge policy wrapper for the lingbot-vla-v2 VLA model.

The network itself (Qwen3-VL-4B VLM + Token-MoE action expert + depth/video
alignment heads) comes from the pinned upstream ``lingbotvla`` package
unmodified, for numerical parity with the upstream benchmark. This module
provides the thin LoongForge-facing layer required by the model registry:

* ``@register_model("lingbot_vla_v2")``
* ``from_pretrained(model_cfg)`` — builds the internal ``LingbotVLAV2Config``
  from the typed dataclass and loads the pretrained checkpoint.
* ``forward(batch)`` — returns ``(loss, log_dict)``.

Teacher models (MoGE/MoRGBD depth, DINO video) are *not* part of this module;
they are frozen inference-only networks owned by the batch enricher so
they never enter FSDP sharding or the optimizer.
"""

import dataclasses
import logging

import torch
import torch.nn as nn

from loongforge.embodied.model.registry import register_model
from loongforge.embodied.model.lingbot_vla_v2.model_configuration_lingbot_vla_v2 import (
    LingbotVLAV2ModelConfig,
)

logger = logging.getLogger(__name__)

# Fields on LingbotVLAV2ModelConfig that are deliberately NOT forwarded to the
# reference LingbotVLAV2Config: LoongForge routing/checkpoint, the training loop,
# the custom_dp collective knobs (inherited from CustomDpConfig),
# the Muon optimizer, and the MoE-bias/expert-lr controls that the load-balance
# hook (moe_load_balance.py) consumes directly. Kept explicit so that adding a new
# model-structure field without wiring it into build_internal_config fails loudly
# (see _assert_full_field_coverage) rather than silently taking a reference default.
_NON_NETWORK_FIELDS = frozenset({
    "model_type", "model_path",
    # MoE bias / expert-lr controls consumed outside the reference config
    "bias_centering", "bias_update_interval", "use_moe_expert_lr",
    # training loop / device
    "gradient_checkpointing", "enable_mixed_precision", "init_device",
    # runtime teacher / compile switches
    "async_teacher", "pipeline_teacher", "flex_compile", "regional_compile",
    # Muon parameter routing
    "muon_exclude_name_patterns",
    # custom_dp collective knobs (CustomDpConfig)
    "grad_reduce_dtype", "param_sync_precision",
    "param_sync_fp8_reprime_interval", "param_sync_fp8_include",
    "param_sync_bf16_with_fp8", "grad_overlap", "param_overlap",
    "comm_bucket_mb", "grad_inflight_mb",
})


def _assert_full_field_coverage(model_cfg, forwarded) -> None:
    """Fail if a config field is neither forwarded nor explicitly LoongForge-only.

    Guards ``build_internal_config`` against drifting out of sync with
    ``LingbotVLAV2ModelConfig``: a newly added model-structure field that is not
    wired below would otherwise be silently dropped and fall back to the reference
    default. Also catches stale entries in ``_NON_NETWORK_FIELDS``.
    """
    all_fields = {f.name for f in dataclasses.fields(model_cfg)}
    forwarded = set(forwarded)
    unclassified = all_fields - forwarded - _NON_NETWORK_FIELDS
    if unclassified:
        raise ValueError(
            "build_internal_config is out of sync with LingbotVLAV2ModelConfig: "
            f"field(s) {sorted(unclassified)} are neither forwarded to the reference "
            "LingbotVLAV2Config nor listed in _NON_NETWORK_FIELDS. Forward them "
            "below, or add them to _NON_NETWORK_FIELDS if they are LoongForge-only."
        )
    stale = _NON_NETWORK_FIELDS - all_fields
    if stale:
        raise ValueError(
            f"_NON_NETWORK_FIELDS lists field(s) {sorted(stale)} that "
            "LingbotVLAV2ModelConfig no longer defines; remove them."
        )


def build_internal_config(model_cfg: LingbotVLAV2ModelConfig):
    """Translate the typed ModelConfig into the internal ``LingbotVLAV2Config``.

    The config class is resolved through :mod:`upstream_adapter`, which returns
    the pinned upstream class.
    """
    from loongforge.embodied.model.lingbot_vla_v2.upstream_adapter import (
        resolve_config_cls,
    )

    LingbotVLAV2Config = resolve_config_cls()

    align_params = model_cfg.resolved_align_params()

    network_kwargs = dict(
        vlm_repo_id=model_cfg.vlm_repo_id,
        tokenizer_path=model_cfg.tokenizer_path,
        post_training=model_cfg.post_training,
        adanorm_time=model_cfg.adanorm_time,
        freeze_vision_encoder=model_cfg.freeze_vision_encoder,
        action_dim=model_cfg.action_dim,
        max_action_dim=model_cfg.max_action_dim,
        max_state_dim=model_cfg.max_state_dim,
        chunk_size=model_cfg.chunk_size,
        vlm_causal=model_cfg.vlm_causal,
        tokenizer_max_length=model_cfg.tokenizer_max_length,
        loss_type=model_cfg.loss_type,
        use_compile=model_cfg.use_compile,
        use_moe=model_cfg.use_moe,
        token_num_experts=model_cfg.token_num_experts,
        token_top_k=model_cfg.token_top_k,
        token_moe_intermediate_size=model_cfg.token_moe_intermediate_size,
        token_shared_intermediate_size=model_cfg.token_shared_intermediate_size,
        bias_update_speed=model_cfg.bias_update_speed,
        sequence_wise_loss_coeff=model_cfg.sequence_wise_loss_coeff,
        sequence_wise_mode=model_cfg.sequence_wise_mode,
        router_z_loss_coeff=model_cfg.router_z_loss_coeff,
        router_activation=model_cfg.router_activation,
        routed_scaling_factor=model_cfg.routed_scaling_factor,
        use_shared_expert_gate=model_cfg.use_shared_expert_gate,
        moe_implementation=model_cfg.moe_implementation,
        split_fused_experts_from_decoder_fsdp=model_cfg.split_fused_experts_from_decoder_fsdp,
        action_fp32=model_cfg.action_fp32,
        precompute_grid_thw=model_cfg.precompute_grid_thw,
        attention_implementation=model_cfg.attention_implementation,
        # Post-processed fields (still count as forwarded for coverage):
        token_moe_layers=list(model_cfg.token_moe_layers),
        align_params=align_params if align_params else None,
    )
    _assert_full_field_coverage(model_cfg, network_kwargs.keys())

    return LingbotVLAV2Config(**network_kwargs)


def _apply_hf_patches(model_cfg: LingbotVLAV2ModelConfig) -> None:
    """Apply the transformers monkey patches (Qwen3-VL + Qwen2 expert).

    Patch entry points are resolved through :mod:`upstream_adapter`, which
    returns the pinned upstream patch functions.
    """
    from loongforge.embodied.model.lingbot_vla_v2.upstream_adapter import (
        resolve_qwen2_patch,
        resolve_qwen3_vl_patch,
    )

    apply_lingbot_qwen3_vl_patch = resolve_qwen3_vl_patch()
    apply_lingbot_qwen2_patch = resolve_qwen2_patch()

    apply_lingbot_qwen3_vl_patch()
    apply_lingbot_qwen2_patch()


# Block lists eligible for regional ``torch.compile``, as submodule keys under
# ``policy.model.qwenvl_with_expert``. Compiling one block at a time rather than
# the whole tower is deliberate:
#   * the backbone's ``flex_attention`` call lives in
#     ``QwenvlWithExpertV2Model.forward``, outside the layers, so a per-layer
#     region leaves the separately compiled ``flex_compile`` path untouched;
#   * ``build_block_mask`` is ``@torch.compiler.disable``, and it also sits
#     outside the layers, so no region has to graph-break around it;
#   * each block is its own graph, so a shape change recompiles one block
#     instead of the model.
_REGIONAL_COMPILE_TARGETS = {
    "vision": "qwenvl.visual.blocks",
    "vlm_decoder": "qwenvl.model.language_model.layers",
    "action_expert": "qwen_expert.model.layers",
}


def apply_regional_compile(policy: nn.Module, targets) -> None:
    """Compile the block lists named by ``model.regional_compile``, per block.

    ``nn.Module.compile`` is used rather than ``torch.compile(module)`` because
    it installs the compiled call on the existing module instead of returning a
    wrapper. By the time this runs, the custom_dp optimizer has
    already captured this module tree, so replacing submodules would leave it
    holding the uncompiled ones.

    The blocks derive from ``GradientCheckpointingLayer``, whose ``__call__``
    forwards to ``nn.Module.__call__``, which is what dispatches to the
    compiled call. The compiled region therefore lands *inside* the checkpoint
    rather than around it, which is the nesting that lets the recompute pass
    reuse the cached graph instead of retracing.
    """
    targets = list(targets)
    if not targets:
        return
    unknown = sorted(set(targets) - set(_REGIONAL_COMPILE_TARGETS))
    if unknown:
        raise ValueError(
            f"unknown model.regional_compile targets: {', '.join(unknown)}; "
            f"valid targets are {sorted(_REGIONAL_COMPILE_TARGETS)}"
        )
    net = policy.model.qwenvl_with_expert
    for target in targets:
        blocks = net.get_submodule(_REGIONAL_COMPILE_TARGETS[target])
        for block in blocks:
            # dynamic=False matches flex_compile: shapes are padded to a
            # multiple of FLEX_SPARSE_BLOCK_SIZE, so specializing is correct
            # and avoids the dynamic-shape guards on every block.
            block.compile(dynamic=False)
        logger.info(
            "Regional compile: %s blocks under %s",
            len(blocks),
            _REGIONAL_COMPILE_TARGETS[target],
        )


@register_model("lingbot_vla_v2")
class LingbotVlaV2ForTraining(nn.Module):
    """Registry wrapper around the reference ``LingbotVlaV2Policy``."""

    def __init__(self, config: LingbotVLAV2ModelConfig, policy: nn.Module):
        super().__init__()
        self.config = config
        self.policy = policy

    @classmethod
    def from_pretrained(cls, model_cfg) -> "LingbotVlaV2ForTraining":
        """Build the upstream policy from the typed config and wrap it."""
        if not isinstance(model_cfg, LingbotVLAV2ModelConfig):
            raise TypeError(
                "LingbotVlaV2ForTraining.from_pretrained expects a typed "
                f"LingbotVLAV2ModelConfig; got {type(model_cfg).__name__}."
            )
        _apply_hf_patches(model_cfg)

        from loongforge.embodied.model.lingbot_vla_v2.upstream_adapter import (
            resolve_build_foundation_model,
        )

        build_foundation_model = resolve_build_foundation_model(model_cfg)

        internal_cfg = build_internal_config(model_cfg)
        # ``enable_mixed_precision`` keeps master weights in fp32 (benchmark
        # semantics); FSDP mp_policy then decides the compute dtype.
        torch_dtype = "float32" if model_cfg.enable_mixed_precision else "bfloat16"
        policy = build_foundation_model(
            config_path=model_cfg.model_path,
            config_cls=internal_cfg,
            weights_path=model_cfg.model_path,
            torch_dtype=torch_dtype,
            init_device=model_cfg.init_device,
            config_kwargs={
                "vlm_repo_id": model_cfg.vlm_repo_id,
                "tokenizer_path": model_cfg.tokenizer_path,
                "post_training": model_cfg.post_training,
                "adanorm_time": model_cfg.adanorm_time,
            },
            moe_implementation=model_cfg.moe_implementation,
        )
        return cls(model_cfg, policy)

    def forward(self, batch):
        """Training forward. ``batch`` is a LingbotVLAV2PreparedBatch whose
        ``data`` dict already contains any teacher targets injected by the
        trainer (``depth_targets`` / ``future_*_targets`` ...).

        Returns (total_loss, log_dict) per the LoongForge model contract.
        """
        inputs = dict(batch.data)
        inputs.pop("rep_id", None)
        inputs.pop("pil_images", None)
        inputs.pop("future_pil_images", None)
        inputs.pop("future_video_effective_fps", None)

        outputs = self.policy(**inputs)
        (
            total_loss,
            vla_loss,
            depth_loss,
            future_depth_loss,
            future_video_loss,
            seq_wise_loss,
            loss_log,
            _depth_preds,
            _future_depth_preds,
            _future_video_preds,
            _current_video_preds,
        ) = outputs

        def _item(x):
            return x.detach().item() if torch.is_tensor(x) else float(x)

        log_dict = {
            "vla_loss": _item(vla_loss),
            "depth_loss": _item(depth_loss),
            "future_depth_loss": _item(future_depth_loss),
            "future_video_loss": _item(future_video_loss),
            "seq_wise_loss": _item(seq_wise_loss),
            "router_z_loss": _item(
                loss_log.get("router_z_loss", loss_log.get("moe_zloss/weighted", 0.0))
            ),
        }
        return total_loss, log_dict

    @torch.no_grad()
    def predict_action_chunk(self, *args, **kwargs):
        """Inference entry — delegates to the reference policy sampler."""
        return self.policy.sample_actions(*args, **kwargs)
