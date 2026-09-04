# Copyright 2026 The LoongForge Authors.
# SPDX-License-Identifier: Apache-2.0

"""LoongForge policy wrapper for the lingbot-vla-v2 VLA model.

The network itself (Qwen3-VL-4B VLM + Token-MoE action expert + depth/video
alignment heads) is vendored unmodified under ``vendor/`` for numerical parity
with the upstream benchmark. This module provides the thin LoongForge-facing
layer required by the model registry:

* ``@register_model("lingbot_vla_v2")``
* ``from_pretrained(model_cfg)`` — builds the internal ``LingbotVLAV2Config``
  from the typed dataclass and loads the pretrained checkpoint.
* ``forward(batch)`` — returns ``(loss, log_dict)``.

Teacher models (MoGE/MoRGBD depth, DINO video) are *not* part of this module;
they are frozen inference-only networks owned by the batch enricher so
they never enter FSDP sharding or the optimizer.
"""

import logging

import torch
import torch.nn as nn

from loongforge.embodied.model.registry import register_model
from loongforge.embodied.model.lingbot_vla_v2.model_configuration_lingbot_vla_v2 import (
    LingbotVLAV2ModelConfig,
)

logger = logging.getLogger(__name__)


def build_internal_config(model_cfg: LingbotVLAV2ModelConfig):
    """Translate the typed ModelConfig into the vendored ``LingbotVLAV2Config``."""
    from loongforge.embodied.model.lingbot_vla_v2.vendor.lingbot_vla.configuration_lingbot_vla import (
        LingbotVLAV2Config,
    )

    align_params = model_cfg.align_params or {}
    if hasattr(align_params, "items") and not isinstance(align_params, dict):
        # OmegaConf DictConfig → plain dict (vendored code expects dict semantics)
        from omegaconf import OmegaConf

        align_params = OmegaConf.to_container(align_params, resolve=True)

    token_moe_layers = list(model_cfg.token_moe_layers)

    return LingbotVLAV2Config(
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
        align_params=align_params if align_params else None,
        use_compile=model_cfg.use_compile,
        use_moe=model_cfg.use_moe,
        token_moe_layers=token_moe_layers,
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
    )


def _apply_hf_patches(model_cfg: LingbotVLAV2ModelConfig) -> None:
    """Apply the vendored transformers monkey patches (Qwen3-VL + Qwen2 expert)."""
    from loongforge.embodied.model.lingbot_vla_v2.vendor.lingbot_vla.qwen2_action_expert import (
        apply_lingbot_qwen2_patch,
    )

    from loongforge.embodied.model.lingbot_vla_v2.vendor.lingbot_vla.qwen3vl_in_vla import (
        apply_lingbot_qwen3_vl_patch,
    )

    apply_lingbot_qwen3_vl_patch()
    apply_lingbot_qwen2_patch()


@register_model("lingbot_vla_v2")
class LingbotVlaV2ForTraining(nn.Module):
    """Registry wrapper around the vendored ``LingbotVlaV2Policy``."""

    def __init__(self, config: LingbotVLAV2ModelConfig, policy: nn.Module):
        super().__init__()
        self.config = config
        self.policy = policy

    @classmethod
    def from_pretrained(cls, model_cfg) -> "LingbotVlaV2ForTraining":
        if not isinstance(model_cfg, LingbotVLAV2ModelConfig):
            raise TypeError(
                "LingbotVlaV2ForTraining.from_pretrained expects a typed "
                f"LingbotVLAV2ModelConfig; got {type(model_cfg).__name__}."
            )
        _apply_hf_patches(model_cfg)

        from loongforge.embodied.model.lingbot_vla_v2.vendor.models_common.auto import (
            build_foundation_model,
        )

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
        """Inference entry — delegates to the vendored policy sampler."""
        return self.policy.sample_actions(*args, **kwargs)
