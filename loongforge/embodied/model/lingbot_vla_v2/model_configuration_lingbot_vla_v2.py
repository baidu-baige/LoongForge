# Copyright 2026 The LoongForge Authors.
# SPDX-License-Identifier: Apache-2.0

"""LingbotVLA-V2 ModelConfig — model-structure parameters (YAML ``model:`` section).

This frozen dataclass is the single source of truth for the model structure and
model-side training switches of the lingbot_vla_v2 VLA model (Qwen3-VL-4B VLM +
Token-MoE action expert + depth/video alignment heads).

Usage rules (per LoongForge-VLA spec):
1. Always read fields via direct attribute access: ``model_cfg.action_dim``.
2. Never use ``getattr(cfg, "x", default)`` / ``cfg.get(...)`` in business code.
3. To add/change a model-structure parameter, edit only this dataclass.

The reference benchmark network is driven by a ``LingbotVLAV2Config`` (a
``PretrainedConfig`` subclass). ``build_internal_config()`` translates this
typed dataclass into that internal config so the two never diverge.
"""

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

from loongforge.embodied.distributed.replicated_sharded_utils.replicated_sharded_config import (
    ReplicatedShardedConfig,
)


@dataclass(frozen=True)
class LingbotVLAV2ModelConfig(ReplicatedShardedConfig):
    """lingbot_vla_v2 model-structure config (maps 1:1 to YAML ``model:`` section).

    The replicated-sharded collective-precision and overlap knobs
    (``grad_reduce_dtype``, ``param_sync_precision``, ``grad_overlap`` ...) are
    inherited from ``ReplicatedShardedConfig``; edit their defaults there.
    """

    model_type: str = "lingbot_vla_v2"

    # ── Pretrained sources ──
    # ``model_path`` holds the 6B lingbot-vla-v2 checkpoint (VLM + expert + heads).
    model_path: Optional[str] = None
    tokenizer_path: Optional[str] = None
    vlm_repo_id: Optional[str] = None
    # ``fused`` matches the benchmark robotwin.yaml (grouped-GEMM experts).
    moe_implementation: str = "fused"

    # ── Task dimensions (shared with data pipeline) ──
    action_dim: int = 55
    max_action_dim: int = 55
    max_state_dim: int = 55
    chunk_size: int = 50  # action horizon / n_action_steps

    # ── VLM / expert structure switches ──
    post_training: bool = True
    adanorm_time: bool = True
    vlm_causal: bool = True
    tokenizer_max_length: int = 72
    loss_type: str = "L1_fm"
    freeze_vision_encoder: bool = False
    action_fp32: bool = False
    attention_implementation: str = "flex_cached"
    precompute_grid_thw: bool = True

    # ── Token-MoE (action expert) ──
    use_moe: bool = True
    token_moe_layers: List[int] = field(default_factory=lambda: list(range(36)))
    token_num_experts: int = 32
    token_top_k: int = 4
    token_moe_intermediate_size: int = 512
    token_shared_intermediate_size: int = 704
    bias_update_speed: float = 0.0
    bias_centering: bool = False
    bias_update_interval: int = 1
    sequence_wise_mode: str = "per_sequence"
    sequence_wise_loss_coeff: float = 1e-3
    router_z_loss_coeff: float = 1e-4
    router_activation: str = "sigmoid"
    routed_scaling_factor: float = 4.0
    use_shared_expert_gate: bool = False
    use_moe_expert_lr: bool = True
    split_fused_experts_from_decoder_fsdp: bool = False

    # ── Model-side training switches (benchmark parity) ──
    gradient_checkpointing: bool = True
    use_compile: bool = True
    enable_mixed_precision: bool = True
    init_device: str = "cuda"
    # Run the frozen depth/video teachers on a side stream + worker thread so they
    # overlap the student forward instead of sitting in front of it.
    async_teacher: bool = True
    # Start each step's teachers one step early, inside the previous step's
    # optimizer window. That window is dominated by NCCL, so it has the tensor-core
    # and latency slack the forward does not, and absorbs most of the teacher.
    # Caveat: it reads the dataloader a step ahead of training, so a checkpoint taken
    # between steps replays without the batches already prefetched.
    pipeline_teacher: bool = True
    # Compile ``flex_attention`` once at import rather than running its eager
    # op-by-op reference path, which materializes the [B, H, Q, KV] scores and costs
    # host time that scales with shape. Set false only to A/B against eager.
    flex_compile: bool = True
    # Regional ``torch.compile`` targets, applied one transformer block at a time.
    # Valid entries are the keys of ``recipe._REGIONAL_COMPILE_TARGETS``; empty means
    # every block runs eager. Off by default: all three targets are worth only a
    # few percent, which does not pay for the first step's inductor compile, and it
    # is not bitwise (loss moves slightly, grad norm unchanged). The ceiling is low
    # by construction: most device-0 kernel time is NCCL, and most of the fusable
    # elementwise work sits outside these blocks.
    regional_compile: List[str] = field(default_factory=list)

    # ── Muon optimizer knobs (used when --optimizer replicated_muon) ──
    muon_momentum: float = 0.95
    muon_nesterov: bool = True
    muon_ns_steps: int = 5
    muon_adjust_lr_fn: str = "match_rms_adamw"
    muon_exclude_name_patterns: Optional[List[str]] = None

    # ── Depth / video alignment (teacher distillation) ──
    # Carried verbatim as a nested dict so the reference heads and the in-loop
    # teachers receive configuration identical to the benchmark robotwin.yaml.
    align_params: Optional[Dict[str, Any]] = None

    def resolved_align_params(self) -> Dict[str, Any]:
        """Return ``align_params`` as a plain resolved dict.

        ``align_params`` arrives as an OmegaConf ``DictConfig`` when loaded from
        YAML; the reference heads and in-loop teachers expect plain-dict
        semantics, so normalise it in one place.
        """
        align = self.align_params or {}
        if not isinstance(align, dict):
            from omegaconf import OmegaConf

            align = OmegaConf.to_container(align, resolve=True)
        return align
