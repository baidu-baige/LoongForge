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

The vendored benchmark network is driven by a ``LingbotVLAV2Config`` (a
``PretrainedConfig`` subclass). ``build_internal_config()`` translates this
typed dataclass into that internal config so the two never diverge.
"""

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional


@dataclass(frozen=True)
class LingbotVLAV2ModelConfig:
    """lingbot_vla_v2 model-structure config (maps 1:1 to YAML ``model:`` section)."""

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
    enable_fp32: bool = True
    enable_full_shard: bool = False
    module_fsdp_enable: bool = True
    vlm_fsdp: bool = True
    init_device: str = "cuda"
    # Run the frozen depth/video teachers on a side stream + worker thread so they
    # overlap the student forward instead of sitting in front of it.
    async_teacher: bool = True
    # Start each step's teachers one step early, inside the previous step's
    # optimizer window. That window is 375.9 of its 389.4 ms NCCL, so it absorbs 79%
    # of the teacher's 145.7 ms against 34% when the teacher runs into the forward.
    # Caveat: it reads the dataloader a step ahead of training, so a checkpoint taken
    # between steps replays without the batches already prefetched.
    pipeline_teacher: bool = True
    # Compile ``flex_attention`` once at import rather than running its eager
    # op-by-op reference path, which materializes the [B, H, Q, KV] scores and costs
    # host time that scales with shape. Set false only to A/B against eager.
    flex_compile: bool = True
    # Regional ``torch.compile`` targets, applied one transformer block at a time.
    # Valid entries are the keys of ``recipe._REGIONAL_COMPILE_TARGETS``; empty means
    # every block runs eager. Off by default: all three targets are worth 1.022x
    # (1282.6 -> 1255.5 ms on 8 GPUs at GBS80), which does not pay for the 4-13 s
    # inductor cost on the first step, and it is not bitwise (loss moves ~0.7%
    # relative, grad norm unchanged). The ceiling is low by construction: NCCL is
    # 50% of device-0 kernel time while the fusable elementwise bucket is 17.3%,
    # most of it outside these blocks.
    regional_compile: List[str] = field(default_factory=list)

    # ── ZeRO-1 collective precision ──
    # Both settings exempt the parameters ``LingbotVlaV2ParameterPolicy`` marks
    # precision-critical (MoE router/gate weights, 1-D norms/biases): expert
    # selection is decided by fp32 gate logits, so a rounded router weight flips
    # top-k instead of merely perturbing it.
    #
    # ``grad_reduce_dtype``: fp32 | bf16 | mixed | compute. The largest collective
    # in the step; bf16 is 1.229x end-to-end (1574.0 -> 1280.9 ms) for 0.792% max
    # loss deviation over 20 iterations. Single-step grad norm moves up to 8%, so
    # switch back to fp32 when bisecting numerics.
    #
    # ``param_sync_dtype``: compute | bf16. "compute" publishes each parameter in
    # its compute dtype (VLM bf16, action expert fp32); "bf16" also rounds the fp32
    # action-expert parameters on the wire, worth a further 1.076x. Decided against:
    # the same bf16 costs ~1e4 more on the state than on the increment, and it takes
    # the action expert from 0.044% to 86.2% of published elements being identical to
    # the previous step -- the dead zone _ACTION_FP32_MARKERS exists to avoid.
    # Opt in per run: bash run_lingbot_vla_v2_zero1.sh model.param_sync_dtype=bf16
    # Rejected alongside it: NCCL_PROTO tuning, larger comm buckets and deeper
    # parameter-sync queues, none of which reproduced a gain.
    grad_reduce_dtype: str = "bf16"
    param_sync_dtype: str = "compute"

    # ── ZeRO-1 collective overlap ──
    # Declared here rather than read from the environment so the launcher cannot
    # disagree with the code about whether overlap is on: a run that does not go
    # through run_lingbot_vla_v2_zero1.sh must get the same defaults as one that
    # does.
    #   grad_overlap : start each gradient bucket's reduce as soon as backward has
    #                  produced it, instead of after the whole backward.
    #   param_overlap: start each owner's broadcast as soon as its master is
    #                  updated, instead of after the whole optimizer step.
    #   comm_bucket_mb: bucket size for both. Overlap wants finer buckets so
    #                  collectives start early; measured 1024 costs 0.94x with
    #                  overlap on. 0 means "pick by grad_overlap" (256 on, 1024 off).
    #   grad_inflight_mb: byte cap on gradient collectives in flight during
    #                  backward. A memory bound, not a throughput dial -- lower it
    #                  on cards with less memory than the validated configuration.
    grad_overlap: bool = True
    param_overlap: bool = True
    comm_bucket_mb: int = 256
    grad_inflight_mb: int = 3072

    # ── Muon optimizer knobs (used when --optimizer Muon) ──
    muon_momentum: float = 0.95
    muon_nesterov: bool = True
    muon_ns_steps: int = 5
    muon_adjust_lr_fn: str = "match_rms_adamw"
    muon_exclude_name_patterns: Optional[List[str]] = None

    # ── Depth / video alignment (teacher distillation) ──
    # Carried verbatim as a nested dict so the vendored heads and the in-loop
    # teachers receive configuration identical to the benchmark robotwin.yaml.
    align_params: Optional[Dict[str, Any]] = None
