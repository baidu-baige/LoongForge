# Copyright 2026 The LoongForge Authors.
# SPDX-License-Identifier: Apache-2.0

"""GigaBrain07 ModelConfig — model-structure parameters (YAML ``model:`` section).

方案B: the GigaBrain-0.7-3.5B-Base checkpoint carries its full architecture in
its own ``config.json`` (proprio_anchor / num_embodiments=8), so the model is
built with ``GigaBrain07Policy.from_pretrained(pretrained)`` and this dataclass
only needs to carry the pretrained path plus the handful of switches the
LoongForge trainer reads back (optimizer hyper-params, loss weight, activation
checkpoint classes).

Mirrors the pi05 style (model/pi05/model_configuration_pi05.py): a frozen
dataclass whose defaults are the single source of truth, overridable by the
YAML ``model:`` section.
"""

from dataclasses import dataclass, field
from typing import List, Optional, Tuple


@dataclass(frozen=True)
class GigaBrain07ModelConfig:
    """GigaBrain07 model-structure config (maps 1:1 to YAML ``model:`` section)."""

    model_type: str = "giga_brain07"

    # ── Checkpoint (arch comes straight from its config.json) ──
    # Local known-good 0.7 base ckpt; overridable from YAML (oc.env GB0_PRETRAINED).
    pretrained: str = "/mnt/cfs/pxy/ckpt/open-gigaai/GigaBrain-0.7-3.5B-Base"

    # ── Semantic switches (mirror giga train/transform config) ──
    vlm_type: str = "paligemma2"
    state_input_mode: str = "proprio_anchor"
    observation_memory_size: int = 1
    num_embodiments: int = 8
    action_horizon: int = 50
    max_action_dim: int = 32

    # llm (next-token) loss weight — giga a2d finetune uses 0.0.
    llm_loss_weight: float = 0.0

    # ── Optimizer hyper-params (giga: AdamW betas(0.9,0.95) eps1e-8 wd1e-10) ──
    lr: float = 2.5e-5
    adam_beta1: float = 0.9
    adam_beta2: float = 0.95
    adam_eps: float = 1.0e-8
    weight_decay: float = 1.0e-10

    # ── LR scheduler (giga WarmupCosineScheduler) ──
    warmup_steps: int = 10
    decay_steps: int = 300
    scheduler_end_value: float = 0.1

    # ── Activation checkpointing (giga: Gemma2DecoderLayerWithExpert first 15) ──
    activation_checkpointing: bool = True
    activation_class_names: List[str] = field(
        default_factory=lambda: ["Gemma2DecoderLayerWithExpert__##__15"]
    )

    # flow-matching multi-sample (giga default: off -> batch_mul=1)
    enable_multi_flow_samples: bool = False
    flow_matching_batch_mul: int = 4

    # ── [compile] Opt-in regional torch.compile (default OFF = eager, unchanged) ──
    # Applied by GigaBrain07Trainer AFTER AC + FSDP2 wrapping by replacing the
    # *inner* module's instance ``forward`` (so FSDP2 hooks and the AC
    # CheckpointWrapper stay eager around the compiled region).
    # compile_scope: comma list of regions:
    #   decoder      -> each Gemma2DecoderLayerWithExpert (26)
    #   vision       -> each SigLIP encoder layer (27)
    #   decoder_mlp  -> only Gemma2MLP submodules inside the decoder
    #   decoder_attn -> only Gemma2AttentionWithExpert submodules
    compile_model: bool = False
    compile_scope: str = "decoder,vision"
    compile_mode: str = "default"
    compile_dynamic: Optional[bool] = False
    compile_fullgraph: bool = False
    # [compile] Opt-in (default OFF): give each decoder layer only ITS OWN
    # past_key_values entry in the 2-stage (prefix-KV -> suffix) train forward.
    # Without it FSDP2's forward-input cast (fp32) copies the WHOLE 26-layer
    # bf16 KV dict at every layer and AC keeps those copies alive until
    # backward -> O(L^2) memory (~45.8GB at bs10). Values unchanged.
    decoder_kv_slice: bool = False

    def adam_betas(self) -> Tuple[float, float]:
        return (self.adam_beta1, self.adam_beta2)
