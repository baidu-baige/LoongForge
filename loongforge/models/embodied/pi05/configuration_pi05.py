# Copyright 2026 The LoongForge Authors.
# SPDX-License-Identifier: Apache-2.0
#
# Modified from lerobot (https://github.com/huggingface/lerobot).
# Copyright 2025 Physical Intelligence and The HuggingFace Inc. team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Configuration objects for the PI05 policy and training presets."""

from dataclasses import dataclass, field
import os
from typing import Callable, Optional
from lerobot.configs.policies import PreTrainedConfig
from lerobot.configs.types import FeatureType, NormalizationMode, PolicyFeature
from lerobot.optim.optimizers import AdamWConfig
from lerobot.optim.schedulers import CosineDecayWithWarmupSchedulerConfig
from lerobot.policies.rtc.configuration_rtc import RTCConfig

DEFAULT_IMAGE_SIZE = 224

@dataclass
class PI05Config(PreTrainedConfig):
    """PI05 model hyperparameters, feature schema, and training knobs."""
    paligemma_variant: str = "gemma_2b"
    action_expert_variant: str = "gemma_300m"
    dtype: str = "float32"  # Options: "bfloat16", "float32"

    n_obs_steps: int = 1
    chunk_size: int = 50  # Number of action steps to predict, in openpi called "action_horizon"
    n_action_steps: int = 50  # Number of action steps to execute

    # Shorter state and action vectors will be padded to these dimensions
    max_state_dim: int = 32
    max_action_dim: int = 32

    # Flow matching parameters: see openpi `PI0Pytorch`
    num_inference_steps: int = 10
    time_sampling_beta_alpha: float = 1.5
    time_sampling_beta_beta: float = 1.0
    time_sampling_scale: float = 0.999
    time_sampling_offset: float = 0.001
    min_period: float = 4e-3
    max_period: float = 4.0

    # Real-Time Chunking (RTC) configuration
    rtc_config: RTCConfig | None = None

    image_resolution: tuple[int, int] = (
        DEFAULT_IMAGE_SIZE,
        DEFAULT_IMAGE_SIZE,
    )  # see openpi `preprocessing_pytorch.py`

    # Add empty images. Used to add empty cameras when no image features are present.
    empty_cameras: int = 0

    tokenizer_name: str = os.environ.get("TOKENIZER_PATH", "")

    tokenizer_local_files_only: bool = True  # Default to offline loading

    normalization_mapping: dict[str, NormalizationMode] = field(
        default_factory=lambda: {
            "VISUAL": NormalizationMode.IDENTITY,
            "STATE": NormalizationMode.QUANTILES,  # Pi0.5 uses quantiles for state
            "ACTION": NormalizationMode.QUANTILES,  # Pi0.5 uses quantiles for action
        }
    )

    # Training settings
    gradient_checkpointing: bool = False  # Enable gradient checkpointing for memory optimization
    compile_model: bool = False  # Whether to use torch.compile for model optimization
    compile_mode: str = "max-autotune"  # Torch compile mode
    device: str | None = None  # Device to use for the model (None = auto-detect)

    # Finetuning settings
    freeze_vision_encoder: bool = False  # Freeze only the vision encoder
    train_expert_only: bool = False  # Freeze entire VLM, train only action expert and projections

    # Optimizer settings: see openpi `AdamW`
    optimizer_lr: float = 2.5e-5  # see openpi `CosineDecaySchedule: peak_lr`
    optimizer_betas: tuple[float, float] = (0.9, 0.95)
    optimizer_eps: float = 1e-8
    optimizer_weight_decay: float = 0.01
    optimizer_grad_clip_norm: float = 1.0

    # Minimal Megatron transformer knobs used by training utilities/FLOP estimator.
    num_layers: int = 1
    hidden_size: int = 1024
    num_attention_heads: int = 16
    ffn_hidden_size: int = 4096
    gated_linear_unit: bool = False  # Keep GLU disabled unless explicitly enabled.
    kv_channels: int | None = None  # Will default to hidden_size // num_attention_heads
    seq_length: int = 200  # align with tokenizer_max_length default
    norm_epsilon: float = 1e-6
    caption_channels: int = 4096
    latent_in_channels: int = 4
    latent_out_channels: int = 8
    latent_patch_size: tuple[int, int, int] = (1, 2, 2)
    latent_space_scale: float = 0.5
    latent_time_scale: float = 1.0
    enable_chunkpipe = False
    # Scheduler settings: see openpi `CosineDecaySchedule`
    # Note: These will auto-scale if --steps < scheduler_decay_steps
    # For example, --steps=3000 will scale warmup to 100 and decay to 3000
    scheduler_warmup_steps: int = 1_000
    scheduler_decay_steps: int = 30_000
    scheduler_decay_lr: float = 2.5e-6

    # Megatron pipeline scheduler flag; default to False for pi05.
    deallocate_pipeline_outputs: bool = False
    use_fp32_dtype_for_param_pattern: list[str] | None = field(
        default_factory=lambda: [
            "vision_tower",
            "multi_modal_projector",
            "input_layernorm",
            "post_attention_layernorm",
            "model.norm",
            "action_in_proj",
            "time_mlp_in",
            "time_mlp_out",
            "action_out_proj",
        ]
    )
    # MoE router flag used by Megatron finalize_model_grads; keep False for non-MoE pi05.
    moe_router_enable_expert_bias: bool = False
    # MoE load-balancing strategy; default matches Megatron transformer config.
    moe_router_load_balancing_type: str = "aux_loss"
    # MoE router expert bias update rate; harmless for non-MoE.
    moe_router_bias_update_rate: float = 1e-3

    tokenizer_max_length: int = 200  # see openpi `__post_init__`

    # generate some random number on CPU to align the loss on XPU with that on GPU
    random_fallback_cpu: bool = False

    param_sync_func: Optional[Callable] = None
    grad_sync_func: Optional[Callable] = None

    def __post_init__(self):
        """Run basic validations and fill derived Megatron parameters."""
        super().__post_init__()

        # Validate configuration
        if self.n_action_steps > self.chunk_size:
            raise ValueError(
                f"n_action_steps ({self.n_action_steps}) cannot be greater than chunk_size ({self.chunk_size})"
            )

        if self.paligemma_variant not in ["gemma_300m", "gemma_2b"]:
            raise ValueError(f"Invalid paligemma_variant: {self.paligemma_variant}")

        if self.action_expert_variant not in ["gemma_300m", "gemma_2b"]:
            raise ValueError(f"Invalid action_expert_variant: {self.action_expert_variant}")

        if self.dtype not in ["bfloat16", "float32"]:
            raise ValueError(f"Invalid dtype: {self.dtype}")

        # Fill Megatron-required derived params.
        if self.kv_channels is None and self.num_attention_heads:
            self.kv_channels = max(1, self.hidden_size // self.num_attention_heads)

    def validate_features(self) -> None:
        """Validate and set up input/output features."""
        for i in range(self.empty_cameras):
            key = f"observation.images.empty_camera_{i}"
            empty_camera = PolicyFeature(
                type=FeatureType.VISUAL,
                shape=(3, *self.image_resolution),  # Use configured image resolution
            )
            self.input_features[key] = empty_camera


        if "observation.state" not in self.input_features:
            state_feature = PolicyFeature(
                type=FeatureType.STATE,
                shape=(self.max_state_dim,),  # Padded to max_state_dim
            )
            self.input_features["observation.state"] = state_feature

        if "action" not in self.output_features:
            action_feature = PolicyFeature(
                type=FeatureType.ACTION,
                shape=(self.max_action_dim,),  # Padded to max_action_dim
            )
            self.output_features["action"] = action_feature

    def get_optimizer_preset(self) -> AdamWConfig:
        """Return the default AdamW configuration for PI05 training."""
        return AdamWConfig(
            lr=self.optimizer_lr,
            betas=self.optimizer_betas,
            eps=self.optimizer_eps,
            weight_decay=self.optimizer_weight_decay,
            grad_clip_norm=self.optimizer_grad_clip_norm,
        )

    def get_scheduler_preset(self):
        """Return the cosine decay scheduler configuration used by PI05."""
        return CosineDecayWithWarmupSchedulerConfig(
            peak_lr=self.optimizer_lr,
            decay_lr=self.scheduler_decay_lr,
            num_warmup_steps=self.scheduler_warmup_steps,
            num_decay_steps=self.scheduler_decay_steps,
        )

    @property
    def observation_delta_indices(self) -> None:
        """PI05 does not use observation deltas."""
        return None

    @property
    def action_delta_indices(self) -> list:
        """Return action indices that participate in delta computations."""
        return list(range(self.chunk_size))

    @property
    def reward_delta_indices(self) -> None:
        """PI05 does not use reward deltas."""
        return None
