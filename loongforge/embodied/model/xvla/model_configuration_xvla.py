# Copyright 2026 The LoongForge Authors.
# SPDX-License-Identifier: Apache-2.0
#
# Modified from X-VLA (https://github.com/2toinf/X-VLA).
# Copyright 2025 2toINF. All rights reserved.
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

"""X-VLA model configuration definitions.

Two families of config coexist here:

* HuggingFace-side ``PretrainedConfig`` subclasses
  (``Florence2VisionConfig``, ``Florence2LanguageConfig``, ``Florence2Config``,
  ``XVLAConfig``) — passed to the ``PreTrainedModel`` constructors, whose
  ``super().__init__(config)`` requires a ``PretrainedConfig`` instance.

* OmegaConf schema dataclass (``XvlaModelConfig``) — used by
  ``embodied.train.parser`` as the structured schema that the YAML
  ``model:`` section is merged into.  Kept as a plain dataclass so
  ``OmegaConf.structured`` accepts it; the merged instance is converted to
  the ``PretrainedConfig`` classes at model-build time
  (see ``XVLAPolicy.from_pretrained``).
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field, is_dataclass
from typing import Any

from transformers import PretrainedConfig


# ─────────────────────────────────────────────────────────────────────────────
# Robot-type -> domain id (single source of truth).
#
# Synced 1:1 with the reference X-VLA ``DATA_DOMAIN_ID`` table
# (datasets/domain_config.py). The domain id selects the
# ``SoftPromptedTransformer``'s per-domain soft prompts / ``DomainAwareLinear``
# weights, so it must match the value used when the checkpoint was trained.
#
# Kept in the model package (not the data layer) so both the training data
# pipeline (``HDF5VLADataset``) and the inference path (``XVLAPolicy`` /
# ``predict_action``) resolve ``domain_id`` from the same table, without the
# eval path importing ``loongforge.embodied.data``. Unknown robot types
# fall back to 0.
# ─────────────────────────────────────────────────────────────────────────────
DOMAIN_ID_MAP: dict[str, int] = {
    # fine-tuning domains
    "Bridge": 0,
    "RT1": 1,
    "Calvin": 2,
    "libero": 3,
    "widowx-air": 4,
    "AIR-AGILEX-HQ": 5,
    "robotwin2_abs_ee": 6,
    "robotwin2_clean": 6,
    "robocasa-human": 7,
    "VLABench": 8,
    "AGIBOT-challenge": 9,
    "AIR-AGILEX": 10,
    "AIRBOT": 18,
    # pretraining domains
    "robomind-franka": 11,
    "robomind-ur": 12,
    "Droid-Left": 13,
    "Droid-Right": 14,
    "AGIBOT": 15,
    "robomind-agilex": 16,
    "robomind-franka-dual": 17,
    # agibot world challenge (reference maps these to 0)
    "agiworld-on-site-pack": 0,
    "agiworld-on-site-pack-extra": 0,
    "agiworld-on-site-conveyor": 0,
    "agiworld-on-site-conveyor-extra": 0,
    "agiworld-on-site-restock": 0,
    "agiworld-on-site-pour": 0,
    "agiworld-on-site-microwave": 0,
    "agiworld-on-site-cloth": 0,
    "agiworld-on-site-cloth-2": 0,
    # lerobot-sim
    "lift2": 0,
    # x2robot
    "x2robot": 0,
}


def resolve_domain_id(robot_type: str | None) -> int:
    """Resolve a robot type to its domain id via :data:`DOMAIN_ID_MAP`.

    Mirrors the training-path generation (``_DOMAIN_ID.get(robot_type, 0)``);
    unknown / empty robot types map to 0.
    """
    return int(DOMAIN_ID_MAP.get(robot_type or "", 0))



def _coerce_to_dict(value: Any) -> Any:
    """Convert dataclass schema instances to plain dicts, pass others through."""
    if is_dataclass(value) and not isinstance(value, type):
        return asdict(value)
    return value


# ─────────────────────────────────────────────────────────────────────────────
# HuggingFace PretrainedConfig classes (consumed by PreTrainedModel subclasses)
# ─────────────────────────────────────────────────────────────────────────────


class Florence2VisionConfig(PretrainedConfig):
    """Configuration for the Florence-2 DaViT vision backbone."""

    model_type = "davit"
    keys_to_ignore_at_inference = ["past_key_values"]

    def __init__(
        self,
        drop_path_rate: float = 0.1,
        patch_size: list[int] | None = None,
        patch_stride: list[int] | None = None,
        patch_padding: list[int] | None = None,
        patch_prenorm: list[bool] | None = None,
        enable_checkpoint: bool = False,
        dim_embed: list[int] | None = None,
        num_heads: list[int] | None = None,
        num_groups: list[int] | None = None,
        depths: list[int] | None = None,
        window_size: int = 12,
        projection_dim: int = 1024,
        visual_temporal_embedding: dict[str, Any] | None = None,
        image_pos_embed: dict[str, Any] | None = None,
        image_feature_source: list[str] | None = None,
        **kwargs,
    ):
        self.drop_path_rate = drop_path_rate
        self.patch_size = patch_size if patch_size is not None else [7, 3, 3, 3]
        self.patch_stride = patch_stride if patch_stride is not None else [4, 2, 2, 2]
        self.patch_padding = patch_padding if patch_padding is not None else [3, 1, 1, 1]
        self.patch_prenorm = (
            patch_prenorm if patch_prenorm is not None else [False, True, True, True]
        )
        self.enable_checkpoint = enable_checkpoint
        self.dim_embed = dim_embed if dim_embed is not None else [256, 512, 1024, 2048]
        self.num_heads = num_heads if num_heads is not None else [8, 16, 32, 64]
        self.num_groups = num_groups if num_groups is not None else [8, 16, 32, 64]
        self.depths = depths if depths is not None else [1, 1, 9, 1]
        self.window_size = window_size
        self.projection_dim = projection_dim
        self.visual_temporal_embedding = visual_temporal_embedding
        self.image_pos_embed = image_pos_embed
        self.image_feature_source = (
            image_feature_source
            if image_feature_source is not None
            else ["spatial_avg_pool", "temporal_avg_pool"]
        )
        super().__init__(**kwargs)


class Florence2LanguageConfig(PretrainedConfig):
    """Configuration for the Florence-2 BART-style language backbone."""

    model_type = "florence2_language"
    keys_to_ignore_at_inference = ["past_key_values"]
    attribute_map = {"num_attention_heads": "encoder_attention_heads", "hidden_size": "d_model"}

    def __init__(
        self,
        vocab_size: int = 51289,
        max_position_embeddings: int = 1024,
        encoder_layers: int = 12,
        encoder_ffn_dim: int = 4096,
        encoder_attention_heads: int = 16,
        decoder_layers: int = 12,
        decoder_ffn_dim: int = 4096,
        decoder_attention_heads: int = 16,
        encoder_layerdrop: float = 0.0,
        decoder_layerdrop: float = 0.0,
        activation_function: str = "gelu",
        d_model: int = 1024,
        dropout: float = 0.1,
        attention_dropout: float = 0.0,
        activation_dropout: float = 0.0,
        init_std: float = 0.02,
        classifier_dropout: float = 0.0,
        scale_embedding: bool = False,
        use_cache: bool = True,
        num_labels: int = 3,
        pad_token_id: int = 1,
        bos_token_id: int = 0,
        eos_token_id: int = 2,
        is_encoder_decoder: bool = True,
        decoder_start_token_id: int = 2,
        forced_eos_token_id: int = 2,
        num_hidden_layers: int | None = None,
        **kwargs,
    ):
        self.vocab_size = vocab_size
        self.max_position_embeddings = max_position_embeddings
        self.encoder_layers = encoder_layers
        self.encoder_ffn_dim = encoder_ffn_dim
        self.encoder_attention_heads = encoder_attention_heads
        self.decoder_layers = decoder_layers
        self.decoder_ffn_dim = decoder_ffn_dim
        self.decoder_attention_heads = decoder_attention_heads
        self.encoder_layerdrop = encoder_layerdrop
        self.decoder_layerdrop = decoder_layerdrop
        self.activation_function = activation_function
        self.d_model = d_model
        self.dropout = dropout
        self.attention_dropout = attention_dropout
        self.activation_dropout = activation_dropout
        self.init_std = init_std
        self.classifier_dropout = classifier_dropout
        self.scale_embedding = scale_embedding
        self.use_cache = use_cache
        self.num_hidden_layers = (
            num_hidden_layers if num_hidden_layers is not None else encoder_layers
        )
        super().__init__(
            num_labels=num_labels,
            pad_token_id=pad_token_id,
            bos_token_id=bos_token_id,
            eos_token_id=eos_token_id,
            is_encoder_decoder=is_encoder_decoder,
            decoder_start_token_id=decoder_start_token_id,
            forced_eos_token_id=forced_eos_token_id,
            **kwargs,
        )


class Florence2Config(PretrainedConfig):
    """Configuration for the composed Florence-2 vision-language model."""

    model_type = "florence2"
    is_composition = False

    def __init__(
        self,
        vision_config: Florence2VisionConfig | dict[str, Any] | None = None,
        text_config: Florence2LanguageConfig | dict[str, Any] | None = None,
        ignore_index: int = -100,
        vocab_size: int = 51289,
        projection_dim: int = 1024,
        bos_token_id: int = 0,
        eos_token_id: int = 2,
        pad_token_id: int = 1,
        torch_dtype: str | None = None,
        is_encoder_decoder: bool = True,
        **kwargs,
    ):
        if vision_config is None:
            vision_config = Florence2VisionConfig(
                image_pos_embed={"type": "learned_abs_2d", "max_pos_embeddings": 50},
                visual_temporal_embedding={"type": "COSINE", "max_temporal_embeddings": 100},
                image_feature_source=["spatial_avg_pool", "temporal_avg_pool"],
            )
        else:
            vision_config = _coerce_to_dict(vision_config)
            if isinstance(vision_config, dict):
                vision_config = Florence2VisionConfig(**vision_config)
        self.vision_config = vision_config

        if text_config is None:
            text_config = Florence2LanguageConfig(
                vocab_size=51289,
                max_position_embeddings=4096,
                d_model=1024,
                encoder_layers=12,
                encoder_ffn_dim=4096,
                encoder_attention_heads=16,
                decoder_layers=12,
                decoder_ffn_dim=4096,
                decoder_attention_heads=16,
            )
        else:
            text_config = _coerce_to_dict(text_config)
            if isinstance(text_config, dict):
                text_config = Florence2LanguageConfig(**text_config)
        self.text_config = text_config

        self.ignore_index = ignore_index
        self.vocab_size = vocab_size
        self.projection_dim = projection_dim
        self.torch_dtype = torch_dtype
        super().__init__(
            bos_token_id=bos_token_id,
            eos_token_id=eos_token_id,
            pad_token_id=pad_token_id,
            is_encoder_decoder=is_encoder_decoder,
            **kwargs,
        )


class XVLAConfig(PretrainedConfig):
    """Configuration for the XVLA policy model."""

    model_type = "xvla"

    def __init__(
        self,
        florence_config: Florence2Config | dict[str, Any] | None = None,
        hidden_size: int = 1024,
        depth: int = 24,
        num_heads: int = 16,
        mlp_ratio: float = 4.0,
        num_domains: int = 30,
        len_soft_prompts: int = 32,
        dim_time: int = 32,
        max_len_seq: int = 512,
        use_hetero_proj: bool = False,
        attn_dropout: float = 0.1,
        mlp_dropout: float = 0.1,
        soft_prompt_length: int = 32,
        max_action_dim: int = 20,
        real_action_dim: int = 20,
        num_actions: int = 30,
        action_mode: str = "ee6d",
        use_proprio: bool = True,
        enable_torch_compile: bool = False,
        robot_type: str = "",
        **kwargs,
    ):
        if florence_config is None:
            florence_config = Florence2Config()
        else:
            florence_config = _coerce_to_dict(florence_config)
            if isinstance(florence_config, dict):
                florence_config = Florence2Config(**florence_config)
        self.florence_config = florence_config

        self.hidden_size = hidden_size
        self.depth = depth
        self.num_heads = num_heads
        self.mlp_ratio = mlp_ratio
        self.num_domains = num_domains
        self.len_soft_prompts = len_soft_prompts
        self.dim_time = dim_time
        self.max_len_seq = max_len_seq
        self.use_hetero_proj = use_hetero_proj
        # Transformer-block dropout (attention / MLP), plumbed to
        # SoftPromptedTransformer -> TransformerBlock.
        self.attn_dropout = attn_dropout
        self.mlp_dropout = mlp_dropout
        self.soft_prompt_length = soft_prompt_length
        self.max_action_dim = max_action_dim
        self.real_action_dim = real_action_dim
        self.num_actions = num_actions
        self.action_mode = action_mode
        self.use_proprio = use_proprio
        self.enable_torch_compile = enable_torch_compile
        # Embodiment/domain key. Resolved to a domain id via ``DOMAIN_ID_MAP``
        # at inference time (see ``XVLAPolicy.predict_action``), mirroring the
        # training path where the dataset maps ``robot_type`` -> domain id.
        self.robot_type = robot_type
        super().__init__(**kwargs)


# ─────────────────────────────────────────────────────────────────────────────
# OmegaConf-schema dataclasses (used by embodied.train.parser).
# Mirror the PretrainedConfig field set 1:1 but stay pure dataclasses so
# ``OmegaConf.structured`` accepts them; converted to PretrainedConfig at
# model-build time (see XVLAPolicy.from_pretrained).
# ─────────────────────────────────────────────────────────────────────────────


@dataclass
class Florence2VisionConfigSchema:
    """ Schema for Florence2VisionConfig. """
    model_type: str = "davit"
    drop_path_rate: float = 0.1
    patch_size: list[int] = field(default_factory=lambda: [7, 3, 3, 3])
    patch_stride: list[int] = field(default_factory=lambda: [4, 2, 2, 2])
    patch_padding: list[int] = field(default_factory=lambda: [3, 1, 1, 1])
    patch_prenorm: list[bool] = field(default_factory=lambda: [False, True, True, True])
    enable_checkpoint: bool = False
    dim_embed: list[int] = field(default_factory=lambda: [256, 512, 1024, 2048])
    num_heads: list[int] = field(default_factory=lambda: [8, 16, 32, 64])
    num_groups: list[int] = field(default_factory=lambda: [8, 16, 32, 64])
    depths: list[int] = field(default_factory=lambda: [1, 1, 9, 1])
    window_size: int = 12
    projection_dim: int = 1024
    visual_temporal_embedding: dict[str, Any] = field(
        default_factory=lambda: {"type": "COSINE", "max_temporal_embeddings": 100}
    )
    image_pos_embed: dict[str, Any] = field(
        default_factory=lambda: {"type": "learned_abs_2d", "max_pos_embeddings": 50}
    )
    image_feature_source: list[str] = field(
        default_factory=lambda: ["spatial_avg_pool", "temporal_avg_pool"]
    )


@dataclass
class Florence2LanguageConfigSchema:
    """ Schema for Florence2LanguageConfig. """
    model_type: str = "florence2_language"
    vocab_size: int = 51289
    activation_dropout: float = 0.1
    activation_function: str = "gelu"
    attention_dropout: float = 0.1
    d_model: int = 1024
    decoder_attention_heads: int = 16
    decoder_layers: int = 12
    encoder_attention_heads: int = 16
    encoder_layers: int = 12
    dropout: float = 0.1
    max_position_embeddings: int = 4096
    num_hidden_layers: int = 12


@dataclass
class Florence2ConfigSchema:
    """ Schema for Florence2Config. """
    model_type: str = "florence2"
    bos_token_id: int = 0
    eos_token_id: int = 2
    ignore_index: int = -100
    pad_token_id: int = 1
    projection_dim: int = 1024
    vocab_size: int = 51289
    torch_dtype: str = "float32"
    is_encoder_decoder: bool = True
    vision_config: Florence2VisionConfigSchema = field(
        default_factory=Florence2VisionConfigSchema
    )
    text_config: Florence2LanguageConfigSchema = field(
        default_factory=Florence2LanguageConfigSchema
    )


@dataclass
class XvlaModelConfig:
    """X-VLA model-structure config (maps 1:1 to YAML ``model:`` section)."""

    model_type: str = "xvla"
    florence_config: Florence2ConfigSchema = field(default_factory=Florence2ConfigSchema)

    # Transformer action head
    hidden_size: int = 1024
    depth: int = 24
    num_heads: int = 16
    mlp_ratio: float = 4.0
    num_domains: int = 30
    len_soft_prompts: int = 32
    dim_time: int = 32
    max_len_seq: int = 512
    use_hetero_proj: bool = False
    attn_dropout: float = 0.1
    mlp_dropout: float = 0.1
    soft_prompt_length: int = 32

    # Action & proprio (shared with data side)
    action_mode: str = "ee6d"
    use_proprio: bool = True
    num_actions: int = 30
    action_horizon: int = 30
    max_action_dim: int = 20
    real_action_dim: int = 20

    enable_torch_compile: bool = False

    # Embodiment/domain key, resolved to a domain id via ``DOMAIN_ID_MAP`` at
    # inference time (see XVLAConfig.robot_type). Set per benchmark/embodiment.
    robot_type: str = ""
