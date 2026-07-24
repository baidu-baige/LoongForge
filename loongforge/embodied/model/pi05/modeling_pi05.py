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


"""Pi05 VLA model.

PaliGemma VLM backbone with Gemma action expert using joint attention
and flow-matching action prediction.
"""

import copy
import logging
import math
import os
from dataclasses import dataclass
from typing import Dict, Literal, Optional

import numpy as np
import torch
import torch.nn.functional as F
from torch import Tensor, nn
from transformers import AutoTokenizer

try:
    from transformers.models.auto import CONFIG_MAPPING
    from transformers.models.gemma import modeling_gemma
    from lerobot.policies.pi_gemma import (
        PaliGemmaForConditionalGenerationWithPiGemma,
        PiGemmaForCausalLM,
        _gated_residual,
        layernorm_forward,
    )
    _transformers_available = True
except ImportError:
    CONFIG_MAPPING = None
    modeling_gemma = None
    PaliGemmaForConditionalGenerationWithPiGemma = None
    PiGemmaForCausalLM = None
    _gated_residual = None
    layernorm_forward = None
    _transformers_available = False

from loongforge.embodied.model.pi05.model_configuration_pi05 import Pi05ModelConfig
from loongforge.embodied.model.registry import register_model

logger = logging.getLogger(__name__)

OPENPI_ATTENTION_MASK_VALUE = -2.3819763e38
DEFAULT_IMAGE_SIZE = 224


# ═══════════════════════════════════════════════════════════════
# Internal low-level config (backbone structure is fixed, not externally visible)
# ═══════════════════════════════════════════════════════════════

@dataclass
class _PI05InternalConfig:
    """Internal parameters for PI05, backbone structure is fixed, populated by PI05Pytorch.__init__."""
    paligemma_variant: str = "gemma_2b"
    action_expert_variant: str = "gemma_300m"
    dtype: str = "bfloat16"
    chunk_size: int = 50
    n_action_steps: int = 50
    max_state_dim: int = 32
    max_action_dim: int = 32
    num_inference_steps: int = 10
    time_sampling_beta_alpha: float = 1.5
    time_sampling_beta_beta: float = 1.0
    time_sampling_scale: float = 0.999
    time_sampling_offset: float = 0.001
    min_period: float = 4e-3
    max_period: float = 4.0
    image_resolution: tuple = (DEFAULT_IMAGE_SIZE, DEFAULT_IMAGE_SIZE)
    tokenizer_local_files_only: bool = True
    tokenizer_max_length: int = 200
    gradient_checkpointing: bool = False
    compile_model: bool = False
    compile_mode: str = "max-autotune"
    compile_scope: str = "backbone"
    compile_fullgraph: bool = True
    compile_dynamic: bool = False
    freeze_vision_encoder: bool = False
    train_expert_only: bool = False
    random_fallback_cpu: bool = False


# ═══════════════════════════════════════════════════════════════
# Utility Functions
# ═══════════════════════════════════════════════════════════════

def get_safe_dtype(target_dtype, device_type):
    """Return a dtype safe for the given device, falling back to float32 on MPS/CPU."""
    if device_type == "mps" and target_dtype == torch.float64:
        return torch.float32
    if device_type == "cpu":
        if target_dtype == torch.bfloat16:
            return torch.float32
        if target_dtype == torch.float64:
            return torch.float64
    return target_dtype


def create_sinusoidal_pos_embedding(time, dimension, min_period, max_period, device=torch.device("cpu")):
    """Create sinusoidal positional embeddings for diffusion timesteps.

    Args:
        time: 1-D tensor of shape (batch_size,) containing timestep values.
        dimension: Embedding dimension; must be even.
        min_period: Minimum sinusoidal period.
        max_period: Maximum sinusoidal period.
        device: Target device for the output tensor.

    Returns:
        Tensor of shape (batch_size, dimension) with sin/cos embeddings.
    """
    if dimension % 2 != 0:
        raise ValueError(f"dimension ({dimension}) must be divisible by 2")
    if time.ndim != 1:
        raise ValueError("The time tensor is expected to be of shape `(batch_size, )`.")
    time = time.to(device=device)
    dtype = get_safe_dtype(torch.float64, device.type)
    fraction = torch.linspace(0.0, 1.0, dimension // 2, dtype=dtype, device=device)
    period = min_period * (max_period / min_period) ** fraction
    scaling_factor = 1.0 / period * 2 * math.pi
    sin_input = scaling_factor[None, :] * time[:, None]
    return torch.cat([torch.sin(sin_input), torch.cos(sin_input)], dim=1)


def sample_beta(alpha, beta, bsize, device):
    """Sample from a Beta(alpha, beta) distribution.

    Returns:
        Tensor of shape (bsize,) on the given device.
    """
    dist = torch.distributions.Beta(
        torch.tensor(alpha, dtype=torch.float32),
        torch.tensor(beta, dtype=torch.float32),
    )
    return dist.sample((bsize,)).to(device)


def make_att_2d_masks(pad_masks, att_masks):
    """Build 2D attention masks from padding and causal masks."""
    if att_masks.ndim != 2:
        raise ValueError(att_masks.ndim)
    if pad_masks.ndim != 2:
        raise ValueError(pad_masks.ndim)
    cumsum = torch.cumsum(att_masks, dim=1)
    att_2d_masks = cumsum[:, None, :] <= cumsum[:, :, None]
    pad_2d_masks = pad_masks[:, None, :] * pad_masks[:, :, None]
    return att_2d_masks & pad_2d_masks


def pad_vector(vector, new_dim):
    """Pad vector to new_dim along last dimension."""
    if vector.shape[-1] >= new_dim:
        return vector
    return F.pad(vector, (0, new_dim - vector.shape[-1]))


def resize_with_pad_torch(
    images: torch.Tensor, height: int, width: int, mode: str = "bilinear"
) -> torch.Tensor:
    """Resize images with padding to maintain aspect ratio."""
    if images.shape[-1] <= 4:
        channels_last = True
        if images.dim() == 3:
            images = images.unsqueeze(0)
        images = images.permute(0, 3, 1, 2)
    else:
        channels_last = False
        if images.dim() == 3:
            images = images.unsqueeze(0)

    batch_size, channels, cur_height, cur_width = images.shape
    ratio = max(cur_width / width, cur_height / height)
    resized_height = int(cur_height / ratio)
    resized_width = int(cur_width / ratio)
    resized_images = F.interpolate(
        images,
        size=(resized_height, resized_width),
        mode=mode,
        align_corners=False if mode == "bilinear" else None,
    )
    if images.dtype == torch.uint8:
        resized_images = torch.round(resized_images).clamp(0, 255).to(torch.uint8)
    elif images.dtype == torch.float32:
        resized_images = resized_images.clamp(0.0, 1.0)
    else:
        raise ValueError(f"Unsupported image dtype: {images.dtype}")
    pad_h0, remainder_h = divmod(height - resized_height, 2)
    pad_h1 = pad_h0 + remainder_h
    pad_w0, remainder_w = divmod(width - resized_width, 2)
    pad_w1 = pad_w0 + remainder_w
    constant_value = 0 if images.dtype == torch.uint8 else 0.0
    padded_images = F.pad(resized_images, (pad_w0, pad_w1, pad_h0, pad_h1), mode="constant", value=constant_value)
    if channels_last:
        padded_images = padded_images.permute(0, 2, 3, 1)
    return padded_images


def tokenize_prompts(
    prompts: list, tokenizer, max_length: int = 200,
    padding: str = "max_length", padding_side: str = "right"
) -> dict:
    """Tokenize a list of prompts with specified padding."""
    original_padding_side = tokenizer.padding_side
    tokenizer.padding_side = padding_side
    try:
        encoded = tokenizer(prompts, return_tensors="pt", max_length=max_length, padding=padding, truncation=True)
    finally:
        tokenizer.padding_side = original_padding_side
    return {"input_ids": encoded["input_ids"], "attention_mask": encoded["attention_mask"]}


def _q99_unnormalize_actions(actions: torch.Tensor, action_stats: Optional[dict]) -> torch.Tensor:
    """Unnormalize PI05 quantile-normalized actions with dataset action stats."""
    if action_stats is None:
        return actions
    if "q01" not in action_stats or "q99" not in action_stats:
        raise ValueError(
            "PI05 is configured for quantile action unnormalization and requires "
            "dataset_stats['action'].q01 and q99"
        )
    q01 = torch.as_tensor(action_stats["q01"], dtype=actions.dtype, device=actions.device)
    q99 = torch.as_tensor(action_stats["q99"], dtype=actions.dtype, device=actions.device)
    dim = min(actions.shape[-1], q01.shape[-1])
    out = actions.clone()
    out[..., :dim] = (out[..., :dim] + 1.0) / 2.0 * (q99[:dim] - q01[:dim]) + q01[:dim]
    return out


def build_tokenizer(tokenizer_name: str, local_files_only: bool = True):
    """Build a HuggingFace tokenizer from the given name or path."""
    if not tokenizer_name:
        raise ValueError("tokenizer_name is empty. Set it or set TOKENIZER_PATH env variable.")
    return AutoTokenizer.from_pretrained(tokenizer_name, local_files_only=local_files_only)


# ═══════════════════════════════════════════════════════════════
# Gemma Config Helpers
# ═══════════════════════════════════════════════════════════════

class _GemmaConfig:
    """Internal Gemma architecture configuration."""

    def __init__(self, width, depth, mlp_dim, num_heads, num_kv_heads, head_dim):
        """Initialize Gemma config with architecture parameters."""
        self.width = width
        self.depth = depth
        self.mlp_dim = mlp_dim
        self.num_heads = num_heads
        self.num_kv_heads = num_kv_heads
        self.head_dim = head_dim


def _get_gemma_config(variant: str) -> _GemmaConfig:
    """Get predefined Gemma config by variant name."""
    if variant == "gemma_300m":
        return _GemmaConfig(width=1024, depth=18, mlp_dim=4096, num_heads=8, num_kv_heads=1, head_dim=256)
    elif variant == "gemma_2b":
        return _GemmaConfig(width=2048, depth=18, mlp_dim=16_384, num_heads=8, num_kv_heads=1, head_dim=256)
    else:
        raise ValueError(f"Unknown variant: {variant}")


# ═══════════════════════════════════════════════════════════════
# compute_layer_complete (gradient checkpointing helper)
# ═══════════════════════════════════════════════════════════════

def _compute_layer_complete(
    layer_idx, inputs_embeds, attention_mask, position_ids,
    adarms_cond, paligemma, gemma_expert
):
    """Compute one transformer layer with joint attention across models."""
    models = [paligemma.model.language_model, gemma_expert.model]
    query_states, key_states, value_states, gates = [], [], [], []
    for i, hidden_states in enumerate(inputs_embeds):
        layer = models[i].layers[layer_idx]
        hidden_states, gate = layernorm_forward(layer.input_layernorm, hidden_states, adarms_cond[i])
        gates.append(gate)
        input_shape = hidden_states.shape[:-1]
        hidden_shape = (*input_shape, -1, layer.self_attn.head_dim)
        query_states.append(layer.self_attn.q_proj(hidden_states).view(hidden_shape).transpose(1, 2))
        key_states.append(layer.self_attn.k_proj(hidden_states).view(hidden_shape).transpose(1, 2))
        value_states.append(layer.self_attn.v_proj(hidden_states).view(hidden_shape).transpose(1, 2))

    query_states = torch.cat(query_states, dim=2)
    key_states = torch.cat(key_states, dim=2)
    value_states = torch.cat(value_states, dim=2)
    dummy_tensor = torch.zeros(query_states.shape[0], query_states.shape[2], query_states.shape[-1],
                               device=query_states.device, dtype=query_states.dtype)
    cos, sin = paligemma.model.language_model.rotary_emb(dummy_tensor, position_ids)
    query_states, key_states = modeling_gemma.apply_rotary_pos_emb(query_states, key_states, cos, sin, unsqueeze_dim=1)
    scaling = paligemma.model.language_model.layers[layer_idx].self_attn.scaling
    att_output, _ = modeling_gemma.eager_attention_forward(
        paligemma.model.language_model.layers[layer_idx].self_attn,
        query_states, key_states, value_states, attention_mask, scaling,
    )
    head_dim = paligemma.model.language_model.layers[layer_idx].self_attn.head_dim
    att_output = att_output.reshape(query_states.shape[0], -1, 1 * 8 * head_dim)

    outputs_embeds = []
    start_pos = 0
    for i, hidden_states in enumerate(inputs_embeds):
        layer = models[i].layers[layer_idx]
        end_pos = start_pos + hidden_states.shape[1]
        if att_output.dtype != layer.self_attn.o_proj.weight.dtype:
            att_output = att_output.to(layer.self_attn.o_proj.weight.dtype)
        out_emb = layer.self_attn.o_proj(att_output[:, start_pos:end_pos])
        out_emb = _gated_residual(hidden_states, out_emb, gates[i])
        after_first_residual = out_emb.clone()
        out_emb, gate = layernorm_forward(layer.post_attention_layernorm, out_emb, adarms_cond[i])
        if layer.mlp.up_proj.weight.dtype == torch.bfloat16:
            out_emb = out_emb.to(dtype=torch.bfloat16)
        out_emb = layer.mlp(out_emb)
        out_emb = _gated_residual(after_first_residual, out_emb, gate)
        outputs_embeds.append(out_emb)
        start_pos = end_pos
    return outputs_embeds


# ═══════════════════════════════════════════════════════════════
# PaliGemmaWithExpertModel
# ═══════════════════════════════════════════════════════════════

class PaliGemmaWithExpertModel(nn.Module):
    """PaliGemma + Gemma action expert with joint attention."""

    _FP32_PARAM_SELECTORS = [
        "vision_tower", "multi_modal_projector",
        "input_layernorm", "post_attention_layernorm", "model.norm",
    ]

    def __init__(self, vlm_config, action_expert_config, use_adarms=None,
                 precision: Literal["bfloat16", "float32"] = "bfloat16",
                 image_size: int = DEFAULT_IMAGE_SIZE,
                 freeze_vision_encoder: bool = False,
                 train_expert_only: bool = False):
        """Initialize PaliGemma with joint attention expert model."""
        if use_adarms is None:
            use_adarms = [False, False]
        super().__init__()
        self.freeze_vision_encoder = freeze_vision_encoder
        self.train_expert_only = train_expert_only

        vlm_config_hf = CONFIG_MAPPING["paligemma"]()
        vlm_config_hf._vocab_size = 257152
        vlm_config_hf.image_token_index = 257152
        vlm_config_hf.text_config.hidden_size = vlm_config.width
        vlm_config_hf.text_config.intermediate_size = vlm_config.mlp_dim
        vlm_config_hf.text_config.num_attention_heads = vlm_config.num_heads
        vlm_config_hf.text_config.head_dim = vlm_config.head_dim
        vlm_config_hf.text_config.num_hidden_layers = vlm_config.depth
        vlm_config_hf.text_config.num_key_value_heads = vlm_config.num_kv_heads
        vlm_config_hf.text_config.hidden_activation = "gelu_pytorch_tanh"
        vlm_config_hf.text_config.dtype = "float32"
        vlm_config_hf.text_config.vocab_size = 257152
        vlm_config_hf.text_config.use_adarms = use_adarms[0]
        vlm_config_hf.text_config.adarms_cond_dim = vlm_config.width if use_adarms[0] else None
        vlm_config_hf.vision_config.image_size = image_size
        vlm_config_hf.vision_config.intermediate_size = 4304
        vlm_config_hf.vision_config.projection_dim = 2048
        vlm_config_hf.vision_config.projector_hidden_act = "gelu_fast"
        vlm_config_hf.vision_config.dtype = "float32"

        action_expert_config_hf = CONFIG_MAPPING["gemma"](
            head_dim=action_expert_config.head_dim,
            hidden_size=action_expert_config.width,
            intermediate_size=action_expert_config.mlp_dim,
            num_attention_heads=action_expert_config.num_heads,
            num_hidden_layers=action_expert_config.depth,
            num_key_value_heads=action_expert_config.num_kv_heads,
            vocab_size=257152,
            hidden_activation="gelu_pytorch_tanh",
            dtype="float32",
            use_adarms=use_adarms[1],
            adarms_cond_dim=action_expert_config.width if use_adarms[1] else None,
        )

        self.paligemma = PaliGemmaForConditionalGenerationWithPiGemma(config=vlm_config_hf)
        self.gemma_expert = PiGemmaForCausalLM(config=action_expert_config_hf)
        self.gemma_expert.model.embed_tokens = None
        self._text_hidden_size = vlm_config_hf.text_config.hidden_size
        self._num_hidden_layers = vlm_config_hf.text_config.num_hidden_layers

        # Attribute hooks — external code (PI05Pytorch._apply_compile_scope) may
        # replace these with torch.compile'd wrappers to shrink compile units and
        # improve DDP backward/allreduce overlap.  See forward() below.
        self._layer_fn = _compute_layer_complete
        self._final_norms_fn = self._final_norms_impl

        if not self._is_on_meta_device():
            self.to_bfloat16_for_selected_params(precision)
        self._set_requires_grad()

    def _is_on_meta_device(self) -> bool:
        """Check if model parameters are on meta device."""
        try:
            return next(self.parameters()).device.type == "meta"
        except StopIteration:
            return False

    def to_bfloat16_for_selected_params(
        self, precision: Literal["bfloat16", "float32"] = "bfloat16"
    ):
        """Convert model to bfloat16 while keeping selected params in fp32."""
        if precision == "bfloat16":
            fp32_saved = {
                name: param.data.clone()
                for name, param in self.named_parameters()
                if any(sel in name for sel in self._FP32_PARAM_SELECTORS)
            }
            self.to(dtype=torch.bfloat16)
            for name, param in self.named_parameters():
                if name in fp32_saved:
                    param.data = fp32_saved[name]
        elif precision == "float32":
            self.to(dtype=torch.float32)
        else:
            raise ValueError(f"Invalid precision: {precision}")

    def _set_requires_grad(self):
        """Freeze parameters based on configuration."""
        if self.freeze_vision_encoder:
            self.paligemma.model.vision_tower.eval()
            for p in self.paligemma.model.vision_tower.parameters():
                p.requires_grad = False
        if self.train_expert_only:
            self.paligemma.eval()
            for p in self.paligemma.parameters():
                p.requires_grad = False

    def train(self, mode: bool = True):
        """Set training mode, keeping frozen modules in eval."""
        super().train(mode)
        if self.freeze_vision_encoder:
            self.paligemma.model.vision_tower.eval()
        if self.train_expert_only:
            self.paligemma.eval()

    def embed_image(self, image: torch.Tensor):
        """Embed image through PaliGemma vision tower and projector."""
        for module in self.paligemma.modules():
            if hasattr(module, "position_ids"):
                pid = module.position_ids
                if pid.device.type == "meta":
                    continue
                n = pid.numel()
                expected = torch.arange(n, device=pid.device, dtype=torch.int64)
                if pid.dtype != torch.int64 or not torch.equal(pid.view(-1), expected):
                    module.register_buffer("position_ids", expected.view(1, -1), persistent=False)
        out_dtype = image.dtype
        if image.dtype != torch.float32:
            image = image.to(torch.float32)
        image_outputs = self.paligemma.model.get_image_features(image)
        img_tensor = image_outputs if isinstance(image_outputs, torch.Tensor) else image_outputs.pooler_output
        features = img_tensor * self._text_hidden_size ** 0.5
        return features.to(out_dtype) if features.dtype != out_dtype else features

    def embed_language_tokens(self, tokens: torch.Tensor):
        """Embed language tokens through PaliGemma language model."""
        return self.paligemma.model.language_model.embed_tokens(tokens)

    def _final_norms_impl(self, inputs_embeds, adarms_cond):
        """Apply final layernorm to each stream's hidden states.

        Promoted from an inline closure so torch.compile can wrap it (see
        PI05Pytorch._apply_compile_scope for the "multi_group" scope).
        """
        models = [self.paligemma.model.language_model, self.gemma_expert.model]
        return [
            layernorm_forward(models[i].norm, h, adarms_cond[i])[0]
            for i, h in enumerate(inputs_embeds)
        ]

    def forward(self, attention_mask=None, position_ids=None,
                past_key_values=None, inputs_embeds=None,
                use_cache=None, adarms_cond=None):
        """Forward pass with joint attention between PaliGemma and expert."""
        if adarms_cond is None:
            adarms_cond = [None, None]
        if inputs_embeds[1] is None:
            prefix_output = self.paligemma.model.language_model.forward(
                inputs_embeds=inputs_embeds[0], attention_mask=attention_mask,
                position_ids=position_ids, past_key_values=past_key_values,
                use_cache=use_cache, adarms_cond=adarms_cond[0],
            )
            return [prefix_output.last_hidden_state, None], prefix_output.past_key_values
        elif inputs_embeds[0] is None:
            suffix_output = self.gemma_expert.model.forward(
                inputs_embeds=inputs_embeds[1], attention_mask=attention_mask,
                position_ids=position_ids, past_key_values=past_key_values,
                use_cache=use_cache, adarms_cond=adarms_cond[1],
            )
            return [None, suffix_output.last_hidden_state], None
        else:
            use_gc = (
                (hasattr(self.gemma_expert.model, "gradient_checkpointing") and
                 self.gemma_expert.model.gradient_checkpointing and self.training)
                or (hasattr(self, "gradient_checkpointing") and self.gradient_checkpointing and self.training)
            )
            for layer_idx in range(self._num_hidden_layers):
                if use_gc:
                    # compile-inside-checkpoint: self._layer_fn may be a
                    # torch.compile'd wrapper (see _apply_compile_scope).
                    inputs_embeds = torch.utils.checkpoint.checkpoint(
                        self._layer_fn, layer_idx, inputs_embeds, attention_mask,
                        position_ids, adarms_cond,
                        use_reentrant=False, preserve_rng_state=False,
                        paligemma=self.paligemma, gemma_expert=self.gemma_expert,
                    )
                else:
                    inputs_embeds = self._layer_fn(
                        layer_idx, inputs_embeds, attention_mask, position_ids, adarms_cond,
                        paligemma=self.paligemma, gemma_expert=self.gemma_expert,
                    )

            outputs_embeds = (
                torch.utils.checkpoint.checkpoint(
                    self._final_norms_fn, inputs_embeds, adarms_cond,
                    use_reentrant=False, preserve_rng_state=False,
                ) if use_gc else self._final_norms_fn(inputs_embeds, adarms_cond)
            )
            return outputs_embeds, None


# ═══════════════════════════════════════════════════════════════
# PI05Pytorch — core training/inference model
# ═══════════════════════════════════════════════════════════════

class PI05Pytorch(nn.Module):
    """Core PI05 PyTorch model, aligned with LeRobot's PI05Pytorch layer."""

    _FP32_PARAM_SELECTORS = [
        "action_in_proj", "action_out_proj", "time_mlp_in", "time_mlp_out",
    ]

    def __init__(self, config: _PI05InternalConfig):
        """Initialize PI05 core model with backbone and action head."""
        super().__init__()
        self.config = config
        paligemma_config = _get_gemma_config(config.paligemma_variant)
        action_expert_config = _get_gemma_config(config.action_expert_variant)

        if config.image_resolution[0] != config.image_resolution[1]:
            raise ValueError(f"PaliGemma expects square image resolution: {config.image_resolution}")

        # ── BACKBONE ──────────────────────────────────────────────────────────
        self.paligemma_with_expert = PaliGemmaWithExpertModel(
            paligemma_config, action_expert_config,
            use_adarms=[False, True],
            precision=config.dtype,
            image_size=config.image_resolution[0],
            freeze_vision_encoder=config.freeze_vision_encoder,
            train_expert_only=config.train_expert_only,
        )
        # ── ACTION HEAD ───────────────────────────────────────────────────────
        self.action_in_proj = nn.Linear(config.max_action_dim, action_expert_config.width)
        self.action_out_proj = nn.Linear(action_expert_config.width, config.max_action_dim)
        self.time_mlp_in = nn.Linear(action_expert_config.width, action_expert_config.width)
        self.time_mlp_out = nn.Linear(action_expert_config.width, action_expert_config.width)
        # ─────────────────────────────────────────────────────────────────────

        self.gradient_checkpointing_enabled = False
        torch.set_float32_matmul_precision("high")

        self._apply_compile_scope(config)

    def _apply_compile_scope(self, config: "_PI05InternalConfig"):
        """Dispatch torch.compile onto backbone sub-functions per `compile_scope`.

        Scopes (only reached when compile_model=True):
          - "backbone"     : legacy single compile over PaliGemmaWithExpertModel.forward.
          - "per_layer"    : compile only _compute_layer_complete; forward stays eager.
          - "multi_group"  : per_layer + final_norms + action-head bundles.

        Rationale: shrinking the compile unit lets Inductor emit one backward
        Function per layer, so grads become ready and can be all-reduced while
        subsequent layers are still computing backward -> better DDP overlap.
        """
        if not config.compile_model:
            return

        scope = config.compile_scope
        mode = config.compile_mode
        dynamic = config.compile_dynamic

        if scope == "backbone":
            # Preserve historical behavior exactly (fullgraph=True by default).
            self.paligemma_with_expert.forward = torch.compile(
                self.paligemma_with_expert.forward,
                mode=mode,
                fullgraph=config.compile_fullgraph,
                dynamic=dynamic,
            )
            logging.info(
                "PI05 torch.compile: scope=backbone mode=%s fullgraph=%s dynamic=%s",
                mode, config.compile_fullgraph, dynamic,
            )
            return

        # per_layer / multi_group both compile _layer_fn.  fullgraph=True is
        # forced on these small sub-graphs (their Python control flow is static;
        # see the plan for justification).
        self.paligemma_with_expert._layer_fn = torch.compile(
            _compute_layer_complete,
            mode=mode,
            fullgraph=True,
            dynamic=dynamic,
        )

        if scope == "per_layer":
            logging.info(
                "PI05 torch.compile: scope=per_layer mode=%s dynamic=%s",
                mode, dynamic,
            )
            return

        if scope == "multi_group":
            self.paligemma_with_expert._final_norms_fn = torch.compile(
                self.paligemma_with_expert._final_norms_impl,
                mode=mode,
                fullgraph=True,
                dynamic=dynamic,
            )

            # Action-head bundles: closed-over Linear modules keep parameters
            # visible to DDP; only the compiled callable wraps the tensor ops.
            action_in_proj = self.action_in_proj
            time_mlp_in = self.time_mlp_in
            time_mlp_out = self.time_mlp_out
            action_out_proj = self.action_out_proj

            def _action_in_bundle(noisy_actions, time_emb):
                action_emb = action_in_proj(noisy_actions)
                adarms_cond = F.silu(time_mlp_out(F.silu(time_mlp_in(time_emb))))
                return action_emb, adarms_cond

            def _action_out_bundle(suffix_out):
                out = suffix_out.to(dtype=action_out_proj.weight.dtype)
                return action_out_proj(out)

            self._action_in_bundle_fn = torch.compile(
                _action_in_bundle, mode=mode, fullgraph=True, dynamic=dynamic,
            )
            self._action_out_bundle_fn = torch.compile(
                _action_out_bundle, mode=mode, fullgraph=True, dynamic=dynamic,
            )
            logging.info(
                "PI05 torch.compile: scope=multi_group mode=%s dynamic=%s",
                mode, dynamic,
            )
            return

        # Unreachable — validated in PI05Policy.__init__.
        raise ValueError(f"Unknown compile_scope: {scope!r}")

    def gradient_checkpointing_enable(self):
        """Enable gradient checkpointing for all sub-models."""
        self.gradient_checkpointing_enabled = True
        self.paligemma_with_expert.paligemma.model.language_model.gradient_checkpointing = True
        self.paligemma_with_expert.paligemma.model.vision_tower.gradient_checkpointing = True
        self.paligemma_with_expert.gemma_expert.model.gradient_checkpointing = True

    def gradient_checkpointing_disable(self):
        """Disable gradient checkpointing for all sub-models."""
        self.gradient_checkpointing_enabled = False
        self.paligemma_with_expert.paligemma.model.language_model.gradient_checkpointing = False
        self.paligemma_with_expert.paligemma.model.vision_tower.gradient_checkpointing = False
        self.paligemma_with_expert.gemma_expert.model.gradient_checkpointing = False

    def _apply_checkpoint(self, func, *args, **kwargs):
        """Apply gradient checkpointing if enabled and in training mode."""
        if self.gradient_checkpointing_enabled and self.training:
            return torch.utils.checkpoint.checkpoint(
                func, *args, use_reentrant=False, preserve_rng_state=False, **kwargs)
        return func(*args, **kwargs)

    def _prepare_attention_masks_4d(self, att_2d_masks):
        """Convert 2D boolean masks to 4D float attention masks."""
        return torch.where(att_2d_masks[:, None, :, :], 0.0, OPENPI_ATTENTION_MASK_VALUE)

    def sample_noise(self, shape, device):
        """Sample Gaussian noise for diffusion."""
        return torch.normal(
            mean=0.0, std=1.0, size=shape, dtype=torch.float32,
            device=(torch.device("cpu") if self.config.random_fallback_cpu else device),
        ).to(device)

    def sample_time(self, bsize, device):
        """Sample diffusion timesteps from a scaled Beta distribution."""
        time_beta = sample_beta(
            self.config.time_sampling_beta_alpha,
            self.config.time_sampling_beta_beta, bsize, device
        )
        return (
            time_beta * self.config.time_sampling_scale
            + self.config.time_sampling_offset
        ).to(dtype=torch.float32, device=device)

    def embed_prefix(self, images, img_masks, tokens, masks):
        """Embed prefix (images + language tokens) for joint attention."""
        embs, pad_masks, att_masks = [], [], []
        for img, img_mask in zip(images, img_masks):
            img_emb = self._apply_checkpoint(self.paligemma_with_expert.embed_image, img)
            bsize, num_img_embs = img_emb.shape[:2]
            embs.append(img_emb)
            pad_masks.append(img_mask[:, None].expand(bsize, num_img_embs))
            att_masks += [0] * num_img_embs

        lang_emb = self._apply_checkpoint(self.paligemma_with_expert.embed_language_tokens, tokens)
        lang_emb = lang_emb * math.sqrt(lang_emb.shape[-1])

        embs.append(lang_emb)
        pad_masks.append(masks)
        att_masks += [0] * lang_emb.shape[1]

        embs = torch.cat([e.to(torch.float32) for e in embs], dim=1)
        pad_masks = torch.cat(pad_masks, dim=1)
        att_masks = torch.tensor(att_masks, dtype=torch.bool, device=pad_masks.device)
        att_masks = att_masks[None, :].expand(pad_masks.shape[0], len(att_masks))
        return embs, pad_masks, att_masks

    def embed_suffix(self, noisy_actions, timestep):
        """Embed suffix (noisy actions + time) for action expert."""
        noisy_actions = noisy_actions.to(dtype=self.action_in_proj.weight.dtype)
        time_emb = create_sinusoidal_pos_embedding(
            timestep, self.action_in_proj.out_features,
            min_period=self.config.min_period, max_period=self.config.max_period,
            device=torch.device("cpu") if self.config.random_fallback_cpu else timestep.device,
        ).to(timestep.device).type(dtype=timestep.dtype)

        if hasattr(self, "_action_in_bundle_fn"):
            # multi_group scope: compiled fused (action_in_proj + time_mlp).
            # We deliberately skip _apply_checkpoint here — the bundle is small
            # and always cheap to keep in the autograd tape.
            action_emb, adarms_cond = self._action_in_bundle_fn(noisy_actions, time_emb)
        else:
            action_emb = self._apply_checkpoint(self.action_in_proj, noisy_actions)

            def _time_mlp(t):
                return F.silu(self.time_mlp_out(F.silu(self.time_mlp_in(t))))
            adarms_cond = self._apply_checkpoint(_time_mlp, time_emb)

        bsize, action_time_dim = action_emb.shape[:2]
        pad_masks = torch.ones(bsize, action_time_dim, dtype=torch.bool, device=timestep.device)
        att_masks_list = [1] + ([0] * (self.config.chunk_size - 1))
        att_masks = torch.tensor(att_masks_list, dtype=action_emb.dtype, device=action_emb.device)
        att_masks = att_masks[None, :].expand(bsize, len(att_masks_list))
        return action_emb, pad_masks, att_masks, adarms_cond

    def forward(self, images, img_masks, tokens, masks, actions, noise=None, time=None) -> Tensor:
        """Training forward: compute flow-matching MSE loss."""
        if noise is None:
            noise = self.sample_noise(actions.shape, actions.device)
        if time is None:
            time = self.sample_time(actions.shape[0], actions.device)

        time_expanded = time[:, None, None]
        x_t = time_expanded * noise + (1 - time_expanded) * actions
        u_t = noise - actions

        prefix_embs, prefix_pad_masks, prefix_att_masks = self.embed_prefix(images, img_masks, tokens, masks)
        suffix_embs, suffix_pad_masks, suffix_att_masks, adarms_cond = self.embed_suffix(x_t, time)

        if (self.paligemma_with_expert.paligemma.model.language_model.layers[0]
                .self_attn.q_proj.weight.dtype == torch.bfloat16):
            suffix_embs = suffix_embs.to(dtype=torch.bfloat16)
            prefix_embs = prefix_embs.to(dtype=torch.bfloat16)

        pad_masks = torch.cat([prefix_pad_masks, suffix_pad_masks], dim=1)
        att_masks = torch.cat([prefix_att_masks, suffix_att_masks], dim=1)
        att_2d_masks = make_att_2d_masks(pad_masks, att_masks)
        position_ids = torch.cumsum(pad_masks, dim=1) - 1
        att_2d_masks_4d = self._prepare_attention_masks_4d(att_2d_masks)

        def _fwd(prefix_embs, suffix_embs, att_2d_masks_4d, position_ids, adarms_cond):
            (_, suffix_out), _ = self.paligemma_with_expert.forward(
                attention_mask=att_2d_masks_4d, position_ids=position_ids,
                inputs_embeds=[prefix_embs, suffix_embs], use_cache=False,
                adarms_cond=[None, adarms_cond],
            )
            return suffix_out

        suffix_out = self._apply_checkpoint(_fwd, prefix_embs, suffix_embs, att_2d_masks_4d, position_ids, adarms_cond)
        suffix_out = suffix_out[:, -self.config.chunk_size:]
        if hasattr(self, "_action_out_bundle_fn"):
            v_t = self._action_out_bundle_fn(suffix_out)
        else:
            suffix_out = suffix_out.to(dtype=self.action_out_proj.weight.dtype)
            v_t = self._apply_checkpoint(self.action_out_proj, suffix_out)
        return F.mse_loss(u_t, v_t, reduction="none")

    @torch.no_grad()
    def sample_actions(self, images, img_masks, tokens, masks, noise=None, num_steps=None) -> Tensor:
        """Sample actions via Euler ODE integration."""
        if num_steps is None:
            num_steps = self.config.num_inference_steps
        bsize = tokens.shape[0]
        device = tokens.device

        if noise is None:
            noise = self.sample_noise((bsize, self.config.chunk_size, self.config.max_action_dim), device)

        prefix_embs, prefix_pad_masks, prefix_att_masks = self.embed_prefix(images, img_masks, tokens, masks)
        prefix_att_2d_masks = make_att_2d_masks(prefix_pad_masks, prefix_att_masks)
        prefix_position_ids = torch.cumsum(prefix_pad_masks, dim=1) - 1
        prefix_att_2d_masks_4d = self._prepare_attention_masks_4d(prefix_att_2d_masks)

        self.paligemma_with_expert.paligemma.model.language_model.config._attn_implementation = "eager"
        _, past_key_values = self.paligemma_with_expert.forward(
            attention_mask=prefix_att_2d_masks_4d, position_ids=prefix_position_ids,
            inputs_embeds=[prefix_embs, None], use_cache=True,
        )

        dt = -1.0 / num_steps
        x_t = noise
        for step in range(num_steps):
            time = torch.tensor(1.0 + step * dt, dtype=torch.float32, device=device).expand(bsize)
            v_t = self._denoise_step(prefix_pad_masks, past_key_values, x_t, time)
            x_t = x_t + dt * v_t
        return x_t

    def _denoise_step(self, prefix_pad_masks, past_key_values, x_t, timestep):
        """Single denoising step using cached prefix KV."""
        suffix_embs, suffix_pad_masks, suffix_att_masks, adarms_cond = self.embed_suffix(x_t, timestep)
        suffix_len = suffix_pad_masks.shape[1]
        batch_size = prefix_pad_masks.shape[0]
        prefix_len = prefix_pad_masks.shape[1]

        prefix_pad_2d_masks = prefix_pad_masks[:, None, :].expand(batch_size, suffix_len, prefix_len)
        suffix_att_2d_masks = make_att_2d_masks(suffix_pad_masks, suffix_att_masks)
        full_att_2d_masks = torch.cat([prefix_pad_2d_masks, suffix_att_2d_masks], dim=2)
        prefix_offsets = torch.sum(prefix_pad_masks, dim=-1)[:, None]
        position_ids = prefix_offsets + torch.cumsum(suffix_pad_masks, dim=1) - 1
        full_att_2d_masks_4d = self._prepare_attention_masks_4d(full_att_2d_masks)

        self.paligemma_with_expert.gemma_expert.model.config._attn_implementation = "eager"
        outputs_embeds, _ = self.paligemma_with_expert.forward(
            attention_mask=full_att_2d_masks_4d, position_ids=position_ids,
            past_key_values=copy.deepcopy(past_key_values),
            inputs_embeds=[None, suffix_embs], use_cache=False,
            adarms_cond=[None, adarms_cond],
        )
        suffix_out = outputs_embeds[1][:, -self.config.chunk_size:].to(dtype=torch.float32)
        return self.action_out_proj(suffix_out)

    def load_pretrained(self, pretrained_path: str, strict: bool = True, device=None):
        """Load pretrained weights from safetensors checkpoint."""
        from pathlib import Path
        from safetensors.torch import load_file

        path = Path(pretrained_path)
        safetensors_file = path / "model.safetensors" if path.is_dir() else path
        load_kwargs = {"device": str(device)} if device is not None else {}
        original_state_dict = load_file(str(safetensors_file), **load_kwargs)

        _PREFIXES = ("model.architecture.pi05_model.", "architecture.pi05_model.")
        fixed = {}
        for key, value in original_state_dict.items():
            new_key = key
            for prefix in _PREFIXES:
                if new_key.startswith(prefix):
                    new_key = new_key[len(prefix):]
                    break
            new_key = new_key.replace(
                "action_time_mlp_in.", "time_mlp_in."
            ).replace("action_time_mlp_out.", "time_mlp_out.")
            if new_key.startswith("state_proj."):
                continue
            if new_key in ("model.paligemma_with_expert.paligemma.lm_head.weight",
                           "paligemma_with_expert.paligemma.lm_head.weight"):
                fixed["paligemma_with_expert.paligemma.model.language_model.embed_tokens.weight"] = value.clone()
            fixed[new_key] = value

        missing, unexpected = self.load_state_dict(fixed, strict=strict)
        assert not missing, f"Missing keys: {missing[:5]}"
        assert not unexpected, f"Unexpected keys: {unexpected[:5]}"
        return self


# ═══════════════════════════════════════════════════════════════
# PI05Policy — Trainer entry point
# ═══════════════════════════════════════════════════════════════

@register_model("pi05")
class PI05Policy(nn.Module):
    """LoongForge policy wrapper around PI05Pytorch, aligned with LeRobot's PI05Policy layer."""

    def __init__(self, config: Pi05ModelConfig):
        """Initialize PI05Policy with given configuration."""
        super().__init__()
        self.config = config

        _VALID_COMPILE_SCOPES = ("backbone", "multi_group", "per_layer")
        if config.compile_scope not in _VALID_COMPILE_SCOPES:
            raise ValueError(
                f"Invalid compile_scope={config.compile_scope!r}; "
                f"expected one of {_VALID_COMPILE_SCOPES}"
            )

        pi05_cfg = _PI05InternalConfig(
            chunk_size=config.action_horizon,
            n_action_steps=config.action_horizon,
            max_action_dim=config.max_action_dim,
            max_state_dim=config.max_state_dim,
            freeze_vision_encoder=config.freeze_vision_encoder,
            train_expert_only=config.train_expert_only,
            gradient_checkpointing=config.gradient_checkpointing,
            compile_model=config.compile_model,
            compile_mode=config.compile_mode,
            compile_scope=config.compile_scope,
            compile_fullgraph=config.compile_fullgraph,
            compile_dynamic=config.compile_dynamic,
        )
        self.model = PI05Pytorch(pi05_cfg)
        if config.gradient_checkpointing:
            self.model.gradient_checkpointing_enable()

        # Backward-compatible alias for existing LoongForge checkpoint/load code.
        self.pi05 = self.model
        self._image_size = DEFAULT_IMAGE_SIZE
        self._tokenizer = None
        self._tokenizer_path = ""

    @property
    def tokenizer(self):
        """Lazily build and cache the tokenizer configured for this model."""
        if self._tokenizer is None:
            path = self._tokenizer_path or os.environ.get("TOKENIZER_PATH", "")
            self._tokenizer = build_tokenizer(path)
        return self._tokenizer

    def forward(self, batch) -> Dict[str, torch.Tensor]:
        """Training forward pass, returns {"action_loss": scalar}."""
        images_list = batch.images_list
        img_masks = batch.img_masks
        tokens = batch.input_ids
        masks = batch.attention_mask
        actions = batch.actions

        target_h = target_w = self._image_size
        resized = []
        for img in images_list:
            if img.shape[-2] != target_h or img.shape[-1] != target_w:
                img_hwc = img.permute(0, 2, 3, 1)
                img_hwc = resize_with_pad_torch(img_hwc, target_h, target_w)
                img = img_hwc * 2.0 - 1.0
                img = img.permute(0, 3, 1, 2)
            else:
                img = img * 2.0 - 1.0
            resized.append(img)

        loss_map = self.model(resized, img_masks, tokens, masks, actions)
        loss_value = loss_map[:, :, :self.config.action_dim].mean()
        return loss_value, {"action_loss": loss_value.detach().item()}

    @torch.no_grad()
    def predict_action_chunk(self, batch, **kwargs) -> torch.Tensor:
        """Predict an action chunk from a preprocessed PI05 batch."""
        self.eval()
        images_list = batch.images_list
        img_masks = batch.img_masks
        tokens = batch.input_ids
        masks = batch.attention_mask

        target_h = target_w = self._image_size
        resized = []
        for img in images_list:
            if img.shape[-2] != target_h or img.shape[-1] != target_w:
                img_hwc = img.permute(0, 2, 3, 1)
                img_hwc = resize_with_pad_torch(img_hwc, target_h, target_w)
                img = img_hwc * 2.0 - 1.0
                img = img.permute(0, 3, 1, 2)
            else:
                img = img * 2.0 - 1.0
            resized.append(img)

        actions = self.model.sample_actions(resized, img_masks, tokens, masks, **kwargs)
        return actions[:, :, :self.config.action_dim]

    @torch.no_grad()
    def select_action(self, batch, **kwargs) -> torch.Tensor:
        """Select the first action from a predicted action chunk."""
        return self.predict_action_chunk(batch, **kwargs)[:, 0]

    @torch.no_grad()
    def predict_action(self, images, instructions, state=None, dataset_stats=None) -> np.ndarray:
        """Inference: Euler ODE denoising, returns ndarray (B, action_horizon, max_action_dim)."""
        from loongforge.embodied.data.datasets.pi05.transforms.pi05_transform import StateDiscretizationTransform
        from loongforge.embodied.data.datasets.transforms.utils.builders import convert_stats
        from loongforge.embodied.data.datasets.transforms.utils.image_transform import ImageTransform

        device = next(self.parameters()).device
        if not isinstance(images[0], list):
            images = [[img] for img in images]
        B = len(images)
        image_transform = ImageTransform(apply_to=[], image_size=self._image_size)
        # Use the number of views provided by the eval client so LIBERO (2) and
        # RoboTwin (3) both work without hard-coding a single camera count.
        num_images = max(len(images[0]), 1)

        images_list, img_masks = [], []
        for v in range(num_images):
            view = []
            mask_vals = []
            for im_list in images:
                if v < len(im_list):
                    view.append(im_list[v])
                    mask_vals.append(True)
                else:
                    view.append(torch.zeros(3, self._image_size, self._image_size))
                    mask_vals.append(False)
            images_list.append(image_transform.process_batch(view).to(device))
            img_masks.append(torch.tensor(mask_vals, dtype=torch.bool, device=device))

        if state is not None:
            if not isinstance(state, torch.Tensor):
                state = torch.as_tensor(state, dtype=torch.float32)
            if state.dim() == 1:
                state = state.unsqueeze(0)
            state_stats = convert_stats(dataset_stats.get("observation.state")) if dataset_stats else None
            state_transform = StateDiscretizationTransform(
                apply_to=["prompt"], state_key="observation.state", task_key="task",
                num_bins=256, normalization_mode="q99", statistics=state_stats,
            )
            prompts = [state_transform.apply({"observation.state": state[i], "task": instructions[i]})["prompt"]
                       for i in range(B)]
        else:
            prompts = [f"Task: {t.strip()};\nAction: " for t in instructions]

        tok = tokenize_prompts(prompts, self.tokenizer, max_length=200)
        tokens = tok["input_ids"].to(device)
        masks = tok["attention_mask"].bool().to(device)

        actions = self.model.sample_actions(images_list, img_masks, tokens, masks)
        action_stats = convert_stats(dataset_stats.get("action")) if dataset_stats else None
        actions = _q99_unnormalize_actions(actions, action_stats)
        return actions.cpu().numpy()

    @classmethod
    def from_pretrained(cls, model_cfg) -> "PI05Policy":
        """Create PI05Policy from a typed Pi05 ModelConfig."""
        if not isinstance(model_cfg, Pi05ModelConfig):
            raise TypeError(
                "PI05Policy.from_pretrained expects a typed Pi05 ModelConfig instance; "
                f"got {type(model_cfg).__name__}. build_model now passes ModelConfig directly."
            )
        return cls(model_cfg)