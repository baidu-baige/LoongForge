# Copyright 2026 The LoongForge Authors.
# SPDX-License-Identifier: Apache-2.0
#
# Modified from Cosmos (NVIDIA cosmos-framework) under the OpenMDW-1.1 License.
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: OpenMDW-1.1

"""Unified MoT (Mixture of Transformers) implementation for Qwen3-VL Dense."""

import json
import logging
from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any

import torch
from torch import nn

from .sequence_packing import (
    FactoredSequencePack,
    from_joint,
    from_und_gen_splits,
    get_device_and_dtype,
    get_gen_seq,
    get_und_seq,
    set_gen_seq,
    set_und_seq,
    zeros_like,
)
from .attention import dispatch_attention, AttentionMaskType
from .configuration_qwen3_vl import (
    Qwen3VLConfig,
    Qwen3VLTextConfig,
    Qwen3VLVisionConfig,
)
from .qwen3_vl import (
    Qwen3VLPreTrainedModel,
    Qwen3VLTextMLP,
    Qwen3VLTextRMSNorm,
    Qwen3VLTextRotaryEmbedding,
    Qwen3VLVisionModel,
)
from .qwen3_vl import (
    apply_rotary_pos_emb as qwen3_vl_apply_rotary_pos_emb,
)

logger = logging.getLogger(__name__)


@dataclass
class _LBLMetadata:
    """Load-balancing loss metadata container."""
    num_tokens_per_expert: torch.Tensor
    num_tokens: torch.Tensor
    mean_router_prob_per_expert: torch.Tensor


# Torch optimization settings
torch._dynamo.config.cache_size_limit = 512
torch._dynamo.config.accumulated_cache_size_limit = 4096

# -----------------------------------------------------------------------------
# Unified MoT (Mixture of Transformers) implementation supporting Qwen3-VL Dense.
#
# Shared components:
#   - PackedAttentionMoT (config-driven QK norm and RoPE)
#   - MoTDecoderLayer (used by all variants)
#   - _impl_* (shared init/forward)
# -----------------------------------------------------------------------------


class LayerTypes:
    """Architecture-family dispatch table for the shared MoT layers."""

    def __init__(self, variant: str):
        """Initialize layer type dispatch table for the given variant."""
        self.variant = variant
        if variant == "qwen3_vl_dense":
            self.mlp = Qwen3VLTextMLP
            self.rms_norm = Qwen3VLTextRMSNorm
            self.rotary_embedding = Qwen3VLTextRotaryEmbedding
            self.apply_rotary_pos_emb = qwen3_vl_apply_rotary_pos_emb
        else:
            raise ValueError(f"Unknown LayerTypes variant: {variant!r}")

# -----------------------------------------------------------------------------
# MoT wrapper configs
# -----------------------------------------------------------------------------


class _MoTConfigBase(object):
    """Shared MoT wrapper logic for architecture families."""

    _full_config_cls: type = type(None)
    _text_config_cls: type = type(None)
    _vision_config_cls: type = type(None)

    def __init__(
        self,
        config_dict: Mapping[str, Any],
        *,
        qk_norm_for_text: bool = True,
        qk_norm_for_diffusion: bool = True,
        include_visual: bool = False,
        text_config_overrides: Mapping[str, Any] | None = None,
    ):
        """Initialize MoT config from a dictionary."""
        self.config_dict = dict(config_dict)
        self.qk_norm_for_text = qk_norm_for_text
        self.qk_norm_for_diffusion = qk_norm_for_diffusion
        self.include_visual = include_visual
        self.text_config_overrides: dict[str, Any] = dict(text_config_overrides) if text_config_overrides else {}

    @property
    def full_config(self) -> Any:
        """Build and return the full model config object."""
        if self._full_config_cls is type(None):
            raise ValueError(f"No _full_config_cls defined for {self.__class__.__name__}")
        return self._full_config_cls(**self.config_dict)

    @property
    def text_config(self) -> Any:
        """Build and return the text config object."""
        nested = self.config_dict.get("text_config")
        text_dict = nested if isinstance(nested, dict) else self.config_dict
        text_dict = self._transform_text_dict(text_dict)
        overrides = getattr(self, "text_config_overrides", None) or {}
        if overrides:
            text_dict = {**text_dict, **overrides}
        if self._text_config_cls is type(None):
            raise ValueError(f"No _text_config_cls defined for {self.__class__.__name__}")
        return self._text_config_cls(**text_dict)

    def _transform_text_dict(self, text_dict: Mapping[str, Any]) -> Mapping[str, Any]:
        """Hook to transform text config dict before construction."""
        return text_dict

    @property
    def vision_config(self) -> Any | None:
        """Build and return the vision config object, or None."""
        if not self.include_visual:
            return None
        vision_dict = self.config_dict.get("vision_config")
        if vision_dict is None:
            raise ValueError(
                "include_visual=True requires a vision_config sub-section in the language-model JSON config."
            )
        if self._vision_config_cls is type(None):
            raise ValueError(f"No _vision_config_cls defined for {self.__class__.__name__}")
        return self._vision_config_cls(**vision_dict)

    @classmethod
    def from_json_file(cls, json_file: str) -> "_MoTConfigBase":
        """Load config from a JSON file."""
        with open(json_file, encoding="utf-8") as reader:
            config_dict = json.load(reader)
        return cls(config_dict=config_dict)


class Qwen3MoTConfig(_MoTConfigBase):
    """MoT wrapper config for the Qwen3 family."""

    _full_config_cls = Qwen3VLTextConfig
    _text_config_cls = Qwen3VLTextConfig


class Qwen3VLMoTConfig(_MoTConfigBase):
    """MoT wrapper config for the dense Qwen3-VL family."""

    _full_config_cls = Qwen3VLConfig
    _text_config_cls = Qwen3VLTextConfig
    _vision_config_cls = Qwen3VLVisionConfig


# -----------------------------------------------------------------------------
# Common layers
# -----------------------------------------------------------------------------


class PackedAttentionMoT(nn.Module):
    """Dual-pathway packed attention for MoT architectures."""

    def __init__(
        self,
        config: Qwen3VLTextConfig,
        *,
        layer_idx: int,
        layer_types: LayerTypes,
        qk_norm_for_text: bool,
        qk_norm_for_diffusion: bool,
    ):
        """Initialize dual-pathway packed attention."""
        super().__init__()
        self.config = config
        self.layer_idx = layer_idx
        self.head_dim = getattr(config, "head_dim", config.hidden_size // config.num_attention_heads)
        self.hidden_size = config.hidden_size
        self.num_attention_heads = config.num_attention_heads
        self.num_key_value_heads = config.num_key_value_heads
        self.num_key_value_groups = self.num_attention_heads // self.num_key_value_heads
        self.scaling = self.head_dim**-0.5
        self.attention_dropout = config.attention_dropout

        eps = config.rms_norm_eps

        # Understanding pathway projections
        self.q_proj = nn.Linear(self.hidden_size, self.num_attention_heads * self.head_dim, bias=config.attention_bias)
        self.k_proj = nn.Linear(self.hidden_size, self.num_key_value_heads * self.head_dim, bias=config.attention_bias)
        self.v_proj = nn.Linear(self.hidden_size, self.num_key_value_heads * self.head_dim, bias=config.attention_bias)
        self.o_proj = nn.Linear(self.num_attention_heads * self.head_dim, self.hidden_size, bias=config.attention_bias)

        # Understanding pathway QK norm
        if qk_norm_for_text:
            self.q_norm = layer_types.rms_norm(self.head_dim, eps=eps)
            self.k_norm = layer_types.rms_norm(self.head_dim, eps=eps)
        else:
            self.q_norm = nn.Identity()
            self.k_norm = nn.Identity()

        # Generation pathway QK norm
        if qk_norm_for_diffusion:
            self.q_norm_moe_gen = layer_types.rms_norm(self.head_dim, eps=eps)
            self.k_norm_moe_gen = layer_types.rms_norm(self.head_dim, eps=eps)
        else:
            self.q_norm_moe_gen = nn.Identity()
            self.k_norm_moe_gen = nn.Identity()

        # Generation pathway linear projections
        self.q_proj_moe_gen = nn.Linear(
            self.hidden_size, self.num_attention_heads * self.head_dim, bias=config.attention_bias
        )
        self.k_proj_moe_gen = nn.Linear(
            self.hidden_size, self.num_key_value_heads * self.head_dim, bias=config.attention_bias
        )
        self.v_proj_moe_gen = nn.Linear(
            self.hidden_size, self.num_key_value_heads * self.head_dim, bias=config.attention_bias
        )
        self.o_proj_moe_gen = nn.Linear(
            self.num_attention_heads * self.head_dim, self.hidden_size, bias=config.attention_bias
        )

        self._apply_rotary_pos_emb = layer_types.apply_rotary_pos_emb
        self.dispatch_attention_fn = dispatch_attention

    def forward(
        self,
        pack: FactoredSequencePack,
        attention_mask: AttentionMaskType,
        packed_position_embeddings: tuple[FactoredSequencePack, FactoredSequencePack],
        natten_metadata: dict | None = None,
    ) -> tuple[FactoredSequencePack, "KVToStore | None"]:
        """Forward pass with dual-pathway QKV projection and attention."""
        q_und_in = self.q_proj(get_und_seq(pack))
        q_gen_in = self.q_proj_moe_gen(get_gen_seq(pack))

        k_und_in = self.k_proj(get_und_seq(pack))
        k_gen_in = self.k_proj_moe_gen(get_gen_seq(pack))

        v_und_in = self.v_proj(get_und_seq(pack))
        v_gen_in = self.v_proj_moe_gen(get_gen_seq(pack))

        q_und = q_und_in.view(-1, self.num_attention_heads, self.head_dim)
        k_und = k_und_in.view(-1, self.num_key_value_heads, self.head_dim)
        v_und = v_und_in.view(-1, self.num_key_value_heads, self.head_dim)

        q_gen = q_gen_in.view(-1, self.num_attention_heads, self.head_dim)
        k_gen = k_gen_in.view(-1, self.num_key_value_heads, self.head_dim)
        v_gen = v_gen_in.view(-1, self.num_key_value_heads, self.head_dim)

        q_und = self.q_norm(q_und)
        k_und = self.k_norm(k_und)

        q_gen = self.q_norm_moe_gen(q_gen)
        k_gen = self.k_norm_moe_gen(k_gen)

        packed_cos = packed_position_embeddings[0]
        packed_sin = packed_position_embeddings[1]

        q_und_, k_und_ = self._apply_rotary_pos_emb(
            q_und, k_und, get_und_seq(packed_cos), get_und_seq(packed_sin), unsqueeze_dim=1,
        )
        q_gen_, k_gen_ = self._apply_rotary_pos_emb(
            q_gen, k_gen, get_gen_seq(packed_cos), get_gen_seq(packed_sin), unsqueeze_dim=1,
        )

        packed_query_states_ = from_und_gen_splits(q_und_, q_gen_, pack)
        packed_key_states_ = from_und_gen_splits(k_und_, k_gen_, pack)
        packed_value_states_ = from_und_gen_splits(v_und, v_gen, pack)

        packed_attn_output = self.dispatch_attention_fn(
            packed_query_states_,
            packed_key_states_,
            packed_value_states_,
            attention_mask,
            natten_metadata=natten_metadata,
        )

        und_seq = self.o_proj(get_und_seq(packed_attn_output))
        gen_seq = self.o_proj_moe_gen(get_gen_seq(packed_attn_output))
        return from_und_gen_splits(und_seq, gen_seq, pack)


def _impl_init(
    self,
    config: Qwen3VLTextConfig,
    *,
    layer_types: LayerTypes,
    qk_norm_for_text: bool,
    qk_norm_for_diffusion: bool,
):
    """Shared __init__ body for MoT text-model variants."""
    self.padding_idx = getattr(config, "pad_token_id", None)
    self.vocab_size = config.vocab_size

    self.embed_tokens = nn.Embedding(config.vocab_size, config.hidden_size, self.padding_idx)

    self.layers = nn.ModuleList()
    for layer_idx in range(config.num_hidden_layers):
        self.layers.append(
            MoTDecoderLayer(
                config=config,
                layer_types=layer_types,
                layer_idx=layer_idx,
                qk_norm_for_text=qk_norm_for_text,
                qk_norm_for_diffusion=qk_norm_for_diffusion,
            )
        )

    self.norm = layer_types.rms_norm(config.hidden_size, eps=config.rms_norm_eps)
    self.norm_moe_gen = layer_types.rms_norm(config.hidden_size, eps=config.rms_norm_eps)
    self.rotary_emb = layer_types.rotary_embedding(config)
    self.post_init()


def _impl_forward(
    self,
    pack: FactoredSequencePack,
    attention_mask,
    position_ids: torch.Tensor,
    natten_metadata_list: list | None = None,
) -> tuple[FactoredSequencePack, dict]:
    """Shared training forward pass for MoT text models."""

    device, dtype = get_device_and_dtype(pack)
    _meta_tensor = torch.tensor([], dtype=dtype, device=device)
    cos, sin = self.rotary_emb(
        _meta_tensor, position_ids=position_ids.unsqueeze(0) if position_ids.ndim == 1 else position_ids.unsqueeze(1)
    )
    cos = cos.squeeze(0)
    sin = sin.squeeze(0)
    position_embeddings = (
        from_joint(cos, pack),
        from_joint(sin, pack),
    )

    lbl_metadata_all = dict(und=[], gen=[])
    hidden_states = pack

    for i, decoder_layer in enumerate(self.layers):
        hidden_states, lbl_metadata_dict = decoder_layer(
            hidden_states,
            attention_mask,
            position_embeddings,
            natten_metadata=None if natten_metadata_list is None else natten_metadata_list[i],
        )

        for pathway, lbl_metadata in lbl_metadata_dict.items():
            lbl_metadata_all[pathway].append(lbl_metadata)

    final_lbl_metadata: dict = dict()
    for pathway, lbl_metadata_list in lbl_metadata_all.items():
        if len(lbl_metadata_list) > 0:
            num_tokens_per_expert = torch.stack(
                [lbl_metadata.num_tokens_per_expert for lbl_metadata in lbl_metadata_list]
            )
            num_tokens = torch.stack([lbl_metadata.num_tokens for lbl_metadata in lbl_metadata_list])
            mean_router_prob_per_expert = torch.stack(
                [lbl_metadata.mean_router_prob_per_expert for lbl_metadata in lbl_metadata_list]
            )
            final_lbl_metadata[pathway] = _LBLMetadata(
                num_tokens_per_expert=num_tokens_per_expert,
                num_tokens=num_tokens,
                mean_router_prob_per_expert=mean_router_prob_per_expert,
            )

    hidden_states_out = zeros_like(hidden_states)
    set_und_seq(hidden_states_out, self.norm(get_und_seq(hidden_states)))
    set_gen_seq(hidden_states_out, self.norm_moe_gen(get_gen_seq(hidden_states)))

    return hidden_states_out, final_lbl_metadata


def _run_mlp(
    mlp: torch.nn.Module,
    input: torch.Tensor,
) -> tuple[torch.Tensor, Any]:
    """Run an MLP block and normalize the return shape across dense / MoE."""
    result = mlp(input)
    if isinstance(result, tuple):
        return result[0], result[1]
    return result, None


class MoTDecoderLayer(nn.Module):
    """Unified MoT decoder layer with dual-pathway attention."""

    def __init__(
        self,
        config: Qwen3VLTextConfig,
        *,
        layer_idx: int,
        layer_types: LayerTypes,
        qk_norm_for_text: bool,
        qk_norm_for_diffusion: bool,
    ):
        """Initialize MoT decoder layer with dual-pathway attention and MLPs."""
        super().__init__()
        self.hidden_size = config.hidden_size
        self.self_attn = PackedAttentionMoT(
            config,
            layer_types=layer_types,
            layer_idx=layer_idx,
            qk_norm_for_text=qk_norm_for_text,
            qk_norm_for_diffusion=qk_norm_for_diffusion,
        )

        self.mlp = layer_types.mlp(config)
        self.mlp_moe_gen = layer_types.mlp(config)

        self.input_layernorm = layer_types.rms_norm(config.hidden_size, eps=config.rms_norm_eps)
        self.input_layernorm_moe_gen = layer_types.rms_norm(config.hidden_size, eps=config.rms_norm_eps)
        self.post_attention_layernorm = layer_types.rms_norm(config.hidden_size, eps=config.rms_norm_eps)
        self.post_attention_layernorm_moe_gen = layer_types.rms_norm(config.hidden_size, eps=config.rms_norm_eps)

    def forward(
        self,
        input: FactoredSequencePack,
        attention_mask,
        packed_position_embeddings: tuple[FactoredSequencePack, FactoredSequencePack],
        natten_metadata: dict | None = None,
    ) -> tuple[FactoredSequencePack, dict, Any]:
        """Forward pass through attention and MLP with residual connections."""
        # Pre-Attention layernorm
        pack_norm_out = from_und_gen_splits(
            self.input_layernorm(get_und_seq(input)),
            self.input_layernorm_moe_gen(get_gen_seq(input)),
            input,
        )

        pack_attn_out = self.self_attn(
            pack_norm_out, attention_mask, packed_position_embeddings,
            natten_metadata=natten_metadata
        )
        residual_und = get_und_seq(input) + get_und_seq(pack_attn_out)
        residual_gen = get_gen_seq(input) + get_gen_seq(pack_attn_out)

        # Pre-MLP layernorm and processing
        lbl_metadata_dict: dict = dict()

        ln_out_und = self.post_attention_layernorm(residual_und)
        ln_out_gen = self.post_attention_layernorm_moe_gen(residual_gen)

        gen_len = pack_attn_out["_num_full_tokens"]
        und_len = pack_attn_out["_num_causal_tokens"]
        ln_out_und_unpadded = ln_out_und[:und_len]
        ln_out_gen_unpadded = ln_out_gen[:gen_len]

        mlp_out_und_unpadded, lbl_metadata_und = _run_mlp(self.mlp, ln_out_und_unpadded)
        mlp_out_gen_unpadded, lbl_metadata_gen = _run_mlp(self.mlp_moe_gen, ln_out_gen_unpadded)

        mlp_out_und = torch.cat([mlp_out_und_unpadded, ln_out_und[und_len:]], dim=0)
        mlp_out_gen = torch.cat([mlp_out_gen_unpadded, ln_out_gen[gen_len:]], dim=0)

        if lbl_metadata_und is not None:
            lbl_metadata_dict["und"] = lbl_metadata_und
        if lbl_metadata_gen is not None:
            lbl_metadata_dict["gen"] = lbl_metadata_gen

        mlp_out_und_seq = residual_und + mlp_out_und
        mlp_out_gen_seq = residual_gen + mlp_out_gen

        return from_und_gen_splits(mlp_out_und_seq, mlp_out_gen_seq, input), lbl_metadata_dict


class Qwen3VLTextModel(Qwen3VLPreTrainedModel):
    """Qwen3VL text model for MoT with dense MLPs."""

    def __init__(
        self,
        config: Qwen3VLTextConfig,
        *,
        qk_norm_for_text: bool,
        qk_norm_for_diffusion: bool,
    ):
        """Initialize text model with MoT layers."""
        super().__init__(config)
        _impl_init(
            self,
            config=config,
            layer_types=LayerTypes("qwen3_vl_dense"),
            qk_norm_for_text=qk_norm_for_text,
            qk_norm_for_diffusion=qk_norm_for_diffusion,
        )

    def forward(self, *args, **kwargs):
        """Forward pass delegating to shared implementation."""
        return _impl_forward(self, *args, **kwargs)


class Qwen3VLTextForCausalLM(Qwen3VLPreTrainedModel):
    """Qwen3VL text causal language model for MoT."""

    _tied_weights_keys = ["lm_head.weight"]

    def __init__(self, config: Qwen3VLMoTConfig):
        """Initialize causal LM with text model, LM head, and optional visual encoder."""
        # Force SDPA attention (flash_attn kernel may not be compiled for this GPU arch)
        full_config = config.full_config
        full_config._attn_implementation = "sdpa"
        if hasattr(full_config, 'text_config') and full_config.text_config is not None:
            full_config.text_config._attn_implementation = "sdpa"
        super().__init__(full_config)

        text_config = config.text_config
        self.model = Qwen3VLTextModel(
            text_config,
            qk_norm_for_text=config.qk_norm_for_text,
            qk_norm_for_diffusion=config.qk_norm_for_diffusion,
        )
        self.vocab_size = text_config.vocab_size
        self.lm_head = nn.Linear(text_config.hidden_size, text_config.vocab_size, bias=False)

        vision_config = config.vision_config
        if vision_config is not None:
            self.visual = Qwen3VLVisionModel._from_config(vision_config)

        self.post_init()

    def init_moe(self) -> None:
        """Copy understanding-pathway weights into the generation-pathway parameters."""
        state_dict = self.state_dict()
        for name, param in self.named_parameters():
            if "moe_gen" not in name:
                continue
            original_name = name.replace("_moe_gen", "").replace("_checkpoint_wrapped_module.", "")
            if original_name in state_dict:
                param.data.copy_(state_dict[original_name].data)
            else:
                raise ValueError(f"Could not find {original_name} in state_dict for initialization of {name}")

    def get_input_embeddings(self) -> nn.Embedding:
        """Return the token embedding layer."""
        return self.model.embed_tokens

    def set_input_embeddings(self, value: nn.Embedding) -> None:
        """Set the token embedding layer."""
        self.model.embed_tokens = value

    def forward(
        self,
        pack: FactoredSequencePack,
        attention_mask,
        position_ids: torch.Tensor,
        natten_metadata_list: list | None = None,
    ) -> tuple[FactoredSequencePack, dict]:
        """Training forward pass."""
        return self.model(
            pack=pack,
            attention_mask=attention_mask,
            position_ids=position_ids,
            natten_metadata_list=natten_metadata_list,
        )

