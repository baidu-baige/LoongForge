# Copyright 2026 The LoongForge Authors.
# SPDX-License-Identifier: Apache-2.0
#
# Modified from Cosmos (NVIDIA cosmos-framework) under the OpenMDW-1.1 License.
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: OpenMDW-1.1

# Source Repository: https://github.com/huggingface/transformers
# This is adapted from src/transformers/models/qwen3_vl/modeling_qwen3_vl.py.
# Commit Hash: 41e5abac5cb49983a08ddef3e8645d6efd23c8f3

"""PyTorch Qwen3-VL model."""

import functools
from typing import Any, Callable, Optional, Union

import torch
import torch.nn as nn
from transformers.activations import ACT2FN
from transformers.cache_utils import Cache, DynamicCache
from transformers.modeling_flash_attention_utils import FlashAttentionKwargs
from transformers.modeling_outputs import BaseModelOutputWithPast
from transformers.modeling_rope_utils import ROPE_INIT_FUNCTIONS, dynamic_rope_update


def _default_rope_init(config, device=None):
    """Default RoPE init for transformers versions that don't register 'default'."""
    dim = config.head_dim if hasattr(config, "head_dim") else config.hidden_size // config.num_attention_heads
    base = config.rope_theta if hasattr(config, "rope_theta") else 10000.0
    import torch
    inv_freq = 1.0 / (base ** (torch.arange(0, dim, 2, dtype=torch.float32, device=device) / dim))
    return inv_freq, 1.0
from transformers.modeling_utils import ALL_ATTENTION_FUNCTIONS, PreTrainedModel
from transformers.processing_utils import Unpack
from transformers.utils.deprecation import deprecate_kwarg

from .qwen3_vl_utils import create_causal_mask
from .configuration_qwen3_vl import Qwen3VLConfig, Qwen3VLTextConfig

TransformersKwargs = Any


class Qwen3VLVisionRotaryEmbedding(nn.Module):
    """Rotary position embedding for vision encoder."""
    def __init__(self, dim: int, theta: float = 10000.0) -> None:
        """Initialize rotary embedding parameters."""
        super().__init__()
        self.dim = dim
        self.theta = theta
        inv_freq = 1.0 / (theta ** (torch.arange(0, dim, 2, dtype=torch.float) / dim))
        self.register_buffer("inv_freq", inv_freq, persistent=False)

    def init_weights(self, buffer_device: torch.device | None = None) -> None:
        """Reinitialize inverse frequency buffer on device."""
        inv_freq = 1.0 / (
            self.theta ** (torch.arange(0, self.dim, 2, dtype=torch.float, device=buffer_device) / self.dim)
        )
        self.register_buffer("inv_freq", inv_freq, persistent=False)

    def forward(self, seqlen: int) -> torch.Tensor:
        """Compute rotary frequencies for given sequence length."""
        seq = torch.arange(seqlen, device=self.inv_freq.device, dtype=self.inv_freq.dtype)  # [seqlen]
        freqs = torch.outer(seq, self.inv_freq)  # [seqlen,dim//2]
        return freqs  # [seqlen,dim//2]


def rotate_half(x):
    """Rotates half the hidden dims of the input."""
    x1 = x[..., : x.shape[-1] // 2]  # [...,head_dim//2]
    x2 = x[..., x.shape[-1] // 2 :]  # [...,head_dim//2]
    return torch.cat((-x2, x1), dim=-1)  # [...,head_dim]


def apply_rotary_pos_emb_vision(
    q: torch.Tensor,  # [N_vision,num_heads,head_dim]
    k: torch.Tensor,  # [N_vision,num_heads,head_dim]
    cos: torch.Tensor,  # [N_vision,head_dim]
    sin: torch.Tensor,  # [N_vision,head_dim]
) -> tuple[torch.Tensor, torch.Tensor]:
    """Apply rotary position embedding to vision query and key tensors."""
    orig_q_dtype = q.dtype
    orig_k_dtype = k.dtype
    q, k = q.float(), k.float()
    cos, sin = cos.unsqueeze(-2).float(), sin.unsqueeze(-2).float()  # [N_vision,1,head_dim]
    q_embed = (q * cos) + (rotate_half(q) * sin)  # [N_vision,num_heads,head_dim]
    k_embed = (k * cos) + (rotate_half(k) * sin)  # [N_vision,num_heads,head_dim]
    q_embed = q_embed.to(orig_q_dtype)
    k_embed = k_embed.to(orig_k_dtype)
    return q_embed, k_embed  # [N_vision,num_heads,head_dim], [N_vision,num_heads,head_dim]


def repeat_kv(hidden_states: torch.Tensor, n_rep: int) -> torch.Tensor:
    """
    This is the equivalent of torch.repeat_interleave(x, dim=1, repeats=n_rep). The hidden states go from (batch,
    num_key_value_heads, seqlen, head_dim) to (batch, num_attention_heads, seqlen, head_dim)
    """
    batch, num_key_value_heads, slen, head_dim = hidden_states.shape
    if n_rep == 1:
        return hidden_states
    hidden_states = hidden_states[:, :, None, :, :].expand(
        batch, num_key_value_heads, n_rep, slen, head_dim
    )  # [B,num_kv_heads,n_rep,N,head_dim]
    return hidden_states.reshape(batch, num_key_value_heads * n_rep, slen, head_dim)  # [B,num_heads,N,head_dim]


def eager_attention_forward(
    module: nn.Module,
    query: torch.Tensor,  # [B,num_heads,N_q,head_dim]
    key: torch.Tensor,  # [B,num_kv_heads,N_kv,head_dim]
    value: torch.Tensor,  # [B,num_kv_heads,N_kv,head_dim]
    attention_mask: Optional[torch.Tensor],
    scaling: float,
    dropout: float = 0.0,
    **kwargs: Unpack[TransformersKwargs],
):
    """Compute eager (non-flash) multi-head attention."""
    key_states = repeat_kv(key, module.num_key_value_groups)  # [B,num_heads,N_kv,head_dim]
    value_states = repeat_kv(value, module.num_key_value_groups)  # [B,num_heads,N_kv,head_dim]

    attn_weights = torch.matmul(query, key_states.transpose(2, 3)) * scaling  # [B,num_heads,N_q,N_kv]
    if attention_mask is not None:
        causal_mask = attention_mask[:, :, :, : key_states.shape[-2]]  # [B,1,N_q,N_kv]
        attn_weights = attn_weights + causal_mask  # [B,num_heads,N_q,N_kv]

    attn_weights = nn.functional.softmax(attn_weights, dim=-1, dtype=torch.float32).to(
        query.dtype
    )  # [B,num_heads,N_q,N_kv]
    attn_weights = nn.functional.dropout(attn_weights, p=dropout, training=module.training)
    attn_output = torch.matmul(attn_weights, value_states)  # [B,num_heads,N_q,head_dim]
    attn_output = attn_output.transpose(1, 2).contiguous()  # [B,N_q,num_heads,head_dim]

    return attn_output, attn_weights


class Qwen3VLTextRotaryEmbedding(nn.Module):
    """Multimodal rotary position embedding for text decoder."""
    def __init__(self, config: Qwen3VLTextConfig):
        """Initialize text rotary embedding."""
        super().__init__()
        if hasattr(config, "rope_scaling") and config.rope_scaling is not None:
            self.rope_type = config.rope_scaling.get("rope_type", "default")
        else:
            self.rope_type = "default"
        self.max_seq_len_cached = config.max_position_embeddings
        self.original_max_seq_len = config.max_position_embeddings

        self.config = config
        self.rope_init_fn = ROPE_INIT_FUNCTIONS.get(self.rope_type, _default_rope_init)

        self.mrope_section = (
            config.rope_scaling.get("mrope_section", [24, 20, 20]) if config.rope_scaling is not None else [24, 20, 20]
        )
        inv_freq, self.attention_scaling = self.rope_init_fn(self.config)
        self.register_buffer("inv_freq", inv_freq, persistent=False)

    def init_weights(self, buffer_device: torch.device | None = None) -> None:
        """Reinitialize inverse frequency buffer on device."""
        inv_freq, self.attention_scaling = self.rope_init_fn(self.config, device=buffer_device)
        self.register_buffer("inv_freq", inv_freq, persistent=False)

    def apply_interleaved_mrope(self, freqs, mrope_section):
        """Apply interleaved MRoPE to 3D rotary embeddings.
        Reorganizes frequency layout from chunked [TTT...HHH...WWW] to
        interleaved [THTHWHTHW...TT], preserving frequency continuity.
        args:
            x: (3, bs, seq_len, head_dim // 2)
            mrope_section: (3,)
        returns:
            x_t: (bs, seq_len, head_dim // 2)
        """
        freqs_t = freqs[0]  # just overwrite the first dimension T
        for dim, offset in enumerate((1, 2), start=1):  # H, W
            length = mrope_section[dim] * 3
            idx = slice(offset, length, 3)
            freqs_t[..., idx] = freqs[dim, ..., idx]
        return freqs_t

    @torch.no_grad()
    @dynamic_rope_update  # power user: used with advanced RoPE types (e.g. dynamic rope)
    def forward(self, x, position_ids):
        """Compute rotary embeddings for text positions."""
        if self.inv_freq.dtype != torch.float32:
            self.inv_freq = self.inv_freq.float()

        # In contrast to other models, Qwen3VL has different position ids for the grids
        # So we expand the inv_freq to shape (3, ...)
        if position_ids.ndim == 2:
            position_ids = position_ids[None, ...].expand(3, position_ids.shape[0], -1)  # [3,B,N]
        inv_freq_expanded = (
            self.inv_freq[None, None, :, None].float().expand(3, position_ids.shape[1], -1, 1).to(x.device)
        )  # [3,B,head_dim//2,1]
        position_ids_expanded = position_ids[:, :, None, :].float()  # [3,B,1,N]

        # ROPE_FP32_FIX: outer FSDP/autocast wraps this forward in bfloat16,
        # which silently downcasts the .float() @ .float() matmul to bf16.
        # On axis-T (position_ids range 15010..15023) bf16 has only 8-bit
        # mantissa, leading to ~45-unit absolute error and divergence between
        # cosmos (fp32) and AIAK (bf16) pods. Disable autocast for this matmul.
        with __import__("torch").amp.autocast(device_type="cuda", enabled=False):
            freqs = (inv_freq_expanded.float() @ position_ids_expanded.float()).transpose(2, 3)  # [3,B,N,head_dim//2]
        freqs = self.apply_interleaved_mrope(freqs, self.mrope_section)  # [B,N,head_dim//2]
        emb = torch.cat((freqs, freqs), dim=-1)  # [B,N,head_dim]
        cos = emb.cos() * self.attention_scaling  # [B,N,head_dim]
        sin = emb.sin() * self.attention_scaling  # [B,N,head_dim]

        return cos.to(dtype=x.dtype), sin.to(dtype=x.dtype)  # each: [B,N,head_dim]


class Qwen3VLTextRMSNorm(nn.Module):
    """RMS normalization layer for Qwen3-VL text model."""
    def __init__(self, hidden_size: int, eps: float = 1e-6) -> None:
        """
        Qwen3VLTextRMSNorm is equivalent to T5LayerNorm
        """
        super().__init__()
        self.weight = nn.Parameter(torch.ones(hidden_size))
        self.variance_epsilon = eps

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        """Apply RMS normalization."""
        input_dtype = hidden_states.dtype
        hidden_states = hidden_states.to(torch.float32)
        variance = hidden_states.pow(2).mean(-1, keepdim=True)
        hidden_states = hidden_states * torch.rsqrt(variance + self.variance_epsilon)
        return self.weight * hidden_states.to(input_dtype)

    def extra_repr(self) -> str:
        """Return string representation of layer parameters."""
        return f"{tuple(self.weight.shape)}, eps={self.variance_epsilon}"


def apply_rotary_pos_emb(q, k, cos, sin, position_ids=None, unsqueeze_dim=1):
    """Applies Rotary Position Embedding to the query and key tensors.

    Args:
        q (`torch.Tensor`): The query tensor.
        k (`torch.Tensor`): The key tensor.
        cos (`torch.Tensor`): The cosine part of the rotary embedding.
        sin (`torch.Tensor`): The sine part of the rotary embedding.
        position_ids (`torch.Tensor`, *optional*):
            Deprecated and unused.
        unsqueeze_dim (`int`, *optional*, defaults to 1):
            The 'unsqueeze_dim' argument specifies the dimension along which to unsqueeze cos[position_ids] and
            sin[position_ids] so that they can be properly broadcasted to the dimensions of q and k. For example, note
            that cos[position_ids] and sin[position_ids] have the shape [batch_size, seq_len, head_dim]. Then, if q and
            k have the shape [batch_size, heads, seq_len, head_dim], then setting unsqueeze_dim=1 makes
            cos[position_ids] and sin[position_ids] broadcastable to the shapes of q and k. Similarly, if q and k have
            the shape [batch_size, seq_len, heads, head_dim], then set unsqueeze_dim=2.
    Returns:
        `tuple(torch.Tensor)` comprising of the query and key tensors rotated using the Rotary Position Embedding.
    """
    cos = cos.unsqueeze(unsqueeze_dim)  # [B,1,N,head_dim]
    sin = sin.unsqueeze(unsqueeze_dim)  # [B,1,N,head_dim]
    q_embed = (q * cos) + (rotate_half(q) * sin)  # [B,num_heads,N,head_dim]
    k_embed = (k * cos) + (rotate_half(k) * sin)  # [B,num_kv_heads,N,head_dim]
    return q_embed, k_embed  # [B,num_heads,N,head_dim], [B,num_kv_heads,N,head_dim]


class Qwen3VLTextAttention(nn.Module):
    """Multi-headed attention from 'Attention Is All You Need' paper"""

    def __init__(self, config: Qwen3VLTextConfig, layer_idx: int):
        """Initialize attention projections and norms."""
        super().__init__()
        self.config = config
        self.layer_idx = layer_idx
        self.head_dim = getattr(config, "head_dim", config.hidden_size // config.num_attention_heads)
        self.num_key_value_groups = config.num_attention_heads // config.num_key_value_heads
        self.scaling = self.head_dim**-0.5
        self.attention_dropout = config.attention_dropout
        self.is_causal = True

        self.q_proj = nn.Linear(
            config.hidden_size, config.num_attention_heads * self.head_dim, bias=config.attention_bias
        )
        self.k_proj = nn.Linear(
            config.hidden_size, config.num_key_value_heads * self.head_dim, bias=config.attention_bias
        )
        self.v_proj = nn.Linear(
            config.hidden_size, config.num_key_value_heads * self.head_dim, bias=config.attention_bias
        )
        self.o_proj = nn.Linear(
            config.num_attention_heads * self.head_dim, config.hidden_size, bias=config.attention_bias
        )
        self.q_norm = Qwen3VLTextRMSNorm(self.head_dim, eps=config.rms_norm_eps)  # unlike olmo, only on the head dim!
        self.k_norm = Qwen3VLTextRMSNorm(
            self.head_dim, eps=config.rms_norm_eps
        )  # thus post q_norm does not need reshape

    @deprecate_kwarg("past_key_value", new_name="past_key_values", version="4.58")
    def forward(
        self,
        hidden_states: torch.Tensor,  # [B,N,hidden_size]
        position_embeddings: tuple[torch.Tensor, torch.Tensor],
        attention_mask: Optional[torch.Tensor],
        past_key_values: Optional[Cache] = None,
        cache_position: Optional[torch.LongTensor] = None,
        **kwargs: Unpack[FlashAttentionKwargs],
    ) -> tuple[torch.Tensor, Optional[torch.Tensor]]:
        """Compute multi-head attention with RoPE."""
        input_shape = hidden_states.shape[:-1]
        hidden_shape = (*input_shape, -1, self.head_dim)

        query_states = self.q_norm(self.q_proj(hidden_states).view(hidden_shape)).transpose(
            1, 2
        )  # [B,num_heads,N,head_dim]
        key_states = self.k_norm(self.k_proj(hidden_states).view(hidden_shape)).transpose(
            1, 2
        )  # [B,num_kv_heads,N,head_dim]
        value_states = self.v_proj(hidden_states).view(hidden_shape).transpose(1, 2)  # [B,num_kv_heads,N,head_dim]

        cos, sin = position_embeddings
        query_states, key_states = apply_rotary_pos_emb(query_states, key_states, cos, sin)
        # query_states: [B,num_heads,N,head_dim], key_states: [B,num_kv_heads,N,head_dim]

        if past_key_values is not None:
            # sin and cos are specific to RoPE models; cache_position needed for the static cache
            cache_kwargs = {"sin": sin, "cos": cos, "cache_position": cache_position}
            key_states, value_states = past_key_values.update(key_states, value_states, self.layer_idx, cache_kwargs)
            # key_states: [B,num_kv_heads,N_cached,head_dim], value_states: [B,num_kv_heads,N_cached,head_dim]

        attention_interface: Callable = eager_attention_forward
        if self.config._attn_implementation != "eager":
            attention_interface = ALL_ATTENTION_FUNCTIONS[self.config._attn_implementation]

        attn_output, attn_weights = attention_interface(
            self,
            query_states,
            key_states,
            value_states,
            attention_mask,
            dropout=0.0 if not self.training else self.attention_dropout,
            scaling=self.scaling,
            **kwargs,
        )  # attn_output: [B,N,num_heads,head_dim]

        attn_output = attn_output.reshape(*input_shape, -1).contiguous()  # [B,N,hidden_size]
        attn_output = self.o_proj(attn_output)  # [B,N,hidden_size]
        return attn_output, attn_weights  # [B,N,hidden_size], [B,num_heads,N,N_cached] or None


class Qwen3VLTextMLP(nn.Module):
    """Gated MLP for Qwen3-VL text decoder."""
    def __init__(self, config):
        """Initialize text MLP layers."""
        super().__init__()
        self.config = config
        self.hidden_size = config.hidden_size
        self.intermediate_size = config.intermediate_size
        self.gate_proj = nn.Linear(self.hidden_size, self.intermediate_size, bias=False)
        self.up_proj = nn.Linear(self.hidden_size, self.intermediate_size, bias=False)
        self.down_proj = nn.Linear(self.intermediate_size, self.hidden_size, bias=False)
        self.act_fn = ACT2FN[config.hidden_act]

    def forward(self, x):
        """Forward pass through the gated MLP."""
        down_proj = self.down_proj(self.act_fn(self.gate_proj(x)) * self.up_proj(x))
        return down_proj


class Qwen3VLTextDecoderLayer(nn.Module):
    """Transformer decoder layer for Qwen3-VL text model."""
    def __init__(self, config: Qwen3VLTextConfig, layer_idx: int):
        """Initialize decoder layer components."""
        super().__init__()
        self.hidden_size = config.hidden_size

        self.self_attn = Qwen3VLTextAttention(config=config, layer_idx=layer_idx)

        self.mlp = Qwen3VLTextMLP(config)
        self.input_layernorm = Qwen3VLTextRMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.post_attention_layernorm = Qwen3VLTextRMSNorm(config.hidden_size, eps=config.rms_norm_eps)

    @deprecate_kwarg("past_key_value", new_name="past_key_values", version="4.58")
    def forward(
        self,
        hidden_states: torch.Tensor,
        position_embeddings: tuple[torch.Tensor, torch.Tensor],
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[Cache] = None,
        use_cache: Optional[bool] = False,
        cache_position: Optional[torch.LongTensor] = None,
        output_attentions: Optional[bool] = False,
        **kwargs: Unpack[TransformersKwargs],
    ) -> tuple[torch.Tensor, Optional[torch.Tensor]]:
        """Forward pass through the decoder layer."""
        residual = hidden_states
        hidden_states = self.input_layernorm(hidden_states)
        # Self Attention
        hidden_states, self_attn_weights = self.self_attn(
            hidden_states=hidden_states,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=past_key_values,
            use_cache=use_cache,
            cache_position=cache_position,
            position_embeddings=position_embeddings,
            **kwargs,
        )
        hidden_states = residual + hidden_states

        # Fully Connected
        residual = hidden_states
        hidden_states = self.post_attention_layernorm(hidden_states)
        hidden_states = self.mlp(hidden_states)
        hidden_states = residual + hidden_states

        outputs = (hidden_states,)
        if output_attentions:
            outputs += (self_attn_weights,)
        return outputs


class Qwen3VLPreTrainedModel(PreTrainedModel):
    """Base pretrained model class for Qwen3-VL."""
    config: Qwen3VLConfig
    base_model_prefix = "model"
    supports_gradient_checkpointing = True
    _no_split_modules = ["Qwen3VLTextDecoderLayer"]
    _skip_keys_device_placement = "past_key_values"
    _supports_flash_attn = True
    _supports_sdpa = True

    _can_compile_fullgraph = True
    _supports_attention_backend = True
    _can_record_outputs = {
        "hidden_states": Qwen3VLTextDecoderLayer,
        "attentions": Qwen3VLTextAttention,
    }

    def _init_weights(self, module: nn.Module, buffer_device: torch.device | None) -> None:
        """Initialize the weights."""
        super()._init_weights(module)

        if isinstance(
            module,
            (
                Qwen3VLVisionRotaryEmbedding,
                Qwen3VLTextRotaryEmbedding,
            ),
        ):
            module.init_weights(buffer_device=buffer_device)

    def init_weights(self, buffer_device: torch.device | None = None) -> None:
        """Initialize all module weights."""
        self.apply(functools.partial(self._init_weights, buffer_device=buffer_device))


class Qwen3VLTextModel(Qwen3VLPreTrainedModel):
    """Qwen3-VL text decoder model."""
    config: Qwen3VLTextConfig
    _no_split_modules = ["Qwen3VLTextDecoderLayer"]

    def __init__(self, config: Qwen3VLTextConfig):
        """Initialize text model components."""
        super().__init__(config)
        self.padding_idx = config.pad_token_id
        self.vocab_size = config.vocab_size

        self.embed_tokens = nn.Embedding(config.vocab_size, config.hidden_size, self.padding_idx)
        self.layers = nn.ModuleList(
            [Qwen3VLTextDecoderLayer(config, layer_idx) for layer_idx in range(config.num_hidden_layers)]
        )
        self.norm = Qwen3VLTextRMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.rotary_emb = Qwen3VLTextRotaryEmbedding(config=config)
        self.gradient_checkpointing = False

        # Initialize weights and apply final processing
        self.post_init()

    def forward(
        self,
        input_ids: Optional[torch.LongTensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[Cache] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
        cache_position: Optional[torch.LongTensor] = None,
        # args for deepstack
        visual_pos_masks: Optional[torch.Tensor] = None,
        **kwargs: Unpack[FlashAttentionKwargs],
    ) -> Union[tuple, BaseModelOutputWithPast]:
        r"""
        visual_pos_masks (`torch.Tensor` of shape `(batch_size, seqlen)`, *optional*):
            The mask of the visual positions.
        """
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        use_cache = use_cache if use_cache is not None else self.config.use_cache
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        if (input_ids is None) ^ (inputs_embeds is not None):
            raise ValueError("You must specify exactly one of input_ids or inputs_embeds")

        # torch.jit.trace() doesn't support cache objects in the output
        if use_cache and past_key_values is None and not torch.jit.is_tracing():
            past_key_values = DynamicCache(config=self.config)

        if inputs_embeds is None:
            inputs_embeds = self.embed_tokens(input_ids)  # [B,N,hidden_size]

        if cache_position is None:
            past_seen_tokens = past_key_values.get_seq_length() if past_key_values is not None else 0
            cache_position = torch.arange(
                past_seen_tokens, past_seen_tokens + inputs_embeds.shape[1], device=inputs_embeds.device
            )  # [N]

        # the hard coded `3` is for temporal, height and width.
        if position_ids is None:
            position_ids = cache_position.view(1, 1, -1).expand(3, inputs_embeds.shape[0], -1)  # [3,B,N]
        elif position_ids.ndim == 2:
            position_ids = position_ids[None, ...].expand(3, position_ids.shape[0], -1)  # [3,B,N]

        if position_ids.ndim == 3 and position_ids.shape[0] == 4:
            text_position_ids = position_ids[0]
            position_ids = position_ids[1:]
        else:
            text_position_ids = position_ids[0]

        attention_mask = create_causal_mask(
            config=self.config,
            input_embeds=inputs_embeds,
            attention_mask=attention_mask,
            cache_position=cache_position,
            past_key_values=past_key_values,
            position_ids=text_position_ids,
        )

        hidden_states = inputs_embeds  # [B,N,hidden_size]

        # create position embeddings to be shared across the decoder layers
        position_embeddings = self.rotary_emb(hidden_states, position_ids)  # each: [B,N,head_dim]

        # Initialize collectors like Qwen3
        all_hidden_states = () if output_hidden_states else None
        all_self_attns = () if output_attentions else None

        # decoder layers
        for layer_idx, decoder_layer in enumerate(self.layers):
            if output_hidden_states:
                all_hidden_states += (hidden_states,)

            layer_outputs = decoder_layer(
                hidden_states,
                attention_mask=attention_mask,
                position_ids=text_position_ids,
                past_key_values=past_key_values,
                cache_position=cache_position,
                position_embeddings=position_embeddings,
                output_attentions=output_attentions,
                **kwargs,
            )
            hidden_states = layer_outputs[0]

            if output_attentions:
                all_self_attns += (layer_outputs[1],)

        hidden_states = self.norm(hidden_states)  # [B,N,hidden_size]

        # add hidden states from the last decoder layer
        if output_hidden_states:
            all_hidden_states += (hidden_states,)

        if not return_dict:
            return tuple(
                v
                for v in [hidden_states, past_key_values if use_cache else None, all_hidden_states, all_self_attns]
                if v is not None
            )

        return BaseModelOutputWithPast(
            last_hidden_state=hidden_states,
            past_key_values=past_key_values,
            hidden_states=all_hidden_states,
            attentions=all_self_attns,
        )

__all__ = [
    "Qwen3VLPreTrainedModel",
    "Qwen3VLTextModel",
]
