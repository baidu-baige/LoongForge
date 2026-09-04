"""Vendor module implementation."""
# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This software may be used and distributed in accordance with
# the terms of the DINOv3 License Agreement.

from typing import TYPE_CHECKING, Callable, List, Optional

import torch
import torch.nn.functional as F
from torch import Tensor, nn

from ..utils import cat_keep_shapes, uncat_with_shapes

from .attention import CausalSelfAttention, SelfAttention
from .ffn_layers import Mlp
from .layer_scale import LayerScale

if TYPE_CHECKING:
    from ..models.inference.kv_cache import LayerKVCache

torch._dynamo.config.automatic_dynamic_shapes = False
torch._dynamo.config.accumulated_cache_size_limit = 1024

try:
    from flash_attn_interface import flash_attn_func as _fa3_flash_attn_func
except Exception:
    _fa3_flash_attn_func = None


def _kv_cache_attention(q: Tensor, k_all: Tensor, v_all: Tensor) -> Tensor:
    """Full attention of current-frame queries against cached K/V.

    q, k_all, v_all: (B, H, S, D). The current frame attends ALL history (no
    causal mask within this call — frame-level causality is enforced by what's
    in the cache). Returns (B, H, S_q, D).
    """
    if _fa3_flash_attn_func is not None and q.is_cuda and q.dtype in (torch.float16, torch.bfloat16):
        # FA3 expects (B, S, H, D).
        out = _fa3_flash_attn_func(
            q.transpose(1, 2).contiguous(),
            k_all.transpose(1, 2).contiguous(),
            v_all.transpose(1, 2).contiguous(),
            causal=False,
        )
        if isinstance(out, tuple):
            out = out[0]
        return out.transpose(1, 2)
    return F.scaled_dot_product_attention(q, k_all, v_all)


class SelfAttentionBlock(nn.Module):
    """Vendor interface."""

    def __init__(
        self,
        dim: int,
        num_heads: int,
        ffn_ratio: float = 4.0,
        qkv_bias: bool = False,
        proj_bias: bool = True,
        ffn_bias: bool = True,
        drop: float = 0.0,
        attn_drop: float = 0.0,
        init_values=None,
        drop_path: float = 0.0,
        act_layer: Callable[..., nn.Module] = nn.GELU,
        norm_layer: Callable[..., nn.Module] = nn.LayerNorm,
        attn_class: Callable[..., nn.Module] = SelfAttention,
        ffn_layer: Callable[..., nn.Module] = Mlp,
        mask_k_bias: bool = False,
        attention_mode: str = "flash_attn3",
        device=None,
    ) -> None:
        """Vendor interface."""
        super().__init__()
        self.norm1 = norm_layer(dim)
        self.attn = attn_class(
            dim,
            num_heads=num_heads,
            qkv_bias=qkv_bias,
            proj_bias=proj_bias,
            attn_drop=attn_drop,
            proj_drop=drop,
            mask_k_bias=mask_k_bias,
            device=device,
        )
        self.ls1 = LayerScale(dim, init_values=init_values, device=device) if init_values else nn.Identity()

        self.norm2 = norm_layer(dim)
        mlp_hidden_dim = int(dim * ffn_ratio)
        self.mlp = ffn_layer(
            in_features=dim,
            hidden_features=mlp_hidden_dim,
            act_layer=act_layer,
            drop=drop,
            bias=ffn_bias,
            device=device,
        )
        self.ls2 = LayerScale(dim, init_values=init_values, device=device) if init_values else nn.Identity()

        self.sample_drop_ratio = drop_path
        self.attention_mode = attention_mode

    @staticmethod
    def _maybe_index_rope(rope: tuple[Tensor, Tensor] | None, indices: Tensor) -> tuple[Tensor, Tensor] | None:
        """Vendor interface."""
        if rope is None:
            return None

        sin, cos = rope
        assert sin.ndim == cos.ndim
        if sin.ndim == 4:
            return sin[indices], cos[indices]
        else:
            return sin, cos

    def _forward(self, x: Tensor, rope=None) -> Tensor:
        """Vendor interface."""
        b, _, _ = x.shape
        sample_subset_size = max(int(b * (1 - self.sample_drop_ratio)), 1)
        residual_scale_factor = b / sample_subset_size

        if self.training and self.sample_drop_ratio > 0.0:
            indices_1 = (torch.randperm(b, device=x.device))[:sample_subset_size]

            x_subset_1 = x[indices_1]
            rope_subset = self._maybe_index_rope(rope, indices_1)
            residual_1 = self.attn(self.norm1(x_subset_1), rope=rope_subset)

            x_attn = torch.index_add(
                x,
                dim=0,
                source=self.ls1(residual_1),
                index=indices_1,
                alpha=residual_scale_factor,
            )

            indices_2 = (torch.randperm(b, device=x.device))[:sample_subset_size]

            x_subset_2 = x_attn[indices_2]
            residual_2 = self.mlp(self.norm2(x_subset_2))

            x_ffn = torch.index_add(
                x_attn,
                dim=0,
                source=self.ls2(residual_2),
                index=indices_2,
                alpha=residual_scale_factor,
            )
        else:
            x_attn = x + self.ls1(self.attn(self.norm1(x), rope=rope))
            x_ffn = x_attn + self.ls2(self.mlp(self.norm2(x_attn)))

        return x_ffn

    def _forward_list(self, x_list: List[Tensor], rope_list=None) -> List[Tensor]:
        """Vendor interface."""
        b_list = [x.shape[0] for x in x_list]
        sample_subset_sizes = [max(int(b * (1 - self.sample_drop_ratio)), 1) for b in b_list]
        residual_scale_factors = [b / sample_subset_size for b, sample_subset_size in zip(b_list, sample_subset_sizes)]

        if self.training and self.sample_drop_ratio > 0.0:
            indices_1_list = [
                (torch.randperm(b, device=x.device))[:sample_subset_size]
                for x, b, sample_subset_size in zip(x_list, b_list, sample_subset_sizes)
            ]
            x_subset_1_list = [x[indices_1] for x, indices_1 in zip(x_list, indices_1_list)]

            if rope_list is not None:
                rope_subset_list = [
                    self._maybe_index_rope(rope, indices_1) for rope, indices_1 in zip(rope_list, indices_1_list)
                ]
            else:
                rope_subset_list = rope_list

            flattened, shapes, num_tokens = cat_keep_shapes(x_subset_1_list)
            norm1 = uncat_with_shapes(self.norm1(flattened), shapes, num_tokens)
            residual_1_list = self.attn.forward_list(norm1, rope_list=rope_subset_list)

            x_attn_list = [
                torch.index_add(
                    x,
                    dim=0,
                    source=self.ls1(residual_1),
                    index=indices_1,
                    alpha=residual_scale_factor,
                )
                for x, residual_1, indices_1, residual_scale_factor in zip(
                    x_list, residual_1_list, indices_1_list, residual_scale_factors
                )
            ]

            indices_2_list = [
                (torch.randperm(b, device=x.device))[:sample_subset_size]
                for x, b, sample_subset_size in zip(x_list, b_list, sample_subset_sizes)
            ]
            x_subset_2_list = [x[indices_2] for x, indices_2 in zip(x_attn_list, indices_2_list)]
            flattened, shapes, num_tokens = cat_keep_shapes(x_subset_2_list)
            norm2_flat = self.norm2(flattened)
            norm2_list = uncat_with_shapes(norm2_flat, shapes, num_tokens)

            residual_2_list = self.mlp.forward_list(norm2_list)

            x_ffn = [
                torch.index_add(
                    x_attn,
                    dim=0,
                    source=self.ls2(residual_2),
                    index=indices_2,
                    alpha=residual_scale_factor,
                )
                for x_attn, residual_2, indices_2, residual_scale_factor in zip(
                    x_attn_list, residual_2_list, indices_2_list, residual_scale_factors
                )
            ]
        else:
            x_out = []
            for x, rope in zip(x_list, rope_list):
                x_attn = x + self.ls1(self.attn(self.norm1(x), rope=rope))
                x_ffn = x_attn + self.ls2(self.mlp(self.norm2(x_attn)))
                x_out.append(x_ffn)
            x_ffn = x_out

        return x_ffn

    def _forward_packed(
        self,
        x: Tensor,
        cu_seqlens: Tensor,
        max_seqlen: int,
        rope_sin: Tensor,
        rope_cos: Tensor,
        flex_extra=None,
    ) -> Tensor:
        """Packed varlen forward path for NaViT."""
        from .attention import rope_apply
        from .flash_attention_packed import flash_attention3_varlen

        attn = self.attn
        x_normed = self.norm1(x)
        qkv = attn.qkv(x_normed)

        C = attn.qkv.in_features
        num_heads = attn.num_heads
        head_dim = C // num_heads
        total_tokens = qkv.shape[0]

        qkv = qkv.reshape(total_tokens, 3, num_heads, head_dim)
        q, k, v = qkv.unbind(1)

        q_dtype, k_dtype = q.dtype, k.dtype
        rope_dtype = rope_sin.dtype
        q = rope_apply(q.to(rope_dtype), rope_sin.unsqueeze(-2), rope_cos.unsqueeze(-2)).to(q_dtype)
        k = rope_apply(k.to(rope_dtype), rope_sin.unsqueeze(-2), rope_cos.unsqueeze(-2)).to(k_dtype)

        if self.attention_mode == "flex_block_causal":
            from .flex_attention_packed import flex_attention_block_causal
            shapes, crops_per_shape, n_prefix, block_masks = flex_extra
            has_regional = getattr(self, "_has_regional_compile", True)
            attn_out = flex_attention_block_causal(
                q.contiguous(), k.contiguous(), v.contiguous(),
                shapes, crops_per_shape, n_prefix, block_masks,
                has_regional_compile=has_regional,
            )
        elif self.attention_mode == "sdpa_block_causal":
            from .sdpa_attention_packed import sdpa_attention_block_causal
            shapes, crops_per_shape, n_prefix, block_masks = flex_extra
            attn_out = sdpa_attention_block_causal(
                q.contiguous(), k.contiguous(), v.contiguous(),
                shapes, crops_per_shape, n_prefix, block_masks,
            )
        else:
            attn_out = flash_attention3_varlen(
                q.contiguous(), k.contiguous(), v.contiguous(),
                cu_seqlens, max_seqlen,
            )
        attn_out = attn_out.reshape(total_tokens, C)
        attn_out = attn.proj(attn_out)

        x = x + self.ls1(attn_out)
        x = x + self.ls2(self.mlp(self.norm2(x)))
        return x

    def _forward_kv_cache(self, x: Tensor, cache: "LayerKVCache", rope: tuple[Tensor, Tensor] | None = None) -> Tensor:
        """Forward one new frame block against cached historical K/V."""
        from .attention import rope_apply

        if self.training:
            raise RuntimeError("_forward_kv_cache is inference-only; call eval() first")
        if x.ndim != 3 or x.shape[0] != 1:
            raise ValueError(f"x must have shape (1, frame_size, D), got {tuple(x.shape)}")

        attn = self.attn
        x_normed = self.norm1(x)
        bsz, n_tokens, dim = x_normed.shape
        num_heads = attn.num_heads
        head_dim = dim // num_heads
        qkv = attn.qkv(x_normed).reshape(bsz, n_tokens, 3, num_heads, head_dim)
        q, k, v = qkv.unbind(2)
        q = q.transpose(1, 2)
        k = k.transpose(1, 2)
        v = v.transpose(1, 2)

        if rope is not None:
            sin, cos = rope
            if sin.shape[0] != n_tokens or cos.shape[0] != n_tokens:
                raise ValueError(f"rope length must match token length {n_tokens}, got {sin.shape[0]}/{cos.shape[0]}")
            q_dtype, k_dtype = q.dtype, k.dtype
            q = rope_apply(q.to(sin.dtype), sin.unsqueeze(0).unsqueeze(0), cos.unsqueeze(0).unsqueeze(0)).to(q_dtype)
            k = rope_apply(k.to(sin.dtype), sin.unsqueeze(0).unsqueeze(0), cos.unsqueeze(0).unsqueeze(0)).to(k_dtype)

        k_all, v_all = cache.append(k, v)
        attn_out = _kv_cache_attention(q, k_all, v_all)
        attn_out = attn_out.transpose(1, 2).reshape(bsz, n_tokens, dim)
        x_attn = x + self.ls1(attn.proj(attn_out))
        return x_attn + self.ls2(self.mlp(self.norm2(x_attn)))

    def forward(self, x_or_x_list, rope_or_rope_list=None) -> List[Tensor]:
        """Vendor interface."""
        packed_seq_info = getattr(self, "_packed_seq_info", None)
        if packed_seq_info is not None:
            # pylint: disable=unpacking-non-sequence
            cu_seqlens, max_seqlen, rope_sin, rope_cos, flex_extra = packed_seq_info
            return self._forward_packed(
                x_or_x_list, cu_seqlens, max_seqlen, rope_sin, rope_cos, flex_extra,
            )
        if isinstance(x_or_x_list, Tensor):
            return self._forward_list([x_or_x_list], rope_list=[rope_or_rope_list])[0]
        elif isinstance(x_or_x_list, list):
            if rope_or_rope_list is None:
                rope_or_rope_list = [None for x in x_or_x_list]
            return self._forward_list(x_or_x_list, rope_list=rope_or_rope_list)
        else:
            raise AssertionError


class CausalSelfAttentionBlock(nn.Module):
    """Vendor interface."""

    def __init__(
        self,
        dim: int,
        num_heads: int,
        ffn_ratio: float = 4.0,
        ls_init_value: Optional[float] = None,
        is_causal: bool = True,
        act_layer: Callable = nn.GELU,
        norm_layer: Callable = nn.LayerNorm,
        dropout_prob: float = 0.0,
    ):
        """Vendor interface."""
        super().__init__()

        self.dim = dim
        self.is_causal = is_causal
        self.ls1 = LayerScale(dim, init_values=ls_init_value) if ls_init_value else nn.Identity()
        self.attention_norm = norm_layer(dim)
        self.attention = CausalSelfAttention(dim, num_heads, attn_drop=dropout_prob, proj_drop=dropout_prob)

        self.ffn_norm = norm_layer(dim)
        ffn_hidden_dim = int(dim * ffn_ratio)
        self.feed_forward = Mlp(
            in_features=dim,
            hidden_features=ffn_hidden_dim,
            drop=dropout_prob,
            act_layer=act_layer,
        )

        self.ls2 = LayerScale(dim, init_values=ls_init_value) if ls_init_value else nn.Identity()

    def init_weights(
        self,
        init_attn_std: float | None = None,
        init_proj_std: float | None = None,
        init_fc_std: float | None = None,
        factor: float = 1.0,
    ) -> None:
        """Vendor interface."""
        init_attn_std = init_attn_std or (self.dim**-0.5)
        init_proj_std = init_proj_std or init_attn_std * factor
        init_fc_std = init_fc_std or (2 * self.dim) ** -0.5
        self.attention.init_weights(init_attn_std, init_proj_std)
        self.attention_norm.reset_parameters()
        nn.init.normal_(self.feed_forward.fc1.weight, std=init_fc_std)
        nn.init.normal_(self.feed_forward.fc2.weight, std=init_proj_std)
        self.ffn_norm.reset_parameters()

    def forward(
        self,
        x: torch.Tensor,
    ):
        """Vendor interface."""
        x_attn = x + self.ls1(self.attention(self.attention_norm(x), self.is_causal))
        x_ffn = x_attn + self.ls2(self.feed_forward(self.ffn_norm(x_attn)))
        return x_ffn
