"""Unified Video ViT for DINOv3: all inputs are 5D video, FA3 attention built directly.

Self-contained — does NOT inherit from DinoVisionTransformer. Builds FA3 attention
blocks in __init__ without the build-then-replace pattern. Only supports 5D video
input (B, C, T, H, W); images should be unsqueezed to T=1 before reaching this model.
"""

import logging
from functools import partial
from typing import Any, Dict, List, Literal, Optional, Sequence, Tuple, Union

import torch
import torch.nn.init
from torch import Tensor, nn

from ..layers.flash_attention import flash_attention3
from loongforge.embodied.model.lingbot_vla_v2.vendor.vision_models.dino_video.lumos_dinov3.layers import (
    LayerScale,
    Mlp,
    PatchEmbed,
    RMSNorm,
    RopePositionEmbedding,
    RopePositionEmbedding3D,
    SelfAttention,
    SelfAttentionBlock,
    SwiGLUFFN,
)
from loongforge.embodied.model.lingbot_vla_v2.vendor.vision_models.dino_video.lumos_dinov3.layers.attention import (
    LinearKMaskedBias,
)
from .vision_transformer import (
    dtype_dict,
    ffn_layer_dict,
    init_weights_vit,
    norm_layer_dict,
)
from loongforge.embodied.model.lingbot_vla_v2.vendor.vision_models.dino_video.lumos_dinov3.utils import named_apply

logger = logging.getLogger("dinov3")


class FlashSelfAttention3(SelfAttention):
    """Drop-in replacement for SelfAttention using Flash Attention 3 on Hopper GPUs."""

    def compute_attention(self, qkv: Tensor, attn_bias=None, rope=None) -> Tensor:
        """Vendor interface."""
        assert attn_bias is None
        B, N, _ = qkv.shape
        C = self.qkv.in_features

        qkv = qkv.reshape(B, N, 3, self.num_heads, C // self.num_heads)
        q, k, v = qkv.unbind(2)

        if rope is not None:
            q_t, k_t = self.apply_rope(q.transpose(1, 2), k.transpose(1, 2), rope)
            q = q_t.transpose(1, 2).contiguous()
            k = k_t.transpose(1, 2).contiguous()
            v = v.contiguous()
        else:
            q = q.contiguous()
            k = k.contiguous()
            v = v.contiguous()

        x = flash_attention3(q, k, v)
        return x.reshape(B, N, C)


class UnifiedVideoViT(nn.Module):
    """Video Vision Transformer with FA3 attention — all inputs are 5D video.

    Built directly with FlashSelfAttention3 blocks (no SDPA build-then-replace).
    Tokenizes per-frame via 2D patch_embed, creates per-frame [CLS, storage, patches],
    and processes the full spatiotemporal sequence with self-attention.
    Spatial-only 2D RoPE (3D temporal RoPE to be added later).
    """

    def __init__(
        self,
        *,
        img_size: int = 224,
        patch_size: int = 16,
        in_chans: int = 3,
        pos_embed_rope_base: float = 100.0,
        pos_embed_rope_min_period: float | None = None,
        pos_embed_rope_max_period: float | None = None,
        pos_embed_rope_normalize_coords: Literal["min", "max", "separate"] = "separate",
        pos_embed_rope_shift_coords: float | None = None,
        pos_embed_rope_jitter_coords: float | None = None,
        pos_embed_rope_rescale_coords: float | None = None,
        pos_embed_rope_temporal_base: float = 10000.0,
        pos_embed_rope_fps: float | None = None,
        pos_embed_rope_base_fps: float = 24.0,
        pos_embed_rope_dtype: str = "bf16",
        pos_embed_rope_3d: bool = False,
        pos_embed_rope_prefix_temporal: bool = False,
        embed_dim: int = 768,
        depth: int = 12,
        num_heads: int = 12,
        ffn_ratio: float = 4.0,
        qkv_bias: bool = True,
        drop_path_rate: float = 0.0,
        layerscale_init: float | None = None,
        norm_layer: str = "layernorm",
        ffn_layer: str = "mlp",
        ffn_bias: bool = True,
        proj_bias: bool = True,
        n_storage_tokens: int = 0,
        mask_k_bias: bool = False,
        untie_cls_and_patch_norms: bool = False,
        untie_global_and_local_cls_norm: bool = False,
        attention_mode: str = "flash_attn3",
        device: Any | None = None,
        **ignored_kwargs,
    ):
        """Vendor interface."""
        super().__init__()
        if ignored_kwargs:
            logger.warning(f"Ignored kwargs: {ignored_kwargs}")

        norm_layer_cls = norm_layer_dict[norm_layer]
        ffn_layer_cls = ffn_layer_dict[ffn_layer]

        self.num_features = self.embed_dim = embed_dim
        self.n_blocks = depth
        self.num_heads = num_heads
        self.patch_size = patch_size
        self.attention_mode = attention_mode

        # --- Patch embedding (2D, applied per-frame) ---
        self.patch_embed = PatchEmbed(
            img_size=img_size,
            patch_size=patch_size,
            in_chans=in_chans,
            embed_dim=embed_dim,
            flatten_embedding=False,
        )

        # --- Learnable tokens ---
        self.cls_token = nn.Parameter(torch.empty(1, 1, embed_dim, device=device))
        self.n_storage_tokens = n_storage_tokens
        if n_storage_tokens > 0:
            self.storage_tokens = nn.Parameter(torch.empty(1, n_storage_tokens, embed_dim, device=device))
        self.mask_token = nn.Parameter(torch.empty(1, embed_dim, device=device))

        # --- RoPE ---
        rope_cls = RopePositionEmbedding3D if pos_embed_rope_3d else RopePositionEmbedding
        rope_kwargs = dict(
            embed_dim=embed_dim,
            num_heads=num_heads,
            base=pos_embed_rope_base,
            min_period=pos_embed_rope_min_period,
            max_period=pos_embed_rope_max_period,
            normalize_coords=pos_embed_rope_normalize_coords,
            shift_coords=pos_embed_rope_shift_coords,
            jitter_coords=pos_embed_rope_jitter_coords,
            rescale_coords=pos_embed_rope_rescale_coords,
            dtype=dtype_dict[pos_embed_rope_dtype],
            device=device,
        )
        if pos_embed_rope_3d:
            rope_kwargs["temporal_base"] = pos_embed_rope_temporal_base
            rope_kwargs["fps"] = pos_embed_rope_fps
            rope_kwargs["base_fps"] = pos_embed_rope_base_fps
        self.rope_embed = rope_cls(**rope_kwargs)
        self.rope_3d = pos_embed_rope_3d
        self.rope_prefix_temporal = pos_embed_rope_prefix_temporal

        # --- Transformer blocks with FA3 (built directly, no rebuild) ---
        self.blocks = nn.ModuleList([
            SelfAttentionBlock(
                dim=embed_dim,
                num_heads=num_heads,
                ffn_ratio=ffn_ratio,
                qkv_bias=qkv_bias,
                proj_bias=proj_bias,
                ffn_bias=ffn_bias,
                drop_path=drop_path_rate,
                norm_layer=norm_layer_cls,
                act_layer=nn.GELU,
                ffn_layer=ffn_layer_cls,
                init_values=layerscale_init,
                mask_k_bias=mask_k_bias,
                attn_class=FlashSelfAttention3,
                attention_mode=attention_mode,
                device=device,
            )
            for _ in range(depth)
        ])

        # --- Output norms ---
        self.norm = norm_layer_cls(embed_dim)

        self.untie_cls_and_patch_norms = untie_cls_and_patch_norms
        self.cls_norm = norm_layer_cls(embed_dim) if untie_cls_and_patch_norms else None

        self.untie_global_and_local_cls_norm = untie_global_and_local_cls_norm
        self.local_cls_norm = norm_layer_cls(embed_dim) if untie_global_and_local_cls_norm else None

        self.head = nn.Identity()

    def init_weights(self):
        """Vendor interface."""
        self.rope_embed._init_weights()
        nn.init.normal_(self.cls_token, std=0.02)
        if self.n_storage_tokens > 0:
            nn.init.normal_(self.storage_tokens, std=0.02)
        nn.init.zeros_(self.mask_token)
        named_apply(init_weights_vit, self)

    # ------------------------------------------------------------------
    # Tokenization (video-only, 5D input)
    # ------------------------------------------------------------------

    def prepare_tokens_with_masks(
        self, x: Tensor, masks: Tensor | None = None
    ) -> Tuple[Tensor, Tuple[int, int, int]]:
        """Tokenize 5D video input with per-frame [CLS, storage, patches] layout.

        Args:
            x: (B, C, T, H, W)
            masks: (B, T*Hp*Wp) tube mask

        Returns:
            tokens: (B, T * (1 + n_storage + Hp*Wp), D)
            shape: (T, Hp, Wp)
        """
        B, C, T, H, W = x.shape

        x = x.permute(0, 2, 1, 3, 4).reshape(B * T, C, H, W)
        x = self.patch_embed(x)
        _, Hp, Wp, D = x.shape
        x = x.reshape(B, T, Hp * Wp, D)

        if masks is not None:
            masks_reshaped = masks.reshape(B, T, Hp * Wp)
            x = torch.where(masks_reshaped.unsqueeze(-1), self.mask_token.to(x.dtype), x)
            cls_token = self.cls_token
        else:
            cls_token = self.cls_token + 0 * self.mask_token

        cls_tokens = cls_token.unsqueeze(0).expand(B, T, 1, -1)

        if self.n_storage_tokens > 0:
            storage = self.storage_tokens.unsqueeze(0).expand(B, T, -1, -1)
        else:
            storage = x.new_empty(B, T, 0, D)

        frame_tokens = torch.cat([cls_tokens, storage, x], dim=2)
        frame_tokens = frame_tokens.reshape(B, T * (1 + self.n_storage_tokens + Hp * Wp), D)

        return frame_tokens, (T, Hp, Wp)

    # ------------------------------------------------------------------
    # RoPE
    # ------------------------------------------------------------------

    def _compute_rope_for_shape(
        self, shape: Tuple[int, int, int], fps: float | None = None
    ) -> Tuple[Tensor, Tensor]:
        """Compute RoPE sin/cos for spatiotemporal shape (T, Hp, Wp).

        Per-frame: identity rotation for [CLS, storage] prefix, RoPE for patches.
        When rope_3d is True, temporal encoding is interleaved into the patches;
        if rope_prefix_temporal is set, the prefix also picks up its frame's
        temporal angle (H/W stays identity).
        """
        from .navit_video_vision_transformer import _make_prefix_rope

        T, Hp, Wp = shape
        n_prefix = 1 + self.n_storage_tokens

        if self.rope_3d:
            sin_sp, cos_sp = self.rope_embed(H=Hp, W=Wp, T=T, fps=fps)
            D = sin_sp.shape[-1]
            patches_per_frame = Hp * Wp
            t_mask = self.rope_embed.temporal_channel_mask
            prefix_temporal = getattr(self, "rope_prefix_temporal", False)

            sin_sp = sin_sp.view(T, patches_per_frame, D)
            cos_sp = cos_sp.view(T, patches_per_frame, D)

            frame_sins, frame_coss = [], []
            for t in range(T):
                sin_pre, cos_pre = _make_prefix_rope(
                    n_prefix, sin_sp[t], cos_sp[t], t_mask,
                    prefix_temporal,
                )
                frame_sins.append(torch.cat([sin_pre, sin_sp[t]], dim=0))
                frame_coss.append(torch.cat([cos_pre, cos_sp[t]], dim=0))
            return torch.cat(frame_sins, dim=0), torch.cat(frame_coss, dim=0)
        else:
            sin_sp, cos_sp = self.rope_embed(H=Hp, W=Wp)
            D = sin_sp.shape[-1]
            sin_prefix = torch.zeros(n_prefix, D, dtype=sin_sp.dtype, device=sin_sp.device)
            cos_prefix = torch.ones(n_prefix, D, dtype=cos_sp.dtype, device=cos_sp.device)
            sin_frame = torch.cat([sin_prefix, sin_sp], dim=0)
            cos_frame = torch.cat([cos_prefix, cos_sp], dim=0)
            return sin_frame.repeat(T, 1), cos_frame.repeat(T, 1)

    # ------------------------------------------------------------------
    # Forward
    # ------------------------------------------------------------------

    def forward_features_list(
        self, x_list: List[Tensor], masks_list: List[Tensor]
    ) -> List[Dict[str, Tensor]]:
        """Process a list of 5D video tensors through the backbone.

        Each element: (B, C, T, H, W) — images should already be unsqueezed to T=1.
        """
        tokens = []
        shapes = []

        for t_x, t_masks in zip(x_list, masks_list):
            t2_x, thw = self.prepare_tokens_with_masks(t_x, t_masks)
            tokens.append(t2_x)
            shapes.append(thw)

        x = tokens
        for blk in self.blocks:
            rope_sincos = [self._compute_rope_for_shape(s) for s in shapes]
            x = blk(x, rope_sincos)

        output = []
        for idx, (xi, masks, shape) in enumerate(zip(x, masks_list, shapes)):
            T, Hp, Wp = shape
            n_patches_per_frame = Hp * Wp
            frame_size = 1 + self.n_storage_tokens + n_patches_per_frame
            n_prefix = 1 + self.n_storage_tokens

            B_v = xi.shape[0]
            xi_frames = xi.reshape(B_v, T, frame_size, -1)

            cls_storage = xi_frames[:, :, :n_prefix, :]
            patches = xi_frames[:, :, n_prefix:, :]

            cls_storage_flat = cls_storage.reshape(B_v, T * n_prefix, -1)
            patches_flat = patches.reshape(B_v, T * n_patches_per_frame, -1)

            if self.untie_cls_and_patch_norms or self.untie_global_and_local_cls_norm:
                if self.untie_global_and_local_cls_norm and self.training and idx == 1:
                    x_norm_cls_storage = self.local_cls_norm(cls_storage_flat)
                elif self.untie_cls_and_patch_norms:
                    x_norm_cls_storage = self.cls_norm(cls_storage_flat)
                else:
                    x_norm_cls_storage = self.norm(cls_storage_flat)
                x_norm_patch = self.norm(patches_flat)
            else:
                x_norm_cls_storage = self.norm(cls_storage_flat)
                x_norm_patch = self.norm(patches_flat)

            x_norm_cls_storage = x_norm_cls_storage.reshape(B_v, T, n_prefix, -1)
            cls_normed = x_norm_cls_storage[:, :, 0, :]
            storage_normed = x_norm_cls_storage[:, :, 1:, :]

            output.append({
                "x_norm_clstoken": cls_normed.mean(dim=1),
                "x_norm_clstokens": cls_normed,
                "x_storage_tokens": storage_normed.reshape(B_v, T * self.n_storage_tokens, storage_normed.shape[-1]),
                "x_norm_patchtokens": x_norm_patch,
                "x_prenorm": xi,
                "masks": masks,
            })

        return output

    def forward_features(
        self, x: Tensor | List[Tensor], masks: Optional[Tensor | List[Tensor]] = None
    ) -> Dict[str, Tensor] | List[Dict[str, Tensor]]:
        """Vendor interface."""
        if isinstance(x, Tensor):
            if x.ndim == 4:
                x = x.unsqueeze(2)
            return self.forward_features_list([x], [masks])[0]
        return self.forward_features_list(x, masks)

    def forward(self, *args, is_training: bool = False, **kwargs) -> List[Dict[str, Tensor]] | Tensor:
        """Vendor interface."""
        ret = self.forward_features(*args, **kwargs)
        if is_training:
            return ret
        if isinstance(ret, list):
            return ret
        return self.head(ret["x_norm_clstoken"])

    # ------------------------------------------------------------------
    # Intermediate layers (for eval / linear probing)
    # ------------------------------------------------------------------

    def get_intermediate_layers(
        self,
        x: Tensor,
        *,
        n: Union[int, Sequence] = 1,
        reshape: bool = False,
        return_class_token: bool = False,
        return_extra_tokens: bool = False,
        norm: bool = True,
    ) -> Tuple:
        """Vendor interface."""
        if x.ndim == 4:
            x = x.unsqueeze(2)

        tokens, (T, Hp, Wp) = self.prepare_tokens_with_masks(x)
        n_prefix = 1 + self.n_storage_tokens

        total_block_len = len(self.blocks)
        blocks_to_take = range(total_block_len - n, total_block_len) if isinstance(n, int) else n

        outputs = []
        for i, blk in enumerate(self.blocks):
            rope_sincos = self._compute_rope_for_shape((T, Hp, Wp))
            tokens = blk([tokens], [rope_sincos])[0]
            if i in blocks_to_take:
                outputs.append(tokens)

        if norm:
            normed = []
            for out in outputs:
                B_v = out.shape[0]
                D = out.shape[-1]
                frames = out.reshape(B_v, T, -1, D)
                cls_storage = frames[:, :, :n_prefix, :]
                patches = frames[:, :, n_prefix:, :]
                if self.untie_cls_and_patch_norms:
                    cls_storage_n = self.cls_norm(cls_storage.reshape(B_v, T * n_prefix, D))
                else:
                    cls_storage_n = self.norm(cls_storage.reshape(B_v, T * n_prefix, D))
                patches_n = self.norm(patches.reshape(B_v, T * Hp * Wp, D))
                # Re-interleave back into the per-frame [cls, storage, patches]
                # layout that the extraction below expects. Concatenating the
                # two blocks as [all cls/storage | all patches] would corrupt the
                # frame_size reshape for T>1 (frame-0 would pick up other frames'
                # prefix tokens); the layouts only coincide at T==1.
                cls_storage_n = cls_storage_n.reshape(B_v, T, n_prefix, D)
                patches_n = patches_n.reshape(B_v, T, Hp * Wp, D)
                interleaved = torch.cat([cls_storage_n, patches_n], dim=2)
                normed.append(interleaved.reshape(B_v, T * (n_prefix + Hp * Wp), D))
            outputs = normed

        # Extract per-frame CLS mean, storage, patches
        frame_size = n_prefix + Hp * Wp
        class_tokens = []
        extra_tokens_list = []
        patch_outputs = []
        for out in outputs:
            frames = out.reshape(out.shape[0], T, frame_size, -1)
            cls = frames[:, :, 0, :].mean(dim=1)
            storage = frames[:, :, 1:n_prefix, :].reshape(out.shape[0], -1, out.shape[-1])
            patches = frames[:, :, n_prefix:, :].reshape(out.shape[0], -1, out.shape[-1])
            class_tokens.append(cls)
            extra_tokens_list.append(storage)
            patch_outputs.append(patches)

        if reshape and T == 1:
            patch_outputs = [
                p.reshape(p.shape[0], Hp, Wp, -1).permute(0, 3, 1, 2).contiguous()
                for p in patch_outputs
            ]

        if not return_class_token and not return_extra_tokens:
            return tuple(patch_outputs)
        elif return_class_token and not return_extra_tokens:
            return tuple(zip(patch_outputs, class_tokens))
        elif not return_class_token and return_extra_tokens:
            return tuple(zip(patch_outputs, extra_tokens_list))
        else:
            return tuple(zip(patch_outputs, class_tokens, extra_tokens_list))


# ------------------------------------------------------------------
# Factory functions
# ------------------------------------------------------------------

def unified_video_vit_small(patch_size=16, **kwargs):
    """Vendor interface."""
    return UnifiedVideoViT(patch_size=patch_size, embed_dim=384, depth=12, num_heads=6, ffn_ratio=4, **kwargs)


def unified_video_vit_base(patch_size=16, **kwargs):
    """Vendor interface."""
    return UnifiedVideoViT(patch_size=patch_size, embed_dim=768, depth=12, num_heads=12, ffn_ratio=4, **kwargs)


def unified_video_vit_large(patch_size=16, **kwargs):
    """Vendor interface."""
    return UnifiedVideoViT(patch_size=patch_size, embed_dim=1024, depth=24, num_heads=16, ffn_ratio=4, **kwargs)


def unified_video_vit_giant2(patch_size=16, **kwargs):
    """Vendor interface."""
    return UnifiedVideoViT(patch_size=patch_size, embed_dim=1536, depth=40, num_heads=24, ffn_ratio=4, **kwargs)
