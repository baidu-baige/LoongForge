"""Vendor module implementation."""
# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the Apache License, Version 2.0
# found in the LICENSE file in the root directory of this source tree.

# References:
#   https://github.com/facebookresearch/dino/blob/main/vision_transformer.py
#   https://github.com/rwightman/pytorch-image-models/tree/master/timm/models/vision_transformer.py

from functools import partial
import math
import logging
from typing import Sequence, Tuple, Union, Callable, Optional, List

import torch
import torch.nn as nn
import torch.utils.checkpoint
from torch.nn.init import trunc_normal_

from ..layers import Mlp, PatchEmbed, SwiGLUFFNFused, MemEffAttention, NestedTensorBlock as Block
from ..layers import PatchEmbedMLP

from .mask_utils import depth_masking

logger = logging.getLogger("dinov2_rgbd")


def named_apply(fn: Callable, module: nn.Module, name="", depth_first=True, include_root=False) -> nn.Module:
    """Vendor interface."""
    if not depth_first and include_root:
        fn(module=module, name=name)
    for child_name, child_module in module.named_children():
        child_name = ".".join((name, child_name)) if name else child_name
        named_apply(fn=fn, module=child_module, name=child_name, depth_first=depth_first, include_root=True)
    if depth_first and include_root:
        fn(module=module, name=name)
    return module


class BlockChunk(nn.ModuleList):
    """Vendor interface."""

    def forward(self, x):
        """Vendor interface."""
        for b in self:
            x = b(x)
        return x


class DinoVisionTransformer(nn.Module):
    """Vendor interface."""

    def __init__(
        self,
        img_size=224,
        patch_size=16,
        in_chans=3,
        embed_dim=768,
        depth=12,
        num_heads=12,
        mlp_ratio=4.0,
        qkv_bias=True,
        ffn_bias=True,
        proj_bias=True,
        drop_path_rate=0.0,
        drop_path_uniform=False,
        init_values=None,  # for layerscale: None or 0 => no layerscale
        embed_layer=PatchEmbed,
        act_layer=nn.GELU,
        block_fn=Block,
        ffn_layer="mlp",
        block_chunks=1,
        num_register_tokens=0,
        interpolate_antialias=False,
        interpolate_offset=0.1,
        img_depth_fuse_mode='',
        depth_mask_ratio: Union[float, List[float]] = 0.6,
        img_mask_ratio: Union[float, List[float]] = 0.0,
        depth_mask_patch_grid_size: int = 1,
        img_mask_patch_grid_size: int = 1,
        depth_emb_mode='',
        # depth_emb_mode='conv_1c'
    ):
        """
        Args:
            img_size (int, tuple): input image size
            patch_size (int, tuple): patch size
            in_chans (int): number of input channels
            embed_dim (int): embedding dimension
            depth (int): depth of transformer
            num_heads (int): number of attention heads
            mlp_ratio (int): ratio of mlp hidden dim to embedding dim
            qkv_bias (bool): enable bias for qkv if True
            proj_bias (bool): enable bias for proj in attn if True
            ffn_bias (bool): enable bias for ffn if True
            drop_path_rate (float): stochastic depth rate
            drop_path_uniform (bool): apply uniform drop rate across blocks
            weight_init (str): weight init scheme
            init_values (float): layer-scale init values
            embed_layer (nn.Module): patch embedding layer
            act_layer (nn.Module): MLP activation layer
            block_fn (nn.Module): transformer block class
            ffn_layer (str): "mlp", "swiglu", "swiglufused" or "identity"
            block_chunks: (int) split block sequence into block_chunks units for FSDP wrap
            num_register_tokens: (int) number of extra cls tokens (so-called "registers")
            interpolate_antialias: (str) flag to apply anti-aliasing when interpolating positional embeddings
            interpolate_offset: (float) work-around offset to apply when interpolating positional embeddings
        """
        super().__init__()
        norm_layer = partial(nn.LayerNorm, eps=1e-6)

        self.num_features = self.embed_dim = embed_dim  # num_features for consistency with other models
        self.num_tokens = 1
        self.n_blocks = depth
        self.num_heads = num_heads
        self.patch_size = patch_size
        self.num_register_tokens = num_register_tokens
        self.interpolate_antialias = interpolate_antialias
        self.interpolate_offset = interpolate_offset

        self.patch_embed = embed_layer(img_size=img_size, patch_size=patch_size, in_chans=in_chans, embed_dim=embed_dim)
        num_patches = self.patch_embed.num_patches

        self.depth_emb_mode = depth_emb_mode
        if self.depth_emb_mode == 'conv_1c':
            self.depth_patch_embed = embed_layer(
                img_size=img_size, patch_size=patch_size, in_chans=1, embed_dim=embed_dim
            )
        else:
            self.depth_patch_embed = None

        self.img_depth_fuse_mode = img_depth_fuse_mode

        self.depth_mask_patch_grid_size = depth_mask_patch_grid_size
        self.img_mask_patch_grid_size = img_mask_patch_grid_size
        assert self.depth_mask_patch_grid_size == 1, "depth_mask_patch_grid_size must be 1 in current version"
        assert self.img_mask_patch_grid_size == 1, "img_mask_patch_grid_size must be 1 in current version"

        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        self.pos_embed = nn.Parameter(torch.zeros(1, num_patches + self.num_tokens, embed_dim))
        assert num_register_tokens >= 0
        self.register_tokens = (
            nn.Parameter(torch.zeros(1, num_register_tokens, embed_dim)) if num_register_tokens else None
        )

        if drop_path_uniform is True:
            dpr = [drop_path_rate] * depth
        else:
            dpr = [x.item() for x in torch.linspace(0, drop_path_rate, depth)]  # stochastic depth decay rule

        if ffn_layer == "mlp":
            logger.info("using MLP layer as FFN")
            ffn_layer = Mlp
        elif ffn_layer == "swiglufused" or ffn_layer == "swiglu":
            logger.info("using SwiGLU layer as FFN")
            ffn_layer = SwiGLUFFNFused
        elif ffn_layer == "identity":
            logger.info("using Identity layer as FFN")

            def f(*args, **kwargs):
                """Vendor interface."""
                return nn.Identity()

            ffn_layer = f
        else:
            raise NotImplementedError

        blocks_list = [
            block_fn(
                dim=embed_dim,
                num_heads=num_heads,
                mlp_ratio=mlp_ratio,
                qkv_bias=qkv_bias,
                proj_bias=proj_bias,
                ffn_bias=ffn_bias,
                drop_path=dpr[i],
                norm_layer=norm_layer,
                act_layer=act_layer,
                ffn_layer=ffn_layer,
                init_values=init_values,
            )
            for i in range(depth)
        ]
        if block_chunks > 0:
            self.chunked_blocks = True
            chunked_blocks = []
            chunksize = depth // block_chunks
            for i in range(0, depth, chunksize):
                # this is to keep the block index consistent if we chunk the block list
                chunked_blocks.append([nn.Identity()] * i + blocks_list[i : i + chunksize])
            self.blocks = nn.ModuleList([BlockChunk(p) for p in chunked_blocks])
        else:
            self.chunked_blocks = False
            self.blocks = nn.ModuleList(blocks_list)

        self.norm = norm_layer(embed_dim)
        self.head = nn.Identity()

        self.mask_token = nn.Parameter(torch.zeros(1, embed_dim))

        self.init_weights()

    @property
    def onnx_compatible_mode(self):
        """Vendor interface."""
        return getattr(self, "_onnx_compatible_mode", False)

    @onnx_compatible_mode.setter
    def onnx_compatible_mode(self, value: bool):
        """Vendor interface."""
        self._onnx_compatible_mode = value

    def init_weights(self):
        """Vendor interface."""
        trunc_normal_(self.pos_embed, std=0.02)
        nn.init.normal_(self.cls_token, std=1e-6)
        if self.register_tokens is not None:
            nn.init.normal_(self.register_tokens, std=1e-6)
        named_apply(init_weights_vit_timm, self)

    def interpolate_pos_encoding(self, x, h, w):
        """Vendor interface."""
        previous_dtype = x.dtype
        npatch = x.shape[1] - 1
        batch_size = x.shape[0]
        N = self.pos_embed.shape[1] - 1
        if not self.onnx_compatible_mode and npatch == N and w == h:
            return self.pos_embed
        pos_embed = self.pos_embed.float()
        class_pos_embed = pos_embed[:, 0, :]
        patch_pos_embed = pos_embed[:, 1:, :]
        dim = x.shape[-1]
        h0, w0 = h // self.patch_size, w // self.patch_size
        M = int(math.sqrt(N))  # Recover the number of patches in each dimension
        assert N == M * M
        kwargs = {}
        if not self.onnx_compatible_mode and self.interpolate_offset > 0:
            # Historical kludge: add a small number to avoid floating point error in interpolation.
            # See https://github.com/facebookresearch/dino/issues/8.
            # Still needed for backward compatibility with output sizes and scale factors.
            sx = float(w0 + self.interpolate_offset) / M
            sy = float(h0 + self.interpolate_offset) / M
            kwargs["scale_factor"] = (sy, sx)
        else:
            # Simply specify an output size instead of a scale factor
            kwargs["size"] = (h0, w0)

        patch_pos_embed = nn.functional.interpolate(
            patch_pos_embed.reshape(1, M, M, dim).permute(0, 3, 1, 2),
            mode="bicubic",
            antialias=self.interpolate_antialias,
            **kwargs,
        )

        assert (h0, w0) == patch_pos_embed.shape[-2:]
        patch_pos_embed = patch_pos_embed.permute(0, 2, 3, 1).flatten(1, 2)
        class_pos_embed = class_pos_embed[:, None, :].expand(patch_pos_embed.shape[0], -1, -1)
        return torch.cat((class_pos_embed, patch_pos_embed), dim=1).to(previous_dtype)

    def interpolate_pos_encoding_without_cls(self, x, h, w, input_pos_embed):
        """Vendor interface."""
        previous_dtype = x.dtype
        npatch = x.shape[1]
        batch_size = x.shape[0]
        N = input_pos_embed.shape[1]
        if not self.onnx_compatible_mode and npatch == N and w == h:
            return input_pos_embed
        patch_pos_embed = input_pos_embed.float()
        dim = x.shape[-1]
        h0, w0 = h // self.patch_size, w // self.patch_size
        M = int(math.sqrt(N))  # Recover the number of patches in each dimension
        assert N == M * M
        kwargs = {}
        if not self.onnx_compatible_mode and self.interpolate_offset > 0:
            # Historical kludge: add a small number to avoid floating point error in interpolation.
            # See https://github.com/facebookresearch/dino/issues/8.
            # Still needed for backward compatibility with output sizes and scale factors.
            sx = float(w0 + self.interpolate_offset) / M
            sy = float(h0 + self.interpolate_offset) / M
            kwargs["scale_factor"] = (sy, sx)
        else:
            # Simply specify an output size instead of a scale factor
            kwargs["size"] = (h0, w0)
        patch_pos_embed = nn.functional.interpolate(
            patch_pos_embed.reshape(1, M, M, dim).permute(0, 3, 1, 2),
            mode="bicubic",
            antialias=self.interpolate_antialias,
            **kwargs,
        )
        assert (h0, w0) == patch_pos_embed.shape[-2:]
        patch_pos_embed = patch_pos_embed.permute(0, 2, 3, 1).flatten(1, 2)
        return patch_pos_embed.to(previous_dtype)

    def prepare_tokens_with_masks(self, x_img, x_depth, x_img_mask=None, x_depth_mask=None, masks=None, **kwargs):
        """Vendor interface."""
        assert masks is None, "extra masks are not supported for this model."
        B, nc, h_img, w_img = x_img.shape
        _, _, h_depth, w_depth = x_depth.shape
        x_depth_raw = x_depth.clone()
        x_depth_raw[x_depth_raw == 0] = -10

        depth_patch_num_h = h_depth // self.patch_size
        depth_patch_num_w = w_depth // self.patch_size

        # patchify, embed image tokens and depth tokens
        x_img = self.patch_embed(x_img) # batch, length_img, dim
        assert self.depth_patch_embed is not None
        x_depth = self.depth_patch_embed(x_depth) # batch, length_depth, dim
        assert depth_patch_num_h * depth_patch_num_w == x_depth.shape[1]

        # get full pose enc of img and depth
        # 1-> img data type enc
        # 2-> depth data type enc
        img_pose_enc = 1 + self.interpolate_pos_encoding_without_cls(
            x_img, h_img, w_img, self.pos_embed[:, 1:]
        ).repeat(B, 1, 1)
        depth_pose_enc = 2 + self.interpolate_pos_encoding_without_cls(
            x_depth, h_depth, w_depth, self.pos_embed[:, 1:]
        ).repeat(B, 1, 1)

        # add pose enc to img and depth
        x_img = x_img + img_pose_enc
        x_depth = x_depth + depth_pose_enc

        ## mask depth tokens
        if kwargs.get('enable_depth_mask', True):
            x_depth_masked, depth_mask_info = depth_masking(
                x_depth,
                depth_patch_num_h,
                depth_patch_num_w,
                depth_values=x_depth_raw,
                depth_mask_threshold_num=[1] * B,
                valid_depth_range=(-9.5, 200.0)
            )
        else:
            x_depth_masked = x_depth
            depth_mask_info = None

        ## mask image tokens
        x_img_masked = x_img
        img_mask_info = None

        # get cls token
        x_cls = self.cls_token.squeeze(0) +  self.pos_embed.squeeze(0)[:1] # 1, dim

        # cat cls, img and depth tokens
        assert self.img_depth_fuse_mode == 'cat_token', "Only cat_token mode is supported for this model."
        x_masked_list = []
        for i in range(B):
            if self.register_tokens is not None:
                x_mased = torch.cat(
                    [x_cls, self.register_tokens.squeeze(0), x_img_masked[i], x_depth_masked[i]],
                    dim=0,
                )  # 1 + num_register_tokens + length_img + length_depth, dim
            else:
                x_mased = torch.cat(
                    [x_cls, x_img_masked[i], x_depth_masked[i]], dim=0
                )  # 1 + length_img + length_depth, dim
            x_mased = x_mased.unsqueeze(0) # 1, 1 + num_register_tokens + length_img + length_depth, dim
            x_masked_list.append(x_mased)

        return x_masked_list

    def _get_intermediate_layers_not_chunked(
        self, x_img, x_depth, x_img_mask=None, x_depth_mask=None, n=1, return_mae_aux=False, **kwargs
    ):
        """Vendor interface."""
        x = self.prepare_tokens_with_masks(x_img, x_depth, x_img_mask, x_depth_mask, **kwargs)

        if not kwargs.get('enable_depth_mask', True):
            x = torch.cat(x, dim=0)

        # If n is an int, take the n last blocks. If it's a list, take them
        output, total_block_len = [], len(self.blocks)
        blocks_to_take = range(total_block_len - n, total_block_len) if isinstance(n, int) else n
        for i, blk in enumerate(self.blocks):
            x = blk(x)
            if i in blocks_to_take:
                output.append(x)
        assert len(output) == len(blocks_to_take), f"only {len(output)} / {len(blocks_to_take)} blocks found"

        if not kwargs.get('enable_depth_mask', True):
            output = [list(torch.split(out, 1, dim=0)) for out in output]
        return output

    def _get_intermediate_layers_chunked(
        self, x_img, x_depth, x_img_mask=None, x_depth_mask=None, n=1, return_mae_aux=False, **kwargs
    ):
        """Vendor interface."""
        x = self.prepare_tokens_with_masks(x_img, x_depth, x_img_mask, x_depth_mask, **kwargs)
        output, i, total_block_len = [], 0, len(self.blocks[-1])
        # If n is an int, take the n last blocks. If it's a list, take them
        blocks_to_take = range(total_block_len - n, total_block_len) if isinstance(n, int) else n
        for block_chunk in self.blocks:
            for blk in block_chunk[i:]:  # Passing the nn.Identity()
                x = blk(x)
                if i in blocks_to_take:
                    output.append(x)
                i += 1
        assert len(output) == len(blocks_to_take), f"only {len(output)} / {len(blocks_to_take)} blocks found"

        return output

    def extract_features(self, outputs, norm=True):
        """Vendor interface."""
        feat_outputs = []
        class_tokens = []
        feat_start_idx = 1 + self.num_register_tokens

        def process_output(out):
            """Vendor interface."""
            normed = self.norm(out) if norm else out
            return normed[:, feat_start_idx:], normed[:, 0]

        for output in outputs:
            if isinstance(output, list):
                feats, tokens = zip(*[process_output(out) for out in output])
                feat_outputs.append(list(feats))
                class_tokens.append(list(tokens))
            else:
                feat, token = process_output(output)
                feat_outputs.append(feat)
                class_tokens.append(token)

        return feat_outputs, class_tokens

    def get_intermediate_layers_mae(
        self,
        x_img: torch.Tensor,
        x_depth: torch.Tensor,
        x_img_mask: torch.Tensor = None,
        x_depth_mask: torch.Tensor = None,
        n: Union[int, Sequence] = 1,  # Layers or n last layers to take
        reshape: bool = False,
        return_class_token: bool = False,
        norm=True,
        return_mae_aux=True,
        **kwargs
    ):
        """Vendor interface."""
        assert reshape is False, "reshape is not supported for now"
        if self.chunked_blocks:
            outputs = self._get_intermediate_layers_chunked(
                x_img, x_depth, x_img_mask, x_depth_mask, n, return_mae_aux=return_mae_aux, **kwargs
            )
        else:
            outputs = self._get_intermediate_layers_not_chunked(
                x_img, x_depth, x_img_mask, x_depth_mask, n, return_mae_aux=return_mae_aux, **kwargs
            )

        feat_outputs, class_tokens = self.extract_features(outputs, norm)

        if return_class_token:
            return tuple(zip(feat_outputs, class_tokens))
        return tuple(feat_outputs)


def init_weights_vit_timm(module: nn.Module, name: str = ""):
    """ViT weight initialization, original timm impl (for reproducibility)"""
    if isinstance(module, nn.Linear):
        trunc_normal_(module.weight, std=0.02)
        if module.bias is not None:
            nn.init.zeros_(module.bias)


def vit_small(patch_size=16, num_register_tokens=0, **kwargs):
    """Vendor interface."""
    model = DinoVisionTransformer(
        patch_size=patch_size,
        embed_dim=384,
        depth=12,
        num_heads=6,
        mlp_ratio=4,
        block_fn=partial(Block, attn_class=MemEffAttention),
        num_register_tokens=num_register_tokens,
        **kwargs,
    )
    return model


def vit_base(patch_size=16, num_register_tokens=0, **kwargs):
    """Vendor interface."""
    model = DinoVisionTransformer(
        patch_size=patch_size,
        embed_dim=768,
        depth=12,
        num_heads=12,
        mlp_ratio=4,
        block_fn=partial(Block, attn_class=MemEffAttention),
        num_register_tokens=num_register_tokens,
        **kwargs,
    )
    return model


def vit_large(patch_size=16, num_register_tokens=0, **kwargs):
    """Vendor interface."""
    model = DinoVisionTransformer(
        patch_size=patch_size,
        embed_dim=1024,
        depth=24,
        num_heads=16,
        mlp_ratio=4,
        block_fn=partial(Block, attn_class=MemEffAttention),
        num_register_tokens=num_register_tokens,
        **kwargs,
    )
    return model


def vit_giant2(patch_size=16, num_register_tokens=0, **kwargs):
    """
    Close to ViT-giant, with embed-dim 1536 and 24 heads => embed-dim per head 64
    """
    model = DinoVisionTransformer(
        patch_size=patch_size,
        embed_dim=1536,
        depth=40,
        num_heads=24,
        mlp_ratio=4,
        block_fn=partial(Block, attn_class=MemEffAttention),
        num_register_tokens=num_register_tokens,
        **kwargs,
    )
    return model
