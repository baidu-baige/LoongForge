"""Vendor module implementation."""
from typing import List, Union
from numbers import Number
import importlib
import itertools
import functools
import sys

import torch
from torch import Tensor
import torch.nn as nn
import torch.nn.functional as F

from .dinov2_rgbd.models.vision_transformer import DinoVisionTransformer
from .utils import wrap_dinov2_attention_with_sdpa, wrap_module_with_gradient_checkpointing


class DINOv2_RGBD_Encoder(nn.Module):  # pylint: disable=invalid-name
    """Vendor interface."""
    backbone: DinoVisionTransformer
    image_mean: torch.Tensor
    image_std: torch.Tensor
    dim_features: int

    def __init__(
            self, backbone: str, intermediate_layers: Union[int, List[int]],
            dim_out: int, ignore_layers: Union[str, List[str]]=None,
            in_chans: int=3, strict: bool=True, img_depth_fuse_mode='', depth_emb_mode='', depth_mask_ratio=0.6,
            img_mask_ratio=0.0, **deprecated_kwargs):
        """Vendor interface."""
        super(DINOv2_RGBD_Encoder, self).__init__()

        self.intermediate_layers = intermediate_layers
        self.strict = strict
        self.ignore_layers = ignore_layers if ignore_layers is not None else []
        self.img_mask_ratio = img_mask_ratio
        # Load the backbone
        self.hub_loader = getattr(importlib.import_module(".dinov2_rgbd.hub.backbones", __package__), backbone)
        self.backbone_name = backbone
        self.backbone = self.hub_loader(pretrained=False,
                                        in_chans=in_chans,
                                        img_depth_fuse_mode=img_depth_fuse_mode,
                                        depth_emb_mode=depth_emb_mode,
                                        depth_mask_ratio=depth_mask_ratio,
                                        img_mask_ratio=img_mask_ratio)

        self.dim_features = self.backbone.blocks[0].attn.qkv.in_features
        self.num_features = intermediate_layers if isinstance(intermediate_layers, int) else len(intermediate_layers)

        if img_mask_ratio > 0:
            self.mask_token_mae = nn.Parameter(torch.zeros(1, 1, self.dim_features))
            torch.nn.init.normal_(self.mask_token_mae, std=.02)

        self.output_projections = nn.ModuleList([
            nn.Conv2d(in_channels=self.dim_features, out_channels=dim_out, kernel_size=1, stride=1, padding=0,)
                for _ in range(self.num_features)
        ])

        self.register_buffer("image_mean", torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1))
        self.register_buffer("image_std", torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1))

    @property
    def onnx_compatible_mode(self):
        """Vendor interface."""
        return getattr(self, "_onnx_compatible_mode", False)

    @onnx_compatible_mode.setter
    def onnx_compatible_mode(self, value: bool):
        """Vendor interface."""
        self._onnx_compatible_mode = value
        self.backbone.onnx_compatible_mode = value

    def init_weights(self):
        """Vendor interface."""
        pretrained_backbone_state_dict = self.hub_loader(pretrained=True).state_dict()
        ignore_layers = []
        if isinstance(self.ignore_layers, str):
            ignore_layers = [self.ignore_layers]
        else:
            ignore_layers = self.ignore_layers

        if len(ignore_layers) == 0:
            self.backbone.load_state_dict(pretrained_backbone_state_dict, strict=self.strict)
        else:
            state_dict = {}
            for k, v in pretrained_backbone_state_dict.items():
                is_ignore = False
                for ig_k in ignore_layers:
                    if ig_k in k:
                        is_ignore = True
                        break
                if not is_ignore:
                    state_dict[k] = v
            self.backbone.load_state_dict(state_dict, strict=self.strict)

    def enable_gradient_checkpointing(self):
        """Vendor interface."""
        for i in range(len(self.backbone.blocks)):
            wrap_module_with_gradient_checkpointing(self.backbone.blocks[i])

    def enable_pytorch_native_sdpa(self):
        """Vendor interface."""
        for i in range(len(self.backbone.blocks)):
            wrap_dinov2_attention_with_sdpa(self.backbone.blocks[i].attn)

    def forward(self,
                image: torch.Tensor,
                depth: torch.Tensor,
                token_rows: Union[int, torch.LongTensor],
                token_cols: Union[int, torch.LongTensor],
                return_class_token: bool = False,
                remap_depth_in: str = 'linear',
                **kwargs):
        """Vendor interface."""
        image_14 = F.interpolate(image, (token_rows * 14, token_cols * 14), mode="bilinear",
                                 align_corners=False, antialias=not self.onnx_compatible_mode)
        image_14 = (image_14 - self.image_mean) / self.image_std

        depth_14 = F.interpolate(depth, (token_rows * 14, token_cols * 14), mode="nearest")

        # set invalid depth value to zero
        depth_14[torch.isinf(depth_14)] = 0.0
        depth_14[torch.isnan(depth_14)] = 0.0
        dmask_14 = (depth_14 > 0.01).detach()
        depth_14 = depth_14 * dmask_14.float()

        if remap_depth_in == 'linear':
            pass # do nothing
        elif remap_depth_in == 'log':
            depth_14 = torch.log(depth_14)
            depth_14[~dmask_14] = 0.0
            depth_14 = torch.nan_to_num(depth_14, nan=0.0, posinf=0.0, neginf=0.0)
        else:
            raise NotImplementedError

        # Get intermediate layers from the backbone
        features = self.backbone.get_intermediate_layers_mae(
            x_img=image_14,
            x_depth=depth_14,
            n=self.intermediate_layers,
            return_class_token=True,
            **kwargs)

        assert self.img_mask_ratio == 0, "img_mask_ratio is not supported in this encoder"

        if isinstance(features[0][0], list):
            num_valid_tokens = token_rows * token_cols
            features = tuple(
                (
                    torch.cat([feat[:, :num_valid_tokens].contiguous() for feat in feats], dim=0),
                    torch.cat(cls_tokens, dim=0)
                )
                for feats, cls_tokens in features
            )

        # Project features to the desired dimensionality
        x = torch.stack([
            proj(feat.permute(0, 2, 1)[:, :, :token_rows * \
                 token_cols].unflatten(2, (token_rows, token_cols)).contiguous())
                for proj, (feat, clstoken) in zip(self.output_projections, features)
        ], dim=1).sum(dim=1)
        cls_token = features[-1][1]

        if return_class_token:
            return x, cls_token, None, None
        else:
            return x, None, None
