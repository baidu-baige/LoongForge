import json
import os
from collections import OrderedDict
from typing import Optional

import accelerate
import torch
from diffusers.configuration_utils import ConfigMixin, register_to_config
from diffusers.loaders import PeftAdapterMixin
from diffusers.models.model_loading_utils import load_model_dict_into_meta
from diffusers.models.modeling_utils import ModelMixin
from diffusers.utils.peft_utils import get_peft_kwargs, set_weights_and_activate_adapters
from einops import rearrange
from peft import LoraConfig, set_peft_model_state_dict
from torch import nn
from torchvision import transforms

from ....acceleration import gather_forward_split_backward, get_sequence_parallel_group, split_forward_gather_backward
from ....utils import load_state_dict
from .modules import FinalLayer, PatchEmbed, TimestepEmbedding, Timesteps, TransformerBlock, VideoRopePosition3DEmb


class GigaWorld0Transformer3DModel(ModelMixin, ConfigMixin, PeftAdapterMixin):
    """Main 3D transformer model for GigaWorld0.

    Supports patch embedding, rotary position encoding, transformer blocks, LoRA/fp8 features, and flexible configuration for video diffusion tasks.
    """

    @register_to_config
    def __init__(
        self,
        max_img_h: int,
        max_img_w: int,
        max_frames: int,
        in_channels: int,
        out_channels: int,
        patch_spatial: tuple,
        patch_temporal: int,
        concat_padding_mask: bool = True,
        block_config: str = 'FA-CA-MLP',
        model_channels: int = 768,
        num_blocks: int = 10,
        num_heads: int = 16,
        mlp_ratio: float = 4.0,
        crossattn_emb_channels: int = 1024,
        adaln_lora_dim: int = 256,
        rope_h_extrapolation_ratio: float = 1.0,
        rope_w_extrapolation_ratio: float = 1.0,
        rope_t_extrapolation_ratio: float = 1.0,
        natten_parameters: list = None,
        moe_parameters: dict = None,
    ) -> None:
        """Initialize the 3D transformer model for GigaWorld0.

        Args:
            max_img_h, max_img_w, max_frames: Maximum input video dimensions.
            in_channels, out_channels: Input/output channels.
            patch_spatial, patch_temporal: Patch sizes for spatial/temporal embedding.
            concat_padding_mask: Whether to add a padding mask channel.
            block_config: Transformer block configuration string.
            model_channels: Embedding dimension.
            num_blocks: Number of transformer blocks.
            num_heads: Number of attention heads.
            mlp_ratio: MLP expansion ratio.
            crossattn_emb_channels: Cross-attention embedding channels.
            adaln_lora_dim: AdaLN/LoRA hidden dim.
            rope_*_extrapolation_ratio: Rotary embedding extrapolation ratios.
            natten_parameters: List of NATTEN configs per block.
            moe_parameters: Mixture-of-Experts config.
        """
        super().__init__()
        # Optionally add a channel for padding mask
        in_channels = in_channels + 1 if concat_padding_mask else in_channels
        # Patch embedding for input video
        self.x_embedder = PatchEmbed(
            spatial_patch_size=patch_spatial,
            temporal_patch_size=patch_temporal,
            in_channels=in_channels,
            out_channels=model_channels,
            bias=False,
        )
        # 3D rotary positional embedding
        self.pos_embedder = VideoRopePosition3DEmb(
            head_dim=model_channels // num_heads,
            len_h=max_img_h // patch_spatial,
            len_w=max_img_w // patch_spatial,
            len_t=max_frames // patch_temporal,
            h_extrapolation_ratio=rope_h_extrapolation_ratio,
            w_extrapolation_ratio=rope_w_extrapolation_ratio,
            t_extrapolation_ratio=rope_t_extrapolation_ratio,
        )
        # Timestep embedding for diffusion
        self.t_embedder = nn.Sequential(
            Timesteps(model_channels),
            TimestepEmbedding(model_channels, model_channels),
        )
        # Stack of transformer blocks
        self.blocks = nn.ModuleDict()
        for idx in range(num_blocks):
            self.blocks[f'block{idx}'] = TransformerBlock(
                block_config=block_config,
                x_dim=model_channels,
                context_dim=crossattn_emb_channels,
                num_heads=num_heads,
                mlp_ratio=mlp_ratio,
                adaln_lora_dim=adaln_lora_dim,
                natten_params=natten_parameters[idx] if natten_parameters is not None else None,
                moe_params=moe_parameters,
            )
        # Final projection layer
        self.final_layer = FinalLayer(
            hidden_size=model_channels,
            spatial_patch_size=patch_spatial,
            temporal_patch_size=patch_temporal,
            out_channels=out_channels,
            adaln_lora_dim=adaln_lora_dim,
        )
        # Affine normalization for timestep embedding
        self.affline_norm = nn.RMSNorm(model_channels, eps=1e-6)

        # FP8 support
        self.fp8 = False
        self.fp8_config = dict()

    def to_fp8(self, **kwargs):
        """Convert model to FP8 using transformer engine (for inference
        speedup)."""
        from ....exports.transformer_engine import convert_model_torch_to_te

        self.fp8_config['convert_cfg'] = kwargs
        convert_model_torch_to_te(self, **kwargs)
        self.fp8 = True

    def state_dict(self, *args, destination=None, **kwargs):
        """Custom state_dict that removes extra state keys unless in FP8
        mode."""
        assert len(args) == 0
        if destination is None:
            destination = OrderedDict()
            destination._metadata = OrderedDict()
        super().state_dict(destination=destination, **kwargs)
        if not self.fp8:
            ignore_keys = []
            for key, val in destination.items():
                if key.endswith('_extra_state'):
                    ignore_keys.append(key)
            for key in ignore_keys:
                destination.pop(key)
        return destination

    def save_config(self, save_directory, *args, **kwargs):
        """Save model config, including FP8 config if present."""
        super().save_config(save_directory, *args, **kwargs)
        if self.fp8:
            save_path = os.path.join(save_directory, 'config_fp8.json')
            json.dump(self.fp8_config, open(save_path, 'w'), indent=2)

    @classmethod
    def load_config(cls, pretrained_model_name_or_path, *args, **kwargs):
        """Load config and attach FP8 config if available."""
        config = super().load_config(pretrained_model_name_or_path, *args, **kwargs)
        fp8_config_path = os.path.join(pretrained_model_name_or_path, 'config_fp8.json')
        if os.path.exists(fp8_config_path):
            fp8_config = json.load(open(fp8_config_path, 'r'))
            config[0]['_fp8_config'] = fp8_config
        return config

    @classmethod
    def from_config(cls, config, *args, **kwargs):
        """Instantiate model from config, restoring FP8 if needed."""
        fp8_config = config.pop('_fp8_config', None)
        model = super().from_config(config, *args, **kwargs)
        if fp8_config is not None:
            model.fp8_config = fp8_config
            model.to_fp8(**fp8_config['convert_cfg'])
        return model

    @classmethod
    def from_pretrained(cls, pretrained_model_name_or_path, *args, **kwargs):
        """Load model weights from disk, handling FP8 and empty-weight loading
        for large models."""
        fp8_config_path = os.path.join(pretrained_model_name_or_path, 'config_fp8.json')
        if os.path.exists(fp8_config_path):
            kwargs['low_cpu_mem_usage'] = False
        try:
            model = super().from_pretrained(pretrained_model_name_or_path, *args, **kwargs)
        except ValueError as _:  # noqa F841
            config = super().load_config(pretrained_model_name_or_path)
            with accelerate.init_empty_weights():
                model = super().from_config(config)
            state_dict = load_state_dict(pretrained_model_name_or_path)
            load_model_dict_into_meta(model, state_dict)
            model.eval()
        return model

    def load_lora(self, model_path_dict, fuse=False):
        """Load LoRA adapters from disk and optionally fuse them for
        inference."""
        if isinstance(model_path_dict, str):
            model_path_dict = {'default': model_path_dict}
        adapter_names = list(model_path_dict.keys())
        lora_scales = []
        for adapter_name in adapter_names:
            model_info = model_path_dict[adapter_name]
            if isinstance(model_info, str):
                model_path, lora_scale = model_info, 1.0
            else:
                model_path, lora_scale = model_info
            state_dict = load_state_dict(model_path)
            rank = {}
            for key, val in state_dict.items():
                if 'lora_B' in key and val.ndim > 1:
                    rank[key] = val.shape[1]
            lora_config_kwargs = get_peft_kwargs(rank, network_alpha_dict=None, peft_state_dict=state_dict)
            lora_config = LoraConfig(**lora_config_kwargs)
            self.add_adapter(lora_config, adapter_name=adapter_name)
            fp8_config_path = os.path.join(model_path, 'config_fp8.json')
            if os.path.exists(fp8_config_path):
                fp8_config = json.load(open(fp8_config_path, 'r'))
                self.fp8_config = fp8_config
                self.to_fp8(**fp8_config['convert_cfg'])
            if not self.fp8:
                keys = list(state_dict.keys())
                for key in keys:
                    if key.endswith('_extra_state'):
                        state_dict.pop(key)
            incompatible_keys = set_peft_model_state_dict(self, state_dict, adapter_name=adapter_name)
            assert len(incompatible_keys.unexpected_keys) == 0
            lora_scales.append(lora_scale)
        set_weights_and_activate_adapters(self, adapter_names=adapter_names, weights=lora_scales)
        if fuse:
            self.fuse_lora(adapter_names=adapter_names)

    def forward(
        self,
        x: torch.Tensor,
        timesteps: torch.Tensor,
        crossattn_emb: torch.Tensor,
        fps: int,
        padding_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Forward pass for the 3D transformer model.

        Args:
            x: Input video tensor.
            timesteps: Diffusion timesteps.
            crossattn_emb: Cross-attention embeddings (e.g., from text encoder).
            fps: Frames per second for rotary embedding.
            padding_mask: Optional mask for padded regions.
        Returns:
            Output tensor after transformer and final projection.
        """
        sp_group = get_sequence_parallel_group()
        # Sequence parallelism for distributed training
        if sp_group is not None:
            x = split_forward_gather_backward(x, dim=2, group=sp_group)
            timesteps = split_forward_gather_backward(timesteps, dim=2, group=sp_group)
        # Optionally concatenate padding mask as a channel
        if self.concat_padding_mask:
            padding_mask = transforms.functional.resize(padding_mask, list(x.shape[-2:]), interpolation=transforms.InterpolationMode.NEAREST)
            padding_mask = padding_mask.unsqueeze(2)
            padding_mask = torch.cat([padding_mask] * x.shape[2], dim=2)
            x = torch.cat([x, padding_mask], dim=1)
        # Patch embedding
        x = self.x_embedder(x)
        # 3D rotary positional embedding
        rope_emb = self.pos_embedder(x, fps=fps)
        # Flatten timesteps for embedding
        timesteps = timesteps.flatten()
        timesteps_emb, adaln_lora = self.t_embedder(timesteps)
        affline_emb = self.affline_norm(timesteps_emb)
        # Pass through transformer blocks
        for name, block in self.blocks.items():
            x = block(
                x,
                affline_emb,
                crossattn_emb,
                rope_emb=rope_emb,
                adaln_lora=adaln_lora,
            )
        # Reshape and project to output
        B, T, H, W = x.shape[:4]
        x = rearrange(x, 'B T H W D -> (B T) (H W) D')
        x = self.final_layer(x, affline_emb, adaln_lora)
        x = rearrange(
            x,
            '(B T) (H W) (p1 p2 t C) -> B C (T t) (H p1) (W p2)',
            B=B,
            H=H,
            W=W,
            p1=self.patch_spatial,
            p2=self.patch_spatial,
            t=self.patch_temporal,
        )
        if sp_group is not None:
            x = gather_forward_split_backward(x, dim=2, group=sp_group)
        return x
