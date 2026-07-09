# Copyright 2026 The LoongForge Authors.
# SPDX-License-Identifier: Apache-2.0

"""Qwen-Image DiT model.

Follows the Wan / Megatron style: the 60 transformer layers are held in
a ``QwenImageTransformerBlock`` (subclass of Megatron ``TransformerBlock``),
which lets ``--recompute-granularity full --recompute-method block
--recompute-num-layers N`` and TP infrastructure flow through unchanged.

Layout convention inside the decoder is [s, b, h] (Megatron standard).
"""

from __future__ import annotations

import math
from typing import Any, Optional

import torch
import torch.nn as nn
from einops import rearrange

from megatron.core.models.common.vision_module.vision_module import VisionModule
from megatron.core.transformer.spec_utils import ModuleSpec

from loongforge.utils import get_args

from .qwen_image_config import QwenImageConfig
from .qwen_image_modules import AdaLayerNorm, RMSNorm, TimestepEmbeddings
from .qwen_image_rope import QwenEmbedLayer3DRope, QwenEmbedRope
from .qwen_image_transformer_block import QwenImageTransformerBlock


class QwenImageModel(VisionModule):
    """Qwen-Image-Edit DiT"""

    def __init__(
        self,
        config: QwenImageConfig,
        transformer_layer_spec: ModuleSpec,
        vocab_size: int = 0,
        max_sequence_length: int = 0,
        pre_process: bool = True,
        post_process: bool = True,
        **kwargs,
    ):
        super().__init__(config=config)
        self.config = config
        self.args = get_args()
        self.pre_process = pre_process
        self.post_process = post_process
        self.hidden_size = config.hidden_size
        self.patch_size = config.patch_size
        self.latent_channels = config.latent_channels
        self.patch_dim = config.patch_dim

        if not config.use_layer3d_rope:
            self.pos_embed = QwenEmbedRope(
                theta=config.rope_theta,
                axes_dim=config.axes_dim,
                scale_rope=config.scale_rope,
            )
        else:
            self.pos_embed = QwenEmbedLayer3DRope(
                theta=config.rope_theta,
                axes_dim=config.axes_dim,
                scale_rope=config.scale_rope,
            )
        self.time_text_embed = TimestepEmbeddings(
            config.time_dim,
            config.hidden_size,
            diffusers_compatible_format=True,
            scale=1000,
            align_dtype_to_timestep=False,
            use_additional_t_cond=config.use_additional_t_cond,
        )
        self.txt_norm = RMSNorm(config.text_dim, eps=config.norm_epsilon)

        # Top-level IO projections stay as plain nn.Linear.
        self.img_in = nn.Linear(config.patch_dim, config.hidden_size)
        self.txt_in = nn.Linear(config.text_dim, config.hidden_size)
        self.proj_out = nn.Linear(config.hidden_size, config.patch_dim)

        # Transformer stack. Using Megatron's TransformerBlock is what
        # makes --recompute-granularity full actually apply.
        self.decoder = QwenImageTransformerBlock(
            config=self.config,
            spec=transformer_layer_spec,
            pre_process=self.pre_process,
            post_process=self.post_process,
            post_layer_norm=False,
        )

        self.norm_out = AdaLayerNorm(config.hidden_size, single=True)

    def set_input_tensor(self, input_tensor):
        """Attach the pipeline-parallel input tensor before the decoder runs."""
        self.input_tensor = input_tensor

    # ------------------------- helpers -------------------------

    def _normalize_latents(self, x: torch.Tensor):
        if x.dim() == 3:
            x = x.unsqueeze(0)
        if x.dim() == 5 and x.shape[0] == 1:
            x = x.squeeze(0)
        if x.dim() != 4:
            raise ValueError(f"expected latent [B,C,H,W], got {tuple(x.shape)}")
        return x

    def _patchify(self, latents: torch.Tensor, height: int, width: int, layer_num: int):
        return rearrange(
            latents,
            "(B N) C (H P) (W Q) -> B (N H W) (C P Q)",
            H=height // 16,
            W=width // 16,
            P=self.patch_size,
            Q=self.patch_size,
            N=layer_num,
        )

    def _unpatchify(self, image: torch.Tensor, height: int, width: int, layer_num: int):
        return rearrange(
            image,
            "B (N H W) (C P Q) -> (B N) C (H P) (W Q)",
            H=height // 16,
            W=width // 16,
            P=self.patch_size,
            Q=self.patch_size,
            B=1,
            N=layer_num,
        )

    @staticmethod
    def _text_lengths(prompt_emb_mask: Optional[torch.Tensor], prompt_emb: torch.Tensor):
        if prompt_emb_mask is None:
            return [prompt_emb.shape[1]] * prompt_emb.shape[0]
        if prompt_emb_mask.dim() == 1:
            prompt_emb_mask = prompt_emb_mask.unsqueeze(0)
        return prompt_emb_mask.sum(dim=1).to(torch.long).tolist()

    # ------------------------- forward -------------------------

    def forward(
        self,
        latents: torch.Tensor,
        timestep: torch.Tensor,
        prompt_emb: torch.Tensor,
        prompt_emb_mask: Optional[torch.Tensor] = None,
        height: Optional[int] = None,
        width: Optional[int] = None,
        edit_latents: Optional[Any] = None,
        layer_num: Optional[int] = None,
        context_latents: Optional[torch.Tensor] = None,
        layer_input_latents: Optional[torch.Tensor] = None,
        edit_rope_interpolation: bool = False,
        zero_cond_t: bool = False,
        **kwargs,
    ):
        """Run the DiT: patchify latents, condition on prompt+timestep, and return the noise prediction."""
        latents = self._normalize_latents(latents)
        if prompt_emb.dim() == 2:
            prompt_emb = prompt_emb.unsqueeze(0)
        if prompt_emb_mask is not None and prompt_emb_mask.dim() == 1:
            prompt_emb_mask = prompt_emb_mask.unsqueeze(0)
        height = int(height or latents.shape[-2] * 8)
        width = int(width or latents.shape[-1] * 8)

        if layer_num is None:
            layer_num = 1
            img_shapes = [(1, latents.shape[2] // 2, latents.shape[3] // 2)]
        else:
            layer_num = layer_num + 1
            img_shapes = [(1, latents.shape[2] // 2, latents.shape[3] // 2)] * layer_num
        txt_seq_lens = self._text_lengths(prompt_emb_mask, prompt_emb)
        timestep = (
            timestep.flatten().to(device=latents.device, dtype=latents.dtype) / 1000
        )

        image = self._patchify(latents, height, width, layer_num)  # [B, S, C']
        image_seq_len = image.shape[1]
        # ``layer_num`` may be incremented below when ``layer_input_latents`` is
        # supplied; ``_unpatchify`` must use the value that matches
        # ``image_seq_len`` (i.e. the layer count that produced the kept tokens).
        output_layer_num = layer_num

        if context_latents is not None:
            context_latents = self._normalize_latents(context_latents)
            img_shapes += [
                (context_latents.shape[0], context_latents.shape[2] // 2, context_latents.shape[3] // 2)
            ]
            context_image = rearrange(
                context_latents,
                "B C (H P) (W Q) -> B (H W) (C P Q)",
                H=context_latents.shape[2] // 2,
                W=context_latents.shape[3] // 2,
                P=self.patch_size,
                Q=self.patch_size,
            )
            image = torch.cat([image, context_image], dim=1)
        if edit_latents is not None:
            edit_latents_list = edit_latents if isinstance(edit_latents, list) else [edit_latents]
            edit_images = []
            for e in edit_latents_list:
                e = self._normalize_latents(e).to(device=latents.device, dtype=latents.dtype)
                img_shapes += [(e.shape[0], e.shape[2] // 2, e.shape[3] // 2)]
                edit_images.append(
                    rearrange(
                        e,
                        "B C (H P) (W Q) -> B (H W) (C P Q)",
                        H=e.shape[2] // 2,
                        W=e.shape[3] // 2,
                        P=self.patch_size,
                        Q=self.patch_size,
                    )
                )
            image = torch.cat([image] + edit_images, dim=1)
        if layer_input_latents is not None:
            layer_num = layer_num + 1
            layer_input_latents = self._normalize_latents(layer_input_latents)
            img_shapes += [
                (
                    layer_input_latents.shape[0],
                    layer_input_latents.shape[2] // 2,
                    layer_input_latents.shape[3] // 2,
                )
            ]
            layer_input_latents = rearrange(
                layer_input_latents,
                "B C (H P) (W Q) -> B (H W) (C P Q)",
                P=self.patch_size,
                Q=self.patch_size,
            )
            image = torch.cat([image, layer_input_latents], dim=1)

        # ---- Trim text tokens to max effective length ----
        # ``txt_seq_lens`` reflects real (unpadded) prompt lengths; ``pos_embed``
        # returns text_freqs of length ``max(txt_seq_lens)``. If the incoming
        # ``prompt_emb`` still carries padding tokens beyond that length, the
        # per-layer RoPE broadcast against text_freqs fails. Crop here so text
        # length and RoPE frequency length always agree.
        max_txt_len = max(txt_seq_lens)
        if prompt_emb.shape[1] > max_txt_len:
            prompt_emb = prompt_emb[:, :max_txt_len]
            if prompt_emb_mask is not None:
                prompt_emb_mask = prompt_emb_mask[:, :max_txt_len]

        # ---- Projections to hidden_size (still [B, S, H]) ----
        image = self.img_in(image)
        text = self.txt_in(self.txt_norm(prompt_emb.to(device=latents.device, dtype=image.dtype)))

        # ---- Time / modulation ----
        if zero_cond_t:
            timestep = torch.cat([timestep, timestep * 0], dim=0)
            modulate_index = torch.tensor(
                [0] * math.prod(img_shapes[0])
                + [1] * sum(math.prod(s) for s in img_shapes[1:]),
                device=timestep.device,
                dtype=torch.int,
            )
        else:
            modulate_index = None
        timestep_mod = self.time_text_embed(
            timestep,
            image.dtype,
            addition_t_cond=None
            if not self.time_text_embed.use_additional_t_cond
            else torch.tensor([0], device=image.device, dtype=torch.long),
        )  # [b (or 2b), h]

        # ---- RoPE ----
        image_freqs, text_freqs = (
            self.pos_embed.forward_sampling(img_shapes, txt_seq_lens, device=latents.device)
            if edit_rope_interpolation
            else self.pos_embed(img_shapes, txt_seq_lens, device=latents.device)
        )

        # ---- [B, S, H] -> [S, B, H] for TransformerBlock ----
        image = image.transpose(0, 1).contiguous()
        text = text.transpose(0, 1).contiguous()

        image = self.decoder(
            hidden_states=image,
            attention_mask=None,
            context=text,
            context_mask=None,
            rotary_pos_emb=(image_freqs, text_freqs),
            packed_seq_params=None,
            inference_context=None,
            timestep_mod=timestep_mod,
            modulate_index=modulate_index,
        )

        # ---- Back to [B, S, H] and head ----
        image = image.transpose(0, 1).contiguous()
        if zero_cond_t:
            conditioning = timestep_mod.chunk(2, dim=0)[0]
        else:
            conditioning = timestep_mod
        image = self.norm_out(image, conditioning)
        image = self.proj_out(image)
        image = image[:, :image_seq_len]
        return self._unpatchify(image, height, width, output_layer_num)
