# Copyright 2026 The LoongForge Authors.
# SPDX-License-Identifier: Apache-2.0
#
# Modified from NVIDIA GR00T under the Apache-2.0 License.
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Qwen3-VL backbone helpers for GR00T-N1.7."""

import inspect
import logging
import os
from dataclasses import dataclass

from huggingface_hub.errors import GatedRepoError
import torch
import transformers
from transformers.feature_extraction_utils import BatchFeature

logger = logging.getLogger(__name__)


def _transformers_major_version() -> int:
    """Return the installed Transformers major version."""
    return int(transformers.__version__.split(".", 1)[0])


try:
    from transformers import Qwen3VLForConditionalGeneration

    _QWEN3VL_AVAILABLE = True
except ImportError:
    _QWEN3VL_AVAILABLE = False


_GATED_BACKBONE_HINT = (
    "Cannot download the VLM backbone '{model_name}', which is a gated Hugging "
    "Face repo. Every GR00T checkpoint loads this backbone, so both inference "
    "and finetuning require access to it."
)

_GATED_MARKERS = ("gated repo", "is restricted", "access to model", "401 client error")


@dataclass(frozen=True)
class _Qwen3VLVisionGraphMetadata:
    grid_signature: tuple[tuple[int, int, int], ...]
    interpolation_indices: torch.Tensor
    interpolation_weights: torch.Tensor
    rotary_position_ids: torch.Tensor
    cu_seqlens: torch.Tensor
    max_seqlen: int
    max_hw: int
    image_split_sizes: tuple[int, ...]

    @property
    def image_token_count(self) -> int:
        """Return the total vision token count across all images."""
        return sum(self.image_split_sizes)


def _coerce_torch_dtype(value) -> torch.dtype | None:
    if isinstance(value, torch.dtype):
        return value
    if isinstance(value, str):
        normalized = value.lower().replace("torch.", "")
        if normalized in {"bf16", "bfloat16"}:
            return torch.bfloat16
        if normalized in {"fp16", "float16", "half"}:
            return torch.float16
        if normalized in {"fp32", "float32", "float"}:
            return torch.float32
    return None


def _is_qwen3vl_flash_attention_module(module: torch.nn.Module) -> bool:
    try:
        config = module.config
    except AttributeError:
        return False
    if config._attn_implementation not in {"flash_attention_2", "flash_attention_3"}:
        return False

    module_type = type(module)
    return "qwen3_vl" in module_type.__module__ and module_type.__name__.startswith("Qwen3VL")


def _qwen3vl_flash_attention_target_dtype(module: torch.nn.Module) -> torch.dtype:
    config = module.config
    for dtype_value in (
        getattr(config, "_pre_quantization_dtype", None),
        getattr(config, "dtype", None),
    ):
        dtype = _coerce_torch_dtype(dtype_value)
        if dtype in {torch.float16, torch.bfloat16}:
            return dtype

    # Cosmos-Reason2 advertises bf16 in text_config.dtype.  In transformers
    # 4.57 this was also the dtype used to cast fp32 q/k/v before FA2.
    return torch.bfloat16


def _patch_qwen3vl_flash_attention_target_dtype() -> None:
    try:
        from transformers.integrations import flash_attention
    except ImportError:
        return

    try:
        original_get_target_dtype = flash_attention.get_target_dtype
    except AttributeError:
        return
    if original_get_target_dtype.__dict__.get("_loongforge_qwen3vl_compat", False):
        return

    def get_target_dtype_compat(query: torch.Tensor, module: torch.nn.Module) -> torch.dtype | None:
        if query.dtype == torch.float32 and _is_qwen3vl_flash_attention_module(module):
            return _qwen3vl_flash_attention_target_dtype(module)
        return original_get_target_dtype(query, module)

    get_target_dtype_compat._loongforge_qwen3vl_compat = True
    get_target_dtype_compat._loongforge_original = original_get_target_dtype
    flash_attention.get_target_dtype = get_target_dtype_compat


def _patch_qwen3_vl_fused_vision_rope() -> bool:
    try:
        from transformers.models.qwen3_vl import modeling_qwen3_vl as qwen_mod
    except ImportError:
        return False
    original = qwen_mod.apply_rotary_pos_emb_vision
    if original.__dict__.get("_loongforge_qwen3_vl_fused_vision_rope", False):
        return False

    def apply_rotary_pos_emb_vision_compat(query, key, cos, sin):
        can_fuse = (
            query.is_cuda
            and key.is_cuda
            and cos.is_cuda
            and sin.is_cuda
            and not any(tensor.requires_grad for tensor in (query, key, cos, sin))
            and query.ndim == 3
            and key.shape == query.shape
            and cos.ndim == 2
            and sin.shape == cos.shape
            and cos.dtype == torch.float32
            and sin.dtype == torch.float32
            and cos.shape == (query.shape[0], query.shape[2])
        )
        if not can_fuse:
            return original(query, key, cos, sin)
        from loongforge.embodied.model.groot_n1_7.qwen3_vl_fused_ops import (
            qwen3_vl_fused_vision_rope_forward,
        )

        return qwen3_vl_fused_vision_rope_forward(query, key, cos, sin)

    apply_rotary_pos_emb_vision_compat._loongforge_qwen3_vl_fused_vision_rope = True
    apply_rotary_pos_emb_vision_compat._loongforge_original = original
    qwen_mod.apply_rotary_pos_emb_vision = apply_rotary_pos_emb_vision_compat
    return True


def _patch_qwen3_vl_fused_text_rope() -> bool:
    try:
        from transformers.models.qwen3_vl import modeling_qwen3_vl as qwen_mod
    except ImportError:
        return False
    original = qwen_mod.apply_rotary_pos_emb
    if original.__dict__.get("_loongforge_qwen3_vl_fused_text_rope", False):
        return False

    def apply_rotary_pos_emb_compat(query, key, cos, sin, unsqueeze_dim=1):
        can_fuse = (
            unsqueeze_dim == 1
            and query.is_cuda
            and key.is_cuda
            and cos.is_cuda
            and sin.is_cuda
            and not any(tensor.requires_grad for tensor in (query, key, cos, sin))
            and query.ndim == 4
            and key.ndim == 4
            and query.shape[0] == key.shape[0]
            and query.shape[2:] == key.shape[2:]
            and cos.ndim == 3
            and sin.shape == cos.shape
            and cos.shape == (query.shape[0], query.shape[2], query.shape[3])
            and query.dtype == key.dtype == cos.dtype == sin.dtype
        )
        if not can_fuse:
            return original(query, key, cos, sin, unsqueeze_dim=unsqueeze_dim)
        from loongforge.embodied.model.groot_n1_7.qwen3_vl_fused_ops import (
            qwen3_vl_fused_text_rope_forward,
        )

        return qwen3_vl_fused_text_rope_forward(query, key, cos, sin)

    apply_rotary_pos_emb_compat._loongforge_qwen3_vl_fused_text_rope = True
    apply_rotary_pos_emb_compat._loongforge_original = original
    qwen_mod.apply_rotary_pos_emb = apply_rotary_pos_emb_compat
    return True


def _patch_qwen3_vl_fused_text_rms_norm() -> bool:
    try:
        from transformers.models.qwen3_vl import modeling_qwen3_vl as qwen_mod
    except ImportError:
        return False

    norm_cls = getattr(qwen_mod, "Qwen3VLTextRMSNorm", None)
    if norm_cls is None:
        return False
    original = norm_cls.forward
    if original.__dict__.get("_loongforge_qwen3_vl_fused_text_rms_norm", False):
        return False

    def forward_compat(self, hidden_states):
        can_fuse = (
            hidden_states.is_cuda
            and hidden_states.is_contiguous()
            and hidden_states.dtype in {torch.bfloat16, torch.float16}
            and self.weight.is_cuda
            and self.weight.is_contiguous()
            and self.weight.dtype == torch.float32
            and self.weight.device == hidden_states.device
            and self.weight.numel() == hidden_states.shape[-1]
            and not hidden_states.requires_grad
            and not self.weight.requires_grad
        )
        if not can_fuse:
            return original(self, hidden_states)
        from loongforge.embodied.model.groot_n1_7.qwen3_vl_fused_ops import (
            qwen3_vl_fused_text_rms_norm_forward,
        )

        return qwen3_vl_fused_text_rms_norm_forward(
            hidden_states,
            self.weight,
            self.variance_epsilon,
        )

    forward_compat._loongforge_qwen3_vl_fused_text_rms_norm = True
    forward_compat._loongforge_original = original
    norm_cls.forward = forward_compat
    return True


def _patch_qwen3_vl_fused_text_silu_mul() -> bool:
    try:
        from transformers.models.qwen3_vl import modeling_qwen3_vl as qwen_mod
    except ImportError:
        return False

    mlp_cls = getattr(qwen_mod, "Qwen3VLTextMLP", None)
    if mlp_cls is None:
        return False
    original = mlp_cls.forward
    if original.__dict__.get("_loongforge_qwen3_vl_fused_text_silu_mul", False):
        return False

    def forward_compat(self, hidden_states):
        can_fuse = (
            hidden_states.is_cuda
            and hidden_states.dtype in {torch.bfloat16, torch.float16, torch.float32}
            and not hidden_states.requires_grad
            and not any(parameter.requires_grad for parameter in self.parameters())
        )
        if not can_fuse:
            return original(self, hidden_states)
        gate = self.gate_proj(hidden_states)
        up = self.up_proj(hidden_states)
        if (
            gate.dtype not in {torch.bfloat16, torch.float16}
            or gate.dtype != up.dtype
            or not gate.is_contiguous()
            or not up.is_contiguous()
        ):
            return self.down_proj(self.act_fn(gate) * up)
        from loongforge.embodied.model.groot_n1_7.qwen3_vl_fused_ops import (
            qwen3_vl_fused_text_silu_mul_forward,
        )

        return self.down_proj(qwen3_vl_fused_text_silu_mul_forward(gate, up))

    forward_compat._loongforge_qwen3_vl_fused_text_silu_mul = True
    forward_compat._loongforge_original = original
    mlp_cls.forward = forward_compat
    return True


def _patch_qwen3vl_output_projection_dtypes(model: torch.nn.Module) -> int:
    try:
        from transformers.models.qwen3_vl import modeling_qwen3_vl as qwen_mod
    except ImportError:
        return 0

    patched = 0
    try:
        text_attention_cls = qwen_mod.Qwen3VLTextAttention
    except AttributeError:
        text_attention_cls = ()
    try:
        vision_attention_cls = qwen_mod.Qwen3VLVisionAttention
    except AttributeError:
        vision_attention_cls = ()

    for module in model.modules():
        proj = None
        if isinstance(module, text_attention_cls):
            proj = module.o_proj
        elif isinstance(module, vision_attention_cls):
            proj = module.proj

        if proj is None or not isinstance(proj, torch.nn.Linear):
            continue
        if proj.__dict__.get("_loongforge_qwen3vl_dtype_patched", False):
            continue

        original_forward = proj.forward

        def forward_compat(input, *args, _original_forward=original_forward, _proj=proj, **kwargs):
            target_dtype = _proj.weight.dtype
            if input.dtype != target_dtype:
                input = input.to(target_dtype)
            return _original_forward(input, *args, **kwargs)

        forward_compat._loongforge_qwen3vl_dtype_patched = True
        proj.forward = forward_compat
        proj._loongforge_qwen3vl_dtype_patched = True
        patched += 1

    if patched:
        logger.info("Patched %d Qwen3-VL output projection(s) to preserve fp32 residual dtype.", patched)
    return patched


def _patch_qwen3vl_skip_unused_lm_head() -> bool:
    """Allow GR00T to call Qwen top-level forward without unused logits projection."""
    try:
        from transformers.models.qwen3_vl import modeling_qwen3_vl as qwen_mod
    except ImportError:
        return False

    cls = getattr(qwen_mod, "Qwen3VLForConditionalGeneration", None)
    output_cls = getattr(qwen_mod, "Qwen3VLCausalLMOutputWithPast", None)
    if cls is None or output_cls is None:
        return False
    if cls.forward.__dict__.get("_loongforge_skip_lm_head_compat", False):
        return False

    original_forward = cls.forward

    def forward_compat(
        self,
        input_ids: torch.LongTensor = None,
        attention_mask: torch.Tensor | None = None,
        position_ids: torch.LongTensor | None = None,
        past_key_values=None,
        inputs_embeds: torch.FloatTensor | None = None,
        labels: torch.LongTensor | None = None,
        pixel_values: torch.Tensor | None = None,
        pixel_values_videos: torch.FloatTensor | None = None,
        image_grid_thw: torch.LongTensor | None = None,
        video_grid_thw: torch.LongTensor | None = None,
        mm_token_type_ids: torch.IntTensor | None = None,
        cache_position: torch.LongTensor | None = None,
        logits_to_keep=0,
        loongforge_skip_lm_head: bool = False,
        **kwargs,
    ):
        if not loongforge_skip_lm_head or labels is not None:
            return original_forward(
                self,
                input_ids=input_ids,
                attention_mask=attention_mask,
                position_ids=position_ids,
                past_key_values=past_key_values,
                inputs_embeds=inputs_embeds,
                labels=labels,
                pixel_values=pixel_values,
                pixel_values_videos=pixel_values_videos,
                image_grid_thw=image_grid_thw,
                video_grid_thw=video_grid_thw,
                mm_token_type_ids=mm_token_type_ids,
                cache_position=cache_position,
                logits_to_keep=logits_to_keep,
                **kwargs,
            )

        outputs = self.model(
            input_ids=input_ids,
            pixel_values=pixel_values,
            pixel_values_videos=pixel_values_videos,
            image_grid_thw=image_grid_thw,
            video_grid_thw=video_grid_thw,
            position_ids=position_ids,
            attention_mask=attention_mask,
            past_key_values=past_key_values,
            inputs_embeds=inputs_embeds,
            cache_position=cache_position,
            mm_token_type_ids=mm_token_type_ids,
            **kwargs,
        )
        hidden_states = outputs[0]
        logits = hidden_states.new_empty((*hidden_states.shape[:-1], 0))
        return output_cls(
            loss=None,
            logits=logits,
            past_key_values=getattr(outputs, "past_key_values", None),
            hidden_states=(hidden_states,),
            attentions=getattr(outputs, "attentions", None),
            rope_deltas=getattr(outputs, "rope_deltas", None),
        )

    forward_compat._loongforge_skip_lm_head_compat = True
    forward_compat._loongforge_original = original_forward
    cls.forward = forward_compat
    return True


def _lookup_qwen3vl_vision_graph_metadata(module, grid_thw):
    if grid_thw is None:
        return None
    by_pointer = module.__dict__.get("_loongforge_cuda_graph_vision_metadata_by_pointer")
    if by_pointer is None:
        return None
    return by_pointer.get(grid_thw.data_ptr())


def _patch_qwen3vl_cuda_graph_vision_metadata() -> bool:
    """Use precomputed grid metadata while retaining vision tensor work in the graph."""
    try:
        from transformers.models.qwen3_vl import modeling_qwen3_vl as qwen_mod
    except ImportError:
        return False

    visual_cls = getattr(qwen_mod, "Qwen3VLVisionModel", None)
    attention_cls = getattr(qwen_mod, "Qwen3VLVisionAttention", None)
    text_cls = getattr(qwen_mod, "Qwen3VLTextModel", None)
    model_cls = getattr(qwen_mod, "Qwen3VLModel", None)
    if visual_cls is None or attention_cls is None or text_cls is None or model_cls is None:
        return False
    if visual_cls.fast_pos_embed_interpolate.__dict__.get(
        "_loongforge_cuda_graph_metadata_compat", False
    ):
        return False

    original_fast_pos_embed_interpolate = visual_cls.fast_pos_embed_interpolate
    original_rot_pos_emb = visual_cls.rot_pos_emb
    original_visual_forward = visual_cls.forward
    original_attention_forward = attention_cls.forward
    original_deepstack_process = text_cls._deepstack_process
    original_model_forward = model_cls.forward
    original_get_image_features = model_cls.get_image_features
    original_get_placeholder_mask = model_cls.get_placeholder_mask

    def fast_pos_embed_interpolate_compat(self, grid_thw):
        metadata = _lookup_qwen3vl_vision_graph_metadata(self, grid_thw)
        if metadata is None:
            return original_fast_pos_embed_interpolate(self, grid_thw)

        pos_embeds = self.pos_embed(metadata.interpolation_indices)
        pos_embeds = pos_embeds * metadata.interpolation_weights[:, :, None]
        patch_pos_embeds = pos_embeds[0] + pos_embeds[1] + pos_embeds[2] + pos_embeds[3]
        patch_pos_embeds = patch_pos_embeds.split(
            [height * width for _, height, width in metadata.grid_signature]
        )

        merge_size = self.config.spatial_merge_size
        permuted = []
        for pos_embed, (num_frames, height, width) in zip(
            patch_pos_embeds, metadata.grid_signature
        ):
            pos_embed = pos_embed.repeat(num_frames, 1)
            pos_embed = (
                pos_embed.view(
                    num_frames,
                    height // merge_size,
                    merge_size,
                    width // merge_size,
                    merge_size,
                    -1,
                )
                .permute(0, 1, 3, 2, 4, 5)
                .flatten(0, 4)
            )
            permuted.append(pos_embed)
        return torch.cat(permuted)

    def rot_pos_emb_compat(self, grid_thw):
        metadata = _lookup_qwen3vl_vision_graph_metadata(self, grid_thw)
        if metadata is None:
            return original_rot_pos_emb(self, grid_thw)
        freq_table = self.rotary_pos_emb(metadata.max_hw)
        return freq_table[metadata.rotary_position_ids].flatten(1)

    def visual_forward_compat(self, hidden_states, grid_thw, **kwargs):
        metadata = _lookup_qwen3vl_vision_graph_metadata(self, grid_thw)
        if metadata is None:
            return original_visual_forward(self, hidden_states, grid_thw, **kwargs)

        kwargs.pop("return_dict", None)
        hidden_states = self.patch_embed(hidden_states)
        hidden_states = hidden_states + self.fast_pos_embed_interpolate(grid_thw)
        rotary_pos_emb = self.rot_pos_emb(grid_thw)

        sequence_length, _ = hidden_states.size()
        hidden_states = hidden_states.reshape(sequence_length, -1)
        rotary_pos_emb = rotary_pos_emb.reshape(sequence_length, -1)
        rotary_embedding = torch.cat((rotary_pos_emb, rotary_pos_emb), dim=-1)
        position_embeddings = (rotary_embedding.cos(), rotary_embedding.sin())

        deepstack_features = []
        for layer_number, block in enumerate(self.blocks):
            hidden_states = block(
                hidden_states,
                cu_seqlens=metadata.cu_seqlens,
                position_embeddings=position_embeddings,
                loongforge_vision_max_seqlen=metadata.max_seqlen,
                **kwargs,
            )
            if layer_number in self.deepstack_visual_indexes:
                merger_index = self.deepstack_visual_indexes.index(layer_number)
                deepstack_features.append(self.deepstack_merger_list[merger_index](hidden_states))

        return qwen_mod.BaseModelOutputWithDeepstackFeatures(
            last_hidden_state=hidden_states,
            pooler_output=self.merger(hidden_states),
            deepstack_features=deepstack_features,
        )

    def attention_forward_compat(
        self,
        hidden_states,
        cu_seqlens,
        rotary_pos_emb=None,
        position_embeddings=None,
        **kwargs,
    ):
        max_seqlen = kwargs.pop("loongforge_vision_max_seqlen", None)
        if max_seqlen is None:
            return original_attention_forward(
                self,
                hidden_states,
                cu_seqlens,
                rotary_pos_emb=rotary_pos_emb,
                position_embeddings=position_embeddings,
                **kwargs,
            )
        if not qwen_mod.is_flash_attention_requested(self.config):
            raise RuntimeError(
                "Cached Qwen3-VL vision metadata currently requires flash attention."
            )

        sequence_length = hidden_states.shape[0]
        query_states, key_states, value_states = (
            self.qkv(hidden_states)
            .reshape(sequence_length, 3, self.num_heads, -1)
            .permute(1, 0, 2, 3)
            .unbind(0)
        )
        cos, sin = position_embeddings
        query_states, key_states = qwen_mod.apply_rotary_pos_emb_vision(
            query_states,
            key_states,
            cos,
            sin,
        )
        query_states = query_states.transpose(0, 1).unsqueeze(0)
        key_states = key_states.transpose(0, 1).unsqueeze(0)
        value_states = value_states.transpose(0, 1).unsqueeze(0)
        attention_interface = qwen_mod.ALL_ATTENTION_FUNCTIONS.get_interface(
            self.config._attn_implementation,
            qwen_mod.eager_attention_forward,
        )
        attention_output, _ = attention_interface(
            self,
            query_states,
            key_states,
            value_states,
            attention_mask=None,
            scaling=self.scaling,
            dropout=0.0 if not self.training else self.attention_dropout,
            cu_seq_lens_q=cu_seqlens,
            cu_seq_lens_k=cu_seqlens,
            max_length_q=max_seqlen,
            max_length_k=max_seqlen,
            is_causal=False,
            **kwargs,
        )
        return self.proj(attention_output.reshape(sequence_length, -1).contiguous())

    def get_image_features_compat(self, pixel_values, image_grid_thw=None, **kwargs):
        metadata = _lookup_qwen3vl_vision_graph_metadata(self.visual, image_grid_thw)
        if metadata is None:
            return original_get_image_features(
                self,
                pixel_values,
                image_grid_thw=image_grid_thw,
                **kwargs,
            )

        kwargs.pop("return_dict", None)
        pixel_values = pixel_values.type(self.visual.dtype)
        vision_output = self.visual(
            pixel_values,
            grid_thw=image_grid_thw,
            return_dict=True,
            **kwargs,
        )
        vision_output.pooler_output = torch.split(
            vision_output.pooler_output,
            metadata.image_split_sizes,
        )
        return vision_output

    def get_placeholder_mask_compat(
        self,
        input_ids,
        inputs_embeds,
        image_features=None,
        video_features=None,
    ):
        by_pointer = self.__dict__.get("_loongforge_cuda_graph_image_tokens_by_pointer")
        expected_image_tokens = None if by_pointer is None or input_ids is None else by_pointer.get(
            input_ids.data_ptr()
        )
        if expected_image_tokens is None:
            return original_get_placeholder_mask(
                self,
                input_ids,
                inputs_embeds,
                image_features=image_features,
                video_features=video_features,
            )
        if video_features is not None:
            raise RuntimeError("GR00T-N1.7 full CUDA graph does not support Qwen video features.")

        special_image_mask = input_ids == self.config.image_token_id
        special_video_mask = input_ids == self.config.video_token_id
        if image_features is not None:
            expected_numel = expected_image_tokens * inputs_embeds.shape[-1]
            if image_features.numel() != expected_numel:
                raise RuntimeError(
                    "Qwen3-VL cached image feature size changed after graph preparation: "
                    f"expected {expected_numel}, got {image_features.numel()}."
                )
        special_image_mask = special_image_mask.unsqueeze(-1).expand_as(inputs_embeds)
        special_video_mask = special_video_mask.unsqueeze(-1).expand_as(inputs_embeds)
        return special_image_mask, special_video_mask

    def deepstack_process_compat(
        self,
        hidden_states,
        visual_pos_masks,
        visual_embeds,
    ):
        visual_indices = self.__dict__.get("_loongforge_cuda_graph_visual_token_indices")
        if visual_indices is None:
            return original_deepstack_process(
                self,
                hidden_states,
                visual_pos_masks,
                visual_embeds,
            )
        visual_embeds = visual_embeds.to(hidden_states.device, hidden_states.dtype)
        flat_hidden_states = hidden_states.reshape(-1, hidden_states.shape[-1])
        output = flat_hidden_states.clone()
        updates = flat_hidden_states.index_select(0, visual_indices) + visual_embeds
        output.index_copy_(0, visual_indices, updates)
        return output.view_as(hidden_states)

    def model_forward_compat(
        self,
        input_ids=None,
        attention_mask=None,
        position_ids=None,
        past_key_values=None,
        inputs_embeds=None,
        pixel_values=None,
        pixel_values_videos=None,
        image_grid_thw=None,
        video_grid_thw=None,
        mm_token_type_ids=None,
        cache_position=None,
        **kwargs,
    ):
        masks_by_pointer = self.__dict__.get(
            "_loongforge_cuda_graph_visual_pos_masks_by_pointer", {}
        )
        visual_pos_mask = (
            None
            if input_ids is None
            else masks_by_pointer.get(input_ids.data_ptr())
        )
        visual_indices = getattr(
            self.language_model,
            "_loongforge_cuda_graph_visual_token_indices",
            None,
        )
        if (
            visual_pos_mask is None
            or visual_indices is None
            or pixel_values is None
            or pixel_values_videos is not None
            or inputs_embeds is not None
            or position_ids is None
        ):
            return original_model_forward(
                self,
                input_ids=input_ids,
                attention_mask=attention_mask,
                position_ids=position_ids,
                past_key_values=past_key_values,
                inputs_embeds=inputs_embeds,
                pixel_values=pixel_values,
                pixel_values_videos=pixel_values_videos,
                image_grid_thw=image_grid_thw,
                video_grid_thw=video_grid_thw,
                mm_token_type_ids=mm_token_type_ids,
                cache_position=cache_position,
                **kwargs,
            )

        inputs_embeds = self.get_input_embeddings()(input_ids)
        image_outputs = self.get_image_features(
            pixel_values,
            image_grid_thw,
            return_dict=True,
        )
        image_embeds = torch.cat(image_outputs.pooler_output, dim=0).to(
            inputs_embeds.device,
            inputs_embeds.dtype,
        )
        flat_inputs = inputs_embeds.reshape(-1, inputs_embeds.shape[-1])
        flat_inputs.index_copy_(0, visual_indices, image_embeds)

        outputs = self.language_model(
            input_ids=None,
            position_ids=position_ids,
            attention_mask=attention_mask,
            past_key_values=past_key_values,
            inputs_embeds=inputs_embeds,
            cache_position=cache_position,
            visual_pos_masks=visual_pos_mask,
            deepstack_visual_embeds=image_outputs.deepstack_features,
            **kwargs,
        )
        return qwen_mod.Qwen3VLModelOutputWithPast(
            **outputs,
            rope_deltas=self.rope_deltas,
        )

    fast_pos_embed_interpolate_compat._loongforge_cuda_graph_metadata_compat = True
    fast_pos_embed_interpolate_compat._loongforge_original = original_fast_pos_embed_interpolate
    rot_pos_emb_compat._loongforge_cuda_graph_metadata_compat = True
    rot_pos_emb_compat._loongforge_original = original_rot_pos_emb
    visual_forward_compat._loongforge_cuda_graph_metadata_compat = True
    visual_forward_compat._loongforge_original = original_visual_forward
    attention_forward_compat._loongforge_cuda_graph_metadata_compat = True
    attention_forward_compat._loongforge_original = original_attention_forward
    get_image_features_compat._loongforge_cuda_graph_metadata_compat = True
    get_image_features_compat._loongforge_original = original_get_image_features
    get_placeholder_mask_compat._loongforge_cuda_graph_metadata_compat = True
    get_placeholder_mask_compat._loongforge_original = original_get_placeholder_mask
    deepstack_process_compat._loongforge_cuda_graph_metadata_compat = True
    deepstack_process_compat._loongforge_original = original_deepstack_process
    model_forward_compat._loongforge_cuda_graph_metadata_compat = True
    model_forward_compat._loongforge_original = original_model_forward
    visual_cls.fast_pos_embed_interpolate = fast_pos_embed_interpolate_compat
    visual_cls.rot_pos_emb = rot_pos_emb_compat
    visual_cls.forward = visual_forward_compat
    attention_cls.forward = attention_forward_compat
    model_cls.get_image_features = get_image_features_compat
    model_cls.get_placeholder_mask = get_placeholder_mask_compat
    model_cls.forward = model_forward_compat
    text_cls._deepstack_process = deepstack_process_compat
    return True


def _qwen3vl_grid_signature(grid_thw: torch.Tensor) -> tuple[tuple[int, int, int], ...]:
    if grid_thw.ndim != 2 or grid_thw.shape[1] != 3:
        raise RuntimeError(
            "Qwen3-VL CUDA graph requires image_grid_thw with shape [N, 3], "
            f"got {tuple(grid_thw.shape)}."
        )
    return tuple(
        (int(row[0]), int(row[1]), int(row[2]))
        for row in grid_thw.detach().cpu().tolist()
    )


def _cpu_tensor_content_key(tensor: torch.Tensor):
    """Build an exact, CPU-only cache key for a fetched metadata tensor."""
    if tensor.device.type != "cpu":
        return None
    contiguous = tensor.detach().contiguous()
    return (tuple(contiguous.shape), str(contiguous.dtype), contiguous.numpy().tobytes())


def _build_qwen3vl_vision_graph_metadata(
    visual: torch.nn.Module,
    grid_signature: tuple[tuple[int, int, int], ...],
) -> _Qwen3VLVisionGraphMetadata:
    if not grid_signature:
        raise RuntimeError("Qwen3-VL CUDA graph received an empty image_grid_thw tensor.")

    merge_size = int(visual.spatial_merge_size)
    interpolation_indices: list[list[int]] = [[] for _ in range(4)]
    interpolation_weights: list[list[float]] = [[] for _ in range(4)]
    rotary_ids = []

    for num_frames, height, width in grid_signature:
        if min(num_frames, height, width) <= 0:
            raise RuntimeError(
                "Qwen3-VL CUDA graph requires positive image grid dimensions, "
                f"got {(num_frames, height, width)}."
            )
        if height % merge_size or width % merge_size:
            raise RuntimeError(
                "Qwen3-VL image grid must be divisible by spatial_merge_size, "
                f"got grid={(num_frames, height, width)} merge_size={merge_size}."
            )

        height_positions = torch.linspace(0, visual.num_grid_per_side - 1, height)
        width_positions = torch.linspace(0, visual.num_grid_per_side - 1, width)
        height_floor = height_positions.int()
        width_floor = width_positions.int()
        height_ceil = (height_floor + 1).clip(max=visual.num_grid_per_side - 1)
        width_ceil = (width_floor + 1).clip(max=visual.num_grid_per_side - 1)
        height_delta = height_positions - height_floor
        width_delta = width_positions - width_floor
        base_height = height_floor * visual.num_grid_per_side
        base_height_ceil = height_ceil * visual.num_grid_per_side

        indices = (
            (base_height[:, None] + width_floor[None]).flatten(),
            (base_height[:, None] + width_ceil[None]).flatten(),
            (base_height_ceil[:, None] + width_floor[None]).flatten(),
            (base_height_ceil[:, None] + width_ceil[None]).flatten(),
        )
        weights = (
            ((1 - height_delta)[:, None] * (1 - width_delta)[None]).flatten(),
            ((1 - height_delta)[:, None] * width_delta[None]).flatten(),
            (height_delta[:, None] * (1 - width_delta)[None]).flatten(),
            (height_delta[:, None] * width_delta[None]).flatten(),
        )
        for index in range(4):
            interpolation_indices[index].extend(indices[index].tolist())
            interpolation_weights[index].extend(weights[index].tolist())

        merged_height = height // merge_size
        merged_width = width // merge_size
        block_rows = torch.arange(merged_height)
        block_cols = torch.arange(merged_width)
        intra_rows = torch.arange(merge_size)
        intra_cols = torch.arange(merge_size)
        row_ids = block_rows[:, None, None, None] * merge_size + intra_rows[None, None, :, None]
        col_ids = block_cols[None, :, None, None] * merge_size + intra_cols[None, None, None, :]
        row_ids = row_ids.expand(merged_height, merged_width, merge_size, merge_size).reshape(-1)
        col_ids = col_ids.expand(merged_height, merged_width, merge_size, merge_size).reshape(-1)
        image_rotary_ids = torch.stack((row_ids, col_ids), dim=-1)
        if num_frames > 1:
            image_rotary_ids = image_rotary_ids.repeat(num_frames, 1)
        rotary_ids.append(image_rotary_ids)

    device = visual.pos_embed.weight.device
    dtype = visual.pos_embed.weight.dtype
    vision_lengths = [
        height * width
        for num_frames, height, width in grid_signature
        for _ in range(num_frames)
    ]
    cu_seqlens = torch.tensor(
        [0, *torch.tensor(vision_lengths, dtype=torch.int64).cumsum(0).tolist()],
        dtype=torch.int32,
        device=device,
    )
    return _Qwen3VLVisionGraphMetadata(
        grid_signature=grid_signature,
        interpolation_indices=torch.tensor(
            interpolation_indices,
            dtype=torch.long,
            device=device,
        ),
        interpolation_weights=torch.tensor(
            interpolation_weights,
            dtype=dtype,
            device=device,
        ),
        rotary_position_ids=torch.cat(rotary_ids).to(device=device),
        cu_seqlens=cu_seqlens,
        max_seqlen=max(vision_lengths),
        max_hw=max(max(height, width) for _, height, width in grid_signature),
        image_split_sizes=tuple(
            num_frames * height * width // merge_size**2
            for num_frames, height, width in grid_signature
        ),
    )


def _is_gated_repo_error(exc: BaseException) -> bool:
    current: BaseException | None = exc
    for _ in range(10):
        if current is None:
            break
        if isinstance(current, GatedRepoError) or any(marker in str(current).lower() for marker in _GATED_MARKERS):
            return True
        current = current.__cause__ or current.__context__
    return False


def _real_inference_device(module: torch.nn.Module) -> torch.device:
    for tensor in module.parameters():
        if tensor.device.type != "meta":
            return tensor.device
    for tensor in module.buffers():
        if tensor.device.type != "meta":
            return tensor.device
    return torch.device("cpu")


def _force_tie_qwen_lm_head(model: torch.nn.Module) -> bool:
    """Restore the Qwen3-VL tied lm_head behavior used by transformers 4.57.x."""
    try:
        lm_head = model.lm_head
    except AttributeError:
        return False
    try:
        lm_head.weight
    except AttributeError:
        return False

    backbone = _unwrap_qwen_backbone(model)
    language_model = backbone.language_model
    try:
        embed_tokens = language_model.embed_tokens
    except AttributeError:
        return False

    try:
        embed_tokens.weight
    except AttributeError:
        return False

    if lm_head.weight is embed_tokens.weight:
        return False

    if lm_head.weight.shape != embed_tokens.weight.shape:
        logger.warning(
            "Cannot tie Qwen3-VL lm_head to input embeddings: shape mismatch %s vs %s.",
            tuple(lm_head.weight.shape),
            tuple(embed_tokens.weight.shape),
        )
        return False

    lm_head.weight = embed_tokens.weight
    tied_keys = model.__dict__.get("all_tied_weights_keys")
    if isinstance(tied_keys, dict):
        tied_keys["lm_head.weight"] = "model.language_model.embed_tokens.weight"
    return True


def _unwrap_qwen_backbone(model: torch.nn.Module) -> torch.nn.Module:
    try:
        return model.model
    except AttributeError:
        return model


def _forward_has_explicit_arg(model: torch.nn.Module, arg_name: str) -> bool:
    try:
        return arg_name in inspect.signature(model.forward).parameters
    except (TypeError, ValueError):
        return False


def _build_mm_token_type_ids(config, input_ids: torch.Tensor) -> torch.Tensor | None:
    token_type_ids = torch.zeros(input_ids.shape, dtype=torch.int32, device=input_ids.device)

    image_token_id = config.image_token_id
    if image_token_id is not None:
        image_mask = input_ids == image_token_id
        token_type_ids = torch.where(
            image_mask,
            torch.ones((), dtype=token_type_ids.dtype, device=token_type_ids.device),
            token_type_ids,
        )

    video_token_id = config.video_token_id
    if video_token_id is not None:
        video_mask = input_ids == video_token_id
        token_type_ids = torch.where(
            video_mask,
            torch.full((), 2, dtype=token_type_ids.dtype, device=token_type_ids.device),
            token_type_ids,
        )

    return token_type_ids


def _build_qwen3vl_compat_position_ids(
    model: torch.nn.Module,
    input_ids: torch.Tensor,
    mm_token_type_ids: torch.Tensor,
    image_grid_thw: torch.Tensor,
    attention_mask: torch.Tensor,
) -> torch.Tensor | None:
    qwen_model = _unwrap_qwen_backbone(model)
    try:
        get_rope_index = qwen_model.get_rope_index
    except AttributeError:
        return None

    try:
        position_ids, rope_deltas = get_rope_index(
            input_ids=input_ids,
            mm_token_type_ids=mm_token_type_ids,
            image_grid_thw=image_grid_thw,
            attention_mask=attention_mask,
        )
    except TypeError:
        return None

    if "rope_deltas" in qwen_model.__dict__:
        qwen_model.rope_deltas = rope_deltas
    return position_ids



def recompute_vision_rotary_inv_freq(
    rotary: torch.nn.Module,
    head_dim_half: int,
    device: torch.device,
) -> torch.Tensor:
    """Rebuild the vision rotary inverse frequency tensor on the target device."""
    with torch.device(device):
        fresh = type(rotary)(head_dim_half)
    return fresh.inv_freq.detach().to(device=device, dtype=torch.float32)


def recompute_text_rotary_inv_freq(
    rotary: torch.nn.Module,
    config,
    device: torch.device,
) -> tuple[torch.Tensor, float]:
    """Rebuild the text rotary inverse frequency tensor and scaling value."""
    with torch.device(device):
        fresh = type(rotary)(config=config, device=device)
    inv_freq = fresh.inv_freq.detach().to(device=device, dtype=torch.float32)
    try:
        attention_scaling = float(fresh.attention_scaling)
    except AttributeError:
        attention_scaling = 1.0
    return inv_freq, attention_scaling


def _assign_inv_freq(
    rotary: torch.nn.Module,
    name: str,
    value: torch.Tensor,
    *,
    persistent: bool,
) -> bool:
    current = rotary._buffers.get(name, vars(rotary).get(name))
    if (
        isinstance(current, torch.Tensor)
        and current.device.type != "meta"
        and current.device == value.device
        and current.shape == value.shape
        and current.dtype == value.dtype
        and torch.equal(current, value)
    ):
        return False
    if name in rotary._buffers:
        rotary.register_buffer(name, value, persistent=persistent)
    else:
        setattr(rotary, name, value)
    return True


class Qwen3Backbone(torch.nn.Module):
    """Qwen3-VL backbone with GR00T-specific compatibility patches."""

    def __init__(
        self,
        model_name: str = "nvidia/Cosmos-Reason2-2B",
        tune_llm: bool = False,
        tune_visual: bool = False,
        select_layer: int = -1,
        reproject_vision: bool = True,
        use_flash_attention: bool = False,
        projector_dim: int = -1,
        load_bf16: bool = False,
        tune_top_llm_layers: int = 0,
        trainable_params_fp32: bool = False,
        transformers_loading_kwargs: dict | None = None,
    ):
        """
        Qwen3Backbone is to generate n_queries to represent the future action hidden states.
        Args:
            model_name: nvidia/Cosmos-Reason2-2B
            tune_llm: whether to tune the LLM model (default: False)
            tune_visual: whether to tune the visual model (default: False)
        """
        if not _QWEN3VL_AVAILABLE:
            raise ImportError(
                "Qwen3VLForConditionalGeneration is not available. "
                "Please upgrade transformers to a version that supports Qwen3-VL: "
                "pip install transformers>=4.57.0"
            )

        super().__init__()
        transformers_loading_kwargs = dict(transformers_loading_kwargs or {})

        # Add attention kwargs
        extra_kwargs = {}
        if use_flash_attention:
            try:
                import flash_attn  # noqa: F401

                _patch_qwen3vl_flash_attention_target_dtype()
                extra_kwargs["attn_implementation"] = "flash_attention_2"
            except ImportError:
                logger.warning(
                    "flash_attn is not installed. Falling back to sdpa attention. "
                    "Install flash-attn for better performance: pip install flash-attn"
                )
                extra_kwargs["attn_implementation"] = "sdpa"
        if load_bf16:
            extra_kwargs["dtype"] = torch.bfloat16

        if (
            str(os.environ.get("HF_HUB_OFFLINE", "")).lower() in {"1", "true", "yes"}
            or str(os.environ.get("TRANSFORMERS_OFFLINE", "")).lower() in {"1", "true", "yes"}
        ):
            transformers_loading_kwargs = {
                **transformers_loading_kwargs,
                "local_files_only": True,
            }

        try:
            self.model = Qwen3VLForConditionalGeneration.from_pretrained(
                model_name,
                **extra_kwargs,
                **transformers_loading_kwargs,
            ).eval()
        except Exception as exc:
            if _is_gated_repo_error(exc):
                raise RuntimeError(_GATED_BACKBONE_HINT.format(model_name=model_name)) from exc
            raise
        if _force_tie_qwen_lm_head(self.model):
            logger.info(
                "Tied Qwen3-VL lm_head.weight to language_model.embed_tokens.weight "
                "for transformers 4.57.x compatibility."
            )
        self._uses_transformers_5_compat = _transformers_major_version() >= 5
        if self._uses_transformers_5_compat:
            _patch_qwen3vl_output_projection_dtypes(self.model)
        if _patch_qwen3vl_cuda_graph_vision_metadata():
            logger.info("Patched Qwen3-VL visual metadata paths for full-iteration CUDA graphs.")
        if _patch_qwen3_vl_fused_vision_rope():
            logger.info("Enabled fused FP32-compatible Qwen3-VL vision RoPE.")
        if _patch_qwen3_vl_fused_text_rope():
            logger.info("Enabled dtype-rounding-compatible Qwen3-VL text RoPE.")
        if _patch_qwen3_vl_fused_text_rms_norm():
            logger.info("Enabled reduction-order-compatible Qwen3-VL text RMSNorm fusion.")
        if _patch_qwen3_vl_fused_text_silu_mul():
            logger.info("Enabled dtype-rounding-compatible Qwen3-VL text MLP pointwise fusion.")
        self._supports_mm_token_type_ids = _forward_has_explicit_arg(self.model, "mm_token_type_ids")

        # needed since we don't use these layers. Also saves compute
        while len(self.language_model.layers) > select_layer:
            self.language_model.layers.pop(-1)

        self._selected_layer_features = None
        if self._uses_transformers_5_compat:
            # Transformers 5.x packages the post-norm tensor as the final hidden
            # state, so capture the configured decoder block output explicitly.
            self.language_model.layers[-1].register_forward_hook(
                self._capture_selected_layer_features
            )

        self.select_layer = select_layer
        self.set_trainable_parameters(tune_llm, tune_visual, tune_top_llm_layers)
        if load_bf16 and trainable_params_fp32:
            # cast trainable parameters to fp32
            for n, p in self.named_parameters():
                if p.requires_grad:
                    p.data = p.data.to(torch.float32)
                    logger.debug(f"Casting trainable parameter {n} to fp32")

        self._skip_lm_head_enabled = True
        if self._skip_lm_head_enabled and _patch_qwen3vl_skip_unused_lm_head():
            logger.info("Patched Qwen3-VL top-level forward to skip unused lm_head logits in GR00T training.")
        self._reset_rotary_inv_freq()
        self._apply_vision_patch_embed_channels_last()

    def _apply_vision_patch_embed_channels_last(self) -> None:
        visual = self.visual
        if visual is None:
            return

        def _to_channels_last_3d(_module, inputs):
            if not inputs or torch.jit.is_tracing():
                return inputs
            x = inputs[0]
            if isinstance(x, torch.Tensor) and x.dim() == 5:
                return (x.contiguous(memory_format=torch.channels_last_3d), *inputs[1:])
            return inputs

        patched = 0
        for module in visual.modules():
            if isinstance(module, torch.nn.Conv3d):
                module.to(memory_format=torch.channels_last_3d)
                module.register_forward_pre_hook(_to_channels_last_3d)
                patched += 1
        if patched:
            logger.debug(
                "Applied channels_last_3d to %d vision patch-embed Conv3d "
                "module(s) (torch>=2.9 cuDNN Conv3d perf workaround).",
                patched,
            )

    def set_trainable_parameters(self, tune_llm: bool, tune_visual: bool, tune_top_llm_layers: int):
        """Enable the requested trainable parameter groups for the backbone."""
        self.tune_llm = tune_llm
        self.tune_visual = tune_visual
        for p in self.parameters():
            p.requires_grad = True
        if not tune_llm:
            self.language_model.requires_grad_(False)
            try:
                self.model.lm_head.requires_grad_(False)
            except AttributeError:
                pass
        if not tune_visual:
            self.visual.requires_grad_(False)

        if tune_top_llm_layers > 0:
            for layer in self.language_model.layers[-tune_top_llm_layers:]:
                for param in layer.parameters():
                    param.requires_grad = True

        logger.debug(f"Tune backbone llm: {self.tune_llm}")
        logger.debug(f"Tune backbone visual: {self.tune_visual}")
        # Check if any parameters are still trainable. If not, log a warning.
        for name, p in self.named_parameters():
            if p.requires_grad:
                logger.debug(f"Backbone trainable parameter: {name}")
        if not any(p.requires_grad for p in self.parameters()):
            logger.warning("No backbone trainable parameters found.")

    def _capture_selected_layer_features(self, _module, _inputs, output) -> None:
        if isinstance(output, (tuple, list)):
            output = output[0]
        self._selected_layer_features = output

    def set_frozen_modules_to_eval_mode(self):
        """
        Huggingface will call model.train() at each training_step. To ensure
        the expected behaviors for modules like dropout, batchnorm, etc., we
        need to call model.eval() for the frozen modules.
        """
        if self.training:
            if self.language_model and not self.tune_llm:
                self.language_model.eval()
            if self.visual and not self.tune_visual:
                self.visual.eval()

    @property
    def language_model(self) -> torch.nn.Module:
        """Return the wrapped language model module."""
        return _unwrap_qwen_backbone(self.model).language_model

    @property
    def visual(self) -> torch.nn.Module:
        """Return the wrapped visual encoder module."""
        return _unwrap_qwen_backbone(self.model).visual

    def _reset_rotary_inv_freq(self) -> None:
        config = self.model.config
        vision_changed = self._reset_vision_rotary_inv_freq(config)
        text_changed = self._reset_language_rotary_inv_freq()
        logger.debug(
            "Qwen3-VL RoPE inv_freq reset (vision_rewritten=%s, text_rewritten=%s).",
            vision_changed,
            text_changed,
        )

    def _reset_vision_rotary_inv_freq(self, config) -> bool:
        visual = self.visual
        try:
            rotary = visual.rotary_pos_emb
        except AttributeError:
            rotary = None
        if rotary is None:
            raise RuntimeError("Qwen3-VL visual rotary_pos_emb/inv_freq not found.")
        try:
            rotary.inv_freq
        except AttributeError:
            raise RuntimeError("Qwen3-VL visual rotary_pos_emb/inv_freq not found.")

        vision_config = config.vision_config
        if vision_config is None:
            raise RuntimeError("Qwen3-VL vision_config missing hidden_size/num_heads.")

        head_dim = vision_config.hidden_size // vision_config.num_heads
        device = _real_inference_device(visual)
        inv_freq = recompute_vision_rotary_inv_freq(rotary, head_dim // 2, device)
        return _assign_inv_freq(rotary, "inv_freq", inv_freq, persistent=False)

    def _reset_language_rotary_inv_freq(self) -> bool:
        language_model = self.language_model
        try:
            rotary = language_model.rotary_emb
        except AttributeError:
            rotary = None
        if rotary is None:
            raise RuntimeError("Qwen3-VL language rotary_emb/inv_freq/config not found.")
        try:
            rotary.inv_freq
        except AttributeError:
            raise RuntimeError("Qwen3-VL language rotary_emb/inv_freq/config not found.")
        try:
            text_config = rotary.config
        except AttributeError:
            text_config = language_model.config
        if text_config is None:
            raise RuntimeError("Qwen3-VL language rotary_emb/inv_freq/config not found.")

        device = _real_inference_device(language_model)
        inv_freq, _attention_scaling = recompute_text_rotary_inv_freq(rotary, text_config, device)
        changed = _assign_inv_freq(rotary, "inv_freq", inv_freq, persistent=False)
        if "original_inv_freq" in rotary.__dict__:
            changed = _assign_inv_freq(rotary, "original_inv_freq", inv_freq.clone(), persistent=False) or changed
        return changed

    def prepare_input(self, batch: dict) -> BatchFeature:
        """Wrap a raw batch in a BatchFeature for the backbone."""
        return BatchFeature(data=batch)

    def prepare_host_position_metadata(self, batch) -> None:
        """Build Qwen position metadata while the fetched batch is still on CPU."""
        self._prepare_host_runtime_metadata(batch)
        if not self._supports_mm_token_type_ids:
            return
        if getattr(batch, "position_ids", None) is not None:
            return

        input_ids = getattr(batch, "input_ids", None)
        image_grid_thw = getattr(batch, "image_grid_thw", None)
        attention_mask = getattr(batch, "attention_mask", None)
        if input_ids is None or image_grid_thw is None or attention_mask is None:
            return

        position_cache = self.__dict__.setdefault(
            "_loongforge_host_position_metadata_cache", {}
        )
        cache_key = (
            _cpu_tensor_content_key(input_ids),
            _cpu_tensor_content_key(image_grid_thw),
            _cpu_tensor_content_key(attention_mask),
        )
        cached = position_cache.get(cache_key)
        if cached is not None:
            cached_position_ids, cached_mm_token_type_ids, cached_rope_deltas = cached
            if cached_position_ids is None:
                batch.mm_token_type_ids = cached_mm_token_type_ids
            else:
                batch.position_ids = cached_position_ids
            qwen_model = _unwrap_qwen_backbone(self.model)
            if cached_rope_deltas is not None and "rope_deltas" in qwen_model.__dict__:
                qwen_model.rope_deltas = cached_rope_deltas
            return

        with torch.no_grad():
            mm_token_type_ids = getattr(batch, "mm_token_type_ids", None)
            if mm_token_type_ids is None:
                mm_token_type_ids = _build_mm_token_type_ids(self.model.config, input_ids)
            position_ids = _build_qwen3vl_compat_position_ids(
                self.model,
                input_ids,
                mm_token_type_ids,
                image_grid_thw,
                attention_mask,
            )
        rope_deltas = getattr(_unwrap_qwen_backbone(self.model), "rope_deltas", None)
        position_cache[cache_key] = (position_ids, mm_token_type_ids, rope_deltas)
        if len(position_cache) > 32:
            position_cache.pop(next(iter(position_cache)))
        if position_ids is None:
            batch.mm_token_type_ids = mm_token_type_ids
        else:
            batch.position_ids = position_ids

    def _prepare_host_runtime_metadata(self, batch) -> None:
        """Prepare dynamic Qwen metadata before the batch crosses to CUDA."""
        input_ids = getattr(batch, "input_ids", None)
        image_grid_thw = getattr(batch, "image_grid_thw", None)
        attention_mask = getattr(batch, "attention_mask", None)
        if input_ids is None or image_grid_thw is None or attention_mask is None:
            return
        image_mask = input_ids == self.model.config.image_token_id
        visual_indices = torch.nonzero(image_mask.reshape(-1), as_tuple=False).flatten()
        batch._loongforge_host_grid_signature = _qwen3vl_grid_signature(image_grid_thw)
        batch._loongforge_host_image_token_count = int(image_mask.sum().item())
        batch._loongforge_host_visual_indices = visual_indices
        batch._loongforge_host_visual_index_signature = tuple(visual_indices.tolist())
        batch._loongforge_host_visual_mask = image_mask
        batch._loongforge_host_attention_mask_all_valid = bool(
            attention_mask.bool().all().item()
        )

    def prepare_vision_metadata_batch(self, batch, grid_signature=None):
        """Cache image-grid metadata for both graph and eager execution.

        ``image_grid_thw`` is already a host-derived batch field.  Reading its
        signature once here lets the patched Vision forward pass a Python
        ``max_seqlen`` to FlashAttention instead of reducing a CUDA
        ``cu_seqlens`` tensor in every vision layer.
        """
        image_grid_thw = getattr(batch, "image_grid_thw", None)
        if image_grid_thw is None:
            return None
        signature = grid_signature or getattr(batch, "_loongforge_host_grid_signature", None)
        if signature is None:
            signature = _qwen3vl_grid_signature(image_grid_thw)
        visual = self.visual
        by_signature = visual.__dict__.setdefault(
            "_loongforge_cuda_graph_vision_metadata_by_signature", {}
        )
        metadata = by_signature.get(signature)
        if metadata is None:
            metadata = _build_qwen3vl_vision_graph_metadata(visual, signature)
            by_signature[signature] = metadata
        by_pointer = visual.__dict__.setdefault(
            "_loongforge_cuda_graph_vision_metadata_by_pointer", {}
        )
        by_pointer[image_grid_thw.data_ptr()] = metadata
        return metadata

    def prepare_cuda_graph_batch(self, batch) -> None:
        """Precompute Qwen dynamic position metadata before graph capture."""
        image_grid_thw = getattr(batch, "image_grid_thw", None)
        if image_grid_thw is not None:
            metadata = self.prepare_vision_metadata_batch(batch)

            input_ids = getattr(batch, "input_ids", None)
            if input_ids is not None:
                actual_image_tokens = getattr(
                    batch,
                    "_loongforge_host_image_token_count",
                    None,
                )
                if actual_image_tokens is None:
                    actual_image_tokens = int(
                        (input_ids == self.model.config.image_token_id).sum().item()
                    )
                if actual_image_tokens != metadata.image_token_count:
                    raise RuntimeError(
                        "Qwen3-VL image token count does not match cached vision features: "
                        f"tokens={actual_image_tokens}, features={metadata.image_token_count}."
                    )
                qwen_model = _unwrap_qwen_backbone(self.model)
                tokens_by_pointer = qwen_model.__dict__.setdefault(
                    "_loongforge_cuda_graph_image_tokens_by_pointer", {}
                )
                tokens_by_pointer[input_ids.data_ptr()] = actual_image_tokens
                visual_indices = getattr(batch, "_loongforge_host_visual_indices", None)
                visual_index_signature = getattr(
                    batch,
                    "_loongforge_host_visual_index_signature",
                    None,
                )
                if visual_indices is None:
                    visual_indices = torch.nonzero(
                        (input_ids == self.model.config.image_token_id).reshape(-1),
                        as_tuple=False,
                    ).flatten()
                elif visual_indices.device != input_ids.device:
                    visual_indices = visual_indices.to(input_ids.device, non_blocking=True)
                if visual_index_signature is None:
                    visual_index_signature = tuple(visual_indices.detach().cpu().tolist())
                visual_index_cache = self.language_model.__dict__.setdefault(
                    "_loongforge_cuda_graph_visual_token_index_cache", {}
                )
                cached_visual_indices = visual_index_cache.get(visual_index_signature)
                if cached_visual_indices is None:
                    cached_visual_indices = visual_indices
                    visual_index_cache[visual_index_signature] = cached_visual_indices
                self.language_model._loongforge_cuda_graph_visual_token_indices = (
                    cached_visual_indices
                )
                visual_masks_by_pointer = qwen_model.__dict__.setdefault(
                    "_loongforge_cuda_graph_visual_pos_masks_by_pointer", {}
                )
                visual_mask = getattr(batch, "_loongforge_host_visual_mask", None)
                if visual_mask is None:
                    visual_mask = input_ids == self.model.config.image_token_id
                elif visual_mask.device != input_ids.device:
                    visual_mask = visual_mask.to(input_ids.device, non_blocking=True)
                visual_masks_by_pointer[input_ids.data_ptr()] = visual_mask

            attention_mask = getattr(batch, "attention_mask", None)
            if attention_mask is not None:
                mask_is_all_valid = getattr(
                    batch,
                    "_loongforge_host_attention_mask_all_valid",
                    None,
                )
                if mask_is_all_valid is None:
                    mask_is_all_valid = bool(attention_mask.bool().all().item())
                mask_by_pointer = self.__dict__.setdefault(
                    "_loongforge_cuda_graph_attention_mask_all_by_pointer", {}
                )
                mask_by_pointer[attention_mask.data_ptr()] = mask_is_all_valid


    def validate_cuda_graph_batch(self, expected_batch, actual_batch) -> None:
        """Reject grid-value changes that would invalidate captured visual metadata."""
        expected_grid = getattr(expected_batch, "image_grid_thw", None)
        actual_grid = getattr(actual_batch, "image_grid_thw", None)
        if expected_grid is None and actual_grid is None:
            return
        if expected_grid is None or actual_grid is None:
            raise RuntimeError("Qwen3-VL image_grid_thw presence changed after CUDA graph capture.")
        expected_signature = _qwen3vl_grid_signature(expected_grid)
        actual_signature = _qwen3vl_grid_signature(actual_grid)
        if expected_signature != actual_signature:
            raise RuntimeError(
                "Qwen3-VL image_grid_thw values changed after CUDA graph capture: "
                f"expected {expected_signature}, got {actual_signature}."
            )
        expected_input_ids = getattr(expected_batch, "input_ids", None)
        actual_input_ids = getattr(actual_batch, "input_ids", None)
        if expected_input_ids is None or actual_input_ids is None:
            raise RuntimeError("Qwen3-VL input_ids disappeared after CUDA graph capture.")
        image_token_id = self.model.config.image_token_id
        expected_visual_mask = expected_input_ids == image_token_id
        actual_visual_mask = actual_input_ids == image_token_id
        if not torch.equal(expected_visual_mask, actual_visual_mask):
            raise RuntimeError(
                "Qwen3-VL visual token positions changed after CUDA graph capture."
            )

    def forward(self, vl_input: BatchFeature) -> BatchFeature:
        """Run the backbone and return the last hidden state plus masks."""
        self.set_frozen_modules_to_eval_mode()
        keys_to_use = ["input_ids", "attention_mask", "pixel_values", "image_grid_thw"]
        source_input = vl_input
        vl_input = {k: source_input[k] for k in keys_to_use}
        for optional_key in ("position_ids", "mm_token_type_ids"):
            if optional_key in source_input and source_input[optional_key] is not None:
                vl_input[optional_key] = source_input[optional_key]
        if self._supports_mm_token_type_ids:
            if "position_ids" not in vl_input and "mm_token_type_ids" not in vl_input:
                mm_token_type_ids = _build_mm_token_type_ids(self.model.config, vl_input["input_ids"])
                position_ids = _build_qwen3vl_compat_position_ids(
                    self.model,
                    vl_input["input_ids"],
                    mm_token_type_ids,
                    vl_input["image_grid_thw"],
                    vl_input["attention_mask"],
                )
                if position_ids is None:
                    vl_input["mm_token_type_ids"] = mm_token_type_ids
                else:
                    vl_input["position_ids"] = position_ids
        model_input = vl_input
        attention_mask = vl_input["attention_mask"]
        # Qwen's text model rebuilds this fixed-length position vector on every
        # replay when it is omitted. Cache it by sequence length so CUDA Graph
        # replays do not launch the same integer fill/arange work repeatedly.
        cache_position = vl_input.get("cache_position")
        if cache_position is None:
            sequence_length = int(vl_input["input_ids"].shape[-1])
            cache_key = (vl_input["input_ids"].device, sequence_length)
            cache_position = self.__dict__.setdefault(
                "_loongforge_cache_positions", {}
            ).get(cache_key)
            if cache_position is None:
                cache_position = torch.arange(
                    sequence_length,
                    device=vl_input["input_ids"].device,
                    dtype=torch.long,
                )
                self.__dict__["_loongforge_cache_positions"][cache_key] = cache_position
            vl_input["cache_position"] = cache_position
        mask_by_pointer = self.__dict__.get(
            "_loongforge_cuda_graph_attention_mask_all_by_pointer", {}
        )
        if mask_by_pointer.get(attention_mask.data_ptr(), False):
            model_input = dict(vl_input)
            model_input["attention_mask"] = None
        self._selected_layer_features = None
        if self._skip_lm_head_enabled:
            model_outputs = self.model(
                **model_input,
                output_hidden_states=not self._uses_transformers_5_compat,
                logits_to_keep=0,
                loongforge_skip_lm_head=True,
            )
        else:
            model_outputs = self.model(
                **model_input,
                output_hidden_states=not self._uses_transformers_5_compat,
            )
        if self._uses_transformers_5_compat:
            if self._selected_layer_features is None:
                raise RuntimeError("Qwen3-VL selected decoder layer did not produce features.")
            outputs = self._selected_layer_features.to(torch.float32)
        else:
            outputs = model_outputs.hidden_states[-1].to(torch.float32)
        image_mask = vl_input["input_ids"] == self.model.config.image_token_id
        attention_mask = vl_input["attention_mask"] == 1
        return BatchFeature(
            data={
                "backbone_features": outputs,
                "backbone_attention_mask": attention_mask,
                "image_mask": image_mask,
            }
        )  # [B, T2, hidden_size]
