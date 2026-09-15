# Copyright 2026 The LoongForge Authors.
# SPDX-License-Identifier: Apache-2.0
"""Checkpoint-compatible GLM-5.3-Flash multimodal model for LoongForge."""

from __future__ import annotations

from copy import copy
from dataclasses import dataclass
from pathlib import Path

import torch
import torch.nn.functional as F
from safetensors.torch import load_file
from torch import nn

from megatron.core import parallel_state
from megatron.core.enums import ModelType
from megatron.core.models.common.embeddings.language_model_embedding import LanguageModelEmbedding
from megatron.core.models.backends import LocalSpecProvider
from megatron.core.process_groups_config import ProcessGroupCollection
from megatron.core.transformer.experimental_attention_variant.dsa import DSAIndexShareCarrier
from megatron.core.transformer.identity_op import IdentityOp
from megatron.core.transformer.spec_utils import build_module

if __package__:
    from loongforge.models.foundation.language_transformer_block import TransformerBlock
else:
    import sys

    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    from language_transformer_block import TransformerBlock

try:
    from .glm5_next_attention import (
        Glm5NextTextLinearAttention,
        KPoolDSAIndexer,
        initialize_kda,
    )
    from .glm5_next_config import Glm5NextConfig
    from .glm5_next_layer_spec import _layer_spec, get_glm5_next_decoder_block_spec
    from .glm5_next_vision import Glm5NextVisionModel, Glm5NextVisionOutput
except ImportError:
    from glm5_next_attention import (
        Glm5NextTextLinearAttention,
        KPoolDSAIndexer,
        initialize_kda,
    )
    from glm5_next_config import Glm5NextConfig
    from glm5_next_layer_spec import _layer_spec, get_glm5_next_decoder_block_spec
    from glm5_next_vision import Glm5NextVisionModel, Glm5NextVisionOutput


@dataclass
class Glm5NextOutput:
    logits: torch.Tensor
    loss: torch.Tensor | None = None


class MeanHyperHead(nn.Module):
    def __init__(self, streams: int, hidden_size: int) -> None:
        super().__init__()
        self.streams = streams
        self.hidden_size = hidden_size

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        return hidden_states.view(*hidden_states.shape[:-1], self.streams, self.hidden_size).mean(-2)


class Glm5NextMTPLayer(nn.Module):
    """GLM-5.3-Flash MTP (nextn) layer, mirroring the released layer-45 weights.

    DeepSeek-V3-style structure: the concatenation of ``hnorm(hidden)`` and
    ``enorm(next-token embedding)`` feeds ``eh_proj`` [2h -> h]; the projected
    states pass a full decoder layer (KPool-DSA attention + MoE) — the released
    MTP layer carries NO hyper connections — and finally the shared-head norm
    before the shared ``lm_head``.
    """

    def __init__(self, config, layer_spec, pg_collection, vp_stage=None) -> None:
        super().__init__()
        self.config = config
        self.enorm = nn.RMSNorm(config.hidden_size, eps=config.layernorm_epsilon)
        self.hnorm = nn.RMSNorm(config.hidden_size, eps=config.layernorm_epsilon)
        # Released layout: eh_proj.weight is [h, 2h], matching nn.Linear.
        self.eh_proj = nn.Linear(config.hidden_size * 2, config.hidden_size, bias=False)
        self.transformer_layer = build_module(
            layer_spec,
            config=config,
            layer_number=config.num_layers + 1,
            pg_collection=pg_collection,
            vp_stage=vp_stage,
            is_mtp_layer=True,
        )
        self.final_layernorm = nn.RMSNorm(config.hidden_size, eps=config.layernorm_epsilon)

    def forward(
        self,
        hidden_states: torch.Tensor,
        next_embeds: torch.Tensor,
        attention_mask: torch.Tensor,
    ) -> torch.Tensor:
        # hidden_states / next_embeds: [s, b, h]
        combined = torch.cat((self.hnorm(hidden_states), self.enorm(next_embeds)), dim=-1)
        projected = self.eh_proj(combined)
        output, _ = self.transformer_layer(
            hidden_states=projected,
            attention_mask=attention_mask.bool(),
            index_share_carrier=DSAIndexShareCarrier(),
        )
        return self.final_layernorm(output)


class Glm5NextMTPBlock(nn.Module):
    """Container for the GLM-5.3-Flash MTP layers (``mtp.layers.{k}``)."""

    def __init__(self, config, pg_collection, vp_stage=None) -> None:
        super().__init__()
        self.config = config
        # The MTP layer mirrors the last decoder layer (KPool-DSA + MoE),
        # without hyper connections: the released nextn layer carries no mHC.
        backend = LocalSpecProvider()
        layer_spec = copy(_layer_spec(config, config.num_layers - 1, backend))
        layer_spec.submodules.self_attention_hyper_connection = IdentityOp
        layer_spec.submodules.mlp_hyper_connection = IdentityOp
        self.layers = nn.ModuleList(
            Glm5NextMTPLayer(config, layer_spec, pg_collection, vp_stage=vp_stage)
            for _ in range(config.mtp_num_layers)
        )

    def forward(
        self,
        hidden_states: torch.Tensor,
        embeds: torch.Tensor,
        attention_mask: torch.Tensor,
    ) -> list[torch.Tensor]:
        outputs = []
        current = hidden_states
        for depth, layer in enumerate(self.layers):
            shifted = torch.roll(embeds, shifts=-(depth + 1), dims=0)
            current = layer(current, shifted, attention_mask)
            outputs.append(current)
        return outputs


def _gather_sequence_parallel(value: torch.Tensor, tp_group) -> torch.Tensor:
    if tp_group.size() == 1:
        return value
    from megatron.core.tensor_parallel import gather_from_sequence_parallel_region

    return gather_from_sequence_parallel_region(value, group=tp_group)


def _scatter_sequence_parallel(value: torch.Tensor, tp_group) -> torch.Tensor:
    if tp_group.size() == 1:
        return value
    from megatron.core.tensor_parallel import scatter_to_sequence_parallel_region

    return scatter_to_sequence_parallel_region(value, group=tp_group)


def _gather_context_parallel(value: torch.Tensor, cp_group) -> torch.Tensor:
    if cp_group.size() == 1:
        return value
    gathered = [torch.empty_like(value) for _ in range(cp_group.size())]
    torch.distributed.all_gather(gathered, value, group=cp_group)
    halves = [tensor.chunk(2, dim=1) for tensor in gathered]
    return torch.cat([pair[0] for pair in halves] + [pair[1] for pair in reversed(halves)], dim=1)


def _scatter_context_parallel(value: torch.Tensor, cp_group) -> torch.Tensor:
    if cp_group.size() == 1:
        return value
    chunks = value.chunk(2 * cp_group.size(), dim=0)
    rank = cp_group.rank()
    return torch.cat((chunks[rank], chunks[2 * cp_group.size() - rank - 1]), dim=0)


class LanguageModel(nn.Module):
    def __init__(
        self,
        config: Glm5NextConfig,
        pre_process: bool,
        post_process: bool,
        pg_collection,
        vp_stage=None,
    ) -> None:
        super().__init__()
        self.config = config
        self.pre_process = pre_process
        self.post_process = post_process
        self.model_type = ModelType.encoder_or_decoder
        if pre_process:
            self.embedding = LanguageModelEmbedding(
                config=config,
                vocab_size=config.vocab_size,
                max_sequence_length=config.max_position_embeddings,
                position_embedding_type="none",
                scatter_to_sequence_parallel=config.sequence_parallel,
                tp_group=pg_collection.tp,
            )
            if self.embedding.word_embeddings.tp_group is None:
                self.embedding.word_embeddings.tp_group = pg_collection.tp
        block_spec = get_glm5_next_decoder_block_spec(
            config, pg_collection=pg_collection, vp_stage=vp_stage
        )
        self.decoder = TransformerBlock(
            config=config,
            spec=block_spec,
            pre_process=pre_process,
            post_process=post_process,
            pg_collection=pg_collection,
            vp_stage=vp_stage,
        )
        if post_process:
            self.decoder.head_hyper_connection = MeanHyperHead(
                config.hc_mult, config.hidden_size
            )
        self.mtp = (
            Glm5NextMTPBlock(config, pg_collection, vp_stage=vp_stage)
            if post_process and (getattr(config, "mtp_num_layers", 0) or 0) > 0
            else None
        )

    @property
    def embed_tokens(self):
        return self.embedding.word_embeddings

    @property
    def layers(self):
        return self.decoder.layers

    @property
    def norm(self):
        return self.decoder.final_layernorm

    def set_input_tensor(self, input_tensor) -> None:
        self.decoder.set_input_tensor(input_tensor)

    def forward(
        self,
        input_ids: torch.Tensor | None,
        attention_mask: torch.Tensor,
        inputs_embeds: torch.Tensor | None = None,
    ) -> torch.Tensor:
        if inputs_embeds is None and self.pre_process:
            if input_ids is None:
                raise ValueError("input_ids is required on the first pipeline stage")
            position_ids = torch.zeros_like(input_ids)
            hidden_states = self.embedding(input_ids=input_ids, position_ids=position_ids)
        elif inputs_embeds is not None:
            hidden_states = inputs_embeds
            if hidden_states.ndim == 3 and hidden_states.shape[:2] == attention_mask.shape:
                hidden_states = hidden_states.transpose(0, 1).contiguous()
            if not self.pre_process:
                self.decoder.set_input_tensor(hidden_states)
        else:
            hidden_states = None
        return self.decoder(
            hidden_states=hidden_states,
            attention_mask=attention_mask.bool(),
            index_share_carrier=DSAIndexShareCarrier(),
        )


class MultimodalContainer(nn.Module):
    def __init__(self, config, pre_process, post_process, pg_collection, vp_stage=None) -> None:
        super().__init__()
        self.visual = Glm5NextVisionModel(config.vision_config) if pre_process else None
        self.language_model = LanguageModel(
            config, pre_process, post_process, pg_collection, vp_stage=vp_stage
        )


class Glm5NextModel(nn.Module):
    config_class = Glm5NextConfig

    def __init__(
        self,
        config: Glm5NextConfig,
        pre_process: bool = True,
        post_process: bool = True,
        parallel_output: bool = True,
        pg_collection=None,
        vp_stage=None,
        **kwargs,
    ) -> None:
        super().__init__()
        self.config = config
        self.model_type = ModelType.encoder_or_decoder
        self.pre_process = pre_process
        self.post_process = post_process
        self.parallel_output = parallel_output
        if not parallel_state.is_initialized():
            raise RuntimeError("initialize Loong-Megatron model parallel before building Glm5NextModel")
        self.pg_collection = pg_collection or ProcessGroupCollection.use_mpu_process_groups()
        self.vp_stage = vp_stage
        self.model = MultimodalContainer(
            config,
            pre_process=pre_process,
            post_process=post_process,
            pg_collection=self.pg_collection,
            vp_stage=vp_stage,
        )
        if post_process:
            self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)
        self._input_tensor = None
        self._initialize_glm_modules()

    def get_input_embeddings(self) -> nn.Embedding:
        return self.model.language_model.embed_tokens

    def set_input_embeddings(self, value: nn.Embedding) -> None:
        self.model.language_model.embed_tokens = value

    def get_output_embeddings(self) -> nn.Linear:
        return self.lm_head

    def set_output_embeddings(self, value: nn.Linear) -> None:
        self.lm_head = value

    def freeze(self) -> None:
        for parameter in self.parameters():
            parameter.requires_grad_(False)

    def set_input_tensor(self, input_tensor: torch.Tensor | list[torch.Tensor] | None) -> None:
        if isinstance(input_tensor, list):
            if len(input_tensor) != 1:
                raise ValueError("GLM-5.3-Flash expects one pipeline input tensor")
            input_tensor = input_tensor[0]
        self._input_tensor = input_tensor
        self.model.language_model.set_input_tensor(input_tensor)

    @classmethod
    def _from_config(cls, config: Glm5NextConfig, **kwargs) -> "Glm5NextModel":
        return cls(config, **kwargs)

    def _initialize_glm_modules(self) -> None:
        for module in self.modules():
            if isinstance(module, KPoolDSAIndexer):
                nn.init.zeros_(module.index_kpool_compress_ape)
                nn.init.ones_(module.index_kpool_compress_gate)
            if isinstance(module, Glm5NextTextLinearAttention):
                initialize_kda(module)

    def _initialize_module(self, module: nn.Module) -> None:
        if isinstance(module, KPoolDSAIndexer):
            nn.init.zeros_(module.index_kpool_compress_ape)
            nn.init.ones_(module.index_kpool_compress_gate)
        if isinstance(module, Glm5NextTextLinearAttention):
            initialize_kda(module)

    def forward(
        self,
        input_ids: torch.Tensor | None = None,
        position_ids: torch.Tensor | None = None,
        attention_mask: torch.Tensor | None = None,
        decoder_input: torch.Tensor | None = None,
        labels: torch.Tensor | None = None,
        pixel_values: torch.Tensor | None = None,
        pixel_values_videos: torch.Tensor | None = None,
        image_grid_thw: torch.Tensor | None = None,
        video_grid_thw: torch.Tensor | None = None,
        inputs_embeds: torch.Tensor | None = None,
        **kwargs,
    ) -> Glm5NextOutput:
        # Preserve the original adapter call model(input_ids, attention_mask).
        if attention_mask is None and position_ids is not None and position_ids.ndim == 2:
            attention_mask, position_ids = position_ids, None
        sequence_first = decoder_input is not None or not self.pre_process
        if decoder_input is not None:
            inputs_embeds = decoder_input
        if not sequence_first and inputs_embeds is not None and inputs_embeds.ndim == 3 and input_ids is not None:
            if inputs_embeds.shape[0] == input_ids.shape[1] and inputs_embeds.shape[1] == input_ids.shape[0]:
                inputs_embeds = inputs_embeds.transpose(0, 1).contiguous()
                sequence_first = True
        has_vision = pixel_values is not None or pixel_values_videos is not None
        if self.pre_process and inputs_embeds is None and has_vision:
            if input_ids is None:
                raise ValueError("input_ids is required when vision inputs are present")
            cp_group = self.pg_collection.cp
            tp_group = self.pg_collection.tp
            global_input_ids = _gather_context_parallel(input_ids, cp_group)
            position_ids = torch.zeros_like(global_input_ids)
            inputs_embeds = self.model.language_model.embedding(global_input_ids, position_ids)
            inputs_embeds = _gather_sequence_parallel(inputs_embeds, tp_group)
            batch_embeds = inputs_embeds.transpose(0, 1).contiguous()
            if pixel_values is not None:
                image_features = self.get_image_features(pixel_values, image_grid_thw).pooler_output
                batch_embeds = self._merge_vision_features(
                    global_input_ids, batch_embeds, image_features, is_video=False
                )
            if pixel_values_videos is not None:
                video_features = self.get_video_features(pixel_values_videos, video_grid_thw).pooler_output
                batch_embeds = self._merge_vision_features(
                    global_input_ids, batch_embeds, video_features, is_video=True
                )
            inputs_embeds = _scatter_context_parallel(
                batch_embeds.transpose(0, 1).contiguous(), cp_group
            )
            inputs_embeds = _scatter_sequence_parallel(inputs_embeds, tp_group)
            sequence_first = True
        elif inputs_embeds is None and self.pre_process and input_ids is None:
            raise ValueError("input_ids or inputs_embeds is required")
        if attention_mask is None:
            if input_ids is not None:
                mask_shape = input_ids.shape
            elif sequence_first:
                mask_shape = (inputs_embeds.shape[1], inputs_embeds.shape[0])
            else:
                mask_shape = inputs_embeds.shape[:2]
            mask_device = input_ids.device if input_ids is not None else inputs_embeds.device
            attention_mask = torch.ones(mask_shape, dtype=torch.long, device=mask_device)
        elif attention_mask.ndim > 2:
            sequence_length = attention_mask.shape[-1]
            attention_mask = attention_mask.reshape(-1, sequence_length, sequence_length)[:, -1]
            if attention_mask.dtype == torch.bool:
                attention_mask = ~attention_mask
            batch_size = input_ids.shape[0] if input_ids is not None else inputs_embeds.shape[1]
            if attention_mask.shape[0] == 1 and batch_size > 1:
                attention_mask = attention_mask.expand(batch_size, -1)
        hidden_states = self.model.language_model(
            input_ids=input_ids if inputs_embeds is None else None,
            inputs_embeds=inputs_embeds,
            attention_mask=attention_mask,
        )
        # The block returns (hidden_states, mhc_multistream) once mtp_num_layers
        # > 0; GLM-5.3's MTP consumes the contracted single-stream hidden.
        if isinstance(hidden_states, tuple):
            hidden_states = hidden_states[0]
        if not self.post_process:
            return hidden_states
        logits = self.lm_head(hidden_states.transpose(0, 1).contiguous())
        loss = None
        if labels is not None:
            loss = F.cross_entropy(
                logits[:, :-1].float().reshape(-1, self.config.vocab_size),
                labels[:, 1:].reshape(-1),
                ignore_index=-100,
            )
            mtp = self.model.language_model.mtp
            if mtp is not None:
                loss = loss + self._mtp_loss(
                    mtp, hidden_states, input_ids, inputs_embeds, attention_mask, labels
                )
        return Glm5NextOutput(logits=logits, loss=loss)

    def _mtp_loss(
        self,
        mtp: "Glm5NextMTPBlock",
        hidden_states: torch.Tensor,
        input_ids: torch.Tensor | None,
        inputs_embeds: torch.Tensor | None,
        attention_mask: torch.Tensor,
        labels: torch.Tensor,
    ) -> torch.Tensor:
        """Compute the scaled MTP auxiliary loss (GLM-5.2-style, DeepSeek-V3 MTP)."""
        if input_ids is not None:
            position_ids = torch.zeros_like(input_ids)
            embeds = self.model.language_model.embedding(input_ids, position_ids)
        elif inputs_embeds is not None:
            # inputs_embeds is [s, b, h] in the pipeline path; roll along the
            # sequence dimension to fetch the next-token embeddings.
            embeds = inputs_embeds
        else:
            return torch.zeros((), device=hidden_states.device, dtype=torch.float32)
        mtp_outputs = mtp(hidden_states, embeds, attention_mask)
        mtp_loss = None
        for depth, mtp_hidden in enumerate(mtp_outputs):
            mtp_logits = self.lm_head(mtp_hidden.transpose(0, 1).contiguous())
            shift = depth + 2
            depth_loss = F.cross_entropy(
                mtp_logits[:, :-shift].float().reshape(-1, self.config.vocab_size),
                labels[:, shift:].reshape(-1),
                ignore_index=-100,
            )
            mtp_loss = depth_loss if mtp_loss is None else mtp_loss + depth_loss
        scale = (
            self.config.mtp_loss_scaling_factor
            if self.config.mtp_loss_scaling_factor is not None
            else 0.1
        ) / len(mtp_outputs)
        return scale * mtp_loss

    def get_image_features(
        self,
        pixel_values: torch.Tensor,
        image_grid_thw: torch.Tensor,
    ) -> Glm5NextVisionOutput:
        if self.model.visual is None:
            raise RuntimeError("vision encoder is only built on the first pipeline stage")
        output = self.model.visual(pixel_values, image_grid_thw)
        split_sizes = (image_grid_thw.prod(-1) // self.model.visual.spatial_merge_size**2).tolist()
        output.pooler_output = torch.split(output.pooler_output, split_sizes)
        return output

    def get_video_features(
        self,
        pixel_values_videos: torch.Tensor,
        video_grid_thw: torch.Tensor,
    ) -> Glm5NextVisionOutput:
        temporal = video_grid_thw[:, 0]
        spatial = torch.repeat_interleave(video_grid_thw[:, 1:], temporal, dim=0)
        flattened_grid = torch.cat((video_grid_thw.new_ones(spatial.shape[0], 1), spatial), dim=1)
        if self.model.visual is None:
            raise RuntimeError("vision encoder is only built on the first pipeline stage")
        output = self.model.visual(pixel_values_videos, flattened_grid)
        split_sizes = (video_grid_thw.prod(-1) // self.model.visual.spatial_merge_size**2).tolist()
        output.pooler_output = torch.split(output.pooler_output, split_sizes)
        return output

    def _merge_vision_features(
        self,
        input_ids: torch.Tensor,
        inputs_embeds: torch.Tensor,
        features: tuple[torch.Tensor, ...],
        is_video: bool,
    ) -> torch.Tensor:
        features_tensor = torch.cat(features, dim=0).to(inputs_embeds.device, inputs_embeds.dtype)
        special_mask = input_ids.eq(self.config.image_token_id)
        in_video_span = input_ids.eq(self.config.video_start_token_id).cumsum(-1) > input_ids.eq(
            self.config.video_end_token_id
        ).cumsum(-1)
        token_mask = special_mask & (in_video_span if is_video else ~in_video_span)
        if token_mask.sum().item() * inputs_embeds.shape[-1] != features_tensor.numel():
            modality = "video" if is_video else "image"
            raise ValueError(
                f"{modality} features and placeholder tokens do not match: "
                f"tokens={token_mask.sum().item()}, features={features_tensor.shape[0]}"
            )
        return inputs_embeds.masked_scatter(token_mask.unsqueeze(-1), features_tensor)

    @staticmethod
    def _remap_hf_state_dict(state: dict[str, torch.Tensor], model: "Glm5NextModel"):
        remapped = {}
        if model.pre_process:
            remapped.update(
                (key, value) for key, value in state.items() if key.startswith("model.visual.")
            )
            remapped["model.language_model.embedding.word_embeddings.weight"] = state[
                "model.language_model.embed_tokens.weight"
            ]
        if model.post_process:
            remapped["lm_head.weight"] = state["lm_head.weight"]
            remapped["model.language_model.decoder.final_layernorm.weight"] = state[
                "model.language_model.norm.weight"
            ]

        def _remap_one_layer(
            source: str, target: str, layer_type: str, mlp_type: str,
            has_hyper_connections: bool = True,
        ) -> None:
            remapped[target + "input_layernorm.weight"] = state[source + "input_layernorm.weight"]
            remapped[target + "pre_mlp_layernorm.weight"] = state[
                source + "post_attention_layernorm.weight"
            ]

            attention_prefix = source + "self_attn."
            if layer_type == "linear_attention":
                for key, value in state.items():
                    if key.startswith(attention_prefix):
                        suffix = key[len(attention_prefix) :]
                        if suffix == "o_norm.weight":
                            suffix = "o_norm.norm.weight"
                        remapped[target + "self_attention." + suffix] = value
            else:
                sparse_attention_names = {
                    "q_a_proj.weight": "linear_q_down_proj.weight",
                    "q_a_layernorm.weight": "q_layernorm.weight",
                    "q_b_proj.weight": "linear_q_up_proj.weight",
                    "kv_a_proj_with_mqa.weight": "linear_kv_down_proj.weight",
                    "kv_a_layernorm.weight": "kv_layernorm.weight",
                    "kv_b_proj.weight": "linear_kv_up_proj.weight",
                    "o_proj.weight": "linear_proj.weight",
                    "indexer.wq_b.weight": "core_attention.indexer.linear_wq_b.weight",
                    "indexer.wk.weight": "core_attention.indexer.linear_wk.weight",
                    "indexer.k_norm.weight": "core_attention.indexer.k_norm.weight",
                    "indexer.k_norm.bias": "core_attention.indexer.k_norm.bias",
                    "indexer.weights_proj.weight": (
                        "core_attention.indexer.linear_weights_proj.weight"
                    ),
                    "indexer.index_kpool_compress_ape": (
                        "core_attention.indexer.index_kpool_compress_ape"
                    ),
                    "indexer.index_kpool_compress_gate": (
                        "core_attention.indexer.index_kpool_compress_gate"
                    ),
                }
                for source_name, target_name in sparse_attention_names.items():
                    source_key = attention_prefix + source_name
                    if source_key in state:
                        remapped[target + "self_attention." + target_name] = state[source_key]

            if has_hyper_connections:
                for source_name, target_name in (
                    ("hc_attn", "self_attention_hyper_connection"),
                    ("hc_ffn", "mlp_hyper_connection"),
                ):
                    remapped[target + target_name + ".mapping_proj.weight"] = state[
                        source + source_name + "_fn"
                    ]
                    remapped[target + target_name + ".bias"] = state[source + source_name + "_base"]
                    scale = state[source + source_name + "_scale"]
                    remapped[target + target_name + ".alpha_pre"] = scale[0:1]
                    remapped[target + target_name + ".alpha_post"] = scale[1:2]
                    remapped[target + target_name + ".alpha_res"] = scale[2:3]

            mlp_source = source + "mlp."
            mlp_target = target + "mlp."
            if mlp_type == "dense":
                remapped[mlp_target + "linear_fc1.weight"] = torch.cat(
                    [state[mlp_source + "gate_proj.weight"], state[mlp_source + "up_proj.weight"]],
                    dim=0,
                )
                remapped[mlp_target + "linear_fc2.weight"] = state[
                    mlp_source + "down_proj.weight"
                ]
            else:
                remapped[mlp_target + "router.weight"] = state[mlp_source + "gate.weight"]
                remapped[mlp_target + "router.expert_bias"] = state[
                    mlp_source + "gate.e_score_correction_bias"
                ]
                for expert_index in range(model.config.n_routed_experts):
                    expert_source = mlp_source + f"experts.{expert_index}."
                    expert_target = mlp_target + f"experts.local_experts.{expert_index}."
                    remapped[expert_target + "linear_fc1.weight"] = torch.cat(
                        [
                            state[expert_source + "gate_proj.weight"],
                            state[expert_source + "up_proj.weight"],
                        ],
                        dim=0,
                    )
                    remapped[expert_target + "linear_fc2.weight"] = state[
                        expert_source + "down_proj.weight"
                    ]
                shared_source = mlp_source + "shared_experts."
                shared_target = mlp_target + "shared_experts."
                remapped[shared_target + "linear_fc1.weight"] = torch.cat(
                    [
                        state[shared_source + "gate_proj.weight"],
                        state[shared_source + "up_proj.weight"],
                    ],
                    dim=0,
                )
                remapped[shared_target + "linear_fc2.weight"] = state[
                    shared_source + "down_proj.weight"
                ]

        for local_layer_index, layer in enumerate(model.model.language_model.layers):
            layer_index = layer.layer_number - 1
            _remap_one_layer(
                f"model.language_model.layers.{layer_index}.",
                f"model.language_model.decoder.layers.{local_layer_index}.",
                model.config.layer_types[layer_index],
                model.config.mlp_layer_types[layer_index],
            )

        # MTP (nextn) layers: HF keeps them as trailing layers
        # (model.language_model.layers.{num_layers + k}); the native model owns
        # them under mtp.layers.{k} with an inner transformer_layer prefix.
        mtp = getattr(model.model.language_model, "mtp", None)
        if mtp is not None:
            for mtp_layer_index in range(model.config.mtp_num_layers):
                source = f"model.language_model.layers.{model.config.num_layers + mtp_layer_index}."
                base = f"model.language_model.mtp.layers.{mtp_layer_index}."
                remapped[base + "eh_proj.weight"] = state[source + "eh_proj.weight"]
                remapped[base + "enorm.weight"] = state[source + "enorm.weight"]
                remapped[base + "hnorm.weight"] = state[source + "hnorm.weight"]
                remapped[base + "final_layernorm.weight"] = state[
                    source + "shared_head.norm.weight"
                ]
                # The MTP layer mirrors the last decoder layer's schedule
                # (KPool-DSA attention + routed MoE) but carries no mHC.
                _remap_one_layer(
                    source,
                    base + "transformer_layer.",
                    model.config.layer_types[-1],
                    model.config.mlp_layer_types[-1],
                    has_hyper_connections=False,
                )

        target_state = model.state_dict()
        for key, value in target_state.items():
            if key.endswith("._extra_state"):
                remapped[key] = value
        parameters = dict(model.named_parameters())
        tp_rank = model.pg_collection.tp.rank()
        tp_size = model.pg_collection.tp.size()
        conformed = {}
        for key, target in target_state.items():
            if key not in remapped:
                continue
            value = remapped[key]
            if value is None or target is None:
                conformed[key] = value
                continue
            if value.shape != target.shape:
                parameter = parameters.get(key)
                if parameter is None or not getattr(parameter, "tensor_model_parallel", False):
                    raise RuntimeError(
                        f"checkpoint shape mismatch for {key}: {tuple(value.shape)} != {tuple(target.shape)}"
                    )
                partition_dim = parameter.partition_dim
                if key.endswith("linear_fc1.weight") and model.config.gated_linear_unit:
                    gate, up = value.chunk(2, dim=partition_dim)
                    value = torch.cat(
                        (gate.chunk(tp_size, dim=partition_dim)[tp_rank],
                         up.chunk(tp_size, dim=partition_dim)[tp_rank]),
                        dim=partition_dim,
                    )
                else:
                    partition_stride = parameter.partition_stride
                    chunks = torch.chunk(value, tp_size * partition_stride, dim=partition_dim)
                    value = torch.cat(chunks[tp_rank::tp_size], dim=partition_dim)
                if value.shape != target.shape:
                    raise RuntimeError(
                        f"TP shard shape mismatch for {key}: {tuple(value.shape)} != {tuple(target.shape)}"
                    )
            conformed[key] = value
        return conformed

    @staticmethod
    def _dequantize_fp8_state(state: dict[str, torch.Tensor]) -> int:
        """Dequantize HF FP8 (e4m3) tensors in place; returns the count.

        GLM-5.3-Flash releases e4m3 weights with 128x128 block-wise
        ``weight_scale_inv`` scales; per-tensor scalar scales are also
        accepted. Consumed scale keys are removed from the state.
        """
        dequantized = 0
        for scale_key in [k for k in state if k.endswith(".weight_scale_inv")]:
            weight_key = scale_key[: -len(".weight_scale_inv")] + ".weight"
            if weight_key not in state:
                continue
            weight, scale = state[weight_key], state[scale_key]
            if "e4m3" not in str(weight.dtype):
                continue
            if scale.ndim == 2:
                out_f, in_f = weight.shape
                out_b, in_b = scale.shape
                if out_f != out_b * 128 or in_f != in_b * 128:
                    raise RuntimeError(
                        f"unsupported block-wise scale shape {tuple(scale.shape)} "
                        f"for FP8 weight {weight_key} {tuple(weight.shape)}"
                    )
                dequantized_weight = (
                    weight.to(torch.float32).view(out_b, 128, in_b, 128)
                    * scale.to(torch.float32).view(out_b, 1, in_b, 1)
                ).view(out_f, in_f)
            else:
                dequantized_weight = weight.to(torch.float32) * scale.to(torch.float32)
            state[weight_key] = dequantized_weight.to(torch.bfloat16)
            del state[scale_key]
            dequantized += 1
        return dequantized

    @classmethod
    def from_checkpoint(
        cls,
        checkpoint: str | Path,
        device: str | torch.device = "cpu",
        config_overrides: dict | None = None,
        **model_kwargs,
    ) -> "Glm5NextModel":
        checkpoint = Path(checkpoint)
        config = Glm5NextConfig.from_checkpoint(checkpoint, **(config_overrides or {}))
        model = cls(config, **model_kwargs).to(device=device, dtype=torch.bfloat16)
        # HF exports either one file or an index plus multiple safetensors shards.
        weight_index = checkpoint / "model.safetensors.index.json"
        if weight_index.exists():
            import json

            index = json.loads(weight_index.read_text())
            state = {}
            for shard_name in dict.fromkeys(index["weight_map"].values()):
                state.update(load_file(checkpoint / shard_name, device=str(device)))
        else:
            candidates = sorted(checkpoint.glob("*.safetensors"))
            if not candidates:
                raise FileNotFoundError(f"no safetensors checkpoint found under {checkpoint}")
            state = {}
            for shard in candidates:
                state.update(load_file(shard, device=str(device)))
        cls._dequantize_fp8_state(state)
        remapped = cls._remap_hf_state_dict(state, model)
        incompatible = model.load_state_dict(remapped, strict=False)
        if incompatible.missing_keys or incompatible.unexpected_keys:
            raise RuntimeError(
                f"checkpoint mapping is incomplete; missing={incompatible.missing_keys}, "
                f"unexpected={incompatible.unexpected_keys}"
            )
        return model
