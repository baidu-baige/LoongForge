# Copyright 2026 The LoongForge Authors.
# SPDX-License-Identifier: Apache-2.0

"""Model provider for the Phase 3 Qwen2.5-VL MiMo path."""

from __future__ import annotations

from typing import Optional

import torch
from transformers import AutoModel
from megatron.core.models.mimo import MimoModel, MimoModelConfig
from megatron.core.models.mimo.config.role import MIMO_LANGUAGE_MODULE_KEY
from megatron.core.process_groups_config import ProcessGroupCollection
from megatron.core.transformer.spec_utils import ModuleSpec

from loongforge.models.encoder.qwen2_vl_vision_models.vision_model import (
    Qwen2VisionModelWithRMSNorm,
)
from loongforge.models.encoder.qwen2_vl_vision_models.adapter import Adapter
from loongforge.models.mimo_models.mimo_vlm_model import LoongForgeMimoVLMModel
from loongforge.models.mimo_models.qwen25_vl import Qwen25VisionModalitySubmodules
from loongforge.utils import get_args, get_model_config


_TOPOLOGY_CACHE = {}


def validate_mimo_vlm_feature_flags(args):
    """Reject MiMo VLM features without a native lifecycle path."""
    if getattr(args, "enable_encoder_hetero_dp", False) or getattr(
        args, "enable_full_hetero_dp", False
    ):
        raise NotImplementedError(
            "MiMo VLM does not support heterogeneous encoder DP in this release"
        )
    if getattr(args, "use_vit_dp_balance", False) or getattr(
        args, "use_vlm_dp_balance", False
    ):
        raise NotImplementedError(
            "MiMo VLM does not support VLM or ViT DP balance in this release"
        )
    if getattr(args, "enable_chunkpipe", False):
        raise NotImplementedError("MiMo VLM does not support ChunkPipe in this release")


def _get_topology(args, image_tp: int):
    """Create one explicit PGC topology shared by all virtual PP chunks."""
    from examples.mimo.training.topology import ModuleGridSpec, create_topology

    world = torch.distributed.get_world_size()
    pp_size = args.pipeline_model_parallel_size
    if args.context_parallel_size != 1:
        raise NotImplementedError("MiMo colocated vision currently requires language CP1")
    vision_ranks = world // pp_size
    if vision_ranks % image_tp:
        raise ValueError("Image encoder TP must divide the number of first-stage language ranks")
    key = (
        world,
        args.tensor_model_parallel_size,
        args.context_parallel_size,
        args.pipeline_model_parallel_size,
        image_tp,
    )
    topology = _TOPOLOGY_CACHE.get(key)
    if topology is None:
        topology = create_topology(
            [
                ModuleGridSpec(
                    name=MIMO_LANGUAGE_MODULE_KEY,
                    num_ranks=world,
                    tp=args.tensor_model_parallel_size,
                    cp=args.context_parallel_size,
                    pp=args.pipeline_model_parallel_size,
                ),
                ModuleGridSpec(
                    name="images",
                    num_ranks=vision_ranks,
                    tp=image_tp,
                    cp=1,
                    pp=1,
                ),
            ]
        )
        _TOPOLOGY_CACHE[key] = topology
    return topology


def _vision_spec(config, vision_pg_collection: ProcessGroupCollection) -> ModuleSpec:
    return ModuleSpec(
        module=Qwen25VisionModalitySubmodules,
        params={"pg_collection": vision_pg_collection},
        submodules={
            "encoders": {
                "image_encoder": ModuleSpec(
                    module=Qwen2VisionModelWithRMSNorm,
                    params={"config": config.image_encoder},
                )
            },
            "input_projections": [
                ModuleSpec(
                    module=Adapter,
                    params={
                        "config": config.image_projector,
                        "input_size": config.image_encoder.hidden_size,
                        "output_size": config.foundation.hidden_size,
                        "spatial_merge_size": getattr(
                            config.image_encoder, "spatial_merge_size", 2
                        ),
                        "use_postshuffle_norm": getattr(
                            config.image_encoder, "use_postshuffle_norm", False
                        ),
                    },
                )
            ],
        },
    )


def mimo_vlm_model_provider(
    pre_process: bool = True,
    post_process: bool = True,
    parallel_output: bool = True,
    vp_stage: Optional[int] = None,
    config=None,
    pg_collection=None,
    **kwargs,
):
    """Build the MiMo VLM backend.

    MCore's provider fallback passes its resolved config and process groups;
    LoongForge still owns the concrete model config and reads it globally.
    """
    args = get_args()
    config = get_model_config()
    validate_mimo_vlm_feature_flags(args)

    foundation = config.foundation
    foundation.padded_vocab_size = args.padded_vocab_size
    foundation.fp16_lm_cross_entropy = args.fp16_lm_cross_entropy
    foundation.rotary_seq_len_interpolation_factor = (
        args.rotary_seq_len_interpolation_factor
    )
    foundation.use_rope_scaling = args.use_rope_scaling
    foundation.rope_scaling_factor = args.rope_scaling_factor
    foundation.max_position_embeddings = args.max_position_embeddings
    foundation.rotary_base = args.rotary_base
    foundation.rotary_percent = args.rotary_percent

    image_encoder = config.image_encoder
    image_encoder.pipeline_model_parallel_size = 1
    image_encoder.context_parallel_size = 1
    image_encoder.sequence_parallel = False
    image_encoder.tp_comm_overlap = False

    language_class = AutoModel._model_mapping[type(foundation)]
    language_spec = ModuleSpec(
        module=language_class,
        params={
            "config": foundation,
            "pre_process": pre_process,
            "post_process": post_process,
            "parallel_output": parallel_output,
            "scatter_embedding_sequence_parallel": False,
            "vp_stage": vp_stage,
        },
    )

    image_token_id = getattr(image_encoder, "image_token_id", None)
    if image_token_id is None:
        raise ValueError("Qwen2.5-VL image_encoder config is missing image_token_id")

    image_tp = image_encoder.tensor_model_parallel_size
    world = torch.distributed.get_world_size()
    if world % image_tp:
        raise ValueError(
            f"world_size={world} is not divisible by image encoder TP={image_tp}"
        )
    # Vision is not VPP-sharded. Build it only in the first virtual chunk;
    # later chunks are language-only views over the same shared topology.
    topology = _get_topology(args, image_tp)
    language_pg = topology.module_pgs[MIMO_LANGUAGE_MODULE_KEY]
    vision_pg = topology.module_pgs["images"]
    language_spec.params["pg_collection"] = language_pg
    owns_vision = vp_stage in (None, 0) and pre_process
    modality_specs = {"images": _vision_spec(config, vision_pg)} if owns_vision else {}
    special_token_ids = {"images": image_token_id} if owns_vision else {}
    mimo_config = MimoModelConfig(
        language_model_spec=language_spec,
        modality_submodules_spec=modality_specs,
        special_token_ids=special_token_ids,
        module_to_grid_map=topology.grids,
        kv_format="sbhd",
    )
    mimo_model = MimoModel(
        mimo_config,
        cp_group=language_pg.cp,
        tp_group=language_pg.tp,
    )
    return LoongForgeMimoVLMModel(
        mimo_model=mimo_model,
        topology=topology,
        model_type=config.model_type,
        pre_process=pre_process,
        post_process=post_process,
        image_spatial_merge_size=getattr(image_encoder, "spatial_merge_size", 2),
    )
