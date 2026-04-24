# Copyright 2026 The LoongForge Authors.
# SPDX-License-Identifier: Apache-2.0

"""wan model provider"""

from megatron.core.transformer.spec_utils import import_module

from loongforge.utils import get_args, build_transformer_config, print_rank_0

from loongforge.models.common import BaseModelStditConfig

from .wan_model import WanModel
from .wan_layer_spec import get_wan_layer_with_te_spec
import torch


def wan2_2_i2v_model_provider(
    pre_process: bool = True,
    post_process: bool = True,
    parallel_output: bool = True,
) -> WanModel:
    """Builds the Wan model.

    Args:
        pre_process (bool, optional): Set to true if you need to compute embedings. Defaults to True.
        post_process (bool, optional): Set to true if you need to want to compute output logits/loss. Defaults to True.
        parallel_output (bool): whether to allgather the output logits

    Returns:
        WanModel: The returned model
    """
    args = get_args()

    print_rank_0("building WAN2_2_I2V model ...")

    config = build_transformer_config(args, config_class=BaseModelStditConfig)
    config.pipeline_dtype = torch.float32

    if args.spec is not None:
        transformer_layer_spec = import_module(args.spec)
    else:
        assert args.transformer_impl == "transformer_engine"
        transformer_layer_spec = get_wan_layer_with_te_spec()

    # Wan-AI/Wan2.2-I2V-A14B
    extra_config = {
        "has_image_input": False,
        "patch_size": [1, 2, 2],
        "in_dim": 36,
        "dim": 5120,
        "ffn_dim": 13824,
        "freq_dim": 256,
        "text_dim": 4096,
        "out_dim": 16,
        "num_heads": 40,
        "num_layers": 40,
        "eps": 1e-6,
        "require_clip_embedding": False,
    }
    model = WanModel(
        config=config,
        transformer_layer_spec=transformer_layer_spec,
        vocab_size=args.padded_vocab_size,
        max_sequence_length=args.max_position_embeddings,
        pre_process=pre_process,
        post_process=post_process,
        fp16_lm_cross_entropy=args.fp16_lm_cross_entropy,
        parallel_output=parallel_output,
        share_embeddings_and_output_weights=False,
        position_embedding_type=args.position_embedding_type,
        rotary_percent=args.rotary_percent,
        **extra_config,
    )

    return model