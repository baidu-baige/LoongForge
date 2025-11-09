"""AIAK Omni 模型提供者。
提供与现有训练框架兼容的模型构建接口。
"""

from __future__ import annotations

from typing import Any, Dict, Optional

from .omni_combination_model import OmniCombinationModel
from aiak_training_omni.utils import get_args
from megatron.core import mpu
import torch
from aiak_training_omni.utils import build_transformer_config, get_model_config
from aiak_training_omni.models.common import BaseModelConfig


def check_model_config(model_config: BaseModelConfig):
    """检查模型配置是否正确,包括隐藏大小和注意力头数等参数"""
    for config in model_config:  # TODO
        # check hidden_size and num_attention_heads
        print(config)
    return model_config


def omni_model_provider(
    pre_process: bool = True,
    post_process: bool = True,
    parallel_output: bool = True,
    # model_config: BaseModelConfig = None,
) -> OmniCombinationModel:
    """
    构建并返回一个Omni组合模型实例。
    该函数根据提供的参数和全局args配置,构建一个多模态组合模型。模型配置从args.model_name获取,
    并支持并行处理和前后处理选项。

    Args:
        pre_process: 是否进行预处理,默认为True
        post_process: 是否进行后处理,默认为True
        parallel_output: 是否启用并行输出,默认为True

    Returns:
        OmniCombinationModel: 构建好的组合模型实例

    Note:
        - 模型配置从args.model_name获取
        - 支持编码器管道并行处理
        - 包含语言模型相关参数配置
    """
    args = get_args()
    model_config = get_model_config()
    # check_model_config(model_config)
    # build_transformer_config(args)
    # FIXME: fix this if model_type is encoder_and_decoder
    if args.encoder_pipeline_model_parallel_size in [0, None]:
        add_encoder = mpu.is_pipeline_first_stage()

    # TODO: fp8 support
    model = OmniCombinationModel(
        model_config,
        parallel_output=parallel_output,
        pre_process=pre_process,
        post_process=post_process,
        add_encoder=add_encoder,
        language_vocab_size=args.padded_vocab_size,
        language_max_sequence_length=args.max_position_embeddings,
        fp16_lm_cross_entropy=args.fp16_lm_cross_entropy,
        share_embeddings_and_output_weights=not args.untie_embeddings_and_output_weights,
        language_position_embedding_type=args.position_embedding_type,
        language_rotary_percent=args.rotary_percent,
        language_rotary_base=args.rotary_base,
        language_rotary_dtype=torch.float32,  # if args.rope_in_fp32 else args.params_dtype,
        seq_len_interpolation_factor=args.rotary_seq_len_interpolation_factor,
    )

    return model
