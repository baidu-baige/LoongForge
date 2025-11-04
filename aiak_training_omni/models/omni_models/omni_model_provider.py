"""AIAK Omni 模型提供者。
提供与现有训练框架兼容的模型构建接口。
"""

from __future__ import annotations

from typing import Any, Dict, Optional

from .omni_combination_model import OmniCombinationModel
from .configuration import OmniModelConfig
from aiak_training_llm.models.factory import  get_model_config_from_model_name, get_model_config_from_file_path
from aiak_training_llm.utils import get_args, print_rank_0
from aiak_training_llm.utils.constants import OmniModulerType
from megatron.core import mpu
from megatron.training.activations import squared_relu
from megatron.core.transformer import TransformerConfig
import torch
import dataclasses
from aiak_training_llm.utils import build_transformer_config

def build_combination_model_config(model_name_list: list[str]):
    """根据模型名称列表构建复合模型配置。"""
    model_config_list = []
    for model_name in model_name_list:
        model_config = get_model_config_from_model_name(model_name)
        model_config_list.append(model_config)
    return OmniModelConfig.build_from_config_list(model_config_list) # TODO

# def overwrite_model_config(input_args, model_config: OmniModelConfig):
#     overwrite_model = input_args.specify_overwrite_model  \ 
#         if input_args.specify_overwrite_model != None \
#         else OmniModulerType.FoundationModel
#     moduler_config_list = model_config.get_all_sub_configs() # TODO
#     for moduler_config in moduler_config_list:
#         for key, value in vars(input_args).items():
#             if hasattr(moduler_config, key):
#                 if moduler_config.META_INFO['model_type'] == overwrite_model:
#                     setattr(moduler_config, key, value)
#             else:
#                 setattr(moduler_config, key, value)
#     return model_config

def check_model_config(model_config: OmniModelConfig):
    """检查模型配置是否正确,包括隐藏大小和注意力头数等参数"""
    for config in model_config.get_all_sub_configs(): # TODO
        # check hidden_size and num_attention_heads
        print(config)
    return model_config

def omni_model_provider(
    pre_process: bool = True,
    post_process: bool = True,
    parallel_output: bool = True,
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
    # assert args.model_name is not None ^ args.config_path is not None \ 
    # ^ args.model_componet is not None, " 必须提供模型名称或配置路径之一。"

    # generate model config
    # if args.config_path:
    #     print_rank_0(f'构建 AIAK Omni 模型: {args.config_path}')
    #     model_config = get_model_config_from_file_path(args.config_path)
    # elif args.model_name:
    #     print_rank_0(f'构建 AIAK Omni 模型: {args.model_name}')
    #     if '-' not in args.model_name:
    #         # building single model
    #         model_config = get_model_config_from_model_name(args.model_name)
    #     else:
    #         # building combination model
    #         model_name_list = args.model_name.split('_')
    #         model_config = build_combination_model_config(model_name_list)
    model_config = get_model_config_from_model_name(args.model_name)

    # model_config = core_transformer_config_from_args(model_src_config, config_class=config_class)
    # TODO: support single model
    # rewrite model config from args
    # overwrite_model_config(args, model_config)
    check_model_config(model_config)
    # build_transformer_config(args)
    # FIXME: fix this if model_type is encoder_and_decoder
    if args.encoder_pipeline_model_parallel_size in [0, None]:
        add_encoder = mpu.is_pipeline_first_stage()


    # TODO: fp8 support
    model = OmniCombinationModel(model_config, 
                                 train_args=args,
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
                                language_rotary_dtype=torch.float32 if args.rope_in_fp32 else args.params_dtype,
                                seq_len_interpolation_factor=args.rotary_seq_len_interpolation_factor) 
    
    return model
