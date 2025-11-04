"""model configuration classes."""

from transformers import PretrainedConfig

from megatron.core.transformer import TransformerConfig
from aiak_training_omni.utils import ConfigAugments
import dataclasses
from dataclasses import fields, asdict
from megatron.training.activations import squared_relu
import torch.nn.functional as F
import torch

class BaseModelConfig(TransformerConfig, PretrainedConfig, ConfigAugments):
    """Base configuration class for AIAK training LLM models."""
    def __init__(self, **kwargs):
        PretrainedConfig.__init__(self, **kwargs)
        ConfigAugments.__init__(self)
        transformer_config_args = {}
        for f in dataclasses.fields(TransformerConfig):
            if hasattr(kwargs, f.name):
                transformer_config_args[f.name] = getattr(kwargs, f.name)
        transformer_config_args['persist_layer_norm'] = not kwargs['no_persist_layer_norm']
        transformer_config_args['layernorm_zero_centered_gamma'] = kwargs['apply_layernorm_1p']
        transformer_config_args['layernorm_epsilon'] = kwargs['norm_epsilon']
        transformer_config_args['deallocate_pipeline_outputs'] = True
        transformer_config_args['pipeline_dtype'] = kwargs['params_dtype']
        transformer_config_args['batch_p2p_comm'] = not kwargs['overlap_p2p_comm']
        transformer_config_args['num_moe_experts'] = kwargs['num_experts']
        transformer_config_args['rotary_interleaved'] = kwargs['rotary_interleaved']
        transformer_config_args['num_layers_in_first_pipeline_stage'] = kwargs['decoder_first_pipeline_num_layers']
        transformer_config_args['num_layers_in_last_pipeline_stage'] = kwargs['decoder_last_pipeline_num_layers']
        transformer_config_args['fp8_param'] = kwargs['fp8_param_gather']

        if 'activation_func_fp8_input_store' in kwargs:
            transformer_config_args['activation_func_fp8_input_store'] = kwargs['activation_func_fp8_input_store']

        if kwargs['swiglu']:
            transformer_config_args['activation_func'] = F.silu
            transformer_config_args['gated_linear_unit'] = True
            transformer_config_args['bias_activation_fusion'] = kwargs['bias_swiglu_fusion']
        else:
            transformer_config_args['bias_activation_fusion'] = kwargs['bias_gelu_fusion']

        if kwargs['squared_relu']:
            assert not kwargs['swiglu']
            transformer_config_args['activation_func'] = squared_relu

        if kwargs['init_method_xavier_uniform']:
            transformer_config_args['init_method'] = torch.nn.init.xavier_uniform_
            transformer_config_args['scaled_init_method'] = torch.nn.init.xavier_uniform_

        if kwargs['group_query_attention']:
            transformer_config_args['num_query_groups'] = kwargs['num_query_groups']
        else:
            transformer_config_args['num_query_groups'] = None

        transformer_config_args['config_logger_dir'] = kwargs['config_logger_dir']

        TransformerConfig.__init__(self, **transformer_config_args)