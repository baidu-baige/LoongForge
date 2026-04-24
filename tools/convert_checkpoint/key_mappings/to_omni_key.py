# Copyright 2026 The LoongForge Authors.
# SPDX-License-Identifier: Apache-2.0

"""Transform external checkpoint keys into LoongForge naming schemes."""

def transform_key(module, prefix_mapping, pipeline_model_parallel_size=1, tensor_model_parallel_size=1):
    """
    Transform keys in module using prefix mapping, supporting VPP.
    
    Args:
        module: The module containing model data
        prefix_mapping: Dictionary mapping old prefixes to new prefixes
        pipeline_model_parallel_size: Pipeline parallel size
        tensor_model_parallel_size: Tensor parallel size  
        num_virtual_stages_per_pipeline_rank: Number of virtual pipeline stages per pipeline rank
    """
    for i in range(len(module)):
        omni_key_model = {}
        for key, value in module[i]['model'].items():
            old_prefix, rest = key.split('.', 1)
            new_prefix = prefix_mapping.get(old_prefix)
            if new_prefix:
                new_key = f"{new_prefix}.{rest}"
            else:
                raise ValueError(f"Invalid prefix: {old_prefix}")
            omni_key_model[new_key] = value
        module[i]['model'] = omni_key_model


def transform_language_model_key(language_model, pipeline_model_parallel_size=1, tensor_model_parallel_size=1, num_virtual_stages_per_pipeline_rank=None):
    """
    Transform language model keys to omni format, supporting VPP.
    
    Args:
        language_model: The language model data
        pipeline_model_parallel_size: Pipeline parallel size
        tensor_model_parallel_size: Tensor parallel size
        num_virtual_stages_per_pipeline_rank: Number of virtual pipeline stages per pipeline rank
    """
    pp = pipeline_model_parallel_size
    tp = tensor_model_parallel_size
    vpp = num_virtual_stages_per_pipeline_rank
    
    if pp > 1:
        for pp_rank in range(pp):
            for tp_rank in range(tp):
                if vpp is not None:
                    for vpp_rank in range(vpp): 
                        omni_key_model = {}
                        cur_model = f'model{str(vpp_rank)}'
                        for key, value in language_model[pp_rank][tp_rank][cur_model].items():
                            _, rest = key.split('.', 1)
                            new_key = f"foundation_model.{rest}"
                            omni_key_model[new_key] = value
                        language_model[pp_rank][tp_rank][cur_model] = omni_key_model
                else:
                    omni_key_model = {}
                    for key, value in language_model[pp_rank][tp_rank]['model'].items():
                        _, rest = key.split('.', 1)
                        new_key = f"foundation_model.{rest}"
                        omni_key_model[new_key] = value
                    language_model[pp_rank][tp_rank]['model'] = omni_key_model
    else:
        for tp_rank in range(tp):
            omni_key_model = {}
            for key, value in language_model[tp_rank]['model'].items():
                _, rest = key.split('.', 1)
                new_key = f"foundation_model.{rest}"
                omni_key_model[new_key] = value
            language_model[tp_rank]['model'] = omni_key_model

