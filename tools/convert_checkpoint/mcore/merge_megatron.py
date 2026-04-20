# Copyright 2026 The LoongForge Authors.
# SPDX-License-Identifier: Apache-2.0

"""Merge Megatron Core checkpoint components into a unified target checkpoint."""

import os
import sys
import argparse
from os.path import dirname

from os.path import dirname
SCRIPT_DIR = dirname(os.path.abspath(__file__))
sys.path.append(dirname(dirname(SCRIPT_DIR)))

from convert_checkpoint.utils.ckpt_util import (
    load_megatron_checkpoint,
    save_megatron_checkpoint,
)
from convert_checkpoint.key_mappings.to_omni_key import (
    transform_key,
    transform_language_model_key,
)


from convert_checkpoint.utils.config_utils import load_config, parallel_param_parser



def parse_args(title=None):
    """Parse all arguments."""
    parser = argparse.ArgumentParser(description='Merger Arguments', allow_abbrev=False)
    group = parser.add_argument_group(title='checkpoint')
    group.add_argument('--language_model_path', type=str, help="Path to language model."),
    group.add_argument('--vision_model_path', type=str, help="Path to vision model."),
    group.add_argument('--vision_patch', type=str, help="Path to vision patch."),
    group.add_argument('--adapter_path', type=str, help="Path to adapter."),
    group.add_argument("--save_ckpt_path", type=str, help="Path to save checkpoint.")
    group.add_argument("--megatron_path", type=str, help="Base directory of Megatron repository")
    group.add_argument("--encoder_tensor_model_parallel_size", type=int, default=None, help="Tensor parallel size for encoder.")
    group.add_argument("--decoder_tensor_model_parallel_size", type=int, default=1, help="Tensor parallel size for decoder.")
    group.add_argument("--pipeline_model_parallel_size", type=int, default=1, help="Pipeline parallel size.")
    group.add_argument("--num_virtual_stages_per_pipeline_rank", type=int, default=None, help="Number of virtual pipeline stages per pipeline parallelism rank")
    group.add_argument('--config_file', type=str, help="Config file for model configuration.")


    return parser.parse_args()

def merge_dict(source, destination):
    """ merge two dictionaries recursively """
    for key, value in source.items():
        if isinstance(value, dict):
            node = destination.setdefault(key, {})
            merge_dict(value, node)
        else:
            destination[key] = value


args = parse_args()
if args.megatron_path is not None:
    sys.path.insert(0, args.megatron_path)

prefix_mapping = {'vision_model': 'encoder_model.image_encoder', 
                                'adapter': 'encoder_model.image_projector'}

print("===== merge megatron checkpoints ======")

language_model = load_megatron_checkpoint(args.language_model_path)
vision_model = load_megatron_checkpoint(args.vision_model_path)
adapter = load_megatron_checkpoint(args.adapter_path)
patch = load_megatron_checkpoint(args.vision_patch)

if args.config_file is None:
    etp_size = args.tensor_model_parallel_size
    dtp_size = args.tensor_model_parallel_size
    pp_size = args.pipeline_model_parallel_size
    etp = args.expert_tensor_parallel_size
    vpp_size = args.num_virtual_stages_per_pipeline_rank
else:
    model_cfg = load_config(args.config_file)
    if hasattr(args, 'encoder_tensor_model_parallel_size') and hasattr(args, 'decoder_tensor_model_parallel_size'):
        etp_size = args.encoder_tensor_model_parallel_size
        dtp_size = args.decoder_tensor_model_parallel_size
    else:
        etp_size = parallel_param_parser(args, model_cfg, 'tensor_model_parallel_size', 'image_encoder')
        dtp_size = parallel_param_parser(args, model_cfg, 'tensor_model_parallel_size', 'foundation')
    pp_size = parallel_param_parser(args, model_cfg, 'pipeline_model_parallel_size', 'foundation')
    vpp_size = parallel_param_parser(args, model_cfg, 'num_virtual_stages_per_pipeline_rank', 'foundation')

    # Validate that etp_size divides dtp_size evenly
    assert dtp_size % etp_size == 0, f"decoder TP size ({dtp_size}) must be divisible by encoder TP size ({etp_size})"
    transform_language_model_key(language_model, pp_size, dtp_size, vpp_size)
    transform_key(vision_model, prefix_mapping, pp_size, etp_size)
    transform_key(adapter, prefix_mapping, pp_size, etp_size)
    transform_key(patch, prefix_mapping, pp_size, etp_size)


# Handle PP=1 case
if pp_size == 1:
    for module in [vision_model, adapter, patch]:
        # Each module has etp_size shards
        for etp_rank in range(etp_size):
            # Modify to staggered allocation for Encoder groups
            dtp_start = etp_rank
            dtp_end = dtp_size
            step = etp_size  # Step size is etp_size
            
            # Merge encoder components into language model
            for dtp_rank in range(dtp_start, dtp_end, step):
                merge_dict(module[etp_rank]['model'],
                        language_model[dtp_rank]['model'])
    
    
    for dtp_rank in range(dtp_size):
        language_model[dtp_rank]['model']['encoder_model.text_encoder.word_embeddings.weight'] = \
            language_model[dtp_rank]['model']['foundation_model.embedding.word_embeddings.weight']


# Handle PP>1 case
else:
    for module in [vision_model, adapter, patch]:
        for etp_rank in range(etp_size):
            # Modify to staggered allocation for Encoder groups
            dtp_start = etp_rank
            dtp_end = dtp_size
            step = etp_size  # Step size is etp_size
            
            # Merge encoder components into language model
            for dtp_rank in range(dtp_start, dtp_end, step):
                if 'model' in language_model[0][dtp_rank].keys():
                    merge_dict(module[etp_rank]['model'],
                                language_model[0][dtp_rank]['model'])
                else:
                    assert 'model0' in language_model[0][dtp_rank].keys()
                    merge_dict(module[etp_rank]['model'],
                                language_model[0][dtp_rank]['model0'])

    # Handle word embeddings for encoder text encoder
    for pp_rank in range(pp_size):
        if 'model' in language_model[pp_rank][0].keys():
            if 'foundation_model.embedding.word_embeddings.weight' in language_model[pp_rank][0]['model']:
                for dtp_rank in range(dtp_size):
                    language_model[pp_rank][dtp_rank]['model']['encoder_model.text_encoder.word_embeddings.weight'] = \
                        language_model[pp_rank][dtp_rank]['model']['foundation_model.embedding.word_embeddings.weight']
        else:
            assert 'model0' in language_model[pp_rank][0].keys()
            if 'foundation_model.embedding.word_embeddings.weight' in language_model[pp_rank][0]['model0']:
                for dtp_rank in range(dtp_size):
                    language_model[pp_rank][dtp_rank]['model0']['encoder_model.text_encoder.word_embeddings.weight'] = \
                        language_model[pp_rank][dtp_rank]['model0']['foundation_model.embedding.word_embeddings.weight']

save_megatron_checkpoint(language_model, args.save_ckpt_path)
