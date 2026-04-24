# Copyright 2026 The LoongForge Authors.
# SPDX-License-Identifier: Apache-2.0

"""Merge Megatron expert checkpoint shards into a unified target checkpoint."""

import os
import sys
import argparse
import torch
import shutil
from os.path import dirname

from os.path import dirname
SCRIPT_DIR = dirname(os.path.abspath(__file__))
sys.path.append(dirname(dirname(SCRIPT_DIR)))

from convert_checkpoint.utils.ckpt_util import (
    load_megatron_checkpoint,
)
from convert_checkpoint.key_mappings.to_omni_key import (
    transform_key,
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
    group.add_argument("--expert_parallel_size", type=int, default=1, help="Expert parallel size.")
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

vision_model = load_megatron_checkpoint(args.vision_model_path)
adapter = load_megatron_checkpoint(args.adapter_path)
patch = load_megatron_checkpoint(args.vision_patch)

if args.config_file is None:
    etp_size = args.tensor_model_parallel_size
    dtp_size = args.tensor_model_parallel_size
    pp_size = args.pipeline_model_parallel_size
    etp = args.expert_tensor_parallel_size
    ep_size = args.expert_parallel_size
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
    ep_size = parallel_param_parser(args, model_cfg, 'expert_parallel_size', 'foundation')
    vpp_size = parallel_param_parser(args, model_cfg, 'num_virtual_stages_per_pipeline_rank', 'foundation')

    # Validate that etp_size divides dtp_size evenly
    assert dtp_size % etp_size == 0, f"decoder TP size ({dtp_size}) must be divisible by encoder TP size ({etp_size})"
    transform_key(vision_model, prefix_mapping, pp_size, etp_size)
    transform_key(adapter, prefix_mapping, pp_size, etp_size)
    transform_key(patch, prefix_mapping, pp_size, etp_size)

    assert ep_size > 1, "This merge file is designed for Expert Parallel and requires EP size must be > 1"


sub_dirs = sorted([x for x in os.listdir(args.language_model_path) if x.startswith("mp_rank")])
modules = [vision_model, adapter, patch]
for sub_dir in sub_dirs:
    splits = sub_dir.split('_')
    t = int(splits[2])
    et_rank = t % etp_size
    checkpoint_name = f"{sub_dir}/model_optim_rng.pt"
    checkpoint_path = os.path.join(args.language_model_path, checkpoint_name)
    print(f"Load Megatron Shard:{checkpoint_path}")
    ckpt = torch.load(checkpoint_path, map_location='cpu', weights_only=False)


    if 'model' in ckpt.keys():
        omni_key_model = {}
        for key, value in ckpt['model'].items():
            prefix, rest = key.split('.', 1)
            if prefix == 'language_model':
                new_key = f"foundation_model.{rest}"
                omni_key_model[new_key] = value
            else:
                omni_key_model[key] = value
        ckpt['model'] = omni_key_model
        if (args.pipeline_model_parallel_size == 1 or int(splits[3]) == 0) and \
                'foundation_model.embedding.word_embeddings.weight' in ckpt['model']:
            ckpt['model']['encoder_model.text_encoder.word_embeddings.weight'] = \
            ckpt['model']['foundation_model.embedding.word_embeddings.weight']
    else:
        for vpp_rank in range(vpp_size): 
            omni_key_model = {}
            cur_model = f'model{str(vpp_rank)}'
            for key, value in ckpt[cur_model].items():
                prefix, rest = key.split('.', 1)
                if prefix == 'language_model':
                    new_key = f"foundation_model.{rest}"
                    omni_key_model[new_key] = value
                else:
                    omni_key_model[key] = value
            ckpt[cur_model] = omni_key_model
        if (args.pipeline_model_parallel_size == 1 or int(splits[3]) == 0) and \
                'foundation_model.embedding.word_embeddings.weight' in ckpt['model0']:
            ckpt['model0']['encoder_model.text_encoder.word_embeddings.weight'] = \
            ckpt['model0']['foundation_model.embedding.word_embeddings.weight']

    if args.pipeline_model_parallel_size == 1 or int(splits[3]) == 0:
        for module in modules:
            if 'model' in ckpt.keys():
                merge_dict(module[et_rank]['model'], ckpt['model'])
                
            else:
                assert 'model0' in ckpt.keys()  # vpp
                merge_dict(module[et_rank]['model'], ckpt['model0'])
    
    try:
        torch.save(ckpt, checkpoint_path)
        print(f"Successfully Update Megatron shard: {checkpoint_path}")
    except Exception as e:
        print(f"Failed to update Megatron shard: {checkpoint_path}")

if os.path.exists(args.save_ckpt_path):
        shutil.rmtree(args.save_ckpt_path)
shutil.move(args.language_model_path, args.save_ckpt_path)
print(f"Saving Megatron shard to: {args.save_ckpt_path}")
