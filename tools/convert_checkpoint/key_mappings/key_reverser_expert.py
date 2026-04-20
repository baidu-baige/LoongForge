# Copyright 2026 The LoongForge Authors.
# SPDX-License-Identifier: Apache-2.0

"""Generate reverse key mappings for expert-parallel LoongForge checkpoints."""

import os
import sys
import argparse
from os.path import dirname
import torch
import shutil

SCRIPT_DIR = dirname(os.path.abspath(__file__))
sys.path.append(dirname(dirname(SCRIPT_DIR)))

from convert_checkpoint.utils.ckpt_util import (
    load_megatron_checkpoint,
    save_megatron_checkpoint,
)


from convert_checkpoint.key_mappings.to_vanilla_key import (
    transform_key_reverse
)

from convert_checkpoint.utils.config_utils import load_config, parallel_param_parser


def reverse_map_single_checkpoint_keys(
    reverse_mappings,
    shard_path,
    vpp_size=1,
    delete_word_embeddings=True
):
    """
    Reverse the keys of a single checkpoint shard using predefined mapping rules.
    
    Args:
        reverse_mappings (dict): Mapping from old key patterns to new ones (used by transform_key_reverse)
        shard_path (str): Path to a single checkpoint shard file (e.g., .pt file)
        vpp_size (int): Number of virtual pipeline stages (>=1). 
                        If > 1, the shard is expected to contain 'model0', 'model1', ..., 'model{vpp_size-1}'.
                        If == 1, the shard uses the standard 'model' key.
        delete_word_embeddings (bool): Whether to delete 'encoder_model.text_encoder.word_embeddings.weight' if present
        
    Returns:
        dict: The checkpoint dict with reversed keys, same top-level structure as input shard
    """
    
    def _reverse_model_keys(model_dict):
        """Reverse keys for a single model dictionary"""
        return {
            transform_key_reverse(key, reverse_mappings): value
            for key, value in model_dict.items()
        }

    # Load the shard
    shard = torch.load(shard_path, map_location='cpu', weights_only=False)

    if vpp_size == 1:
        # Standard mode: single 'model' key
        model_dict = shard['model']
        if delete_word_embeddings and 'encoder_model.text_encoder.word_embeddings.weight' in model_dict:
            del model_dict['encoder_model.text_encoder.word_embeddings.weight']
        shard['model'] = _reverse_model_keys(model_dict)

    else:
        # VPP mode: multiple models named 'model0', 'model1', ..., 'model{vpp_size-1}'
        for vpp_rank in range(vpp_size):
            model_key = f'model{vpp_rank}'
            if model_key not in shard:
                raise KeyError(f"Expected key '{model_key}' in shard (vpp_size={vpp_size}), but not found. "
                               f"Available keys: {list(shard.keys())}")
            
            model_dict = shard[model_key]
            if delete_word_embeddings and 'encoder_model.text_encoder.word_embeddings.weight' in model_dict:
                del model_dict['encoder_model.text_encoder.word_embeddings.weight']
            shard[model_key] = _reverse_model_keys(model_dict)
    
    return shard


def process_checkpoint_shards(input_dir, output_dir, reverse_mappings, pipeline_parallel_size=1, tensor_parallel_size=1, num_virtual_stages_per_pipeline_rank=None):
    """
    Process all checkpoint shards by loading, reversing keys, and saving.

    Args:
        input_dir: Path to directory containing checkpoint shards (must exist)
        output_dir: Path to save reversed checkpoint shards (will be created)
        pipeline_parallel_size: Pipeline parallel configuration
        tensor_parallel_size: Tensor parallel configuration
        num_virtual_stages_per_pipeline_rank: Number of virtual pipeline stages per pipeline rank
    """
    # Ensure output directory exists
    os.makedirs(dirname(output_dir), exist_ok=True)

    print(f"🚀 Starting processing: {input_dir}")

    # Load all checkpoint shards from the directory
    print("📥 Loading checkpoint shards...")
    sub_dirs = sorted([x for x in os.listdir(input_dir) if x.startswith("mp_rank")])
    for sub_dir in sub_dirs:
        splits = sub_dir.split('_')
        t = int(splits[2])
        checkpoint_name = f"{sub_dir}/model_optim_rng.pt"
        checkpoint_path = os.path.join(input_dir, checkpoint_name)
        print(f"Load Megatron Shard:{checkpoint_path}")
        print("🔄 Reversing keys...")
        reversed_shard = reverse_map_single_checkpoint_keys(reverse_mappings, checkpoint_path)
        save_checkpoint_path = os.path.join(output_dir, checkpoint_name)
        os.makedirs(os.path.join(output_dir, sub_dir), exist_ok=True)
        print("💾 Saving reversed shard...")
        torch.save(reversed_shard, save_checkpoint_path)
        print(f"Saved Megatron Shard:{save_checkpoint_path}")


def parse_args(title=None):
    """Parse all arguments."""
    parser = argparse.ArgumentParser(description='Reverse checkpoint shard keys using predefined mappings')
    parser.add_argument("--load_omni_ckpt_path", type=str, help="Path to load the omni checkpoint.")
    parser.add_argument("--save_original_ckpt_path", type=str, help="Path to save the original checkpoint.")
    # parser.add_argument("--encoder_tensor_model_parallel_size", type=int, default=None, help="Tensor parallel size for encoder.")
    parser.add_argument("--decoder_tensor_model_parallel_size", type=int, default=1, help="Tensor parallel size for decoder.") # Use this as tp
    parser.add_argument("--pipeline_model_parallel_size", type=int, default=1, help="Pipeline parallel size.")
    parser.add_argument("--num_virtual_stages_per_pipeline_rank", type=int, default=None, help="Number of virtual pipeline stages per pipeline parallelism rank")
    parser.add_argument('--config_file', type=str, help="Config file for model configuration.")


    return parser.parse_args()


def main():
    args = parse_args()

    model_cfg = load_config(args.config_file)

    # In heterogeneous case, only need to use dtp size
    # etp_size = parallel_param_parser(args, model_cfg, 'tensor_model_parallel_size', 'image_encoder')
    if hasattr(args, 'decoder_tensor_model_parallel_size'):
        dtp_size = args.decoder_tensor_model_parallel_size
    else:
        dtp_size = parallel_param_parser(args, model_cfg, 'tensor_model_parallel_size', 'foundation')
    pp_size = parallel_param_parser(args, model_cfg, 'pipeline_model_parallel_size', 'foundation')
    vpp_size = parallel_param_parser(args, model_cfg, 'num_virtual_stages_per_pipeline_rank', 'foundation')

    # Validate and normalize input path
    input_dir = os.path.abspath(args.load_omni_ckpt_path)
    if not os.path.exists(input_dir):
        print(f"❌ Error: Input directory does not exist: {input_dir}")
        sys.exit(1)

    # Auto-generate output directory path when not specified
    if args.save_original_ckpt_path is None:
        parent_dir = os.path.dirname(input_dir)
        dir_name = os.path.basename(input_dir)
        args.output_dir = os.path.join(parent_dir, f"{dir_name}_reversed")

    # Create output directory
    output_dir = os.path.abspath(args.save_original_ckpt_path)
    os.makedirs(output_dir, exist_ok=True)

    print(f"📂 Input Path: {input_dir}")
    print(f"📂 Output Path: {output_dir}")
    print(f"⚙️  Pipeline Parallel Size: {pp_size}")
    print(f"⚙️  Decoder Tensor Parallel Size: {dtp_size}")
    print(f"⚙️  Virtual Stages per PP Rank: {vpp_size}")
    print("-" * 50)
    
    # Validate PP and VPP combination
    if args.pipeline_model_parallel_size == 1:
        assert args.num_virtual_stages_per_pipeline_rank is None, "VPP requires PP > 1"
    
    # Get reverse mapping

    reverse_mapping = {'encoder_model.image_encoder': 'vision_model',
                        'encoder_model.image_projector': 'adapter',
                        'foundation_model': 'language_model'}
    
    # Process shards
    process_checkpoint_shards(
        input_dir,
        output_dir,
        reverse_mapping,
        pp_size,
        dtp_size,
        vpp_size,
    )


if __name__ == "__main__":
    main()