# Copyright 2026 The LoongForge Authors.
# SPDX-License-Identifier: Apache-2.0

"""Generate reverse key mappings between LoongForge and external checkpoint names."""

import os
import sys
import argparse
from os.path import dirname

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


def reverse_map_checkpoint_keys(reverse_mappings, shard_data, pipeline_parallel_size=1, tensor_parallel_size=1, num_virtual_stages_per_pipeline_rank=None):
    """
    Reverse the keys of checkpoint shards using predefined mapping rules.
    
    Args:
        shard_data: List of checkpoint shards or nested list for parallel models
        pipeline_parallel_size: Number of pipeline parallel ranks (default: 1)
        tensor_parallel_size: Number of tensor parallel ranks (default: 1)
        num_virtual_stages_per_pipeline_rank: Number of virtual pipeline stages per pipeline rank (default: None)
        
    Returns:
        List of shards with reversed keys, preserving original structure
    """
    
    def _reverse_model_keys(model_dict):
        """Reverse keys for a single model dictionary"""
        return {
            transform_key_reverse(key, reverse_mappings): value
            for key, value in model_dict.items()
        }
    
    if pipeline_parallel_size == 1:
        # Single pipeline parallel case
        if num_virtual_stages_per_pipeline_rank is None:
            return [
                {
                    'model': _reverse_model_keys(shard['model']),
                    'iteration': shard['iteration'],
                    'args': shard.get('args', None),
                    'checkpoint_version': shard.get('checkpoint_version', None)
                }
                for shard in shard_data
            ]
        else:
            raise ValueError("VPP requires pipeline_parallel_size > 1")
    else:
        # Multiple pipeline parallel case
        reversed_shard = []
        for pp_rank in range(pipeline_parallel_size):
            pp_reversed = []
            for tp_rank in range(tensor_parallel_size):
                shard = shard_data[pp_rank][tp_rank]
                
                if num_virtual_stages_per_pipeline_rank is None:
                    # Standard PP without VPP
                    if 'encoder_model.text_encoder.word_embeddings.weight' in shard['model']:
                        del shard['model']['encoder_model.text_encoder.word_embeddings.weight']
                    reversed_model = _reverse_model_keys(shard['model'])
                    pp_reversed.append({
                        'model': reversed_model,
                        'iteration': shard['iteration'],
                        'args': shard.get('args', None),
                        'checkpoint_version': shard.get('checkpoint_version', None)
                    })
                else:
                    # PP with VPP
                    reversed_models = {}
                    for vpp_rank in range(num_virtual_stages_per_pipeline_rank):
                        model_key = f'model{vpp_rank}'
                        if model_key in shard:
                            # Handle word embeddings deletion for VPP case
                            if 'encoder_model.text_encoder.word_embeddings.weight' in shard[model_key]:
                                del shard[model_key]['encoder_model.text_encoder.word_embeddings.weight']
                            reversed_models[model_key] = _reverse_model_keys(shard[model_key])
                    
                    pp_reversed.append({
                        **reversed_models,
                        'iteration': shard['iteration'],
                        'args': shard.get('args', None),
                        'checkpoint_version': shard.get('checkpoint_version', None)
                    })
            reversed_shard.append(pp_reversed)
        return reversed_shard


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
    os.makedirs(output_dir, exist_ok=True)

    print(f"🚀 Starting processing: {input_dir}")

    try:
        # Load all checkpoint shards from the directory
        print("📥 Loading checkpoint shards...")
        all_shards_data = load_megatron_checkpoint(input_dir)

        if not all_shards_data:
            print("❌ No checkpoint data found in directory")
            return

        # Reverse map keys for all loaded shards
        print("🔄 Reversing keys...")
        reversed_shards_data = reverse_map_checkpoint_keys(
            reverse_mappings, all_shards_data, pipeline_parallel_size, tensor_parallel_size, num_virtual_stages_per_pipeline_rank
        )

        # Save all reversed shards
        print("💾 Saving reversed shards...")
        save_megatron_checkpoint(reversed_shards_data, output_dir)

        print(f"✅ All shards processed successfully! Output: {output_dir}")

    except FileNotFoundError as e:
        print(f"❌ File not found error: {e}")
    except PermissionError as e:
        print(f"❌ Permission error: {e}")
    except Exception as e:
        print(f"❌ Unexpected error during processing: {e}")
        import traceback
        traceback.print_exc()

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