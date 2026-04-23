# Copyright 2026 The LoongForge Authors.
# SPDX-License-Identifier: Apache-2.0

import argparse
import os
import sys
from typing import Union
from hydra import compose, initialize_config_dir
from hydra.core.global_hydra import GlobalHydra
from omegaconf import OmegaConf

project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
sys.path.insert(0, project_root)
sys.path.insert(0, os.path.join(project_root, 'tools'))

from tools.dist_checkpoint.config.parallel_config import ParallelConfig
from loongforge.utils.config_map import get_config_from_model_name


class Parser:
    """Parse args object (from parse_train_args) into configuration"""

    def __init__(self, args: Union[argparse.Namespace, None] = None):
        """
        Initialize Parser from args object.

        Args:
            args: argparse.Namespace object from parse_train_args()
        """
        assert args is not None, "args is required"
        self.args = args
        self.type = None
        self.param_dict = None

        self._parse_from_args(args)

    def _parse_from_args(self, args):
        """Parse args to extract configuration."""
        # Step 1: Get model config path from model-name or config-file
        if args.config_file is None and args.model_name is not None:
            model_name = args.model_name
            config_path, config_name = get_config_from_model_name(model_name)
            model_config_file = f"{config_path}/{config_name}.yaml"
        elif args.config_file is not None:
            model_config_file = args.config_file
        else:
            raise ValueError('Either --model-name or --config-file must be provided.')

        # Step 2: Load model config YAML to get convert_file
        model_cfg = load_config(model_config_file)

        self.config_file = model_config_file 

        # Step 3: Check if this is a VLM model by looking at defaults
        is_vlm = self._is_vlm_model(model_cfg)

        if is_vlm:
            self.type = 'vlm'
            self.convert_file = model_cfg.model.foundation.convert_file
            self.vision_patch_convert_file = model_cfg.model.image_encoder.convert_file
            self.adapter_convert_file = model_cfg.model.image_projector.convert_file

            
        else:
            self.type = 'llm'
            self.convert_file = model_cfg.convert_file
            self.vision_patch_convert_file = None
            self.adapter_convert_file = None


        # Step 4: Build param_dict from args
        self.param_dict = self._build_param_dict(args)

    def _is_vlm_model(self, model_cfg):
        """Check if model is a VLM by looking for multiple modules in config."""
        # Check if model_cfg has 'model' key and it contains all VLM components
        if hasattr(model_cfg, 'model'):
            model = model_cfg.model
            has_image_encoder = hasattr(model, 'image_encoder')
            has_image_projector = hasattr(model, 'image_projector')
            has_foundation = hasattr(model, 'foundation')
            # VLM has all three modules
            if has_image_encoder and has_image_projector and has_foundation:
                return True
        return False


    def _build_param_dict(self, args) -> dict:
        """Build param_dict from args object."""
        param_dict = {}

        # Training related parameters
        param_dict['model_name'] = args.model_name
        param_dict['seq_length'] = getattr(args, 'seq_length', None)
        param_dict['max_position_embeddings'] = getattr(args, 'max_position_embeddings', None)
        param_dict['micro_batch_size'] = getattr(args, 'micro_batch_size', None)
        param_dict['global_batch_size'] = getattr(args, 'global_batch_size', None)
        param_dict['train_iters'] = getattr(args, 'train_iters', None)
        param_dict['training_phase'] = getattr(args, 'training_phase', None)
        param_dict['load'] = getattr(args, 'load', None)
        param_dict['save_hf_path'] = getattr(args, 'save_hf_path', None)

        # Parallel related parameters (for ParallelConfig)
        param_dict['tp_size'] = getattr(args, 'tensor_model_parallel_size', 1) or 1
        param_dict['pp_size'] = getattr(args, 'pipeline_model_parallel_size', 1) or 1
        param_dict['encoder_tp_size'] = getattr(args, 'encoder_tensor_model_parallel_size', None)
        if args.num_experts is not None:
            param_dict['ep_size'] = getattr(args, 'expert_model_parallel_size', None)
            param_dict['etp_size'] = getattr(args, 'expert_tensor_parallel_size', None)
            if param_dict['etp_size'] == param_dict['tp_size']:
                param_dict['etp_size'] = None
        else:
            param_dict['ep_size'] = None
            param_dict['etp_size'] = None
        param_dict['vpp_size'] = getattr(args, 'num_virtual_stages_per_pipeline_rank', None)
        param_dict['custom_pipeline_layers'] = getattr(args, 'custom_pipeline_layers', None)
        param_dict['decoder_first_pipeline_num_layers'] = getattr(args, 'decoder_first_pipeline_num_layers', None)
        param_dict['decoder_last_pipeline_num_layers'] = getattr(args, 'decoder_last_pipeline_num_layers', None)
        param_dict['moe_grouped_gemm'] = getattr(args, 'moe_grouped_gemm', False)
        param_dict['lora_alpha'] = getattr(args, 'lora_alpha', None)
        param_dict['lora_dim'] = getattr(args, 'lora_dim', None)
        # ===== Parameters marked with # are not in training args, only in convert ckpt args =====
        param_dict['vpp_scheduler'] = getattr(args, 'vpp_scheduler', None) #
        param_dict['safetensors'] = getattr(args, 'safetensors', True) #
        param_dict['max_workers'] = getattr(args, 'max_workers', 1) # 
        param_dict['fp8_force_no_requant'] = getattr(args, 'fp8_force_no_requant', False) # 
        param_dict['force_pow_2_scales'] = getattr(args, 'force_pow_2_scales', False) # 
        param_dict['amax_epsilon'] = getattr(args, 'amax_epsilon', 0.0) # 
        # ============================================================================

        param_dict['mtp_num_layers'] = getattr(args, 'mtp_num_layers', None)

        # Remove None values
        return {k: v for k, v in param_dict.items() if v is not None}



    def get_parallel_config(self):
        """Build ParallelConfig from param_dict."""
        parallel_params = {}

        parallel_fields = [
            'tp_size', 'encoder_tp_size', 'pp_size', 'ep_size', 'etp_size', 'vpp_size',
            'custom_pipeline_layers', 'decoder_first_pipeline_num_layers',
            'decoder_last_pipeline_num_layers', 'moe_grouped_gemm',
            'vpp_scheduler', 'tp_ranks', 'pp_ranks', 'ep_ranks',
            'etp_ranks', 'safetensors',
            'max_workers', 'fp8_force_no_requant', 'force_pow_2_scales',
            'amax_epsilon', 'mtp_num_layers', 'lora_alpha', 'lora_dim', 'enable_full_hetero_dp'
        ]

        for field in parallel_fields:
            if field in self.param_dict:
                parallel_params[field] = self.param_dict[field]

        return ParallelConfig(**parallel_params)


def get_module_convert_file(model_cfg, module_type):
    """Get convert_file path for a specific module from model_cfg."""
    try:
        if module_type == 'image_encoder':
            return model_cfg.model.image_encoder.convert_file
        elif module_type == 'image_projector':
            return model_cfg.model.image_projector.convert_file
        else:  # foundation
            return model_cfg.model.foundation.convert_file
    except AttributeError as e:
        # Debug: print what's available
        print(f"DEBUG: Failed to get convert_file for {module_type}")
        print(f"DEBUG: model_cfg.model keys: {list(model_cfg.model.keys()) if hasattr(model_cfg, 'model') else 'no model'}")
        if hasattr(model_cfg, 'model'):
            if hasattr(model_cfg.model, module_type):
                print(f"DEBUG: model_cfg.model.{module_type} keys: {list(model_cfg.model[module_type].keys())}")
            else:
                print(f"DEBUG: model_cfg.model does not have {module_type}")
        return None




def load_config(config_path, config_name=None, hydra_overrides=None):
    """
    Load configuration using the Hydra API
    """
    config_path = os.path.abspath(config_path)
    config_dir = os.path.dirname(config_path)
    config_name = os.path.basename(config_path)[:-5]

    if hydra_overrides is None:
        hydra_overrides = []
    # Convert dict to list if needed (Hydra supports both formats)
    elif isinstance(hydra_overrides, dict):
        hydra_overrides = [f"{k}={v}" for k, v in hydra_overrides.items()]

    GlobalHydra.instance().clear()

    with initialize_config_dir(config_dir=config_dir, version_base=None):
        cfg = compose(config_name=config_name, overrides=hydra_overrides)

    return cfg


if __name__ == "__main__":
    # Test example - qwen2_5_vl_7b (VLM)
    sys.argv = [
        'script',
        '--model-name', 'qwen2_5-vl-7b',
        '--tensor-model-parallel-size', '1',
        '--pipeline-model-parallel-size', '1',
        '--encoder-tensor-model-parallel-size', '2',
        '--seq-length', '4096',
        '--max-position-embeddings', '4096',
        '--micro-batch-size', '1',
        '--global-batch-size', '8',
        '--train-iters', '0',
        '--load', '/workspace/loongforge-ckpt/Qwen2-VL-7B-Instruct',
        '--save-hf-path', '/workspace/loongforge-ckpt/output',
        '--save-hf', 'true',
        '--training-phase', 'pretrain',
    ]

    from loongforge.train.parser import parse_train_args

    args = parse_train_args()
    parser = Parser(args)

    print("=" * 50)
    print("Model type:", parser.type)
    print("=" * 50)
    print("param_dict:")
    for k, v in parser.param_dict.items():
        print(f"  {k}: {v}")

    print("=" * 50)
    print("parallel_config:")
    print(f"  {parser.get_parallel_config()}")

    print("=" * 50)
    print(parser.config_file)
    print(parser.convert_file)
    print(parser.vision_patch_convert_file)
    print(parser.adapter_convert_file)
