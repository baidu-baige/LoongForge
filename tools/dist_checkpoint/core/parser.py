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
from tools.convert_checkpoint.common.common_config import CommonConfig
from aiak_training_omni.utils.config_map import get_config_from_model_name


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
        self.data = None
        self.type = None
        self.config = None
        self.param_dict = None
        self.mappings = {}
        self.module_names = {}

        self._parse_from_args(args)

    def _parse_from_args(self, args):
        """Parse args to extract configuration."""
        # Step 1: Get model config path from model-name
        model_name = args.model_name
        config_path, config_name = get_config_from_model_name(model_name)

        # Step 2: Load model config YAML to get convert_file
        model_config_file = f"{config_path}/{config_name}.yaml"
        model_cfg = load_config(model_config_file)

        # Initialize config dict early
        self.config = {
            'model_config_file': model_config_file,
        }

        # Step 3: Check if this is a VLM model by looking at defaults
        is_vlm = self._is_vlm_model(model_cfg)

        if is_vlm:
            # VLM: Parse multiple modules
            self._parse_vlm_config(model_cfg, config_name)
            self.type = 'vlm'

            # update the prefix keys
            self._update_foundation_model_cfg()
            self._update_image_encoder_cfg()
            self._update_image_projector_cfg()
            
        else:
            # LLM: Parse single module (existing logic)
            self._parse_llm_config(model_cfg, config_name)
            self.type = 'llm'

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

    def _parse_vlm_config(self, model_cfg, config_name):
        """Parse VLM configuration with multiple modules."""
        # Get module names from defaults using parse_at_configs
        module_names = self._parse_module_names(model_cfg)

        # Process each module
        for module_type in ['image_encoder', 'image_projector', 'foundation']:
            if module_type not in module_names:
                continue

            module_info = module_names[module_type]
            module_name = module_info['name']
            prefix = module_info['prefix']
            convert_file = get_module_convert_file(model_cfg, module_type)

            if convert_file is None:
                continue

            # Load module config
            # For convert_file with defaults like "- qwen2.5@module: ???"
            # we need to override the key like "qwen2.5@module" with module_name
            override_key = f"{prefix}@module"
            hydra_overrides = {override_key: module_name}
            cfg = load_config(
                convert_file,
                hydra_overrides=hydra_overrides
            )
            OmegaConf.set_struct(cfg, False)

            # Build CommonConfig
            c_config = CommonConfig()
            c_config.load_convert_data(cfg)
            update_overwrite(model_cfg, c_config, module_type)

            # Store in mappings
            if module_type == 'foundation':
                self.mappings['foundation'] = c_config
            else:
                self.mappings[module_type] = c_config

    def _parse_llm_config(self, model_cfg, config_name):
        """Parse LLM configuration with single module."""
        # Get convert_file path from model_cfg
        convert_file = model_cfg.get('convert_file', None)
        assert convert_file is not None, f"convert_file not found in model config"

        # Get module_type from convert_file path (e.g., "qwen2.5" from ".../qwen2.5/ckpt_convert/...")
        module_type = convert_file.split('/')[-3]

        # Load convert_file with model config override
        # For convert_file with defaults like "- qwen2.5@module: ???"
        # we need to override the key like "qwen2.5@module" with config_name
        override_key = f"{module_type}@module"
        hydra_overrides = {override_key: config_name}
        cfg = load_config(
            convert_file,
            hydra_overrides=hydra_overrides
        )
        OmegaConf.set_struct(cfg, False)

        # Build CommonConfig
        c_config = CommonConfig()
        c_config.load_convert_data(cfg)
        update_overwrite(model_cfg, c_config, module_type)
        self.mappings['language_model'] = c_config

        # Update config for compatibility
        self.config['convert_file'] = convert_file

    def _parse_module_names(self, model_cfg):
        """Parse module names from model config defaults."""
        # Read the YAML file directly to parse @ syntax
        config_path, model_name = get_config_from_model_name(self.args.model_name)
        model_config_file = f"{config_path}/{model_name}.yaml"

        try:
            with open(model_config_file, 'r') as f:
                lines = f.readlines()
            result = parse_at_configs(lines)
            # print(f"DEBUG: parse_at_configs result: {result}")
            return result
        except Exception as e:
            # Fallback: try to get from model_cfg attributes
            print(f"DEBUG: Failed to parse YAML: {e}")
            module_names = {}
            if hasattr(model_cfg, 'model'):
                if hasattr(model_cfg.model, 'image_encoder'):
                    module_names['image_encoder'] = getattr(model_cfg.model.image_encoder, '_name', 'default')
                if hasattr(model_cfg.model, 'image_projector'):
                    module_names['image_projector'] = getattr(model_cfg.model.image_projector, '_name', 'default')
                if hasattr(model_cfg.model, 'foundation'):
                    module_names['foundation'] = getattr(model_cfg.model.foundation, '_name', 'default')
            print(f"DEBUG: fallback module_names: {module_names}")
            return module_names

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
        # =====加#的参数都不在train的args里，只在convert ckpt的args =====================
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


    def _update_foundation_model_cfg(self):
        for key, value in self.mappings['foundation'].data['name_map']['mcore'].items():
            if isinstance(value, str) and 'language_model' in value:
                self.mappings['foundation'].data['name_map']['mcore'][key] = value.replace('language_model', 'foundation_model')


    def _update_image_encoder_cfg(self):
        for key, value in self.mappings['image_encoder'].data['name_map']['mcore'].items():
            if isinstance(value, str) and 'vision_model' in value:
                self.mappings['image_encoder'].data['name_map']['mcore'][key] = value.replace('vision_model', 'encoder_model.image_encoder')

        for key, value in self.mappings['image_encoder'].data['vision_patch'].items():
            if isinstance(key, str) and 'vision_model' in key:
                self.mappings['image_encoder'].data['vision_patch'][key.replace('vision_model', 'encoder_model.image_encoder')] = value
                del self.mappings['image_encoder'].data['vision_patch'][key]


    def _update_image_projector_cfg(self):
        for key, value in self.mappings['image_projector'].data.items():
            if isinstance(key, str) and 'adapter' in key:
                self.mappings['image_projector'].data[key.replace('adapter', 'encoder_model.image_projector')] = value
                del self.mappings['image_projector'].data[key]

    
    def get_language_model_cfg(self):
        return self.mappings['language_model']

    def get_foundation_model_cfg(self):
        return self.mappings['foundation']

    def get_image_encoder_cfg(self):
        return self.mappings['image_encoder']
    
    def get_image_projector_cfg(self):
        return self.mappings['image_projector']


    def get_parallel_config(self):
        """Build ParallelConfig from param_dict."""
        parallel_params = {}

        parallel_fields = [
            'tp_size', 'pp_size', 'ep_size', 'etp_size', 'vpp_size',
            'custom_pipeline_layers', 'decoder_first_pipeline_num_layers',
            'decoder_last_pipeline_num_layers', 'moe_grouped_gemm',
            'vpp_scheduler', 'tp_ranks', 'pp_ranks', 'ep_ranks',
            'etp_ranks', 'safetensors',
            'max_workers', 'fp8_force_no_requant', 'force_pow_2_scales',
            'amax_epsilon', 'mtp_num_layers'
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


def parse_at_configs(yaml_lines):
    """
    Parse configuration lines with @ symbol in YAML to extract module names.

    Args:
        yaml_lines (list): List of YAML file content lines

    Returns:
        dict: Dictionary containing module info, e.g.:
            {
                'image_encoder': {'prefix': 'image_encoder', 'name': 'qwen2_5_vit'},
                'image_projector': {'prefix': 'image_projector', 'name': 'qwen_mlp_adapter'},
                'foundation': {'prefix': 'qwen2.5', 'name': 'qwen2_5_7b'}
            }
    """
    result = {}
    for line in yaml_lines:
        line = line.strip()
        # Match lines like: - ../../models/image_encoder@model.image_encoder: qwen2_5_vit
        if line.startswith('- ') and '@' in line and ':' in line:
            # Remove the leading "- "
            config_str = line[2:].strip()
            # Split key and value
            key_part, value = config_str.split(':', 1)
            key_part = key_part.strip()
            value = value.strip()
            # Extract the part before @ as the prefix (e.g., qwen2.5 from "../../models/qwen2.5@model.foundation")
            if '@' in key_part:
                # Get the part before @, e.g., "qwen2.5" or "image_encoder"
                prefix = key_part.split('@')[0].split('/')[-1]
                # Get the last part after @, e.g., "image_encoder" from "model.image_encoder"
                config_key = key_part.split('@')[1].split('.')[-1]
                result[config_key] = {'prefix': prefix, 'name': value}
    return result


def update_overwrite(model_cfg, module_cfg, module_type):
    if module_type != 'foundation':
        for key in module_cfg.data.module.keys():
            try:
                module_cfg.module[key] = model_cfg['model'][module_type][key]
            except:
                continue
    else:
        for key in module_cfg.data.module.keys():
            try:
                module_cfg.module[key] = model_cfg['model']['foundation'][key]
            except:
                continue


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
        '--seq-length', '4096',
        '--max-position-embeddings', '4096',
        '--micro-batch-size', '1',
        '--global-batch-size', '8',
        '--train-iters', '0',
        '--load', '/workspace/aiak-ckpt/Qwen2-VL-7B-Instruct',
        '--save-hf-path', '/workspace/aiak-ckpt/output',
        '--training-phase', 'pretrain',
    ]

    from aiak_training_omni.train.parser import parse_train_args

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
    print("mappings keys:", list(parser.mappings.keys()))
    if 'foundation' in parser.mappings:
        print("foundation config loaded:", parser.mappings['foundation'] is not None)
        print("foundation config:", parser.get_foundation_model_cfg().data)
    if 'image_encoder' in parser.mappings:
        print("image_encoder config loaded:", parser.mappings['image_encoder'] is not None)
        print("image_encoder config:", parser.get_image_encoder_cfg().data)
    if 'image_projector' in parser.mappings:
        print("image_projector config loaded:", parser.mappings['image_projector'] is not None)
        print("image_projector config:", parser.get_image_projector_cfg().data)
