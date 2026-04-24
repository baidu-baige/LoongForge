# Copyright 2026 The LoongForge Authors.
# SPDX-License-Identifier: Apache-2.0

"""Configuration loading and parsing utilities for checkpoint conversion."""

import os
from hydra import compose, initialize_config_dir
from hydra.core.global_hydra import GlobalHydra
from omegaconf import DictConfig, OmegaConf

from convert_checkpoint.common.common_config import CommonConfig
from convert_checkpoint.common.common_checkpoint import (
    BIAS,
    EXTRA_DATA,
    FIRST_LAYER_NAMES,
    LAST_LAYER_NAMES,
    LAYER_EXTRA_DATA,
    LAYER_IGNORE_TP,
    LAYER_IS_DIRECT_NAME,
    LAYER_IS_LAYERNORM,
    LAYER_NAME,
    LAYER_PREFIX,
    LAYERNORM_BIAS,
    LAYERNORM_WEIGHT,
    VISION_MAP,
    VISION_WORD_EMBEDDINGS,
    WEIGHT,
    WORD_EMBEDDINGS
)


def load_config(config_path, config_name=None, hydra_overrides=None):
    """
    Load configuration using the Hydra API, supports both directory+name and full path.

    Args:
        config_path: Either a directory path or full yaml file path
        config_name: Config file name without .yaml (required if config_path is directory)
        hydra_overrides: Optional list of override strings

    Returns:
        Hydra config object
    """
    # Convert to absolute path
    config_path = os.path.abspath(config_path)
    
    # Handle full file path case
    if config_path.endswith('.yaml'):
        config_dir = os.path.dirname(config_path)
        config_name = os.path.basename(config_path)[:-5]  # remove .yaml
    else:
        config_dir = config_path
        if config_name is None:
            raise ValueError("config_name is required when config_path is a directory")

    if hydra_overrides is None:
        hydra_overrides = []

    # Clear previous Hydra instance
    GlobalHydra.instance().clear()

    # Initialize and compose config
    with initialize_config_dir(config_dir=config_dir, version_base=None):
        cfg = compose(config_name=config_name, overrides=hydra_overrides)

    return cfg

def parse_at_configs(yaml_lines):
    """
    Parse configuration lines with @ symbol in YAML to extract key-value pairs.

    Args:
        yaml_lines (list): List of YAML file content lines

    Returns:
        dict: Dictionary containing configuration values in format:
            {
                'image_encoder': 'qwen_vit_rmsnorm_test',
                'image_projector': 'mlp_adapter_test',
                'qwen': 'qwen2_5_7b_test'
            }
    """
    result = {}
    for line in yaml_lines:
        line = line.strip()
        if line.startswith('- ') and '@' in line and ':' in line:
            # Remove the leading "- "
            config_str = line[2:].strip()
            # Split key and value
            key_part, value = config_str.split(':', 1)
            key_part = key_part.strip()
            value = value.strip()
            # Extract the part before @ as the key
            if '@' in key_part:
                config_key = key_part.split('@')[0].split('/')[-1]
                result[config_key.strip()] = value
    return result

def parallel_param_parser(args, model_cfg, parallel_param, module_type):
    parallel_param_name = 'model.'+module_type + '.' + parallel_param
    if OmegaConf.select(model_cfg, parallel_param_name): # Returns None if not exists
        parallel_size = model_cfg.model[module_type][parallel_param]
        setattr(args, parallel_param, parallel_size)
    elif hasattr(args, parallel_param):
        parallel_size = getattr(args, parallel_param)
    else:
        raise ValueError(f"Please provide {parallel_param} either in yaml or args")

    return parallel_size

def update_overwrite(model_cfg, module_cfg, module_type):
    for key in module_cfg.data.module.keys():
        try:
            module_cfg.data.module[key] = model_cfg['model'][module_type][key]
        except:
            continue

prefix_mapping = {'vision_model': 'encoder_model.image_encoder',
                  'adapter': 'encoder_model.image_projector'}

def get_adapter_config(config_file, convert_file):
    with open(config_file, 'r') as f:
        module_names = parse_at_configs(f.readlines())
    module_type = convert_file.split('/')[-3]
    name_map = load_config(convert_file, hydra_overrides = {module_type+'@module='+module_names[module_type]})
    adapter_name_map = {}
    for k1, k2 in name_map.items():
        if k1 != 'name_map' and k1 != 'module':
            adapter_name_map[k1] = k2

    return adapter_name_map

def get_vision_patch_config(config_file, convert_file):
    with open(config_file, 'r') as f:
            module_names = parse_at_configs(f.readlines())
    module_type = convert_file.split('/')[-3]
    cfg = load_config(convert_file, hydra_overrides = {module_type+'@module='+module_names[module_type]})

    return cfg.vision_patch

def parse_yaml_config(config_file, convert_file):
    c_config = CommonConfig()
    with open(config_file, 'r') as f:
        module_names = parse_at_configs(f.readlines())
    module_type = convert_file.split('/')[-3]
    # Filter out PEFT config keys (e.g., 'lora') which are not model module types
    module_keys = [k for k in module_names.keys() if k not in ['lora']]
    if len(module_keys) == 0: # llm
        cfg = load_config(convert_file, hydra_overrides={module_type+'@module='+config_file.split("/")[-1].split(".")[0]})
    else: # omni vlm
        cfg = load_config(convert_file, hydra_overrides = {module_type+'@module='+module_names[module_type]})
    OmegaConf.set_struct(cfg, False)

    model_cfg = load_config(config_file)
    if module_type != 'image_encoder':
        module_type = 'foundation'

    c_config.load_convert_data(cfg)

    update_overwrite(model_cfg, c_config, module_type)
    return c_config

def get_yaml_config(config_file, convert_file, adapter_convert_file=None, for_vlm=False):
    c_config = parse_yaml_config(config_file, convert_file)
    if adapter_convert_file is not None:
        adapter = get_adapter_config(config_file, adapter_convert_file)
        vision_patch = get_vision_patch_config(config_file, convert_file)
    else:
        adapter = None
        vision_patch = None

    return convert_vlm_config(c_config, adapter=adapter, vision_patch=vision_patch, for_vlm=for_vlm)

def convert_vlm_config(c_config, adapter=None, vision_patch=None, for_vlm=False):
    if adapter is not None:
        c_config = replace_vlm_config(c_config, adapter, vision_patch)
    if for_vlm:
        for name in [LAYER_PREFIX] + FIRST_LAYER_NAMES + LAST_LAYER_NAMES:
            if name not in c_config.get("name_map")["mcore"]:
                continue
            old_key = c_config.get("name_map")["mcore"].get(name, None)
            if old_key is None:
                continue
            if isinstance(old_key, (dict, DictConfig)):
                node_is_dict = True
                old_key = old_key[LAYER_NAME]
            else:
                node_is_dict = False
            old_prefix, rest = old_key.split('.', 1)
            if old_prefix == "language_model":
                new_key = f"foundation_model.{rest}"
                if node_is_dict:
                    c_config.get("name_map")["mcore"][name][LAYER_NAME] = new_key
                else:
                    c_config.get("name_map")["mcore"][name] = new_key

        word_embeddings = c_config.get("name_map")["mcore"].get(WORD_EMBEDDINGS, None)
        if word_embeddings == "foundation_model.embedding.word_embeddings":
            c_config.get("name_map")["huggingface"][VISION_WORD_EMBEDDINGS] = c_config.get("name_map")["huggingface"][WORD_EMBEDDINGS]
            c_config.get("name_map")["mcore"][VISION_WORD_EMBEDDINGS] = "encoder_model.text_encoder.word_embeddings"
    return c_config

def replace_vlm_config(c_config, adapter, vision_patch):
    name_map = {}
    for k1, k2 in adapter.items():
        if k1 in name_map:
            continue
        extra_data = True
        if k1.startswith("adapter.linear_fc1") or k1.startswith("adapter.linear_fc2"):
            extra_data = False
        name_map[k1] = {
            LAYER_NAME: k2,
            LAYER_EXTRA_DATA: extra_data
        }
    for k1, k2 in vision_patch.items():
        if k1 in name_map:
            continue
        name_map[k1] = {
            LAYER_NAME: k2,
            LAYER_EXTRA_DATA: False
        }
    c_config.get("name_map")["vision_patch"] = None

    hf_dict = {}

    # Key suffix list
    key_suffixes = [f".{WEIGHT}", f".{BIAS}", f".{LAYERNORM_WEIGHT}", f".{LAYERNORM_BIAS}"]

    for key, value in name_map.items():
        hf_name = value[LAYER_NAME]
        mcore_name = key
        hf_is_direct = True
        mcore_is_direct = True
        mcore_is_layernorm = False
        old_prefix, rest = mcore_name.split('.', 1)
        new_prefix = prefix_mapping.get(old_prefix)
        if new_prefix:
            mcore_name = f"{new_prefix}.{rest}"
        for suffix in key_suffixes:
            if value[LAYER_NAME].endswith(suffix):
                hf_name = value[LAYER_NAME][:-len(suffix)]
                hf_is_direct = False
            if key.endswith(suffix):
                mcore_name = mcore_name[:-len(suffix)]
                mcore_is_direct = False
                if suffix in [f".{LAYERNORM_WEIGHT}", f".{LAYERNORM_BIAS}"]:
                    mcore_is_layernorm = True
        if hf_name not in hf_dict:
            c_config.get("name_map")["huggingface"][f"{VISION_MAP}_{hf_name}"] = {LAYER_NAME: hf_name, LAYER_IS_DIRECT_NAME: hf_is_direct}
            c_config.get("name_map")["mcore"][f"{VISION_MAP}_{hf_name}"] = {LAYER_NAME: mcore_name, LAYER_IS_DIRECT_NAME: mcore_is_direct,
                                   LAYER_IS_LAYERNORM: mcore_is_layernorm, LAYER_EXTRA_DATA: value[LAYER_EXTRA_DATA]}
            hf_dict[hf_name] = True

    layer_prefix = c_config.get("name_map")["mcore"].get(LAYER_PREFIX, None)
    if layer_prefix:
        old_prefix, rest = layer_prefix.split('.', 1)
        new_prefix = prefix_mapping.get(old_prefix)
        if new_prefix:
            new_key = f"{new_prefix}.{rest}"
            c_config.get("name_map")["mcore"]["layer_prefix"] = new_key
    return c_config
