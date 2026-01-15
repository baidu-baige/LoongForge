import os
from hydra import compose, initialize_config_dir
from hydra.core.global_hydra import GlobalHydra
from omegaconf import OmegaConf


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
    解析YAML中带@符号的配置行，提取键值对
    
    参数:
        yaml_lines (list): YAML文件内容行列表
        
    返回:
        dict: 包含配置值的字典，格式为:
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
            # 移除行首的"- "
            config_str = line[2:].strip()
            # 分割键和值
            key_part, value = config_str.split(':', 1)
            key_part = key_part.strip()
            value = value.strip()
            # 提取@前面的部分作为键
            if '@' in key_part:
                config_key = key_part.split('@')[0].split('/')[-1]
                result[config_key.strip()] = value
    return result

def parallel_param_parser(args, model_cfg, parallel_param, module_type):
    parallel_param_name = 'model.'+module_type + '.' + parallel_param
    if OmegaConf.select(model_cfg, parallel_param_name): # 若不存在则为None
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
