""" Huggingface checkpoint converter """

import os

from dist_checkpoint.config.parallel_config import ParallelConfig
from dist_checkpoint.checkpoint.hf_checkpoint_converter import HfCheckpointConverter
from omegaconf import OmegaConf
from tools.convert_checkpoint.utils.config_utils import parse_at_configs, load_config, update_overwrite

from tools.convert_checkpoint.common.common_config import CommonConfig

def test_hf_to_mcore():
    parallel_config = ParallelConfig()
    parallel_config.vpp_size = 2
    parallel_config.vpp_scheduler = None
    parallel_config.tp_size = 2
    parallel_config.pp_size = 2
    parallel_config.custom_pipeline_layers = None
    parallel_config.safetensors = True
    parallel_config.ep_size = None
    parallel_config.pp_ranks = [1]
    parallel_config.ep_ranks = None
    parallel_config.tp_ranks = [1]
    parallel_config.etp_ranks = None
    config_file = os.environ.get('MODEL_CONFIG_FILE')
    convert_file = os.environ.get('CONVERT_FILE')
    ckpt_path = os.environ.get('LOAD')

    c_config = CommonConfig()
    with open(config_file, 'r') as f:
        module_names = parse_at_configs(f.readlines())
    module_type = convert_file.split('/')[-3]
    if module_names == {}: # llm
        cfg = load_config(convert_file, hydra_overrides={module_type+'@module='+config_file.split("/")[-1].split(".")[0]})
    else: # omni vlm
        cfg = load_config(convert_file, hydra_overrides = {module_type+'@module='+module_names[module_type]})
    OmegaConf.set_struct(cfg, False)

    model_cfg = load_config(config_file)
    if module_type != 'image_encoder':
        module_type = 'foundation'

    c_config.load_convert_data(cfg)

    update_overwrite(model_cfg, c_config, module_type)
    hf_convert = HfCheckpointConverter(parallel_config, c_config)
    hf_convert.set_mapping_cfg(c_config)
    hf_convert.load_hf(ckpt_path, c_config)
    m_dict = hf_convert.get_mcore_ckpt()
    return m_dict

def test_moe_hf_to_mcore():
    parallel_config = ParallelConfig()
    parallel_config.vpp_size = 2
    parallel_config.vpp_scheduler = None
    parallel_config.tp_size = 2
    parallel_config.pp_size = 2
    parallel_config.custom_pipeline_layers = None
    parallel_config.safetensors = True
    parallel_config.ep_size = 8
    parallel_config.etp_size = 1
    parallel_config.pp_ranks = [1]
    parallel_config.ep_ranks = [2,3]
    parallel_config.tp_ranks = [0,1]
    parallel_config.etp_ranks = [0]
    config_file = os.environ.get('MODEL_CONFIG_FILE')
    convert_file = os.environ.get('CONVERT_FILE')
    ckpt_path = os.environ.get('LOAD')

    c_config = CommonConfig()
    with open(config_file, 'r') as f:
        module_names = parse_at_configs(f.readlines())
    module_type = convert_file.split('/')[-3]
    if module_names == {}: # llm
        cfg = load_config(convert_file, hydra_overrides={module_type+'@module='+config_file.split("/")[-1].split(".")[0]})
    else: # omni vlm
        cfg = load_config(convert_file, hydra_overrides = {module_type+'@module='+module_names[module_type]})
    OmegaConf.set_struct(cfg, False)

    model_cfg = load_config(config_file)
    if module_type != 'image_encoder':
        module_type = 'foundation'

    c_config.load_convert_data(cfg)

    update_overwrite(model_cfg, c_config, module_type)
    hf_convert = HfCheckpointConverter(parallel_config, c_config)
    hf_convert.set_mapping_cfg(c_config)
    hf_convert.load_hf(ckpt_path, c_config)
    m_dict = hf_convert.get_mcore_ckpt()
    return m_dict

if __name__ == "__main__":
    test_model = os.environ.get('TEST_MODEL')
    if test_model == "mimo":
        m_dict = test_hf_to_mcore()
        for key in m_dict:
            print(f"{key}: {m_dict[key].keys()}")
    if test_model == "MiniMax":
        m_dict = test_moe_hf_to_mcore()
        for key in m_dict:
            for k in m_dict[key]:
                print(f"{key}: {k}, {m_dict[key][k].keys()}")