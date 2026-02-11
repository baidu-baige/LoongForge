""" Huggingface checkpoint converter """

import os

from dist_checkpoint.config.parallel_config import ParallelConfig
from dist_checkpoint.checkpoint.hf_checkpoint_converter import HfCheckpointConverter
from omegaconf import OmegaConf
from convert_checkpoint.utils.config_utils import parse_at_configs, load_config, update_overwrite

from convert_checkpoint.common.common_config import CommonConfig

def test_hf_to_mcore(tp, pp, vpp, pp_ranks, tp_ranks):
    parallel_config = ParallelConfig()
    parallel_config.vpp_size = vpp
    parallel_config.vpp_scheduler = None
    parallel_config.tp_size = tp
    parallel_config.pp_size = pp
    parallel_config.custom_pipeline_layers = None
    parallel_config.safetensors = True
    parallel_config.ep_size = None
    parallel_config.pp_ranks = pp_ranks
    parallel_config.ep_ranks = None
    parallel_config.tp_ranks = tp_ranks
    parallel_config.etp_ranks = None
    parallel_config.moe_grouped_gemm = True
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
        m_dict = test_hf_to_mcore(2, 2, 2, [1], [0, 1])
        for p in m_dict:
            print(f"{p=}: {m_dict[p].keys()}")
    if test_model == "qwen2":
        m_dict = test_hf_to_mcore(2, 2, None, [1], [1])
        for p in m_dict:
            for t in m_dict[p]:
                print(f"{p=}, {t=}: {m_dict[p][t]['model']['decoder.layers.0.mlp.linear_fc1.weight'].shape}")
    if test_model == "qwen3moe":
        m_dict = test_moe_hf_to_mcore()
        for p in m_dict:
            print(f"{p=}: {m_dict[p].keys()}")
            for e in m_dict[p]:
                print(f"{p=}: {e=}, {m_dict[p][e].keys()}")