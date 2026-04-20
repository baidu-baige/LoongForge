# Copyright 2026 The LoongForge Authors.
# SPDX-License-Identifier: Apache-2.0

""" Huggingface checkpoint converter """

import os
import shutil
import time

from dist_checkpoint.config.parallel_config import ParallelConfig
from dist_checkpoint.checkpoint.hf_checkpoint_converter import HfCheckpointConverter
from omegaconf import OmegaConf
from convert_checkpoint.utils.config_utils import get_yaml_config

from convert_checkpoint.common.common_config import CommonConfig

from convert_checkpoint.utils.utils import(
    check_all_done,
    make_hf_sub_checkpoints
)


def test_hf_to_mcore(tp, pp, vpp, pp_ranks, tp_ranks, encoder_tp_size=None):
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
    #vlm
    parallel_config.encoder_tp_size = encoder_tp_size
    vision_patch_convert_file = os.environ.get('VISION_PATCH_CONVERT_FILE', None)
    adapter_convert_file = os.environ.get('ADAPTER_CONVERT_FILE', None)

    c_config = get_yaml_config(config_file, convert_file, for_vlm=(vision_patch_convert_file is not None))
    c_vision_patch_config = get_yaml_config(
        config_file, vision_patch_convert_file,
        adapter_convert_file=adapter_convert_file) if vision_patch_convert_file is not None else None

    hf_convert = HfCheckpointConverter(parallel_config, c_config, vision_patch_config=c_vision_patch_config)
    m_dict = hf_convert.get_mcore_ckpt(ckpt_path)
    return m_dict

def test_mcore_to_hf(tp, pp, vpp, pp_ranks, tp_ranks, mcore_dict, encoder_tp_size=None):
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
    ckpt_path = os.environ.get('SAVE')
    #vlm
    parallel_config.encoder_tp_size = encoder_tp_size
    vision_patch_convert_file = os.environ.get('VISION_PATCH_CONVERT_FILE', None)
    adapter_convert_file = os.environ.get('ADAPTER_CONVERT_FILE', None)

    c_config = get_yaml_config(config_file, convert_file, for_vlm=(vision_patch_convert_file is not None))
    c_vision_patch_config = get_yaml_config(
        config_file, vision_patch_convert_file,
        adapter_convert_file=adapter_convert_file) if vision_patch_convert_file is not None else None

    hf_convert = HfCheckpointConverter(parallel_config, c_config, vision_patch_config=c_vision_patch_config)
    hf_convert.save_hf_ckpt(mcore_dict, ckpt_path)

    rank_id = int(os.getenv('RANK', '0'))
    if rank_id == 0:
        done_dir = os.path.join(ckpt_path, "dones")
        while True:
            checked_done = check_all_done(done_dir, pp, ep)
            if checked_done:
                shutil.rmtree(done_dir)
                break
        make_hf_sub_checkpoints(ckpt_path)

def test_moe_hf_to_mcore(tp, pp, vpp, ep, etp, pp_ranks, tp_ranks, ep_ranks, etp_ranks, encoder_tp_size=None):
    parallel_config = ParallelConfig()
    parallel_config.vpp_size = vpp
    parallel_config.vpp_scheduler = None
    parallel_config.tp_size = tp
    parallel_config.pp_size = pp
    parallel_config.custom_pipeline_layers = None
    parallel_config.safetensors = True
    parallel_config.ep_size = ep
    parallel_config.etp_size = etp
    parallel_config.pp_ranks = pp_ranks
    parallel_config.ep_ranks = ep_ranks
    parallel_config.tp_ranks = tp_ranks
    parallel_config.etp_ranks = etp_ranks
    parallel_config.moe_grouped_gemm = True
    config_file = os.environ.get('MODEL_CONFIG_FILE')
    convert_file = os.environ.get('CONVERT_FILE')
    ckpt_path = os.environ.get('LOAD')
    #vlm
    parallel_config.encoder_tp_size = encoder_tp_size
    vision_patch_convert_file = os.environ.get('VISION_PATCH_CONVERT_FILE', None)
    adapter_convert_file = os.environ.get('ADAPTER_CONVERT_FILE', None)

    c_config = get_yaml_config(config_file, convert_file, for_vlm=(vision_patch_convert_file is not None))
    c_vision_patch_config = get_yaml_config(
        config_file, vision_patch_convert_file,
        adapter_convert_file=adapter_convert_file) if vision_patch_convert_file is not None else None

    hf_convert = HfCheckpointConverter(parallel_config, c_config, vision_patch_config=c_vision_patch_config)
    m_dict = hf_convert.get_mcore_ckpt(ckpt_path)
    return m_dict

def test_moe_mcore_to_hf(tp, pp, vpp, ep, etp, pp_ranks, tp_ranks, ep_ranks, etp_ranks, mcore_dict, encoder_tp_size=None):
    parallel_config = ParallelConfig()
    parallel_config.vpp_size = vpp
    parallel_config.vpp_scheduler = None
    parallel_config.tp_size = tp
    parallel_config.pp_size = pp
    parallel_config.custom_pipeline_layers = None
    parallel_config.safetensors = True
    parallel_config.ep_size = ep
    parallel_config.etp_size = etp
    parallel_config.pp_ranks = pp_ranks
    parallel_config.ep_ranks = ep_ranks
    parallel_config.tp_ranks = tp_ranks
    parallel_config.etp_ranks = etp_ranks
    parallel_config.moe_grouped_gemm = True
    config_file = os.environ.get('MODEL_CONFIG_FILE')
    convert_file = os.environ.get('CONVERT_FILE')
    ckpt_path = os.environ.get('SAVE')
    #vlm
    parallel_config.encoder_tp_size = encoder_tp_size
    vision_patch_convert_file = os.environ.get('VISION_PATCH_CONVERT_FILE', None)
    adapter_convert_file = os.environ.get('ADAPTER_CONVERT_FILE', None)

    c_config = get_yaml_config(config_file, convert_file, for_vlm=(vision_patch_convert_file is not None))
    c_vision_patch_config = get_yaml_config(
        config_file, vision_patch_convert_file,
        adapter_convert_file=adapter_convert_file) if vision_patch_convert_file is not None else None

    hf_convert = HfCheckpointConverter(parallel_config, c_config, vision_patch_config=c_vision_patch_config)
    hf_convert.save_hf_ckpt(mcore_dict, ckpt_path)

    rank_id = int(os.getenv('RANK', '0'))
    if rank_id == 0:
        done_dir = os.path.join(ckpt_path, "dones")
        while True:
            checked_done = check_all_done(done_dir, pp, ep)
            if checked_done:
                shutil.rmtree(done_dir)
                break
        make_hf_sub_checkpoints(ckpt_path)


if __name__ == "__main__":
    test_model = os.environ.get('TEST_MODEL')
    tp = int(os.environ.get('TP_SIZE')) if os.environ.get('TP_SIZE') is not None else 1
    encoder_tp_size = int(os.environ.get('ENCODER_TP_SIZE')) if os.environ.get('ENCODER_TP_SIZE') is not None else 1
    pp = int(os.environ.get('PP_SIZE')) if os.environ.get('PP_SIZE') is not None else 1
    vpp = int(os.environ.get('VPP_SIZE')) if os.environ.get('VPP_SIZE') is not None else None
    pp_ranks = [int(x.strip()) for x in os.environ.get('PP_RANKS').split(',') if x.strip()]
    tp_ranks = [int(x.strip()) for x in os.environ.get('TP_RANKS').split(',') if x.strip()]
    ep = int(os.environ.get('EP_SIZE')) if os.environ.get('EP_SIZE') is not None else None
    etp = int(os.environ.get('ETP_SIZE')) if os.environ.get('ETP_SIZE') is not None else None
    ep_ranks = [int(x.strip()) for x in (os.environ.get('EP_RANKS') if os.environ.get('EP_RANKS') is not None else "").split(',') if x.strip()]
    etp_ranks = [int(x.strip()) for x in (os.environ.get('ETP_RANKS') if os.environ.get('ETP_RANKS') is not None else "").split(',') if x.strip()]
    if test_model == "mimo":
        m_dict = test_hf_to_mcore(tp, pp, vpp, pp_ranks, tp_ranks)
        for p in m_dict:
            print(f"{p=}: {m_dict[p].keys()}")
        test_mcore_to_hf(tp, pp, vpp, pp_ranks, tp_ranks, m_dict)
    if test_model == "qwen2":
        m_dict = test_hf_to_mcore(tp, pp, vpp, pp_ranks, tp_ranks)
        for p in m_dict:
            for t in m_dict[p]:
                print(f"{p=}, {t=}: {m_dict[p][t]['model']['decoder.layers.0.mlp.linear_fc1.weight'].shape}")
    if test_model == "qwen3moe":
        m_dict = test_moe_hf_to_mcore(tp, pp, vpp, ep, etp, pp_ranks, tp_ranks, ep_ranks, etp_ranks)
        for p in m_dict:
            print(f"{p=}: {m_dict[p].keys()}")
            for e in m_dict[p]:
                for t in m_dict[p][e]:
                    print(f"{p=}, {e=}, {t=}")
        test_moe_mcore_to_hf(tp, pp, vpp, ep, etp, pp_ranks, tp_ranks, ep_ranks, etp_ranks, m_dict)
    if test_model == "qwen2.5_vl_3b":
        m_dict = test_hf_to_mcore(tp, pp, vpp, pp_ranks, tp_ranks, encoder_tp_size)
        for p in m_dict:
            for t in m_dict[p]:
                print(f"{p=}, {m_dict[p].keys()}")
        test_mcore_to_hf(tp, pp, vpp, pp_ranks, tp_ranks, m_dict, encoder_tp_size)
    if test_model == "qwen3_vl_30b_a3b":
        m_dict = test_moe_hf_to_mcore(tp, pp, vpp, ep, etp, pp_ranks, tp_ranks, ep_ranks, etp_ranks, encoder_tp_size)
        for p in m_dict:
            print(f"{p=}: {m_dict[p].keys()}")
            for e in m_dict[p]:
                for t in m_dict[p][e]:
                    print(f"{p=}, {e=}, {t=}")
        test_moe_mcore_to_hf(tp, pp, vpp, ep, etp, pp_ranks, tp_ranks, ep_ranks, etp_ranks, m_dict, encoder_tp_size)
