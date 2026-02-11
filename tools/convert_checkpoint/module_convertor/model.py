#!/usr/bin/env python3
# -*- coding: UTF-8 -*-
################################################################################
#
# Copyright (c) 2024 Baidu.com, Inc. All Rights Reserved
#
################################################################################

import argparse
import os
import sys
import shutil
import time
import torch
import concurrent.futures

import logging

logging.basicConfig(level=logging.INFO)

from os.path import dirname
SCRIPT_DIR = dirname(os.path.abspath(__file__))
sys.path.append(dirname(dirname(SCRIPT_DIR)))

from convert_checkpoint.huggingface.huggingface_checkpoint import HuggingFaceCheckpoint
from convert_checkpoint.huggingface.huggingface_config import HuggingFaceConfig
from convert_checkpoint.mcore.mcore_checkpoint import McoreCheckpoint
from convert_checkpoint.mcore.mcore_config import McoreConfig
from convert_checkpoint.common.common_config import CommonConfig
from convert_checkpoint.arguments import parse_args, set_args
from convert_checkpoint.utils import utils

from convert_checkpoint.utils.utils import(
    get_pipeline_by_rank_id,
    get_layer_ids,
    check_all_done,
    get_ep_map,
    convert_layout_to_custom_pipeline_layers,
)


from omegaconf import OmegaConf
from convert_checkpoint.utils.config_utils import parse_at_configs, load_config, parallel_param_parser, update_overwrite


BIG_MODEL_LIST = ['llama2-70b', 'qwen-72b', 'codellama-70b', 'codellama-34b']


class Model():
    """
        Model
    """
    def __init__(self, c_config):
        self.config = c_config
        self.delay_convert_optimizer = False

    @staticmethod
    def check_done_files(platform, save_path, layer_dict, expert_dict):
        if platform == 'mcore':
            return McoreCheckpoint.check_done_files(save_path, layer_dict, expert_dict=expert_dict)
        if platform == 'huggingface':
            return HuggingFaceCheckpoint.check_done_files(save_path, layer_dict, expert_dict=expert_dict)
        return False

    def convert_from_common(self, platform, target_config, layer_dict, expert_dict=None):
        """
            Convert common checkpoint to the platform checkpoint.

            Args:
                platform (str): name of platform 
                args (dict): arguments
        """

        args = parse_args()
        if platform == 'mcore':
            m_ckpt = McoreCheckpoint(self.config, args)
            return m_ckpt.convert_from_common(self.c_ckpt, target_config, layer_dict, expert_dict=expert_dict)
        if platform == 'huggingface':
            hf_ckpt = HuggingFaceCheckpoint(self.config, args)
            return hf_ckpt.convert_from_common(self.c_ckpt, layer_dict, expert_dict=expert_dict, save_path=args.save_ckpt_path)
        self.common_ckpt.clear()

    def convert_config(self, platform):
        """
            Convert common config to the platform config.

            Args:
                platform (str): name of platform 
        """
        if platform == 'mcore':
            return self.config.convert(McoreConfig)

        if platform == 'huggingface':
            return self.config.convert(HuggingFaceConfig)

    def convert_to_common(self, args, layer_dict, expert_dict=None):
        """
            Load checkpoint and config.
        """

        if not hasattr(args, "common_config_path") or args.common_config_path is None:
            assert hasattr(args, "model_type_custom"), "model_type_custom or common_config_path is required"
            base_dir  = os.path.dirname(os.path.abspath(__file__))
            args.common_config_path = os.path.join(base_dir, f"config/{args.model_type_custom}.json")
        else:
            model_type_custom = os.path.splitext(os.path.basename(args.common_config_path))[0]
            setattr(args, "model_type_custom", model_type_custom)

        platform = args.load_platform
        ckpt_path = args.load_ckpt_path

        # load common config
        assert isinstance(self.config, CommonConfig)

        cargs = self.config.get_args("common")
        mtp_num_layers = args.mtp_num_layers if args.mtp_num_layers is not None else cargs.get("mtp_num_layers", 0)

        # load checkpoint
        if platform == 'huggingface':
            hf_ckpt = HuggingFaceCheckpoint(self.config, args)
            assert len(layer_dict.keys()) == 1, f"layer_dict keys: {layer_dict.keys()}"
            p = list(layer_dict.keys())[0]
            layer_ids = layer_dict[p]
            expert_ids=expert_dict.values() if expert_dict is not None else None
            hf_ckpt.load(ckpt_path, args.safetensors, self.config, layer_ids, expert_ids=expert_ids, mtp_num_layers=mtp_num_layers)
            self.c_ckpt = hf_ckpt.convert_to_common(layer_dict, expert_dict=expert_dict)

        # load checkpoint
        if platform == 'mcore':
            self.delay_convert_optimizer = args.model_type_custom in BIG_MODEL_LIST
            m_ckpt = McoreCheckpoint(self.config, args)
            m_ckpt.load(ckpt_path, layer_dict, expert_dict=expert_dict)
            self.c_ckpt = m_ckpt.convert_to_common(layer_dict, expert_dict=expert_dict)


    def update_args(self, args, group):
        """ update config accoding to args """
        self.config.update_args(vars(args), group)


def main():
    """ main """
    args = parse_args()

    config_path = args.common_config_path
    c_config = CommonConfig()
    if config_path is not None:
        c_config.load(config_path)
        tp = args.tensor_model_parallel_size
        pp = args.pipeline_model_parallel_size
        ep = args.expert_parallel_size
        etp = args.expert_tensor_parallel_size
    else:
        with open(args.config_file, 'r') as f:
            module_names = parse_at_configs(f.readlines())
        module_type = args.convert_file.split('/')[-3]
        if module_names == {}: # llm
            cfg = load_config(args.convert_file, hydra_overrides={module_type+'@module='+args.config_file.split("/")[-1].split(".")[0]})
        else: # omni vlm
            cfg = load_config(args.convert_file, hydra_overrides = {module_type+'@module='+module_names[module_type]})
        OmegaConf.set_struct(cfg, False)

        model_cfg = load_config(args.config_file)
        if module_type != 'image_encoder':
            module_type = 'foundation'
        tp = parallel_param_parser(args, model_cfg, 'tensor_model_parallel_size', module_type)
        pp = parallel_param_parser(args, model_cfg, 'pipeline_model_parallel_size', module_type)
        ep = parallel_param_parser(args, model_cfg, 'expert_parallel_size', module_type)
        etp = parallel_param_parser(args, model_cfg, 'expert_tensor_parallel_size', module_type)
        vpp = parallel_param_parser(args, model_cfg, 'num_virtual_stages_per_pipeline_rank', module_type)

        c_config.load_convert_data(cfg)

        update_overwrite(model_cfg, c_config, module_type)

    if args.megatron_path is not None:
        sys.path.insert(0, args.megatron_path)

    if not args.distributed_convert:
        os.environ['RANK'] = '0'
        os.environ['WORLD_SIZE'] = '1'
    rank_id = int(os.getenv('RANK', '0'))
    world_size = int(os.getenv("WORLD_SIZE", '1'))
    if utils.LOADED_STATE_DICT is None:
        if etp is not None:
            assert (ep * pp // world_size) % (tp // etp) == 0, f"(ep * pp // world_size) % (tp // etp) must be 0"
        p_dict = get_pipeline_by_rank_id(rank_id, world_size, pp, ep=ep)
    else:
        p_dict = {}
        for p, ep_ids in utils.LOADED_STATE_DICT.items():
            p_dict[p] = []
            for ep_id in ep_ids:
                p_dict[p].append(ep_id)

    if args.pipeline_model_parallel_layout is not None:
        assert args.custom_pipeline_layers is None, \
            "custom_pipeline_layers and pipeline_model_parallel_layout can not be set at the same time"

        args.custom_pipeline_layers = convert_layout_to_custom_pipeline_layers(
            args.pipeline_model_parallel_layout)

        split = [int(x) for x in args.custom_pipeline_layers.split(',') if x.strip()]
        if args.num_virtual_stages_per_pipeline_rank is None:
            assert len(split) % args.pipeline_model_parallel_size == 0, \
                "len(args.custom_pipeline_layers) must be divisible by pipeline_model_parallel_size"
            args.num_virtual_stages_per_pipeline_rank = len(split) // args.pipeline_model_parallel_size

    cargs = c_config.get_args("common")
    num_experts = cargs.get("num_experts", None)
    if num_experts is not None:
        assert num_experts > 0, "num_experts must be greater than zero"
        if ep is None:
            args.expert_parallel_size = 1  # if ep is not set, will set ep=1

    def convert_one_p(p, cur_ep_ids=None):
        model = Model(c_config)
        layer_dict = {}
        layer_dict[p] = get_layer_ids(c_config, args, p)
        ep_expert_mapping = None
        if cur_ep_ids is not None:
            expert_local_mapping, expert_ep_mapping, ep_expert_mapping = get_ep_map(num_experts, ep)

        if (Model.check_done_files(args.save_platform, args.save_ckpt_path, layer_dict, expert_dict=ep_expert_mapping)):
            logging.info(f"{args.save_ckpt_path=}, {layer_dict=}, " \
                    f"expert_dict={ep_expert_mapping}. already converted. pass.")
            return
        model.convert_to_common(args, layer_dict, expert_dict=ep_expert_mapping)
        for group in ["common", "megatron", "huggingface"]:
            _args = parse_args(group)
            model.update_args(_args, group)
            if group == "megatron":
                model.update_args(_args, "mcore")

        target_config = model.convert_config(args.save_platform)
        save_optim = not args.no_save_optim and not model.delay_convert_optimizer
        target_ckpt = model.convert_from_common(args.save_platform, target_config, layer_dict, expert_dict=ep_expert_mapping)
        save_option = dict()
        save_option["save_optim"] = save_optim
        if isinstance(target_ckpt, HuggingFaceCheckpoint):
            save_option["save_safe"] = args.safetensors

    if args.max_workers > 1 and ep is None:
        futures = []
        with concurrent.futures.ThreadPoolExecutor(max_workers=args.max_workers) as executor:
            for p, cur_ep_ids in p_dict.items():
                futures.append(executor.submit(convert_one_p, p=p, cur_ep_ids=cur_ep_ids))
        concurrent.futures.wait(futures)
        for future in futures:
            try:
                result = future.result()
            except Exception as e:
                logging.info(f"An error occurred: {e}")
                raise e
    else:
        for p, cur_ep_ids in p_dict.items():
            convert_one_p(p, cur_ep_ids)

    if rank_id == 0 and utils.LOADED_STATE_DICT is None:
        done_dir = os.path.join(args.save_ckpt_path, "dones")
        while True:
            checked_done = check_all_done(done_dir, pp, ep)
            if checked_done:
                shutil.rmtree(done_dir)
                break
            else:
                if world_size == 1:
                    raise Exception(f"{done_dir} is not complete. please retry it again")
                logging.info(f"Waiting for the other rank to finish. {world_size=}")
                time.sleep(10)
        if args.save_platform == "huggingface":
            make_hf_sub_checkpoints(args.save_ckpt_path)

    logging.info(f"Finished convert checkpoint {args.load_platform} -> {args.save_platform}")


def verl_convert_mcore_to_hf_v3(v3_params, args):
    if os.path.exists(args.save_ckpt_path):
        for filename in os.listdir(args.save_ckpt_path):
            if filename.endswith(".safetensors") or filename == "model.safetensors.index.json":
                file_path = os.path.join(args.save_ckpt_path, filename)
                try:
                    os.remove(file_path)
                except Exception as e:
                    logging.info(f"Failed to delete {file_path}: {str(e)}")
    p_keys = set(v3_params.keys())
    utils.LOADED_STATE_DICT = {}
    for p in p_keys:
        utils.LOADED_STATE_DICT[p] = v3_params[p]
        del v3_params[p]
    set_args(args)
    main()


def test():
    tp = 1
    pp = 2
    ep = 4
    num_experts = 16
    custom_pipeline_layers=None
    args = argparse.Namespace()
    args.load_platform = "mcore"
    args.save_platform = "huggingface"
    args.load_ckpt_path = None
    args.save_ckpt_path = "/mnt/cluster/deepseek-ai/DeepSeek_V3_Lite_hf"
    args.common_config_path = "./convert_checkpoint/config/deepseek-v3-lite.json"
    args.megatron_path = None
    args.model_type_custom = None
    args.vpp_scheduler = None
    args.num_virtual_stages_per_pipeline_rank = None
    args.decoder_first_pipeline_num_layers = None
    args.decoder_last_pipeline_num_layers = None
    args.tensor_model_parallel_size = tp
    args.pipeline_model_parallel_size = pp
    args.data_parallel_size = 1
    args.expert_parallel_size = ep
    args.num_layers_per_virtual_pipeline_stage = None
    args.max_workers = 1
    args.num_experts = num_experts
    args.moe_grouped_gemm = True
    args.custom_pipeline_layers = custom_pipeline_layers
    args.safetensors = True
    args.save_sub_checkpoint_by_pp = True
    args.convert_to_fp8 = False
    args.pretrain_as_fp8 = False
    args.quant_method = 'te'
    args.amax_epsilon = 0.0
    args.distributed_convert = False
    logging.info(f"{args=}")

    v3_params = {}
    for p in range(pp):
        v3_params[p] = {}
        for e in range(ep):
            v3_params[p][e] = torch.load(f'/mnt/cluster/deepseek-ai/DeepSeek_V3_tp1pp2ep4/release/mp_rank_00_{p:03d}_{e:03d}/model_optim_rng.pt')
    verl_convert_mcore_to_hf_v3(v3_params, args)

from convert_checkpoint.utils.utils import make_hf_sub_checkpoints

def test_merge_hf_ckpt():
    make_hf_sub_checkpoints('/mnt/cluster/deepseek-ai/DeepSeek_V3_Lite_hf')

if __name__ == "__main__":
    #test()
    main()