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


SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.dirname(SCRIPT_DIR))

from convert_checkpoint.huggingface_checkpoint import HuggingFaceCheckpoint
from convert_checkpoint.megatron_checkpoint import MegatronCheckpoint
from convert_checkpoint.mcore_checkpoint import McoreCheckpoint
from convert_checkpoint.huggingface_config import HuggingFaceConfig
from convert_checkpoint.megatron_config import MegatronConfig
from convert_checkpoint.mcore_config import McoreConfig
from convert_checkpoint.common_config import CommonConfig
from convert_checkpoint.arguments import parse_args, set_args
from convert_checkpoint import utils

from convert_checkpoint.utils import(
    get_pipeline_by_rank_id,
    get_layer_ids,
    check_all_done,
    get_ep_map,
)


BIG_MODEL_LIST = ['llama2-70b', 'qwen-72b', 'codellama-70b', 'codellama-34b']


class Model():
    """
        Model
    """
    def __init__(self):
        self.config = CommonConfig()
        self.delay_convert_optimizer = False

    def convert_checkpoint(self, platform, save_path=None, m_config=None, save_optim=True, p=None, layer_ids=[],
                           cur_ep_ids=None, expert_ids=None):
        """
            Convert common checkpoint to the platform checkpoint.

            Args:
                platform (str): name of platform 
                args (dict): arguments
        """

        if platform == 'megatron':
            return self.ckpt.convert(MegatronCheckpoint, self.config)

        if platform == 'mcore':
            return self.ckpt.convert(
                McoreCheckpoint, self.config, save_path, m_config, save_optim, p=p, cur_ep_ids=cur_ep_ids)

        if platform == 'huggingface':
            return self.ckpt.convert(HuggingFaceCheckpoint, self.config, p=p, layer_ids=layer_ids,
                                     cur_ep_ids=cur_ep_ids, expert_ids=expert_ids)

    def convert_config(self, platform):
        """
            Convert common config to the platform config.

            Args:
                platform (str): name of platform 
        """
        if platform == 'megatron':
            return self.config.convert(MegatronConfig)
    
        if platform == 'mcore':
            return self.config.convert(McoreConfig)

        if platform == 'huggingface':
            return self.config.convert(HuggingFaceConfig)

    def load(self, args, p=None, layer_ids=[], cur_ep_ids=None, expert_ids=None):
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
        config_path = args.common_config_path
        self.config.load(config_path)
        common_args = self.config.get_args()
        num_layers = common_args['num_layers']
        platform_args = self.config.get_args(platform)
        dtype = self.config.get_dtype()
        name_map = self.config.get_name_map(platform)

        # load checkpoint
        if platform == 'huggingface':
            hf_ckpt = HuggingFaceCheckpoint(num_layers)
            hf_ckpt.set_dtype(dtype)
            hf_ckpt.load(ckpt_path, args.safetensors, c_config=self.config, layer_ids=layer_ids, expert_ids=expert_ids)
            self.ckpt = hf_ckpt.convert_to_common(self.config, layer_ids=layer_ids)
            return hf_ckpt

        # load checkpoint
        if platform == 'mcore':
            self.delay_convert_optimizer = args.model_type_custom in BIG_MODEL_LIST
            m_config = McoreConfig()
            m_config.load(ckpt_path)
            m_config.update({"tensor_parallel_dim": self.config.get("tensor_parallel_dim")})
            m_ckpt = McoreCheckpoint(num_layers, args.load_ckpt_path)
            m_ckpt.set_dtype(dtype)
            load_optim = not self.delay_convert_optimizer and not args.no_load_optim
            m_ckpt.load(ckpt_path, m_config, name_map, load_optim, p=p, cur_ep_ids=cur_ep_ids)
            self.ckpt = m_ckpt.convert_to_common(self.config, p=p, cur_ep_ids=cur_ep_ids)
            return m_ckpt

    def update_args(self, args, group):
        """ update config accoding to args """
        self.config.update_args(vars(args), group)


def main():
    """ main """
    args = parse_args()

    if args.megatron_path is not None:
        sys.path.insert(0, args.megatron_path)

    if args.load_platform == "mcore"  or args.save_platform == "mcore":
        assert args.transformer_impl == "transformer_engine", \
            "Only support transformer_engine implemenation for mcore now!"
    
        args.no_load_optim = True
        args.no_save_optim = True
        logging.info(f"<< Warning: not support mcore optimizer now, so no_load_optim and no_save_optim are set to True! >>")

    model = Model()
    if not args.distributed_convert:
        os.environ['RANK'] = '0'
        os.environ['WORLD_SIZE'] = '1'
    rank_id = int(os.getenv('RANK', '0'))
    world_size = int(os.getenv("WORLD_SIZE", '1'))
    tp = args.tensor_model_parallel_size
    pp = args.pipeline_model_parallel_size
    ep = args.expert_parallel_size
    etp = args.expert_tensor_parallel_size
    num_experts = args.num_experts
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
    config_path = args.common_config_path
    c_config = CommonConfig()
    c_config.load(config_path)

    def convert_one_p(p, cur_ep_ids=None):
        expert_ids = None
        layer_ids = get_layer_ids(c_config, args, p)
        if cur_ep_ids is not None:
            expert_local_mapping, expert_ep_mapping, ep_expert_mapping = get_ep_map(
                    num_experts, ep, num_experts_for_test=None)
            expert_ids = []
            for ep_id in cur_ep_ids:
                expert_ids += ep_expert_mapping[ep_id]

        source_ckpt = model.load(args, p=p, layer_ids=layer_ids, cur_ep_ids=cur_ep_ids, expert_ids=expert_ids)
        for group in ["common", "megatron", "huggingface"]:
            _args = parse_args(group)
            model.update_args(_args, group)
            if group == "megatron":
                model.update_args(_args, "mcore")

        target_config = model.convert_config(args.save_platform)
        save_optim = not args.no_save_optim and not model.delay_convert_optimizer
        target_ckpt = model.convert_checkpoint(
            args.save_platform, args.save_ckpt_path, target_config, save_optim, p=p, layer_ids=layer_ids,
            cur_ep_ids=cur_ep_ids, expert_ids=expert_ids)
        save_option = dict()
        save_option["save_optim"] = save_optim
        if isinstance(target_ckpt, HuggingFaceCheckpoint):
            save_option["save_safe"] = args.safetensors

        if not args.no_save_optim and model.delay_convert_optimizer:
            from convert_checkpoint.optim import change_tp, change_pp, change_dp
            is_change_pp = (source_ckpt.pp != target_ckpt.pp) or (source_ckpt.num_stages != target_ckpt.num_stages)
            is_change_tp = (source_ckpt.tp != target_ckpt.tp)
            assert not (is_change_pp and is_change_tp), "cann't change tp and pp at the same time"
            if is_change_tp:
                change_tp(source_ckpt, target_ckpt, args.load_ckpt_path, args.save_ckpt_path, model.config)
            elif is_change_pp:
                change_pp(source_ckpt, target_ckpt, args.load_ckpt_path, args.save_ckpt_path, model.config)
            else:
                change_dp(source_ckpt, target_ckpt, args.load_ckpt_path, args.save_ckpt_path)

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
                    logging.info(f"删除失败 {file_path}: {str(e)}")
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
    args.torch_dtype = None
    args.vocab_size = None
    args.vpp_scheduler = None
    args.num_virtual_stages_per_pipeline_rank = None
    args.decoder_first_pipeline_num_layers = None
    args.decoder_last_pipeline_num_layers = None
    args.use_distributed_optimizer = False
    args.tensor_model_parallel_size = tp
    args.pipeline_model_parallel_size = pp
    args.data_parallel_size = 1
    args.expert_parallel_size = ep
    args.pad_vocab_size_to = None
    args.num_layers_per_virtual_pipeline_stage = None
    args.transformer_impl = 'transformer_engine'
    args.checkpoint_format = None
    args.max_workers = 1
    args.num_experts = num_experts
    args.no_load_optim = True
    args.no_save_optim = True
    args.no_te = True
    args.moe_grouped_gemm = True
    args.resume_convert = False
    args.cache_path = None
    args.layer_for_test = None
    args.num_experts_for_test = None
    args.sub_num_layers_for_save = None
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

from convert_checkpoint.utils import make_hf_sub_checkpoints

def test_merge_hf_ckpt():
    make_hf_sub_checkpoints('/mnt/cluster/deepseek-ai/DeepSeek_V3_Lite_hf')

if __name__ == "__main__":
    #test()
    main()
