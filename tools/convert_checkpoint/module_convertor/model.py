# Copyright 2026 The LoongForge Authors.
# SPDX-License-Identifier: Apache-2.0

"""Checkpoint conversion orchestration for HuggingFace and Megatron Core formats."""

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
from convert_checkpoint.common.common_checkpoint import CommonCheckpoint
from convert_checkpoint.arguments import parse_args, set_args
from convert_checkpoint.utils import utils

from convert_checkpoint.utils.utils import(
    get_pipeline_by_rank_id,
    get_layer_ids,
    check_all_done,
    get_ep_map,
    convert_layout_to_custom_pipeline_layers
)

from convert_checkpoint.utils.config_utils import get_yaml_config, replace_vlm_config


BIG_MODEL_LIST = ['llama2-70b', 'qwen-72b', 'codellama-70b', 'codellama-34b']


class Model():
    """
        Model
    """
    def __init__(self, c_config, c_vision_patch_config=None):
        self.config = c_config
        self.c_vision_patch_config = c_vision_patch_config
        self.delay_convert_optimizer = False

    @staticmethod
    def check_done_files(platform, save_path, layer_dict, expert_dict):
        if platform == 'mcore':
            return McoreCheckpoint.check_done_files(save_path, layer_dict, expert_dict=expert_dict)
        if platform == 'huggingface':
            return HuggingFaceCheckpoint.check_done_files(save_path, layer_dict, expert_dict=expert_dict)
        return False

    @staticmethod
    def get_pipeline_args(args, c_config):
        cargs = c_config.get_args("common")
        num_layers = cargs["num_layers"]

        tp = args.tensor_model_parallel_size
        pp = args.pipeline_model_parallel_size
        ep = args.expert_parallel_size
        etp = args.expert_tensor_parallel_size
        num_layers_per_stage = args.num_layers_per_virtual_pipeline_stage
        if num_layers_per_stage:
            vpp = num_layers // pp // num_layers_per_stage
        else:
            vpp = args.num_virtual_stages_per_pipeline_rank or 1
        return (tp, pp, vpp), (ep, etp)

    def get_visual_args(args):
        visual_args = argparse.Namespace()
        visual_args.tensor_model_parallel_size = args.encoder_tensor_model_parallel_size \
            if args.encoder_tensor_model_parallel_size is not None else args.tensor_model_parallel_size
        visual_args.num_virtual_stages_per_pipeline_rank = 1
        visual_args.vpp_scheduler = None
        visual_args.pipeline_model_parallel_size = 1
        visual_args.expert_tensor_parallel_size = None
        visual_args.expert_parallel_size = None
        visual_args.custom_pipeline_layers = None
        visual_args.safetensors = args.safetensors
        visual_args.decoder_first_pipeline_num_layers = None
        visual_args.decoder_last_pipeline_num_layers = None
        visual_args.num_layers_per_virtual_pipeline_stage = None
        visual_args.save_ckpt_path = args.save_ckpt_path
        visual_args.load_ckpt_path = args.load_ckpt_path
        visual_args.convert_to_fp8 = args.convert_to_fp8
        visual_args.max_workers = args.max_workers
        visual_args.moe_grouped_gemm = args.moe_grouped_gemm
        visual_args.fp8_force_no_requant = args.fp8_force_no_requant
        visual_args.force_pow_2_scales = args.force_pow_2_scales
        visual_args.amax_epsilon = args.amax_epsilon
        visual_args.mtp_num_layers = 0
        visual_args.load_lora_ckpt_path = args.load_lora_ckpt_path
        visual_args.lora_alpha = args.lora_alpha
        visual_args.lora_dim = args.lora_dim
        visual_args.vit_in_first_virtual_stage_only = False
        visual_args.enable_full_hetero_dp = args.enable_full_hetero_dp
        visual_args.hf_checkpoint_device = args.hf_checkpoint_device
        visual_args.sub_file_tag = args.sub_file_tag
        return visual_args

    def convert_from_common(self, platform, target_c_config, layer_dict, expert_dict=None, target_c_vision_config=None):
        """
            Convert common checkpoint to the platform checkpoint.

            Args:
                platform (str): name of platform 
                args (dict): arguments
        """

        args = parse_args()
        assert len(layer_dict.keys()) == 1, f"layer_dict keys: {layer_dict.keys()}"
        p = list(layer_dict.keys())[0]
        if platform == 'mcore':
            (tp, pp, vpp), (ep, etp) = Model.get_pipeline_args(args, self.config)
            m_ckpt = McoreCheckpoint(self.config, args)
            no_encoder: bool = self.c_vision_patch_config is None or (not args.enable_full_hetero_dp and p > 0)
            if no_encoder:
                m_ckpt.convert_from_common(self.c_ckpt, target_c_config, layer_dict, expert_dict=expert_dict)
            else:
                visual_model_id = 0 if vpp > 1 else None
                visual_args = Model.get_visual_args(args)
                m_vision_ckpt = McoreCheckpoint(
                    self.c_vision_patch_config, visual_args, model_id=visual_model_id)
                McoreCheckpoint.convert_from_common_vlm(m_ckpt, m_vision_ckpt, self.c_vision_patch_config, self.c_ckpt,
                        self.c_vision_ckpt, target_c_config, target_c_vision_config, args.save_ckpt_path, layer_dict, expert_dict)

        if platform == 'huggingface':
            hf_ckpt = HuggingFaceCheckpoint(self.config, args)
            if p > 0 or self.c_vision_patch_config is None:
                hf_ckpt.convert_from_common(self.c_ckpt, layer_dict, expert_dict=expert_dict, save_path=args.save_ckpt_path)
            else:
                visual_args = Model.get_visual_args(args)
                hf_vision_ckpt = HuggingFaceCheckpoint(self.c_vision_patch_config, visual_args)
                HuggingFaceCheckpoint.save_vlm_checkpoint(
                    hf_ckpt, hf_vision_ckpt, self.c_vision_patch_config, self.c_ckpt,
                    self.c_vision_ckpt, args.save_ckpt_path, layer_dict, expert_dict=expert_dict)

    def convert_config(self, platform):
        """
            Convert common config to the platform config.

            Args:
                platform (str): name of platform 
        """
        if platform == 'mcore':
            if self.c_vision_patch_config is None:
                return self.config.convert(McoreConfig), None
            else:
                return self.config.convert(McoreConfig), self.c_vision_patch_config.convert(McoreConfig)

        if platform == 'huggingface':
            if self.c_vision_patch_config is None:
                return self.config.convert(HuggingFaceConfig), None
            else:
                return self.config.convert(HuggingFaceConfig), self.c_vision_patch_config.convert(HuggingFaceConfig)

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

        assert len(layer_dict.keys()) == 1, f"layer_dict keys: {layer_dict.keys()}"
        p = list(layer_dict.keys())[0]
        # load checkpoint
        if platform == 'huggingface':
            hf_ckpt = HuggingFaceCheckpoint(self.config, args)
            layer_ids = layer_dict[p]
            expert_ids=expert_dict.values() if expert_dict is not None else None
            hf_ckpt.load(ckpt_path, args.safetensors, self.config, layer_ids, expert_ids=expert_ids,
                         mtp_num_layers=mtp_num_layers)
            self.c_ckpt = hf_ckpt.convert_to_common(layer_dict, expert_dict=expert_dict)
            has_encoder: bool = self.c_vision_patch_config is not None and (args.enable_full_hetero_dp or p == 0)
            if has_encoder:
                visual_args = Model.get_visual_args(args)
                hf_vision_ckpt = HuggingFaceCheckpoint(self.c_vision_patch_config, visual_args)
                vision_num_layers = self.c_vision_patch_config.get_args("common")["num_layers"]
                vision_layer_dict = {}
                vision_layer_dict[0] = list(range(vision_num_layers)) 
                hf_vision_ckpt.load(ckpt_path, args.safetensors, self.c_vision_patch_config, vision_layer_dict[0])
                self.c_vision_ckpt = hf_vision_ckpt.convert_to_common(vision_layer_dict)
        # load checkpoint
        if platform == 'mcore':
            self.delay_convert_optimizer = args.model_type_custom in BIG_MODEL_LIST
            (tp, pp, vpp), (ep, etp) = Model.get_pipeline_args(args, self.config)
            vit_model_id = 0 if args.vit_in_first_virtual_stage_only else None
            m_ckpt = McoreCheckpoint(self.config, args, model_id=vit_model_id)
            m_ckpt.load(ckpt_path, layer_dict, expert_dict=expert_dict, lora_load_path=args.load_lora_ckpt_path)
            self.c_ckpt = m_ckpt.convert_to_common(layer_dict, expert_dict=expert_dict)
            if p == 0 and self.c_vision_patch_config is not None:
                visual_model_id = 0 if vpp > 1 else None
                visual_args = Model.get_visual_args(args)
                m_vision_ckpt = McoreCheckpoint(
                    self.c_vision_patch_config, visual_args, model_id=visual_model_id)
                vision_num_layers = self.c_vision_patch_config.get_args("common")["num_layers"]
                vision_layer_dict = {}
                vision_layer_dict[0] = list(range(vision_num_layers)) 
                m_vision_ckpt.m_dict = m_ckpt.m_dict
                self.c_vision_ckpt = m_vision_ckpt.convert_to_common(vision_layer_dict)

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
        c_vision_patch_config = None
    else:
        c_config = get_yaml_config(args.config_file, args.convert_file, for_vlm=(args.vision_patch_convert_file is not None))
        c_vision_patch_config = get_yaml_config(args.config_file, args.vision_patch_convert_file,
                                                adapter_convert_file=args.adapter_convert_file) \
                if args.vision_patch_convert_file is not None else None

    tp = args.tensor_model_parallel_size
    pp = args.pipeline_model_parallel_size
    ep = args.expert_parallel_size
    etp = args.expert_tensor_parallel_size

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
        model = Model(c_config, c_vision_patch_config)
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

        target_c_config, target_c_vision_config = model.convert_config(args.save_platform)
        model.convert_from_common(args.save_platform, target_c_config, layer_dict,
                                  expert_dict=ep_expert_mapping, target_c_vision_config=target_c_vision_config)

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