""" Huggingface checkpoint converter """

import argparse

from dist_checkpoint.config.parallel_config import ParallelConfig
from convert_checkpoint.huggingface.huggingface_checkpoint import HuggingFaceCheckpoint
from convert_checkpoint.mcore.mcore_checkpoint import McoreCheckpoint
from convert_checkpoint.common.common_config import CommonConfig
from convert_checkpoint.utils.utils import(
    get_layer_ids
)

from convert_checkpoint.utils.utils import get_ep_map
from tools.convert_checkpoint.module_convertor.model import Model

class HfCheckpointConverter:
    """Converter for Huggingface checkpoint."""

    def __init__(self, parallel_config: ParallelConfig, config: CommonConfig, vision_patch_config=None):
        self.args = argparse.Namespace()
        self.args.tensor_model_parallel_size = parallel_config.tp_size
        self.args.num_virtual_stages_per_pipeline_rank = parallel_config.vpp_size
        self.args.vpp_scheduler = parallel_config.vpp_scheduler
        self.args.pipeline_model_parallel_size = parallel_config.pp_size
        self.args.expert_tensor_parallel_size = parallel_config.etp_size
        self.args.expert_parallel_size = parallel_config.ep_size
        self.args.custom_pipeline_layers = parallel_config.custom_pipeline_layers
        self.args.safetensors = parallel_config.safetensors
        self.args.decoder_first_pipeline_num_layers = parallel_config.decoder_first_pipeline_num_layers
        self.args.decoder_last_pipeline_num_layers = parallel_config.decoder_last_pipeline_num_layers
        self.args.num_layers_per_virtual_pipeline_stage = None
        self.args.vit_in_first_virtual_stage_only = parallel_config.vit_in_first_virtual_stage_only
        self.args.save_ckpt_path = None
        self.args.load_ckpt_path = None
        self.args.convert_to_fp8 = False
        self.args.max_workers = parallel_config.max_workers
        self.args.moe_grouped_gemm = parallel_config.moe_grouped_gemm
        self.args.fp8_force_no_requant = parallel_config.fp8_force_no_requant
        self.args.force_pow_2_scales = parallel_config.force_pow_2_scales
        self.args.amax_epsilon = parallel_config.amax_epsilon
        self.args.mtp_num_layers = parallel_config.mtp_num_layers
        self.args.encoder_tensor_model_parallel_size = parallel_config.encode_tp_size
        self.args.load_lora_ckpt_path = None
        self.args.lora_alpha = parallel_config.lora_alpha
        self.args.lora_dim = parallel_config.lora_dim
        self.ep_size = parallel_config.ep_size
        self.pp_ranks = parallel_config.pp_ranks
        self.ep_ranks = parallel_config.ep_ranks
        self.tp_ranks = parallel_config.tp_ranks
        self.etp_ranks = parallel_config.etp_ranks

        self.config = config
        self.vision_patch_config = vision_patch_config
        cargs = self.config.get_args("common")
        self.num_experts = cargs.get("num_experts", None)
        self.layer_ids = []
        self.layer_dict = {}
        for p in self.pp_ranks:
            layer_id_list = get_layer_ids(self.config, self.args, p)
            self.layer_dict[p] = layer_id_list
            self.layer_ids.extend(layer_id_list)
        if self.ep_ranks is None:
            self.expert_dict = None
        else:
            self.expert_dict = []
            _, _, ep_expert_mapping = get_ep_map(self.num_experts, self.ep_size)
            self.expert_dict = {key: value for key, value in ep_expert_mapping.items() if key in self.ep_ranks}

        (tp, pp, vpp), (ep, etp) = Model.get_pipeline_args(self.args, config)
        self.hf_ckpt = HuggingFaceCheckpoint(config, self.args)
        self.m_ckpt = McoreCheckpoint(config, self.args, tp, pp, vpp, ep, etp)
        if vision_patch_config is not None:
            vision_num_layers = vision_patch_config.get_args("common")["num_layers"]
            self.vision_layer_dict = {}
            self.vision_layer_dict[0] = list(range(vision_num_layers)) 
            self.hf_vision_ckpt = HuggingFaceCheckpoint(vision_patch_config, self.args)
            self.m_vision_ckpt = McoreCheckpoint(
                c_config=vision_patch_config, args=self.args, tp=self.args.encoder_tensor_model_parallel_size, pp=1, vpp=1)

    def get_mcore_ckpt(self, ckpt_path):
        expert_ids=self.expert_dict.values() if self.expert_dict is not None else None
        mcore_dict = {}
        for p in self.pp_ranks:
            cur_layer_dict = {p: self.layer_dict[p]}
            self.hf_ckpt.load(ckpt_path, self.args.safetensors, self.config, self.layer_ids, expert_ids=expert_ids,
                         mtp_num_layers=self.args.mtp_num_layers)
            c_ckpt = self.hf_ckpt.convert_to_common(cur_layer_dict, expert_dict=self.expert_dict)
            if p > 0 or self.vision_patch_config is None:
                mcore_dict[p] = self.m_ckpt.convert_from_common(
                        c_ckpt, None, cur_layer_dict, expert_dict=self.expert_dict, save_file=False,
                        tp_ranks=self.tp_ranks, etp_ranks=self.etp_ranks)[p]
            else:
                self.hf_vision_ckpt.load(ckpt_path, self.args.safetensors, self.vision_patch_config, self.vision_layer_dict[0])
                self.c_vision_ckpt = self.hf_vision_ckpt.convert_to_common(self.vision_layer_dict)
                mcore_dict[p] = McoreCheckpoint.convert_from_common_vlm(self.m_ckpt, self.m_vision_ckpt, self.vision_patch_config, c_ckpt,
                        self.c_vision_ckpt, None, None, None, cur_layer_dict, self.expert_dict, save_file=False)[p]
        return mcore_dict

    def save_hf_ckpt(self, mcore_dict, save_path):
        for p in self.pp_ranks:
            cur_layer_dict = {p: self.layer_dict[p]}
            self.m_ckpt.load(None, layer_dict=cur_layer_dict, expert_dict=self.expert_dict, mcore_dict=mcore_dict)
            c_ckpt = self.m_ckpt.convert_to_common(cur_layer_dict, expert_dict=self.expert_dict)
            if p > 0 or self.vision_patch_config is None:
                self.hf_ckpt.convert_from_common(c_ckpt, cur_layer_dict, expert_dict=self.expert_dict, save_path=save_path)
            else:
                self.m_vision_ckpt.m_dict = self.m_ckpt.m_dict
                c_vision_ckpt = self.m_vision_ckpt.convert_to_common(self.vision_layer_dict)
                HuggingFaceCheckpoint.save_vlm_checkpoint(
                    self.hf_ckpt, self.hf_vision_ckpt, self.vision_patch_config, c_ckpt,
                    c_vision_ckpt, save_path, cur_layer_dict, expert_dict=self.expert_dict)
