# Copyright 2026 The LoongForge Authors.
# SPDX-License-Identifier: Apache-2.0

""" Mcore_checkpoint converter for megatron lm. """

import os
import torch
import logging
import argparse

logging.basicConfig(level=logging.INFO)

import concurrent.futures
from convert_checkpoint.arguments import parse_args
from convert_checkpoint.common.abstact_checkpoint import AbstractCheckpoint
from convert_checkpoint.common.common_checkpoint import VISION_MAP, VISION_WORD_EMBEDDINGS, CommonCheckpoint
from convert_checkpoint.mcore.mcore_base import McoreBase
from convert_checkpoint.mcore.mcore_moe import McoreMoe
from convert_checkpoint.utils.utils import (
    touch_file,
    get_done_keys,
    get_virtual_partition,
    get_num_layers_in_vp_map,
    get_etp_map,
)

from convert_checkpoint.common.common_checkpoint import (
    TRANSFORMER, TRANSFORMER_TPL, MTP_LAYER_PREFIX, WORD_EMBEDDINGS,
    FIRST_LAYER_NAMES, BASE_NAMES, MOE_EXPERT_PROJS, LAST_LAYER_NAMES, MTP_NAMES,
    MTP_SHARED_HEAD_HEAD, MOE_SHARED_EXPERT, MOE_EXPERT, MTP_NAME_PREFIX_FOR_LAYER
)


class McoreCheckpoint(AbstractCheckpoint):
    """
        McoreCheckpoint
    """

    def __init__(self, c_config, args, model_id=None):
        super().__init__(c_config)
        self.args = args
        self.tp = args.tensor_model_parallel_size
        self.pp = args.pipeline_model_parallel_size
        self.ep = args.expert_parallel_size
        self.etp = args.expert_tensor_parallel_size
        self.m_base = McoreBase(c_config, args)
        self.m_moe = McoreMoe(c_config, args)
        self.iteration = 0
        self.checkpoint_version = 3.0
        self.rng_state = None
        self.model_id = model_id
        margs = c_config.get_args("mcore")
        cargs = c_config.get_args("common")
        num_layers = cargs["num_layers"]
        num_layers_per_stage = self.args.num_layers_per_virtual_pipeline_stage

        if num_layers_per_stage:
            stage = num_layers // self.pp // num_layers_per_stage
        else:
            stage = self.args.num_virtual_stages_per_pipeline_rank or 1
        self.num_stages = stage or 1
        self.name_map = self.c_config.get("name_map")["mcore"]
        self.optim_state_dict = None
        self.name_prefix_for_layer = self.name_map[MTP_NAME_PREFIX_FOR_LAYER] if MTP_NAME_PREFIX_FOR_LAYER in self.name_map else None


    @staticmethod
    def check_done_files(save_path, layer_dict, expert_dict=None):
        done_dir = os.path.join(save_path, "dones")
        need_check_dones, done_keys = McoreCheckpoint.get_need_check_dones(done_dir, layer_dict, expert_dict)
        if not need_check_dones:
            return False
        p = list(layer_dict.keys())[0]
        if expert_dict is None:
            if p not in done_keys:
                return False
        else:
            for ep_id in expert_dict.keys():
                if (p, ep_id) not in done_keys:
                    return False
        return True

    @staticmethod
    def get_need_check_dones(done_dir, layer_dict, expert_dict=None):
        p = list(layer_dict.keys())[0]
        need_check_dones = False
        if os.path.exists(done_dir):
            need_check_dones = True
            if expert_dict is None:
                done_keys = get_done_keys(done_dir, p)
            else:
                done_keys = get_done_keys(done_dir, p, expert_dict.keys())
        else:
            done_keys = []
            rank_id = int(os.getenv('RANK', '0'))
            if rank_id == 0:
                os.makedirs(done_dir, exist_ok=True)
            else:
                import time
                while(not os.path.exists(done_dir)):
                    time.sleep(10)
                    logging.info(f"Rank {rank_id} waiting for done file dir: {done_dir}.")
        return need_check_dones, done_keys


    def convert_from_common(self, c_ckpt, m_config, layer_dict, expert_dict=None, save_file=True, tp_ranks=None, etp_ranks=None):
        """
        Convert common checkpoint to mcore checkpoint.

        Args:
            c_ckpt: CommonCheckpoint
        """
        logging.info("\n==================== Common -> Mcore ====================")

        name_map = self.c_config.get("name_map")["mcore"]
        cargs = self.c_config.get_args("common")
        margs = self.c_config.get_args("mcore")

        dualpipev = self.args.vpp_scheduler == 'dualpipev'
        custom_pipeline_layers = self.args.custom_pipeline_layers

        mtp_num_layers = self.args.mtp_num_layers if self.args.mtp_num_layers is not None else cargs.get("mtp_num_layers", 0)
        num_layers = cargs["num_layers"]
        stage = self.args.num_virtual_stages_per_pipeline_rank or 1
        num_layers_in_first_pipeline_stage = self.args.decoder_first_pipeline_num_layers
        num_layers_in_last_pipeline_stage = self.args.decoder_last_pipeline_num_layers
        if num_layers_in_first_pipeline_stage is not None or num_layers_in_last_pipeline_stage is not None:
            assert self.args.num_virtual_stages_per_pipeline_rank is not None, "num_virtual_stages_per_pipeline_rank is required"

        num_layers_in_vp = get_num_layers_in_vp_map(
            stage, num_layers, self.pp, mtp_num_layers=mtp_num_layers,
            custom_pipeline_layers=custom_pipeline_layers,
            num_layers_in_first_pipeline_stage=num_layers_in_first_pipeline_stage,
            num_layers_in_last_pipeline_stage=num_layers_in_last_pipeline_stage)

        self.iteration = c_ckpt.other_args.get("iteration", self.iteration)
        self.checkpoint_version = c_ckpt.other_args.get("checkpoint_version", self.checkpoint_version)
        self.args = c_ckpt.other_args.get("args", self.args)
        self.rng_state = c_ckpt.other_args.get("rng_state", self.rng_state)

        assert layer_dict != None and len(layer_dict) == 1, "layer_dict must be provided and size == 1"
        p = list(layer_dict.keys())[0]
        layer_ids = layer_dict[p]

        etp_to_tp_mapping, _ = get_etp_map(self.tp, self.ep, self.etp)

        # check dones dir and mkdir release
        if save_file:
            save_path = self.args.save_ckpt_path
            done_dir = os.path.join(save_path, "dones")
            need_check_dones, done_keys = McoreCheckpoint.get_need_check_dones(done_dir, layer_dict, expert_dict)
            release_dir, save_margs = self.pre_save(save_path, m_config)
        else:
            mcore_dict = {}
            mcore_dict[p] = {}
            if expert_dict is None:
                mcore_dict[p] = {}
            else:
                for ep_id in expert_dict.keys():
                    mcore_dict[p][ep_id] = {}

        def convert_one_ep_from_common(ep_id=None):
            if save_file and need_check_dones:
                if ep_id is None and p in done_keys:
                    logging.info(f"> p: {p} already converted. pass...")
                    return
                if ep_id is not None and (p, ep_id) in done_keys:
                    logging.info(f"> p: {p}, ep_id: {ep_id} already converted. pass...")
                    return
            m_dict = {}
            if ep_id is None or self.etp is None:
                for t in range(self.tp):
                    if tp_ranks is not None and t not in tp_ranks:
                        continue
                    m_dict[t] = {}
            else:
                for et in range(self.etp):
                    if etp_ranks is not None and et not in etp_ranks:
                        continue
                    m_dict[et] = {}
            if p == 0:
                t_name = self.get_transformer_name(0)
                for c_name in FIRST_LAYER_NAMES:
                    self.m_base.common_to_mcore(c_name, c_ckpt, m_dict, t_name, ep_id=ep_id)
                for c_name in name_map.keys():
                    if c_name.startswith(VISION_MAP):
                        self.m_base.common_to_mcore(c_name, c_ckpt, m_dict, t_name, ep_id=ep_id)
            elif self.args.enable_full_hetero_dp:
                t_name = self.get_transformer_name(0)
                self.m_base.common_to_mcore(VISION_WORD_EMBEDDINGS, c_ckpt, m_dict, t_name, ep_id=ep_id)

            for stage_index in range(stage):
                virtual_p, mcore_layer_offset, = get_virtual_partition(dualpipev, stage_index, p, self.pp, num_layers_in_vp)
                t_name = self.get_transformer_name(stage_index)
                for cur_layer_id in range(num_layers_in_vp[virtual_p]):
                    layer_id = cur_layer_id + mcore_layer_offset
                    if layer_id >= num_layers:
                        m_layer_id = layer_id - num_layers
                        layer_prefix = name_map[MTP_LAYER_PREFIX]
                        name_prefix = self.name_prefix_for_layer if layer_prefix is not None else None
                    else:
                        m_layer_id = cur_layer_id
                        layer_prefix = None
                        name_prefix = None
                    for c_name in BASE_NAMES:
                        self.m_base.common_to_mcore(c_name, c_ckpt, m_dict, t_name, layer_id, m_layer_id,
                                                    layer_prefix=layer_prefix, ep_id=ep_id, name_prefix=name_prefix)
                    # ====moe shared_expert
                    for c_name in MOE_EXPERT_PROJS:
                        self.m_base.common_to_mcore(c_name, c_ckpt, m_dict, t_name, layer_id, m_layer_id, layer_prefix=layer_prefix,
                                                    ep_id=ep_id, expert_name=MOE_SHARED_EXPERT, name_prefix=name_prefix)

                    # EXPERT
                    if expert_dict is not None:
                        for expert_id in expert_dict[ep_id]:
                            for c_name in MOE_EXPERT_PROJS:
                                self.m_moe.common_e_to_mcore(MOE_EXPERT, c_name, c_ckpt, m_dict, t_name, layer_id, m_layer_id,
                                                                ep_id, expert_id, layer_prefix=layer_prefix, name_prefix=name_prefix)

                    # MTP
                    if layer_id >= num_layers:
                        for c_name in MTP_NAMES:
                            if c_name == MTP_SHARED_HEAD_HEAD:
                                continue
                            self.m_base.common_to_mcore(c_name, c_ckpt, m_dict, t_name, layer_id, m_layer_id, layer_prefix=layer_prefix, ep_id=ep_id)

                    # final pp
                    if layer_id == num_layers - 1:
                        for c_name in LAST_LAYER_NAMES:
                            self.m_base.common_to_mcore(c_name, c_ckpt, m_dict, t_name, ep_id=ep_id)

            for mt in m_dict.keys():
                if ep_id is None:
                    t = mt
                    if save_file:
                        self.save_model_file(
                            release_dir, save_margs, p, t, None, m_dict[mt],
                            self.optim_state_dict[p][t] if self.optim_state_dict is not None else None, layer_ids)
                    else:
                        mcore_dict[p][t] = m_dict[mt]
                else:
                    if self.etp is None:
                        t = mt
                    else:
                        et = mt
                        t = etp_to_tp_mapping[ep_id][et]
                    if save_file:
                        self.save_model_file(
                            release_dir, save_margs, p, t, ep_id, m_dict[mt],
                            self.optim_state_dict[p][ep_id][et] if self.optim_state_dict is not None else None,
                            layer_ids)
                    else:
                        mcore_dict[p][ep_id][t] = m_dict[mt]

            if save_file:
                touch_file(done_dir=done_dir, p=p, ep_id=ep_id)
                logging.info(f"Finish saving {p=} {ep_id=} {layer_ids=}.")

        if expert_dict is None:
            convert_one_ep_from_common(ep_id=None)
        else:
            if self.args.max_workers > 1:
                futures = []
                with concurrent.futures.ThreadPoolExecutor(max_workers=self.args.max_workers) as executor:
                    for ep_id in expert_dict.keys():
                        futures.append(executor.submit(convert_one_ep_from_common, ep_id=ep_id))
                concurrent.futures.wait(futures)
                for future in futures:
                    try:
                        result = future.result()
                    except Exception as e:
                        logging.info(f"An error({p=}) occurred: {e}")
                        raise e
            else:
                for ep_id in expert_dict.keys():
                    convert_one_ep_from_common(ep_id=ep_id)
        logging.info(f"Finish saving mcore checkpoint. {p=} {layer_ids=}.")
        if not save_file:
            return mcore_dict

    def load_state_dict(self, load_path, p, t, e=None):
        checkpoint_name = "model_optim_rng.pt"
        if e is None or self.ep == 1:
            sub_dir_name = f"mp_rank_{t:02d}" if self.pp == 1 \
                    else f"mp_rank_{t:02d}_{p:03d}"
            checkpoint_path = os.path.join(load_path, sub_dir_name, checkpoint_name)
            logging.info(f"load checkpoint: {checkpoint_path}")
            return torch.load(checkpoint_path, map_location="cpu", weights_only=False)
        else:
            sub_dir_name = f"mp_rank_{t:02d}_{e:03d}" if self.pp == 1 \
                else f"mp_rank_{t:02d}_{p:03d}_{e:03d}"
            checkpoint_path = os.path.join(load_path, sub_dir_name, checkpoint_name)
            logging.info(f"load checkpoint: {checkpoint_path}")
            return torch.load(checkpoint_path, map_location="cpu", weights_only=False)

    def load_state_dict_from_mcore(self, load_path, p, ep_ids=None, tp_to_ep=None, etp_to_tp_mapping=None, mcore_dict=None):
        tp = self.tp
        # return {ep_id: {tp: state_dict}}
        m_dict = {}
        if ep_ids is None:
            for t in range(tp):
                if mcore_dict is None:
                    m_dict[t] = self.load_state_dict(load_path, p, t)
                else:
                    m_dict[t] = mcore_dict[p][t]
                self.checkpoint_version = m_dict[t].get('checkpoint_version', self.checkpoint_version)
            ep_mcore_state_dict = None
        elif self.etp is None:
            loaded_keys = {}
            if 0 in ep_ids:
                for ep_id in ep_ids:
                    for t in range(tp):
                        if mcore_dict is None:
                            m_dict[t] = self.load_state_dict(load_path, p, t, e=ep_id)
                        else:
                            m_dict[t] = mcore_dict[p][ep_id][t]
                        self.checkpoint_version = m_dict[t].get('checkpoint_version', self.checkpoint_version)
                        loaded_keys[f"{p}_{t}_{ep_id}"] = m_dict[t]
            else:
                m_dict = None
            ep_mcore_state_dict = {}
            for ep_id in ep_ids:
                ep_mcore_state_dict[ep_id] = {}
                for t in range(tp):
                    key = f"{p}_{t}_{ep_id}"
                    if key in loaded_keys:
                        ep_mcore_state_dict[ep_id][t] = loaded_keys[key]
                    else:
                        if mcore_dict is None:
                            ep_mcore_state_dict[ep_id][t] = self.load_state_dict(load_path, p, t, e=ep_id)
                        else:
                            ep_mcore_state_dict[ep_id][t] = mcore_dict[p][ep_id][t]
        else:
            assert tp_to_ep is not None, f"tp_to_ep is not provided, {ep_ids=}"
            assert etp_to_tp_mapping is not None, f"etp_to_tp_mapping is not provided, {ep_ids=}"
            loaded_keys = {}
            if 0 in ep_ids:
                for t in range(tp):
                    ep_id = tp_to_ep[t]
                    if mcore_dict is None:
                        m_dict[t] = self.load_state_dict(load_path, p, t, e=ep_id)
                    else:
                        m_dict[t] = mcore_dict[p][ep_id][t]
                    self.checkpoint_version = m_dict[t].get('checkpoint_version', self.checkpoint_version)
                    loaded_keys[f"{p}_{t}_{ep_id}"] = m_dict[t]
            else:
                m_dict = None
            ep_mcore_state_dict = {}
            for ep_id in ep_ids:
                assert ep_id in etp_to_tp_mapping, f"{etp_to_tp_mapping=} does not contain {ep_id=}"
                ep_mcore_state_dict[ep_id] = {}
                etp_to_tp = etp_to_tp_mapping[ep_id]
                for et in range(self.etp):
                    t = etp_to_tp[et]
                    key = f"{p}_{t}_{ep_id}"
                    if key in loaded_keys:
                        ep_mcore_state_dict[ep_id][et] = loaded_keys[key]
                    else:
                        if mcore_dict is None:
                            ep_mcore_state_dict[ep_id][et] = self.load_state_dict(load_path, p, t, e=ep_id)
                        else:
                            ep_mcore_state_dict[ep_id][et] = mcore_dict[p][ep_id][t]

        if m_dict is not None:
            assert len(m_dict) > 0, f"m_dict must not be empty"
            self.checkpoint_version = m_dict[0].get('checkpoint_version', self.checkpoint_version)
            self.rng_state = m_dict[0].get('rng_state', None)
        return m_dict, ep_mcore_state_dict

    def load(self, load_path, layer_dict, expert_dict=None, mcore_dict=None, lora_load_path=None):
        p = list(layer_dict.keys())[0]
        if expert_dict is None:
            self.m_dict, self.ep_mcore_state_dict = self.load_state_dict_from_mcore(load_path, p, mcore_dict=mcore_dict)
            if lora_load_path is not None:
                lora_m_dict, _ = self.load_state_dict_from_mcore(lora_load_path, p)
                for t in self.m_dict.keys():
                    for key in self.m_dict[t].keys():
                        if not key.startswith("model"):
                            continue
                        self.m_dict[t][key].update(lora_m_dict[t][key])
        else:
            ep_ids = list(expert_dict.keys())
            etp_to_tp_mapping, tp_to_ep = get_etp_map(self.tp, self.ep, self.etp)
            self.m_dict, self.ep_mcore_state_dict = self.load_state_dict_from_mcore(
                    load_path, p, ep_ids=ep_ids, tp_to_ep=tp_to_ep, etp_to_tp_mapping=etp_to_tp_mapping, mcore_dict=mcore_dict)
            if lora_load_path is not None:
                lora_m_dict, lora_ep_mcore_state_dict = self.load_state_dict_from_mcore(
                        lora_load_path, p, ep_ids=ep_ids, tp_to_ep=tp_to_ep, etp_to_tp_mapping=etp_to_tp_mapping)
                for t in self.m_dict.keys():
                    for key in self.m_dict[t].keys():
                        if not key.startswith("model"):
                            continue
                        self.m_dict[t][key].update(lora_m_dict[t][key])
                for e in self.ep_mcore_state_dict.keys():
                    for t in self.ep_mcore_state_dict[e].keys():
                        for key in self.ep_mcore_state_dict[e][t].keys():
                            if not key.startswith("model"):
                                continue
                            self.ep_mcore_state_dict[e][t][key].update(lora_ep_mcore_state_dict[e][t][key])

    def convert_to_common(self, layer_dict, expert_dict=None):
        """
        Convert Mcore checkpoint to common checkpoint.
            Args:
                load_path: str, the path of the mcore checkpoint.
                layer_dict: dict, the mapping between mcore layer name and common layer name.
                expert_dict: dict, {p -> {ep_id -> expert_ids}}.
            Returns:
                c_ckpt: CommonCheckpoint, the converted common checkpoint.
        """

        logging.info("\n==================== Mcore -> Common ====================")

        name_map = self.c_config.get("name_map")["mcore"]
        cargs = self.c_config.get_args("common")

        dualpipev = self.args.vpp_scheduler == 'dualpipev'
        custom_pipeline_layers = self.args.custom_pipeline_layers

        mtp_num_layers = self.args.mtp_num_layers if self.args.mtp_num_layers is not None else cargs.get("mtp_num_layers", 0)
        num_layers = cargs["num_layers"]
        stage = self.args.num_virtual_stages_per_pipeline_rank or 1
        num_layers_in_first_pipeline_stage = self.args.decoder_first_pipeline_num_layers
        num_layers_in_last_pipeline_stage = self.args.decoder_last_pipeline_num_layers
        if num_layers_in_first_pipeline_stage is not None or num_layers_in_last_pipeline_stage is not None:
            assert self.args.num_virtual_stages_per_pipeline_rank is not None, "num_virtual_stages_per_pipeline_rank is required"

        c_ckpt = CommonCheckpoint(self.c_config)

        num_layers_in_vp = get_num_layers_in_vp_map(
            stage, num_layers, self.pp, mtp_num_layers=mtp_num_layers,
            custom_pipeline_layers=custom_pipeline_layers,
            num_layers_in_first_pipeline_stage=num_layers_in_first_pipeline_stage,
            num_layers_in_last_pipeline_stage=num_layers_in_last_pipeline_stage)

        assert layer_dict != None and len(layer_dict) == 1, "layer_dict must be provided and size == 1"
        p = list(layer_dict.keys())[0]
 
        def convert_one_ep_to_common(ep_id=None):
            if p == 0:
                t_name = self.get_transformer_name(0)
                for c_name in FIRST_LAYER_NAMES:
                    self.m_base.mcore_to_common(c_name, c_ckpt, self.m_dict, t_name)
                for c_name in name_map.keys():
                    if c_name.startswith(VISION_MAP):
                        self.m_base.mcore_to_common(c_name, c_ckpt, self.m_dict, t_name)

            for stage_index in range(stage):
                virtual_p, mcore_layer_offset, = get_virtual_partition(dualpipev, stage_index, p, self.pp, num_layers_in_vp)
                t_name = self.get_transformer_name(stage_index)
                for cur_layer_id in range(num_layers_in_vp[virtual_p]):
                    layer_id = cur_layer_id + mcore_layer_offset
                    if layer_id >= num_layers:
                        m_layer_id = layer_id - num_layers
                        layer_prefix = name_map[MTP_LAYER_PREFIX]
                        name_prefix = self.name_prefix_for_layer if layer_prefix is not None else None
                    else:
                        layer_prefix = None
                        m_layer_id = cur_layer_id
                        name_prefix = None

                    for c_name in BASE_NAMES:
                        self.m_base.mcore_to_common(c_name, c_ckpt, self.m_dict, t_name, layer_id, m_layer_id,
                                                    layer_prefix=layer_prefix, name_prefix=name_prefix)
                    # ====moe shared_expert
                    for c_name in MOE_EXPERT_PROJS:
                        self.m_base.mcore_to_common(c_name, c_ckpt, self.m_dict, t_name, layer_id, m_layer_id,
                                                    expert_name=MOE_SHARED_EXPERT, layer_prefix=layer_prefix, name_prefix=name_prefix)

                    # EXPERT
                    if expert_dict is not None:
                        expert_ids = expert_dict[ep_id]
                        e_m_dict = self.ep_mcore_state_dict[ep_id]
                        for expert_id in expert_ids:
                            for c_name in MOE_EXPERT_PROJS:
                                self.m_moe.mcore_e_to_common(MOE_EXPERT, c_name, c_ckpt, e_m_dict, t_name,
                                                            layer_id, m_layer_id, expert_id, layer_prefix=layer_prefix, name_prefix=name_prefix)

                    # MTP
                    if layer_id >= num_layers:
                        for c_name in MTP_NAMES:
                            self.m_base.mcore_to_common(c_name, c_ckpt, self.m_dict, t_name, layer_id, m_layer_id, layer_prefix=layer_prefix)                                                                                                

                    # final pp
                    if layer_id == num_layers - 1:
                        for c_name in LAST_LAYER_NAMES:
                            self.m_base.mcore_to_common(c_name, c_ckpt, self.m_dict, t_name)

        if expert_dict is None:
            convert_one_ep_to_common(ep_id=None)
        else:
            if self.args.max_workers > 1:
                futures = []
                with concurrent.futures.ThreadPoolExecutor(max_workers=self.args.max_workers) as executor:
                    for ep_id in expert_dict.keys():
                        futures.append(executor.submit(convert_one_ep_to_common, ep_id=ep_id))
                concurrent.futures.wait(futures)
                for future in futures:
                    try:
                        result = future.result()
                    except Exception as e:
                        logging.info(f"An error({p=}) occurred: {e}")
                        raise e
            else:
                for ep_id in expert_dict.keys():
                    convert_one_ep_to_common(ep_id=ep_id)

        c_ckpt.other_args["iteration"] = self.iteration
        c_ckpt.other_args["checkpoint_version"] = self.checkpoint_version
        c_ckpt.other_args["args"] = self.args
        c_ckpt.other_args["rng_state"] = self.rng_state

        return c_ckpt

    def get_transformer_name(self, stage_index):
        """ get transformer name """
        if self.model_id is not None:
            return self.name_map[TRANSFORMER_TPL] % self.model_id
        elif self.num_stages > 1:
            return self.name_map[TRANSFORMER_TPL] % stage_index
        else:
            return self.name_map[TRANSFORMER]

    def pre_save(self, save_path, m_config=None):
        """
        Before saving the model, delete the old save directory,
        create a new save directory, and update the tracking file.
        If 'm_config' is not provided, the current 'mcore' configuration will be used.

        Args:
            save_path (str): Path where the model should be saved.
            m_config (Optional[dict], optional): Optional `mcore` configuration dictionary, default to None.

        Returns:
            tuple(str, dict): Returns a tuple containing two elements: the first is the new saved directory path,
                and the second is the updated `mcore` configuration dictionary.
        """
        os.makedirs(save_path, exist_ok=True)
        # Saving the tracker file
        tracker_filepath = os.path.join(save_path, "latest_checkpointed_iteration.txt")
        with open(tracker_filepath, "w") as f:
            f.write(str(self.iteration or "release"))

        # create `release` dir in args.load_path
        folder_name = f"iter_{self.iteration:07d}" if self.iteration > 0 else "release"
        release_dir = os.path.join(save_path, folder_name)
        os.makedirs(release_dir, exist_ok=True)

        # mcore config
        margs = self.args
        if m_config is not None:
            for k, v in m_config.data.items():
                setattr(margs, k, v)
        logging.info(f"Saving mcore args {margs}")
        return release_dir, margs

    def save_model_file(self, release_dir, margs, p, t, e, state_dict_node, optim_state_dict_node, saved_models_str):
        """
        Save the model file, including model parameters, optimizer state, and random seed.
        If the number of iterations is None, use mp_rank as the directory name; otherwise,
        use mp_rank and epoch as the directory name.

        Args:
            release_dir (str): The path of the release directory.
            margs (Optional[Namespace], optional): Namespace object of command line parameters, default is None.
            p (int): process number mp_rank.
            t (int): task number mp_rank.
            e (Optional[int], optional): The number of epochs, default to None.
            state_dict_node (Dict[str, Any]): Model parameter dictionary.
            optim_state_dict_node (Dict[str, Any]): Optimizer state dictionary.

        Returns:
            None.

        Raises:
            None.
        """
        state_dict_node["checkpoint_version"] = self.checkpoint_version
        if e is None or self.ep == 1:
            checkpoint_dir = (
                f"mp_rank_{t:02d}"
                if self.pp == 1
                else f"mp_rank_{t:02d}_{p:03d}"
            )
        else:
            checkpoint_dir = (
                f"mp_rank_{t:02d}_{e:03d}"
                if self.pp == 1
                else f"mp_rank_{t:02d}_{p:03d}_{e:03d}"
            )

        checkpoint_name = "model_optim_rng.pt"
        if optim_state_dict_node is not None:
            state_dict_node.update(optim_state_dict_node.to_dict())
        if margs is not None:
            state_dict_node['args'] = margs
        if self.rng_state is not None:
            state_dict_node['rng_state'] = self.rng_state
        state_dict_node["iteration"] = self.iteration
        checkpoint_dir = os.path.join(release_dir, checkpoint_dir)
        os.makedirs(checkpoint_dir, exist_ok=True)
        checkpoint_path = os.path.join(checkpoint_dir, checkpoint_name)
        torch.save(state_dict_node, checkpoint_path)
        logging.info(f"Saving mcore checkpoint {state_dict_node.keys()} to: {checkpoint_path}, {saved_models_str}")

    @staticmethod
    def convert_from_common_vlm(m_ckpt, m_vision_ckpt, c_vision_patch_config, c_ckpt, c_vision_ckpt, target_c_config,
                            target_c_vision_config, save_path, layer_dict, expert_dict, save_file=True):
        p = list(layer_dict.keys())[0]
        vision_num_layers = c_vision_patch_config.get_args("common")["num_layers"]
        vision_layer_dict = {}
        vision_layer_dict[0] = list(range(vision_num_layers))
        encoder_tp = m_vision_ckpt.tp
        state_dict = m_ckpt.convert_from_common(c_ckpt, target_c_config, layer_dict, expert_dict=expert_dict, save_file=False)
        vision_dict = m_vision_ckpt.convert_from_common(c_vision_ckpt, target_c_vision_config, vision_layer_dict, save_file=False)
        if save_file:
            done_dir = os.path.join(save_path, "dones")
            need_check_dones, done_keys = McoreCheckpoint.get_need_check_dones(done_dir, layer_dict, expert_dict)
            if need_check_dones:
                if p in done_keys:
                    logging.info(f"> p: {p} already converted. pass...")
                    return
            release_dir, save_margs = m_ckpt.pre_save(save_path, target_c_config)
        layer_ids = layer_dict[p]
        if expert_dict is None:
            for t in state_dict[p].keys():
                encode_t = t % encoder_tp
                for model in state_dict[p][t].keys():
                    if model in ("model", "model0"):
                        state_dict[p][t][model].update(vision_dict[0][encode_t][model])
                if save_file:
                    m_ckpt.save_model_file(
                        release_dir, save_margs, p, t, None, state_dict[p][t], None, layer_ids)
            if save_file:
                touch_file(done_dir=done_dir, p=p, ep_id=None)
                logging.info(f"Finish saving {p=} ep_id=None {layer_ids=}.")
        else:
            for e, t_v in state_dict[p].items():
                for t in t_v.keys():
                    encode_t = t % encoder_tp
                    for model in state_dict[p][e][t].keys():
                        if model in ("model", "model0"):
                            state_dict[p][e][t][model].update(vision_dict[0][encode_t][model])
                    if save_file:
                        m_ckpt.save_model_file(
                            release_dir, save_margs, p, t, e, state_dict[p][e][t], None, layer_ids)
                if save_file:
                    touch_file(done_dir=done_dir, p=p, ep_id=e)
                    logging.info(f"Finish saving {p=} ep_id={e} {layer_ids=}.")
        if not save_file:
            return state_dict

if __name__ == "__main__":
    pass

