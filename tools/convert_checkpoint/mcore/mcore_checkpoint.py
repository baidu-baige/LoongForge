""" Mcore_checkpoint converter for aiak megatron. """
import os
import torch
import logging
import argparse

logging.basicConfig(level=logging.INFO)

import concurrent.futures
from convert_checkpoint.arguments import parse_args
from convert_checkpoint.common.abstact_checkpoint import AbstractCheckpoint
from convert_checkpoint.common.common_checkpoint import CommonCheckpoint
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

    def __init__(self, c_config):
        super().__init__(c_config)
        self.args = parse_args()
        self.m_base = McoreBase(c_config)
        self.m_moe = McoreMoe(c_config)
        self.iteration = 0
        self.checkpoint_version = 3.0
        self.rng_state = None
        margs = c_config.get_args("mcore")
        cargs = c_config.get_args("common")
        num_layers = cargs["num_layers"]
        num_layers_per_stage = self.args.num_layers_per_virtual_pipeline_stage

        self.tp = self.args.tensor_model_parallel_size
        self.pp = self.args.pipeline_model_parallel_size
        self.ep = self.args.expert_parallel_size
        self.etp = self.args.expert_tensor_parallel_size if hasattr(self.args, 'expert_tensor_parallel_size') else None

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


    def convert_from_common(self, c_ckpt, m_config, layer_dict, expert_dict=None):
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
        mtp_has_word_embeddings = cargs.get("mtp_has_word_embeddings", False)

        num_nextn_predict_layers = cargs.get("num_nextn_predict_layers", 0)
        num_layers = cargs["num_layers"]
        stage = self.args.num_virtual_stages_per_pipeline_rank or 1
        num_layers_in_first_pipeline_stage = self.args.decoder_first_pipeline_num_layers
        num_layers_in_last_pipeline_stage = self.args.decoder_last_pipeline_num_layers
        if num_layers_in_first_pipeline_stage is not None or num_layers_in_last_pipeline_stage is not None:
            assert self.args.num_virtual_stages_per_pipeline_rank is not None, "num_virtual_stages_per_pipeline_rank is required"

        num_layers_in_vp = get_num_layers_in_vp_map(
            stage, num_layers, self.pp, num_nextn_predict_layers=num_nextn_predict_layers,
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

        etp_to_tp_mapping, tp_to_ep = get_etp_map(self.tp, self.ep, self.etp)

        # check dones dir and mkdir release
        save_path = self.args.save_ckpt_path
        done_dir = os.path.join(save_path, "dones")
        need_check_dones, done_keys = McoreCheckpoint.get_need_check_dones(done_dir, layer_dict, expert_dict)
        release_dir, save_margs = self.pre_save(save_path, m_config)

        def convert_one_ep_from_common(ep_id=None):
            if need_check_dones:
                if ep_id is None and p in done_keys:
                    logging.info(f"> p: {p} already converted. pass...")
                    return
                if ep_id is not None and (p, ep_id) in done_keys:
                    logging.info(f"> p: {p}, ep_id: {ep_id} already converted. pass...")
                    return
            m_dict = {}
            if ep_id is None or self.etp is None:
                for t in range(self.tp):
                    m_dict[t] = {}
            else:
                for et in range(self.etp):
                    m_dict[et] = {}
            if p == 0:
                t_name = self.get_transformer_name(0)
                for c_name in FIRST_LAYER_NAMES:
                    self.m_base.common_to_mcore(c_name, c_ckpt, m_dict, t_name, ep_id=ep_id)

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
                        if mtp_has_word_embeddings:
                            self.m_base.common_to_mcore(WORD_EMBEDDINGS, c_ckpt, m_dict, t_name, ep_id=ep_id)

                    # final pp
                    if layer_id == num_layers - 1:
                        for c_name in LAST_LAYER_NAMES:
                            self.m_base.common_to_mcore(c_name, c_ckpt, m_dict, t_name, ep_id=ep_id)

            for mt in m_dict.keys():
                if ep_id is None:
                    t = mt
                    self.save_model_file(
                        release_dir, save_margs, p, t, None, m_dict[t],
                        self.optim_state_dict[p][t] if self.optim_state_dict is not None else None, layer_ids)
                else:
                    if self.etp is None:
                        t = mt
                    else:
                        et = mt
                        t = etp_to_tp_mapping[ep_id][et]
                    self.save_model_file(
                        release_dir, save_margs, p, t, ep_id, m_dict[mt],
                        self.optim_state_dict[p][ep_id][et] if self.optim_state_dict is not None else None,
                        layer_ids)

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

    def load_state_dict_from_mcore(self, load_path, p, ep_ids=None, tp_to_ep=None, etp_to_tp_mapping=None):
        tp = self.tp
        # return {ep_id: {tp: state_dict}}
        m_dict = {}
        if ep_ids is None:
            for t in range(tp):
                m_dict[t] = self.load_state_dict(load_path, p, t)
                self.checkpoint_version = m_dict[t]['checkpoint_version']
            ep_mcore_state_dict = None
        elif self.etp is None:
            loaded_keys = {}
            for ep_id in ep_ids:
                for t in range(tp):
                    m_dict[t] = self.load_state_dict(load_path, p, t, e=ep_id)
                    self.checkpoint_version = m_dict[t]['checkpoint_version']
                    loaded_keys[f"{p}_{t}_{ep_id}"] = m_dict[t]
            ep_mcore_state_dict = {}
            for ep_id in ep_ids:
                ep_mcore_state_dict[ep_id] = {}
                for t in range(tp):
                    key = f"{p}_{t}_{ep_id}"
                    if key in loaded_keys:
                        ep_mcore_state_dict[ep_id][t] = loaded_keys[key]
                    else:
                        ep_mcore_state_dict[ep_id][t] = self.load_state_dict(load_path, p, t, e=ep_id)
        else:
            assert tp_to_ep is not None, f"tp_to_ep is not provided, {ep_ids=}"
            assert etp_to_tp_mapping is not None, f"etp_to_tp_mapping is not provided, {ep_ids=}"
            loaded_keys = {}
            for t in range(tp):
                ep_id = tp_to_ep[t]
                m_dict[t] = self.load_state_dict(load_path, p, t, e=ep_id)
                self.checkpoint_version = m_dict[t]['checkpoint_version']
                loaded_keys[f"{p}_{t}_{ep_id}"] = m_dict[t]
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
                        ep_mcore_state_dict[ep_id][et] = self.load_state_dict(load_path, p, t, e=ep_id)

        assert len(m_dict) > 0, f"m_dict must not be empty"
        self.checkpoint_version = m_dict[0].get('checkpoint_version', 3.0)
        self.rng_state = m_dict[0].get('rng_state', None)
        return m_dict, ep_mcore_state_dict


    def load(self, load_path, layer_dict, expert_dict=None):
        p = list(layer_dict.keys())[0]
        if expert_dict is None:
            self.m_dict, self.ep_mcore_state_dict = self.load_state_dict_from_mcore(load_path, p)
        else:
            ep_ids = list(expert_dict.keys())
            etp_to_tp_mapping, tp_to_ep = get_etp_map(self.tp, self.ep, self.etp)
            self.m_dict, self.ep_mcore_state_dict = self.load_state_dict_from_mcore(
                    load_path, p, ep_ids=ep_ids, tp_to_ep=tp_to_ep, etp_to_tp_mapping=etp_to_tp_mapping)

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
        margs = self.c_config.get_args("mcore")

        dualpipev = self.args.vpp_scheduler == 'dualpipev'
        custom_pipeline_layers = self.args.custom_pipeline_layers

        num_nextn_predict_layers = cargs.get("num_nextn_predict_layers", 0)
        num_layers = cargs["num_layers"]
        stage = self.args.num_virtual_stages_per_pipeline_rank or 1
        num_layers_in_first_pipeline_stage = self.args.decoder_first_pipeline_num_layers
        num_layers_in_last_pipeline_stage = self.args.decoder_last_pipeline_num_layers
        if num_layers_in_first_pipeline_stage is not None or num_layers_in_last_pipeline_stage is not None:
            assert self.args.num_virtual_stages_per_pipeline_rank is not None, "num_virtual_stages_per_pipeline_rank is required"

        c_ckpt = CommonCheckpoint(self.c_config)

        num_layers_in_vp = get_num_layers_in_vp_map(
            stage, num_layers, self.pp, num_nextn_predict_layers=num_nextn_predict_layers,
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
        if self.args.vit_in_first_virtual_stage_only:
            return self.name_map[TRANSFORMER_TPL] % 0
        if self.num_stages > 1:
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

if __name__ == "__main__":
    pass

