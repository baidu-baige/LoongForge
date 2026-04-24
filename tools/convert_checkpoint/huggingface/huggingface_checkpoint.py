# Copyright 2026 The LoongForge Authors.
# SPDX-License-Identifier: Apache-2.0

"""Load, save, and convert HuggingFace checkpoints within the common conversion pipeline."""

import os
import torch
import json
import logging

logging.basicConfig(level=logging.INFO)

import concurrent.futures
from convert_checkpoint.common.abstact_checkpoint import AbstractCheckpoint
from convert_checkpoint.arguments import parse_args
from convert_checkpoint.common.common_checkpoint import VISION_MAP, VISION_WORD_EMBEDDINGS, CommonCheckpoint

from convert_checkpoint.utils.utils import (
    get_done_keys,
    touch_file,
)

from convert_checkpoint.common.common_checkpoint import (
    TRANSFORMER, MTP_TRANSFORMER, MTP_LAYER_PREFIX, LAYER_PREFIX, MOE_EXPERT, MOE_SHARED_EXPERT, LAYER_IS_DICT_FOR_EXPERT,
    FIRST_LAYER_NAMES, BASE_NAMES, MOE_EXPERT_PROJS, LAST_LAYER_NAMES, MTP_NAMES, MTP_WORD_EMBEDDING,
    MOE_EXPERT_H_TO_4H, MOE_EXPERT_4H_TO_H, MOE_SHARED_EXPERT_H_TO_4H, MOE_SHARED_EXPERT_4H_TO_H,
    MTP_MOE_EXPERT_H_TO_4H, MTP_MOE_EXPERT_4H_TO_H, MTP_MOE_SHARED_EXPERT_H_TO_4H, MTP_MOE_SHARED_EXPERT_4H_TO_H
)

from convert_checkpoint.common.common_checkpoint import CommonCheckpoint
from convert_checkpoint.huggingface.huggingface_base import HuggingfaceBase
from convert_checkpoint.huggingface.huggingface_moe import HuggingfaceMoe

def get_hf_checkpoint_names(c_config, weight_map, layer_ids, expert_ids=None, mtp_num_layers=0, args=None):
    name_map = c_config.get("name_map")["huggingface"]
    cargs = c_config.get_args("common")
    hargs = c_config.get_args("huggingface")
    ori_num_layers = cargs["num_layers"]
    num_layers = ori_num_layers + mtp_num_layers
    mtp_transformer = name_map.get(MTP_TRANSFORMER, None)
    mtp_layer_id = hargs.get("mtp_layer_id", None)

    filenames_in_the_layer = set()

    if 0 in layer_ids or num_layers - 1 in layer_ids:
        for c_name in FIRST_LAYER_NAMES:
            if c_name in name_map:
                if name_map[c_name] is None:
                    continue
                name = name_map[c_name] + ".weight"
                if name in weight_map:
                    filenames_in_the_layer.add(weight_map[name])
    if args is not None and args.enable_full_hetero_dp:
        c_name = VISION_WORD_EMBEDDINGS
        if c_name in name_map and name_map[c_name] is not None:
            name = name_map[c_name] + ".weight"
            if name in weight_map:
                filenames_in_the_layer.add(weight_map[name])

    if 0 in layer_ids:
        for c_name in name_map.keys():
            if c_name.startswith(VISION_MAP):
                hf_name, _, _, _, no_layer_id, _, _ = HuggingfaceBase.get_hf_name_and_args(name_map[c_name])
                for ext in ["", ".weight", ".bias"]:
                    name = hf_name + ext
                    if name in weight_map:
                        filenames_in_the_layer.add(weight_map[name])

    if (num_layers - 1) in layer_ids:
        for c_name in LAST_LAYER_NAMES:
            if c_name in name_map:
                if name_map[c_name] is None:
                    continue
                name = name_map[c_name] + ".weight"
                if name in weight_map:
                    filenames_in_the_layer.add(weight_map[name])
        if mtp_num_layers > 0:
            for c_name in MTP_NAMES:
                if c_name in name_map:
                    if name_map[c_name] is None:
                        continue
                    hf_name, _, _, _, no_layer_id, _, _ = HuggingfaceBase.get_hf_name_and_args(name_map[c_name])
                    if not no_layer_id:
                        continue
                    name = hf_name + ".weight"
                    if name in weight_map:
                        filenames_in_the_layer.add(weight_map[name])

    ori_transformer = name_map[TRANSFORMER]
    layer_prefix = name_map[LAYER_PREFIX]
    if expert_ids is not None:
        moe_expert = name_map[MOE_EXPERT]
    for layer_id in layer_ids:
        transformer = ori_transformer
        cur_layer_id = layer_id
        if layer_id >= ori_num_layers and mtp_num_layers > 0 and mtp_transformer is not None:
            transformer = mtp_transformer
            if mtp_layer_id is not None:
                cur_layer_id = mtp_layer_id
        for key, value in weight_map.items():
            name_prefix = f"{transformer}.{layer_prefix}.{cur_layer_id}."
            if key.startswith(name_prefix) and value not in filenames_in_the_layer:
                if expert_ids is None or not key.startswith(f"{name_prefix}.{moe_expert}."):
                    filenames_in_the_layer.add(value)
                else:
                    for expert_id in expert_ids:
                        expert_prefix = f"{name_prefix}.{moe_expert}.{expert_id}."
                        if key.startswith(expert_prefix):
                            filenames_in_the_layer.add(value)
                            break
    return list(filenames_in_the_layer)

def merge_transformers_sharded_states(path, checkpoint_names, load_safe=False, max_workers=1):
    """
    Merge sharded checkpoints from transformers into a single checkpoint.

    Args:
        path (str): the path to the sharded checkpoints
        checkpoint_names (list): the names of the checkpoints to merge
    """
    if load_safe:
        from safetensors.torch import load_file
    state_dict = {}
    current_chunks = [None] * len(checkpoint_names)
    def load_files(checkpoint_path, i):
        if load_safe:
            current_chunks[i] = load_file(checkpoint_path, device="cpu")
        else:
            current_chunks[i] = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
        logging.info(f"Loaded huggingface checkpoint: {checkpoint_path}")
    if max_workers is None:
        for i in range(len(checkpoint_names)):
            load_files(os.path.join(path, checkpoint_names[i]), i)
    else:
        futures = []
        with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
            for i in range(len(checkpoint_names)):
                futures.append(executor.submit(load_files, os.path.join(path, checkpoint_names[i]), i))
        concurrent.futures.wait(futures)
        for future in futures:
            try:
                result = future.result()
            except Exception as e:
                logging.info(f"An error occurred: {e}")
                raise e
    for i in range(len(checkpoint_names)):
        state_dict.update(current_chunks[i])
    return state_dict


class HuggingFaceCheckpoint(AbstractCheckpoint):
    """
       HuggingFaceCheckpoint
    """

    def __init__(self, c_config, args):
        super().__init__(c_config)
        self.args = args
        self.margs = self.c_config.get_args("mcore")
        self.cargs = self.c_config.get_args("common")
        self.h_base = HuggingfaceBase(c_config, args)
        self.h_moe = HuggingfaceMoe(c_config, args)
        self.state_dict = {}

    @staticmethod
    def check_done_files(save_path, layer_dict, expert_dict=None):
        done_dir = os.path.join(save_path, "dones")
        p = list(layer_dict.keys())[0]
        ep_ids = expert_dict.keys() if expert_dict is not None else None
        if os.path.exists(done_dir):
            done_keys = get_done_keys(done_dir, p, ep_ids)
            if ep_ids is None:
                if (p, None) in done_keys:
                    logging.info(f"> p: {p} already converted. pass...")
                    return True
            else:
                all_done = True
                for ep_id in ep_ids:
                    if not (p, ep_id) in done_keys:
                        all_done = False
                if all_done:
                    logging.info(f"> p: {p}, ep_id: {ep_ids} already converted. pass...")
                    return True
        else:
            os.makedirs(done_dir, exist_ok=True)
        return False


    def convert_from_common(self, c_ckpt, layer_dict, expert_dict=None, save_path=None, save_file=True):
        """
        Convert HuggingFace checkpoint to common checkpoint.
        """

        logging.info("==================== Common -> HuggingFace ====================")

        cargs = self.c_config.get_args("common")
        hargs = self.c_config.get_args("huggingface")
        num_layers = cargs["num_layers"]
        mtp_layer_id = hargs.get("mtp_layer_id", None)
        name_map = self.c_config.get("name_map")["huggingface"]
        mtp_transformer = name_map.get(MTP_TRANSFORMER, None)
        mtp_layer_prefix = name_map.get(MTP_LAYER_PREFIX, None)

        p = list(layer_dict.keys())[0]
        layer_ids = layer_dict[p]
        ep_ids = list(expert_dict.keys()) if expert_dict is not None else None

        if save_file and self.check_done_files(save_path, layer_dict, expert_dict=expert_dict):
            return

        if 0 in layer_ids:
            for c_name in FIRST_LAYER_NAMES:
                self.h_base.common_to_hf(c_name, c_ckpt, self.state_dict)
            for c_name in name_map.keys():
                if c_name.startswith(VISION_MAP):
                    self.h_base.common_to_hf(c_name, c_ckpt, self.state_dict)

        for layer_id in layer_ids:
            hf_layer_id = mtp_layer_id + (layer_id - num_layers) if (layer_id >= num_layers and mtp_layer_id is not None) else layer_id
            transformer = mtp_transformer if layer_id >= num_layers else None
            layer_prefix = mtp_layer_prefix if layer_id >= num_layers else None
            for c_name in BASE_NAMES:
                self.h_base.common_to_hf(c_name, c_ckpt, self.state_dict, layer_id=layer_id,
                                         hf_layer_id=hf_layer_id, transformer=transformer, layer_prefix=layer_prefix)
            # ====moe shared_expert
            for c_name in MOE_EXPERT_PROJS:
                if c_name == MOE_EXPERT_H_TO_4H:
                    spec_name = MTP_MOE_SHARED_EXPERT_H_TO_4H if layer_id >= num_layers else MOE_SHARED_EXPERT_H_TO_4H
                else:
                    spec_name = MTP_MOE_SHARED_EXPERT_4H_TO_H if layer_id >= num_layers else MOE_SHARED_EXPERT_4H_TO_H
                self.h_base.common_to_hf(c_name, c_ckpt, self.state_dict, layer_id=layer_id,
                                         hf_layer_id=hf_layer_id, expert_name=MOE_SHARED_EXPERT,
                                         transformer=transformer, layer_prefix=layer_prefix, spec_name=spec_name)

            # EXPERT
            if expert_dict is not None:
                for ep_id, expert_ids in expert_dict.items():
                    for expert_id in expert_ids:
                        for c_name in MOE_EXPERT_PROJS:
                            if c_name == MOE_EXPERT_H_TO_4H:
                                spec_name = MTP_MOE_EXPERT_H_TO_4H if layer_id >= num_layers else None
                            else:
                                spec_name = MTP_MOE_EXPERT_4H_TO_H if layer_id >= num_layers else None
                            self.h_moe.common_e_to_hf(MOE_EXPERT, c_name, c_ckpt, self.state_dict, layer_id=layer_id,
                                                      hf_layer_id=hf_layer_id, expert_id=expert_id,
                                                      transformer=transformer, layer_prefix=layer_prefix, spec_name=spec_name)
            self.merge_dict_tensor(self.state_dict)
            # MTP
            if layer_id >= num_layers:
                for c_name in MTP_NAMES:
                    self.h_base.common_to_hf(c_name, c_ckpt, self.state_dict, layer_id=layer_id,
                                             hf_layer_id=hf_layer_id, transformer=transformer, layer_prefix=layer_prefix)

        if num_layers - 1 in layer_ids:
            for c_name in LAST_LAYER_NAMES:
                self.h_base.common_to_hf(c_name, c_ckpt, self.state_dict, layer_prefix=layer_prefix)

        if save_file:
            self.save_ckpt_file(save_path, p, ep_ids, self.state_dict)
        else:
            return self.state_dict

    def save_ckpt_file(self, save_path, p, ep_ids, state_dict):
        done_dir = os.path.join(save_path, "dones")
        if ep_ids is None or len(ep_ids) == 0:
            self.save(f"{save_path}/sub_checkpoint/{p}", state_dict, None)
            touch_file(done_dir=done_dir, p=p)
            logging.info(f"touch file: {done_dir=}, {p=}")
        else:
            self.save(f"{save_path}/sub_checkpoint/{p * 1000 + ep_ids[0]}", state_dict, None)
            for ep_id in ep_ids:
                touch_file(done_dir=done_dir, p=p, ep_id=ep_id)
                logging.info(f"touch file: {done_dir=}, {p=}, {ep_id=}")


    def convert_to_common(self, layer_dict, expert_dict=None):
        """
        Convert HuggingFace checkpoint to common checkpoint.
        """

        logging.info("==================== HuggingFace -> Common ====================")

        cargs = self.c_config.get_args("common")
        hargs = self.c_config.get_args("huggingface")
        c_ckpt = CommonCheckpoint(self.c_config)
        num_layers = cargs["num_layers"]
        mtp_layer_id = hargs.get("mtp_layer_id", None)
        name_map = self.c_config.get("name_map")["huggingface"]
        mtp_transformer = name_map.get(MTP_TRANSFORMER, None)
        mtp_layer_prefix = name_map.get(MTP_LAYER_PREFIX, None)

        p = list(layer_dict.keys())[0]
        layer_ids = layer_dict[p]

        if 0 in layer_ids:
            for c_name in FIRST_LAYER_NAMES:
                self.h_base.hf_to_common(c_name, c_ckpt, self.state_dict)
            for c_name in name_map.keys():
                if c_name.startswith(VISION_MAP):
                    self.h_base.hf_to_common(c_name, c_ckpt, self.state_dict)
        elif self.args.enable_full_hetero_dp:
            self.h_base.hf_to_common(VISION_WORD_EMBEDDINGS, c_ckpt, self.state_dict)

        for layer_id in layer_ids:
            hf_layer_id = mtp_layer_id if (layer_id >= num_layers and mtp_layer_id is not None) else layer_id
            transformer = mtp_transformer if layer_id >= num_layers else None
            layer_prefix = mtp_layer_prefix if layer_id >= num_layers else None
            for c_name in BASE_NAMES:
                self.h_base.hf_to_common(c_name, c_ckpt, self.state_dict, layer_id=layer_id,
                                         hf_layer_id=hf_layer_id, transformer=transformer, layer_prefix=layer_prefix)
            # ====moe shared_expert
            for c_name in MOE_EXPERT_PROJS:
                if c_name == MOE_EXPERT_H_TO_4H:
                    spec_name = MTP_MOE_SHARED_EXPERT_H_TO_4H if layer_id >= num_layers else MOE_SHARED_EXPERT_H_TO_4H
                else:
                    spec_name = MTP_MOE_SHARED_EXPERT_4H_TO_H if layer_id >= num_layers else MOE_SHARED_EXPERT_4H_TO_H
                self.h_base.hf_to_common(c_name, c_ckpt, self.state_dict, layer_id=layer_id, hf_layer_id=hf_layer_id,
                                         transformer=transformer, expert_name=MOE_SHARED_EXPERT, layer_prefix=layer_prefix, spec_name=spec_name)

            # EXPERT
            if expert_dict is not None:
                for ep_id, expert_ids in expert_dict.items():
                    for expert_id in expert_ids:
                        for c_name in MOE_EXPERT_PROJS:
                            if c_name == MOE_EXPERT_H_TO_4H:
                                spec_name = MTP_MOE_EXPERT_H_TO_4H if layer_id >= num_layers else None
                            else:
                                spec_name = MTP_MOE_EXPERT_4H_TO_H if layer_id >= num_layers else None
                            self.h_moe.hf_e_to_common(MOE_EXPERT, c_name, c_ckpt, self.state_dict,
                                                      layer_id=layer_id, hf_layer_id=hf_layer_id,
                                                      transformer=transformer, expert_id=expert_id, layer_prefix=layer_prefix, spec_name=spec_name)

            # MTP
            if layer_id >= num_layers:
                for c_name in MTP_NAMES:
                    self.h_base.hf_to_common(c_name, c_ckpt, self.state_dict, layer_id=layer_id,
                                             hf_layer_id=hf_layer_id, transformer=transformer, layer_prefix=layer_prefix)
                

        if num_layers - 1 in layer_ids:
            for c_name in LAST_LAYER_NAMES:
                self.h_base.hf_to_common(c_name, c_ckpt, self.state_dict)

        return c_ckpt

    def load(self, load_path, load_safe=False, c_config=None, layer_ids=[], expert_ids=None, mtp_num_layers=0):
        """ load ckpt """
        if load_safe:
            from safetensors.torch import load_file
            sub_dirs = [x for x in os.listdir(load_path) if x.endswith("safetensors")]
            if not os.path.exists(os.path.join(load_path, "model.safetensors.index.json")):
                checkpoint_name = "model.safetensors"
                self.state_dict = load_file(os.path.join(load_path, checkpoint_name), device="cpu")
                logging.info(f"Load HuggingFace from: {os.path.join(load_path, checkpoint_name)}")
            else:
                meta_path = f"{load_path}/model.safetensors.index.json"
                with open(meta_path, 'r') as f:
                    file_content = json.load(f)
                weight_map = file_content["weight_map"]
                checkpoint_names = get_hf_checkpoint_names(c_config, weight_map, layer_ids, expert_ids=expert_ids,
                                                           mtp_num_layers=mtp_num_layers, args=self.args)
                self.state_dict = merge_transformers_sharded_states(load_path, checkpoint_names, load_safe=True, max_workers=self.args.max_workers)
                logging.info(f"merge_transformers_sharded_states: {load_path}")
        else:
            sub_dirs = [x for x in os.listdir(load_path) if x.startswith("pytorch_model")]
            if len(sub_dirs) == 1:
                checkpoint_name = "pytorch_model.bin"
                self.state_dict = torch.load(os.path.join(load_path, checkpoint_name),
                                             map_location="cpu", weights_only=False)
                logging.info(f"Load HuggingFace from: {os.path.join(load_path, checkpoint_name)}")
            else:
                meta_path = f"{load_path}/pytorch_model.bin.index.json"
                with open(meta_path, 'r') as f:
                    file_content = json.load(f)
                weight_map = file_content["weight_map"]
                checkpoint_names = get_hf_checkpoint_names(c_config, weight_map, layer_ids, expert_ids=expert_ids,
                                                           mtp_num_layers=mtp_num_layers, args=self.args)
                self.state_dict = merge_transformers_sharded_states(load_path, checkpoint_names, max_workers=self.args.max_workers)
                logging.info(f"merge_transformers_sharded_states: {load_path}")


    def print_memory_usage(self, desc):
        import psutil
        process = psutil.Process(os.getpid())
        mem = process.memory_info().rss / 1024**2  # Convert to MB
        logging.info(f"{desc} memory usage: {mem:.2f} MB")

    def save(self, save_path, state_dict, h_config=None, save_optim=False):
        """ save ckpt """
        from huggingface_hub import split_torch_state_dict_into_shards
        from transformers.modeling_utils import SAFE_WEIGHTS_INDEX_NAME
        from safetensors.torch import save_file
        os.makedirs(save_path, exist_ok=True)
        state_dict_split = split_torch_state_dict_into_shards(state_dict)
        self.print_memory_usage(f"before save {save_path}")
        has_safetensor_file = False

        def save_hf_shard(tensors, shard_file):
            shard = {}
            for tensor in tensors:
                shard[tensor] = state_dict[tensor].clone().contiguous()
                del state_dict[tensor]
            shard_path = os.path.join(save_path, shard_file)
            save_file(shard, shard_path, metadata={"format": "pt"})
            logging.info(f"Saving HuggingFace shard to: {shard_path}")

        if self.args.max_workers > 1:
            futures = []
            with concurrent.futures.ThreadPoolExecutor(max_workers=self.args.max_workers) as executor:
                for shard_file, tensors in state_dict_split.filename_to_tensors.items():
                    has_safetensor_file = True
                    futures.append(executor.submit(save_hf_shard, tensors=tensors, shard_file=shard_file))
            concurrent.futures.wait(futures)
            for future in futures:
                try:
                    result = future.result()
                except Exception as e:
                    logging.info(f"An error occurred: {e}")
                    raise e
        else:
            for shard_file, tensors in state_dict_split.filename_to_tensors.items():
                has_safetensor_file = True
                save_hf_shard(tensors, shard_file)
        self.print_memory_usage(f"after save {save_path}")

        if state_dict_split.is_sharded:
            index = {
                "metadata": state_dict_split.metadata,
                "weight_map": state_dict_split.tensor_to_filename,
            }
            save_index_file = SAFE_WEIGHTS_INDEX_NAME
            save_index_file = os.path.join(save_path, save_index_file)
            with open(save_index_file, "w", encoding="utf-8") as f:
                content = json.dumps(index, indent=2, sort_keys=True) + "\n"
                f.write(content)
        elif has_safetensor_file:
            for key in state_dict_split.tensor_to_filename.keys():
                if state_dict_split.tensor_to_filename[key] == "model.safetensors":
                    state_dict_split.tensor_to_filename[key] = "model-00001-of-00001.safetensors"
            index = {
                "metadata": state_dict_split.metadata,
                "weight_map": state_dict_split.tensor_to_filename,
            }
            save_index_file = SAFE_WEIGHTS_INDEX_NAME
            save_index_file = os.path.join(save_path, save_index_file)
            with open(save_index_file, "w", encoding="utf-8") as f:
                content = json.dumps(index, indent=2, sort_keys=True) + "\n"
                f.write(content)
            os.rename(os.path.join(save_path, 'model.safetensors'), \
                      os.path.join(save_path, 'model-00001-of-00001.safetensors'))

        if h_config is not None:
            h_config.save(save_path)

    def merge_dict_tensor(self, state_dict):
        for key, value in state_dict.items():
            if isinstance(value, dict) and LAYER_IS_DICT_FOR_EXPERT in value and value[LAYER_IS_DICT_FOR_EXPERT]:
                value.pop(LAYER_IS_DICT_FOR_EXPERT)
                sorted_items = sorted(value.items())
                tensors = [tensor for _, tensor in sorted_items]
                state_dict[key] = torch.stack(tensors, dim=0)

    @staticmethod
    def save_vlm_checkpoint(hf_ckpt, hf_vision_ckpt, c_vision_patch_config, c_ckpt, c_vision_ckpt, save_path, layer_dict, expert_dict=None):
        if hf_ckpt.check_done_files(save_path, layer_dict, expert_dict=expert_dict):
            return
        vision_num_layers = c_vision_patch_config.get_args("common")["num_layers"]
        vision_layer_dict = {}
        vision_layer_dict[0] = list(range(vision_num_layers)) 
        state_dict = hf_ckpt.convert_from_common(c_ckpt, layer_dict, expert_dict=expert_dict, save_path=save_path, save_file=False)
        vision_ckpt = hf_vision_ckpt.convert_from_common(c_vision_ckpt, vision_layer_dict, save_file=False)
        state_dict.update(vision_ckpt)
        # save checkpoint file
        ep_ids = list(expert_dict.keys()) if expert_dict is not None else None
        hf_ckpt.save_ckpt_file(save_path, 0, ep_ids, state_dict)
