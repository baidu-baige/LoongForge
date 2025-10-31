#!/usr/bin/env python3
# -*- coding: UTF-8 -*-
################################################################################
#
# Copyright (c) 2024 Baidu.com, Inc. All Rights Reserved
#
################################################################################

import os
import sys
import torch
import json
from os.path import dirname
from safetensors.torch import load_file, save_file
from huggingface_hub import split_torch_state_dict_into_shards
from transformers.modeling_utils import SAFE_WEIGHTS_INDEX_NAME, SAFE_WEIGHTS_NAME


def merge_transformers_sharded_states(path, num_checkpoints):
    """
    Merge sharded checkpoints from transformers into a single checkpoint.

    Args:
        path (str): the path to the sharded checkpoints
        num_checkpoints (int): the number of checkpoints to merge
    """
    state_dict = {}
    for i in range(1, num_checkpoints + 1):
        checkpoint_path = os.path.join(path, f"model-{i:05d}-of-{num_checkpoints:05d}.safetensors")
        current_chunk = load_file(checkpoint_path)
        state_dict.update(current_chunk)
    return state_dict

def load_huggingface_checkpoint(load_path):
    """ load ckpt """
    state_dict = {}
    sub_dirs = [x for x in os.listdir(load_path) if x.endswith("safetensors")]
    if len(sub_dirs) == 1:
        checkpoint_name = "model.safetensors"
        state_dict = load_file(os.path.join(load_path, checkpoint_name), device="cpu")
    else:
        num_checkpoints = len(sub_dirs)
        state_dict = merge_transformers_sharded_states(load_path, num_checkpoints)
    return state_dict


def save_huggingface_checkpoint(state_dict, save_path):
    """ save ckpt """
    os.makedirs(save_path, exist_ok=True)
    checkpoint_path = os.path.join(save_path, "model.safetensors")

    state_dict_split = split_torch_state_dict_into_shards(state_dict)
    for shard_file, tensors in state_dict_split.filename_to_tensors.items():
        shard = {}
        for tensor in tensors:
            shard[tensor] = state_dict[tensor].contiguous()
            del state_dict[tensor]
        shard_path = os.path.join(save_path, shard_file)
        save_file(shard, shard_path, metadata={"format": "pt"})
        print(f"Saving HuggingFace shard to: {shard_path}")

    if state_dict_split.is_sharded:
        index = {
            "metadata": state_dict_split.metadata,
            "weight_map": state_dict_split.tensor_to_filename,
        }
        save_index_file = os.path.join(save_path, SAFE_WEIGHTS_INDEX_NAME)
        with open(save_index_file, "w", encoding="utf-8") as f:
            content = json.dumps(index, indent=2, sort_keys=True) + "\n"
            f.write(content)


def load_megatron_checkpoint(load_path):
    """ load ckpt """
    state_dict = []
    sub_dirs = sorted([x for x in os.listdir(load_path) if x.startswith("mp_rank")])
    last_dir = sub_dirs[-1].split('_')
    if len(last_dir) == 4:
        tp = int(last_dir[-2]) + 1
        pp = int(last_dir[-1]) + 1
        for p in range(pp):
            state_dict.append([])
            for t in range(tp):
                checkpoint_name = f"mp_rank_{t:02d}_{p:03d}/model_optim_rng.pt"
                ckpt = torch.load(os.path.join(load_path, checkpoint_name), map_location='cpu', weights_only=False)
                state_dict[p].append(ckpt)
        return state_dict
    else:
        for t in range(len(sub_dirs)):
            checkpoint_name = f"mp_rank_{t:02d}/model_optim_rng.pt"
            ckpt = torch.load(os.path.join(load_path, checkpoint_name), map_location='cpu', weights_only=False)
            state_dict.append(ckpt)
        return state_dict


def save_megatron_checkpoint(state_dict, save_path):
    """ save ckpt """
    if isinstance(state_dict[0], list):
        for p in range(len(state_dict)):
            for t in range(len(state_dict[p])):
                sub_dir_name = f"mp_rank_{t:02d}_{p:03d}"
                os.makedirs(os.path.join(save_path, sub_dir_name), exist_ok=True)
                checkpoint_path = os.path.join(save_path, sub_dir_name, "model_optim_rng.pt")
                torch.save(state_dict[p][t], checkpoint_path)
                print(f"Saving Megatron shard to: {checkpoint_path}")
    else:
        for t in range(len(state_dict)):
            sub_dir_name = f"mp_rank_{t:02d}"
            os.makedirs(os.path.join(save_path, sub_dir_name), exist_ok=True)
            checkpoint_path = os.path.join(save_path, sub_dir_name, "model_optim_rng.pt")
            torch.save(state_dict[t], checkpoint_path)
            print(f"Saving Megatron shard to: {checkpoint_path}")