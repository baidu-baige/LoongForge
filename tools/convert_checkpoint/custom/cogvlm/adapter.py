#!/usr/bin/env python3
# -*- coding: UTF-8 -*-
################################################################################
#
# Copyright (c) 2024 Baidu.com, Inc. All Rights Reserved
#
################################################################################

import os
import io
import sys
import json
import torch
from os.path import dirname
from einops import rearrange
from safetensors.torch import load_file, save_file

SCRIPT_DIR = dirname(os.path.abspath(__file__))
sys.path.append(dirname(dirname(dirname(SCRIPT_DIR))))

from convert_checkpoint.arguments import parse_args
from convert_checkpoint.custom.cogvlm.util import (
    load_megatron_checkpoint,
    save_megatron_checkpoint,
    load_huggingface_checkpoint,
    save_huggingface_checkpoint,
)


args = parse_args()
name_map = {} # megatron -> huggingface
with open(args.common_config_path, 'r', encoding='utf-8') as f:
    name_map = json.loads(f.read())

if (args.load_platform, args.save_platform) == ('mcore', 'huggingface'):
    """ megatron to huggingface """
    print(" ====== convert adapters from Megatron Core to HuggingFace ======")
    if args.megatron_path is not None:
        sys.path.insert(0, args.megatron_path)
    target = {}
    source = {}
    state_dict = load_megatron_checkpoint(args.load_ckpt_path)
    tp = len(state_dict)
    state_dict = [state_dict[t]['model'] for t in range(tp)]
    for k, v in state_dict[0].items():
        if k.startswith("adapter") and isinstance(v, torch.Tensor):
            if k == "adapter.mlp.linear_fc1.weight":
                source[k] = torch.cat(tuple(state_dict[t][k] for t in range(tp)), dim = 0)
            elif k == "adapter.mlp.linear_fc2.weight":
                source[k] = torch.cat(tuple(state_dict[t][k] for t in range(tp)), dim = 1)
            else:
                source[k] = state_dict[0][k].clone()
            print(f" > {k}")
    for k1, k2 in name_map.items(): # mcore -> huggingface
        if 'adapter.mlp.linear_fc1.weight' == k1:
            source[k1] = rearrange(source[k1], "(TP N D) H -> (N TP D) H",
                N = 2, TP = tp, D = 14336 // tp, H = 4096)
        if isinstance(k2, str):
            target[k2] = source[k1]
        elif isinstance(k2, list):
            weights = torch.chunk(source[k1], len(k2))
            for i, k in enumerate(k2):
                target[k] = weights[i].clone()
        else:
            raise ValueError
    save_huggingface_checkpoint(target, args.save_ckpt_path)

elif (args.load_platform, args.save_platform) == ('huggingface', 'mcore'):
    """ huggingface to megatron """
    print(" ====== convert adapters from HuggingFace to Megatron Core ======")
    tp = args.tensor_model_parallel_size
    source = load_huggingface_checkpoint(args.load_ckpt_path)
    target = {}
    for k1, k2 in name_map.items(): # mcore -> huggingface
        if isinstance(k2, str):
            target[k1] = source[k2]
        elif isinstance(k2, list):
            target[k1] = torch.cat([source[i] for i in k2])
        else:
            raise ValueError
        if 'adapter.mlp.linear_fc1.weight' == k1:
            target[k1] = rearrange(target[k1], "(N TP D) H -> (TP N D) H",
                N = 2, TP = tp, D = 14336 // tp, H = 4096)
    state_dict = [{'model': {}} for i in range(tp)]
    for k, v in target.items():
        for t in range(tp):
            if k == "adapter.mlp.linear_fc1.weight":
                state_dict[t]['model'][k] = v.chunk(tp, dim = 0)[t].clone()
            elif k == "adapter.mlp.linear_fc2.weight":
                state_dict[t]['model'][k] = v.chunk(tp, dim = 1)[t].clone()
            else:
                state_dict[t]['model'][k] = v.clone()

            if "linear" in k:
                a = io.BytesIO()
                torch.save(None, a)
                state_dict[t]['model'][k.replace('.weight', '._extra_state')] = a
        print(f" > {k}")
    save_megatron_checkpoint(state_dict, os.path.join(args.save_ckpt_path, 'release'))

else:
    raise NotImplementedError

