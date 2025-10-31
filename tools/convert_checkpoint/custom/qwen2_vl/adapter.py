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
from copy import deepcopy
from safetensors.torch import load_file, save_file

SCRIPT_DIR = dirname(os.path.abspath(__file__))
sys.path.append(dirname(dirname(dirname(SCRIPT_DIR))))

from convert_checkpoint.arguments import parse_args
from convert_checkpoint.custom.qwen2_vl.util import (
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
    if args.megatron_path is not None:
        sys.path.insert(0, args.megatron_path)
    print(" ====== convert adapter from Megatron Core to HuggingFace ======")
    target = {}
    state_dict = load_megatron_checkpoint(args.load_ckpt_path)
    source = state_dict[0]['model'] if args.pipeline_model_parallel_size == 1 else state_dict[0][0]['model']
    for k1, k2 in name_map.items():
        target[k2] = source[k1]
    save_huggingface_checkpoint(target, args.save_ckpt_path)

elif (args.load_platform, args.save_platform) == ('huggingface', 'mcore'):
    """ huggingface to megatron """
    print(" ====== convert adapter from HuggingFace to Megatron Core ======")
    tp = args.tensor_model_parallel_size
    source = load_huggingface_checkpoint(args.load_ckpt_path)
    target = {}
    for k1, k2 in name_map.items():
        target[k1] = source[k2]
        print(f" > {k1}")
    for k in ['adapter.linear_fc1._extra_state', 'adapter.linear_fc2._extra_state']:
        extra_state = io.BytesIO()
        torch.save(None, extra_state)
        target[k] = extra_state
    state_dict = [{'model': deepcopy(target)} for i in range(tp)]
    save_megatron_checkpoint(state_dict, os.path.join(args.save_ckpt_path, 'release'))

else:
    raise NotImplementedError

