#!/usr/bin/env python3
# -*- coding: UTF-8 -*-
################################################################################
#
# Copyright (c) 2024 Baidu.com, Inc. All Rights Reserved
#
################################################################################

import os
import sys
import json
import torch
import argparse
from os.path import dirname
from copy import deepcopy
from einops import rearrange
from safetensors.torch import load_file, save_file

SCRIPT_DIR = dirname(os.path.abspath(__file__))
sys.path.append(dirname(dirname(dirname(SCRIPT_DIR))))

from convert_checkpoint.custom.qwen2_vl.util import (
    load_megatron_checkpoint,
    save_megatron_checkpoint,
)


def parse_args(title=None):
    """Parse all arguments."""
    parser = argparse.ArgumentParser(description='Merger Arguments', allow_abbrev=False)
    group = parser.add_argument_group(title='checkpoint')
    group.add_argument('--language_model_path', type=str, help="Path to language model."),
    group.add_argument('--vision_model_path', type=str, help="Path to vision model."),
    group.add_argument('--vision_patch', type=str, help="Path to vision patch."),
    group.add_argument('--adapter_path', type=str, help="Path to adapter."),
    group.add_argument("--save_ckpt_path", type=str, help="Path to save checkpoint.")
    group.add_argument("--megatron_path", type=str, help="Base directory of Megatron repository")
    group.add_argument("--tensor_model_parallel_size", type=int, default=1, help="Tensor parallel size.")
    group.add_argument("--pipeline_model_parallel_size", type=int, default=1, help="Pipeline parallel size.")

    return parser.parse_args()


def merge_dict(source, destination):
    """ merge two dictionaries recursively """
    for key, value in source.items():
        if isinstance(value, dict):
            node = destination.setdefault(key, {})
            merge_dict(value, node)
        else:
            destination[key] = value


args = parse_args()
if args.megatron_path is not None:
    sys.path.insert(0, args.megatron_path)

print("===== merge megatron checkpoints ======")

language_model = load_megatron_checkpoint(args.language_model_path)
vision_model = load_megatron_checkpoint(args.vision_model_path)
adapter = load_megatron_checkpoint(args.adapter_path)
patch = load_megatron_checkpoint(args.vision_patch)

if args.pipeline_model_parallel_size == 1:
    for module in [vision_model, adapter, patch]:
        assert len(module) == len(language_model)
        for i in range(len(module)):
            merge_dict(module[i]['model'], language_model[i]['model'])
else:
    for module in [vision_model, adapter, patch]:
        assert len(module) == len(language_model[0])
        for i in range(len(module)):
            merge_dict(module[i]['model'], language_model[0][i]['model'])

save_megatron_checkpoint(language_model, args.save_ckpt_path)