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

from convert_checkpoint.custom.cogvlm.util import (
    load_megatron_checkpoint,
    save_megatron_checkpoint,
)


def parse_args(title=None):
    """Parse all arguments."""
    parser = argparse.ArgumentParser(description='Merger Arguments', allow_abbrev=False)
    group = parser.add_argument_group(title='checkpoint')
    group.add_argument('--language_expert_path', type=str, help="Path to language expert model."),
    group.add_argument('--vision_expert_path', type=str, help="Path to vision expert model."),
    group.add_argument('--vision_model_path', type=str, help="Path to vision model."),
    group.add_argument('--vision_patch', type=str, help="Path to vision patch."),
    group.add_argument('--adapter_path', type=str, help="Path to adapter."),
    group.add_argument("--save_ckpt_path", type=str, help="Path to save checkpoint.")
    group.add_argument("--megatron_path", type=str, help="Base directory of Megatron repository")

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

language_expert = load_megatron_checkpoint(args.language_expert_path)
vision_expert = load_megatron_checkpoint(args.vision_expert_path)
vision_model = load_megatron_checkpoint(args.vision_model_path)
adapter = load_megatron_checkpoint(args.adapter_path)
patch = load_megatron_checkpoint(args.vision_patch)

for module in [vision_expert, vision_model, adapter, patch]:
    assert len(module) == len(language_expert)
    for i in range(len(module)):
        merge_dict(module[i]['model'], language_expert[i]['model'])

save_megatron_checkpoint(language_expert, args.save_ckpt_path)