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

from convert_checkpoint.custom.internvl.util import (
    load_huggingface_checkpoint,
    save_huggingface_checkpoint,
)

def parse_args():
    """Parse all arguments."""
    parser = argparse.ArgumentParser(description='Merger Arguments', allow_abbrev=False)
    group = parser.add_argument_group(title='checkpoint')
    group.add_argument('--language_model_path', type=str, help="Path to language expert model."),
    group.add_argument('--vision_model_path', type=str, help="Path to vision model."),
    group.add_argument('--vision_patch', type=str, help="Path to vision patch."),
    group.add_argument('--adapter_path', type=str, help="Path to adapter."),
    group.add_argument("--save_ckpt_path", type=str, help="Path to save checkpoint.")
    group.add_argument("--megatron_path", type=str, help="Base directory of Megatron repository")

    # add prefix when finally convert to InternVLChatModel
    group.add_argument("--language_model_prefix", type=str, default="")
    group.add_argument("--vision_model_prefix", type=str, default="")
    group.add_argument("--adapter_prefix", type=str, default="")

    return parser.parse_args()

def merge_dict(source, destination):
    """ merge two dictionaries recursively """
    for key, value in source.items():
        if isinstance(value, dict):
            node = destination.setdefault(key, {})
            merge_dict(value, node)
        else:
            destination[key] = value


def _add_prefix_to_keys(state_dict, prefix):
    if not prefix:
        return
    for key in list(state_dict.keys()):
        state_dict[prefix + key] = state_dict.pop(key)

def merge_huggingface_checkpoint(args):
    language_model = load_huggingface_checkpoint(args.language_model_path)
    vision_model = load_huggingface_checkpoint(args.vision_model_path)
    patch = load_huggingface_checkpoint(args.vision_patch)
    adapter = load_huggingface_checkpoint(args.adapter_path)

    _add_prefix_to_keys(language_model, args.language_model_prefix)
    _add_prefix_to_keys(vision_model, args.vision_model_prefix)
    _add_prefix_to_keys(patch, args.vision_model_prefix)
    _add_prefix_to_keys(adapter, args.adapter_prefix)

    for module in [vision_model, adapter, patch]:
        merge_dict(module, language_model)

    save_huggingface_checkpoint(language_model, args.save_ckpt_path)

if __name__ == '__main__':
    args = parse_args()
    if args.megatron_path is not None:
        sys.path.insert(0, args.megatron_path)
    print("===== merge huggingface checkpoints ======")
    merge_huggingface_checkpoint(args)