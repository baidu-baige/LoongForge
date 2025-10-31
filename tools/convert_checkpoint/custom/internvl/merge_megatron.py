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
import shutil

SCRIPT_DIR = dirname(os.path.abspath(__file__))
sys.path.append(dirname(dirname(dirname(SCRIPT_DIR))))

from convert_checkpoint.custom.internvl.util import (
    load_megatron_checkpoint
)


def parse_args(title=None):
    """Parse all arguments."""
    parser = argparse.ArgumentParser(description='Merger Arguments', allow_abbrev=False)
    group = parser.add_argument_group(title='checkpoint')
    group.add_argument('--language_model_path', type=str, help="Path to language model."),
    group.add_argument('--vision_model_path', type=str, help="Path to vision model."),
    group.add_argument('--vision_patch', type=str, help="Path to vision patch."),
    group.add_argument('--adapter_path', type=str, default=None, help="Path to adapter."),
    group.add_argument("--save_ckpt_path", type=str, help="Path to save checkpoint.")
    group.add_argument("--megatron_path", type=str, help="Base directory of Megatron repository")
    group.add_argument("--tensor_model_parallel_size", type=int, default=1, help="Tensor parallel size.")
    group.add_argument("--pipeline_model_parallel_size", type=int, default=1, help="Pipeline parallel size.")
    group.add_argument("--expert_parallel_size", type=int, default=-1, help="Expert parallel size.")

    return parser.parse_args()

def merge_dict(source, destination):
    """ merge two dictionaries recursively """
    for key, value in source.items():
        if isinstance(value, dict):
            node = destination.setdefault(key, {})
            merge_dict(value, node)
        else:
            destination[key] = value

def merge_megatron_checkpoint(args):
    vision_model_original = load_megatron_checkpoint(args.vision_model_path)
    adapter = load_megatron_checkpoint(args.adapter_path) if args.adapter_path else None
    patch = load_megatron_checkpoint(args.vision_patch)

    if len(vision_model_original) == 1 and len(vision_model_original) != args.tensor_model_parallel_size:
        vision_model = [vision_model_original[0]] * args.tensor_model_parallel_size
    else:
        vision_model = vision_model_original

    modules = [vision_model, patch]
    if adapter:
        modules.append(adapter)

    sub_dirs = sorted([x for x in os.listdir(args.language_model_path) if x.startswith("mp_rank")])
    for sub_dir in sub_dirs:
        splits = sub_dir.split('_')
        t = int(splits[2])
        if args.pipeline_model_parallel_size == 1 or int(splits[3]) == 0:
            checkpoint_name = f"{sub_dir}/model_optim_rng.pt"
            checkpoint_path = os.path.join(args.language_model_path, checkpoint_name)
            print(f"Load Megatron Shard:{checkpoint_path}")
            ckpt = torch.load(checkpoint_path, map_location='cpu', weights_only=False)
            for module in modules:
                if 'model' in ckpt.keys():
                    merge_dict(module[t]['model'], ckpt['model'])
                else:
                    assert 'model0' in ckpt.keys()  # vpp
                    merge_dict(module[t]['model'], ckpt['model0'])
            try:
                torch.save(ckpt, checkpoint_path)
                print(f"Successfully Update Megatron shard: {checkpoint_path}")
            except Exception as e:
                print(f"Failed to update Megatron shard: {checkpoint_path}")

    if os.path.exists(args.save_ckpt_path):
        shutil.rmtree(args.save_ckpt_path)
    shutil.move(args.language_model_path, args.save_ckpt_path)
    print(f"Saving Megatron shard to: {args.save_ckpt_path}")

if __name__ == '__main__':
    args = parse_args()
    if args.megatron_path is not None:
        sys.path.insert(0, args.megatron_path)
    print("===== merge megatron checkpoints ======")
    merge_megatron_checkpoint(args)