# Copyright 2026 The LoongForge Authors.
# SPDX-License-Identifier: Apache-2.0

"""Convert vision patch checkpoints between Megatron Core and HuggingFace formats."""

import os
import sys

from os.path import dirname
from copy import deepcopy

from os.path import dirname
SCRIPT_DIR = dirname(os.path.abspath(__file__))
sys.path.append(dirname(dirname(SCRIPT_DIR)))

from convert_checkpoint.arguments import parse_args
from convert_checkpoint.utils.ckpt_util import (
    load_megatron_checkpoint,
    save_megatron_checkpoint,
    load_huggingface_checkpoint,
    save_huggingface_checkpoint
)

from convert_checkpoint.utils.config_utils import parse_at_configs, load_config, parallel_param_parser


args = parse_args()
with open(args.config_file, 'r') as f:
        module_names = parse_at_configs(f.readlines())
module_type = args.convert_file.split('/')[-3]
cfg = load_config(args.convert_file, hydra_overrides = {module_type+'@module='+module_names[module_type]})
name_map = cfg.vision_patch

model_cfg = load_config(args.config_file)

if (args.load_platform, args.save_platform) == ('mcore', 'huggingface'):
    """ megatron to huggingface """
    pp = parallel_param_parser(args, model_cfg, 'pipeline_model_parallel_size', 'foundation')
    if args.megatron_path is not None:
        sys.path.insert(0, args.megatron_path)
    print(" ====== convert vision patch from Megatron Core to HuggingFace ======")
    target = {}
    state_dict = load_megatron_checkpoint(args.load_ckpt_path)
    if pp == 1:
        source = state_dict[0]['model']
    else:
        if 'model' in state_dict[0][0].keys():  # pp
            source = state_dict[0][0]['model']
        else:
            assert 'model0' in state_dict[0][0].keys()  # vpp
            source = state_dict[0][0]['model0']
    for k1, k2 in name_map.items():
        target[k2] = source[k1]
    save_huggingface_checkpoint(target, args.save_ckpt_path)

elif (args.load_platform, args.save_platform) == ('huggingface', 'mcore'):
    """ huggingface to megatron """
    pp = parallel_param_parser(args, model_cfg, 'pipeline_model_parallel_size', module_type)
    print(" ====== convert vision patch from HuggingFace to Megatron Core ======")
    tp = parallel_param_parser(args, model_cfg, 'tensor_model_parallel_size', module_type)
    source = load_huggingface_checkpoint(args.load_ckpt_path)
    target = {}
    for k1, k2 in name_map.items():
        target[k1] = source[k2]
        print(f" > {k1}")
    state_dict = [{'model': deepcopy(target)} for i in range(tp)]
    save_megatron_checkpoint(state_dict, os.path.join(args.save_ckpt_path, 'release'))

else:
    raise NotImplementedError

