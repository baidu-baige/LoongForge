#!/usr/bin/env python3
# -*- coding: UTF-8 -*-
################################################################################
#
# Copyright (c) 2024 Baidu.com, Inc. All Rights Reserved
#
################################################################################

import os
import torch
import types
from tqdm import tqdm
import pprint

from convert_checkpoint.abstact_config import AbstractConfig
from convert_checkpoint.common_config import CommonConfig


class MegatronConfig(AbstractConfig):
    """
        MegatronConfig
    """

    def __init__(self):
        super().__init__()

    @staticmethod
    def convert_from_common(c_config):
        """
            return megatron config converted from common config 
        """
        config = MegatronConfig()
        cargs = c_config.get_args("common")
        margs = c_config.get_args("megatron")
        config.update(cargs)
        config.update(margs)

        num_layers_per_stage = margs["num_layers_per_virtual_pipeline_stage"]
        if num_layers_per_stage is not None:
            num_layers = cargs["num_layers"]
            pp = margs["pipeline_model_parallel_size"]
            stage = num_layers // pp // num_layers_per_stage
            config.update({"virtual_pipeline_model_parallel_size": stage})
        return config

    def load(self, load_path):
        """ load config """
        sub_dirs = os.listdir(load_path)
        possible_sub_dirs = ["mp_rank_00", "mp_rank_00_000"]
        rank0_checkpoint_name = None
        rank0_checkpoint_path = None
        for sub_dir in possible_sub_dirs:
            if sub_dir in sub_dirs:
                rank0_checkpoint_name = os.listdir(os.path.join(load_path, sub_dir))[0]
                rank0_checkpoint_path = os.path.join(load_path, sub_dir, rank0_checkpoint_name)
                break
        print(f"Loading Megatron-LM config from: {rank0_checkpoint_path}")

        rank0_state_dict = torch.load(rank0_checkpoint_path, map_location="cpu")
        if "args" not in rank0_state_dict:
            raise ValueError(
                "Megatron-LM checkpoint does not contain arguments. This utility only supports Megatron-LM checkpoints"
                " containing all the megatron arguments. This is because it loads all config related to model"
                " architecture, the tensor and pipeline model parallel size from the checkpoint insead of user having to"
                " manually specify all the details. Please save Megatron-LM checkpoint along with all the megatron"
                " arguments to use this utility."
            )
        self.data = vars(rank0_state_dict["args"])

    def save(self, save_path):
        """ save config """

        release_dir = os.path.join(save_path, "release")
        os.makedirs(release_dir, exist_ok=True)

        tp = self.get("tensor_model_parallel_size")
        pp = self.get("pipeline_model_parallel_size")
        pbar = tqdm(range(tp*pp), desc='Saving Megatron-LM Config')
        for p in range(pp):
            for t in range(tp):
                sub_dir_name = f"mp_rank_{t:02d}" if pp == 1 \
                        else f"mp_rank_{t:02d}_{p:03d}"
                checkpoint_name = os.listdir(os.path.join(release_dir, sub_dir_name))[0]
                checkpoint_path = os.path.join(release_dir, sub_dir_name, checkpoint_name)
                tp_state_dict = torch.load(checkpoint_path, map_location="cpu")
                tp_state_dict["args"] = self.to_namespace()
                torch.save(tp_state_dict, checkpoint_path)
                pbar.update(1)

    def to_namespace(self):
        margs = types.SimpleNamespace()
        for k, v in self.data.items():
            setattr(margs, k, v)
        return margs
