# Copyright 2026 The LoongForge Authors.
# SPDX-License-Identifier: Apache-2.0

"""Common configuration helpers for checkpoint conversion."""

import torch
import json
from omegaconf import OmegaConf
from omegaconf.dictconfig import DictConfig

from convert_checkpoint.common.abstact_config import AbstractConfig


class CommonConfig(AbstractConfig):
    """
       CommonConfig 
    """

    def __init__(self):
        super().__init__()

    @staticmethod
    def convert_from_common(*args, **kwargs):
        """
            return config converted from common config 
        """
        raise NotImplementedError()

    def convert(self, config_class, *args, **kwargs):
        """
            convert common config to config_class
        """
        return config_class.convert_from_common(self, *args, **kwargs)

    def get_args(self, platform='common'):
        """ return args of platform """
        return self.get("args").get(platform, {})

    def update_args(self, args, platform='common'):
        """ update args for platform """
        res = {k: v for k, v in args.items() if v is not None}
        self.get("args").setdefault(platform, {})
        if isinstance(self.get("args"), DictConfig):
            OmegaConf.set_struct(self.get("args"), False)
        self.get("args").get(platform).update(res)
        if isinstance(self.get("args").get(platform), DictConfig):
            OmegaConf.set_struct(self.get("args").get(platform), False)
        if res.get("torch_dtype") is not None:
            self.update({"torch_dtype": res.get("torch_dtype")})

    def get_name_map(self, platform):
        """ return args of platform """
        return self.get("name_map").get(platform, {})

    def get_dtype(self):
        """
            return data type accoding to config
        """
        target_params_dtype = self.get("torch_dtype")
        if target_params_dtype == "float16":
            return torch.float16
        elif target_params_dtype == "bfloat16":
            return torch.bfloat16
        elif target_params_dtype == "float32":
            return torch.float32
        else:
            raise TypeError("Unsupported dtype {}".format(target_params_dtype))

    def load(self, config_path):
        """
            load config
        """
        with open(config_path, 'r', encoding='utf-8') as f:
            self.data = json.loads(f.read())

    def load_convert_data(self, convert_data):
        self.data = convert_data