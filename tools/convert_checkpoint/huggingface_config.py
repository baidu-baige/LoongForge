#!/usr/bin/env python3
# -*- coding: UTF-8 -*-
################################################################################
#
# Copyright (c) 2024 Baidu.com, Inc. All Rights Reserved
#
################################################################################

import json

import os
from convert_checkpoint.abstact_checkpoint import AbstractCheckpoint
from convert_checkpoint.abstact_config import AbstractConfig
from convert_checkpoint.common_config import CommonConfig
from pprint import pprint


class HuggingFaceConfig(AbstractConfig):
    """
        HuggingFaceConfig
    """

    def __init__(self):
        super().__init__()

    @staticmethod
    def convert_from_common(c_config):
        """
        return HuggingFace config converted from Common config.

            Args:
                cc_config: CommonConfig
        """
        config = HuggingFaceConfig()
        config.update(c_config.get_args("common"))
        config.update(c_config.get_args("huggingface"))
        return config

    def save(self, save_path):
        """
            save config
        """
        os.makedirs(save_path, exist_ok=True)
        config_path = os.path.join(save_path, "config.json")
        with open(config_path, 'w', encoding='utf-8') as f:
            json.dump(self.data, f, ensure_ascii=False, indent=4)
        print(f"Saving HuggingFace config to {config_path}")
        pprint(self.data)
        