# Copyright 2026 The LoongForge Authors.
# SPDX-License-Identifier: Apache-2.0

"""HuggingFace configuration conversion and serialization helpers."""

import json

import os
from convert_checkpoint.common.abstact_config import AbstractConfig
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
        