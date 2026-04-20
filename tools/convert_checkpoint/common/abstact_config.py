# Copyright 2026 The LoongForge Authors.
# SPDX-License-Identifier: Apache-2.0

"""Abstract configuration interfaces for checkpoint conversion."""

from abc import ABC, abstractmethod


class AbstractConfig(ABC):
    """
       AbstractConfig 
    """

    def __init__(self):
        self.data = {}
        
    @staticmethod
    @abstractmethod
    def convert_from_common():
        """
            return config converted from common config 
        """
        raise NotImplementedError()

    def update(self, config):
        """ update data by given config(dict) """
        self.data.update(config)

    def get(self, *args, **kwargs):
        """ return args """
        return self.data.get(*args, **kwargs)

    def load(self, config_path):
        """
            load config
        """
        raise NotImplementedError()
        
    def save(self, config_path):
        """
            save config
        """
        raise NotImplementedError()