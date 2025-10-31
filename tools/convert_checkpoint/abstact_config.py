#!/usr/bin/env python3
# -*- coding: UTF-8 -*-
################################################################################
#
# Copyright (c) 2024 Baidu.com, Inc. All Rights Reserved
#
################################################################################

import torch
import json
from abc import ABC, abstractmethod
from convert_checkpoint.utils import get_element_from_dict_by_path


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