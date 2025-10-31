#!/usr/bin/env python3
# -*- coding: UTF-8 -*-
################################################################################
#
# Copyright (c) 2024 Baidu.com, Inc. All Rights Reserved
#
################################################################################

import torch
from abc import ABC, abstractmethod


class AbstractCheckpoint(ABC):
    """
       AbstractCheckpoint 
    """
    
    def __init__(self, num_layers):
        self.num_layers = num_layers
        self.state_dict = {}

    @staticmethod
    @abstractmethod
    def convert_from_common(*args, **kwargs):
        """
            return checkpoints converted from common checkpoint 
        """
        raise NotImplementedError()
    
    @abstractmethod
    def convert_to_common(self, *args, **kwargs):
        """
            convert checkpoints to common checkpoint 
        """
        raise NotImplementedError()

    def set_dtype(self, dtype):
        """ set dtype """
        self.dtype = dtype

    def load(self, ckpt_path):
        """
            load checkpoint
        """
        self.state_dict = torch.load(ckpt_path)
    
    def save(self, ckpt_path):
        """
            save checkpoint
        """
        torch.save(self.state_dict, ckpt_path)
          
    