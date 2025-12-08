#!/usr/bin/env python3
# -*- coding: UTF-8 -*-
################################################################################
#
# Copyright (c) 2024 Baidu.com, Inc. All Rights Reserved
#
################################################################################

from abc import ABC, abstractmethod


class AbstractCheckpoint(ABC):
    """
       AbstractCheckpoint 
    """
    
    def __init__(self, c_config):
        self.c_config = c_config

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

    def load(self, ckpt_path):
        """
            load checkpoint
        """
        raise NotImplementedError()
    
    def save(self, ckpt_path):
        """
            save checkpoint
        """
        raise NotImplementedError()
          
    