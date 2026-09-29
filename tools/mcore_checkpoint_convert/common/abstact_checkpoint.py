# Copyright 2026 The LoongForge Authors.
# SPDX-License-Identifier: Apache-2.0

"""Abstract checkpoint interfaces for format conversion."""

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

    def save(self, ckpt_path):
        """
            save checkpoint
        """
        raise NotImplementedError()
          
    