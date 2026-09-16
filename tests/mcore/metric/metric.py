# Copyright 2026 The LoongForge Authors.
# SPDX-License-Identifier: Apache-2.0

"""Metric class."""
from dataclasses import dataclass, field


@dataclass
class Metric:
    """
    Metric class to store and convert metrics related to a model.
    """
    model_name: str = None
    global_batch_size: list = field(default_factory=list)
    throughput: list = field(default_factory=list)
    elapsed_time_match: list = field(default_factory=list)
    lm_loss_list: list = field(default_factory=list)
    grad_norm_list: list = field(default_factory=list)
    mem_allocated_avg_MB: list = field(default_factory=list)
    mem_max_allocated_avg_MB: list = field(default_factory=list)

    def obj_to_dict(self):
        """
        Converts the object attributes to a dictionary and returns it.
        """
        return vars(self)