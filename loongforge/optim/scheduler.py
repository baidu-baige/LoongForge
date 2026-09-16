# Copyright 2026 The LoongForge Authors.
# SPDX-License-Identifier: Apache-2.0
#
# Modified from Cosmos (NVIDIA cosmos-framework) under the OpenMDW-1.1 License.
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: OpenMDW-1.1

"""Learning-rate schedules and optimizer scheduler construction."""
import math
from torch.optim.lr_scheduler import LambdaLR
from transformers import get_scheduler

import numpy as np
import logging

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)


class LambdaLinearScheduler:
    """
    Linear instead of cosine decay for the main part of the cycle.
    """

    def __init__(self, warm_up_steps, f_min, f_max, f_start, cycle_lengths, verbosity_interval=0):
        """__init__."""
        assert len(warm_up_steps) == len(f_min) == len(f_max) == len(f_start) == len(cycle_lengths)
        self.lr_warm_up_steps = warm_up_steps
        self.f_start = f_start
        self.f_min = f_min
        self.f_max = f_max
        self.cycle_lengths = cycle_lengths
        self.cum_cycles = np.cumsum([0] + list(self.cycle_lengths))
        self.last_f = 0.0
        self.verbosity_interval = verbosity_interval

    def find_in_interval(self, n):
        """find_in_interval."""
        interval = 0
        for cl in self.cum_cycles[1:]:
            if n <= cl:
                return interval
            interval += 1

    def __call__(self, n, **kwargs):
        """__call__."""
        return self.schedule(n, **kwargs)

    def schedule(self, n, **kwargs):
        """schedule."""
        cycle = self.find_in_interval(n)
        n = n - self.cum_cycles[cycle]
        if self.verbosity_interval > 0:
            if n % self.verbosity_interval == 0:
                logger.info(f"current step: {n}, recent lr-multiplier: {self.last_f}, current cycle {cycle}")

        if n < self.lr_warm_up_steps[cycle]:
            f = (self.f_max[cycle] - self.f_start[cycle]) / self.lr_warm_up_steps[cycle] * n + self.f_start[cycle]
            self.last_f = f
            return f
        else:
            f = self.f_min[cycle] + (self.f_max[cycle] - self.f_min[cycle]) * (self.cycle_lengths[cycle] - n) / (
                self.cycle_lengths[cycle] - self.lr_warm_up_steps[cycle]
            )
            self.last_f = f
            return f


def build_scheduler(optimizer, training_args):
    """Build LR scheduler from CLI training_args."""
    style = training_args.lr_decay_style

    if style == "lambda_linear":

        _scheduler = LambdaLinearScheduler(
            warm_up_steps=[training_args.lr_warmup_iters],
            f_min=[training_args.lambda_f_min],
            f_max=[training_args.lambda_f_max],
            f_start=[training_args.lambda_f_start],
            cycle_lengths=[training_args.lambda_cycle_length]
        )

        logger.info(
            f"LambdaLinear scheduler: f_max={training_args.lambda_f_max}, "
            f"f_min={training_args.lambda_f_min}, warmup={training_args.lr_warmup_iters}, "
            f"cycle_len={training_args.lambda_cycle_length}"
        )

        return LambdaLR(optimizer, _scheduler.schedule)

    if style in {"cosine_with_min_lr", "cosine_warmup_with_min_lr"} and training_args.custom_lr_lambda:
        peak_lr = float(optimizer.defaults["lr"])
        end_lr = float(training_args.min_lr if training_args.min_lr is not None else peak_lr * 0.1)
        num_warmup_steps = int(training_args.lr_warmup_iters)
        num_training_steps = int(training_args.lr_decay_iters or training_args.train_iters)

        def lr_lambda(current_step: int):
            if current_step < num_warmup_steps:
                init_lr = peak_lr / (num_warmup_steps + 1)
                current_lr = init_lr + (peak_lr - init_lr) * current_step / num_warmup_steps
                return current_lr / peak_lr

            decay_steps = num_training_steps - num_warmup_steps
            progress = min(1.0, (current_step - num_warmup_steps) / max(1, decay_steps))
            cos = 0.5 * (1 + math.cos(math.pi * progress))
            current_lr = end_lr + (peak_lr - end_lr) * cos
            return current_lr / peak_lr

        return LambdaLR(optimizer, lr_lambda)
    else:
        kwargs = {}
        if style in {"cosine_with_min_lr", "cosine_warmup_with_min_lr"}:
            kwargs["min_lr"] = training_args.min_lr
        elif style == "polynomial":
            kwargs["lr_end"] = training_args.lr_end
            kwargs["power"] = training_args.polynomial_power
        elif style == "cosine_with_restarts":
            kwargs["num_cycles"] = training_args.num_cycles

        num_training_steps = int(
            training_args.lr_decay_iters or training_args.train_iters
        )
        return get_scheduler(
            name=style,
            optimizer=optimizer,
            num_warmup_steps=training_args.lr_warmup_iters,
            num_training_steps=num_training_steps,
            scheduler_specific_kwargs=kwargs,
        )
