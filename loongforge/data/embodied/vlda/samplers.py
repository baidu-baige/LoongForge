# Copyright 2026 The LoongForge Authors.
# SPDX-License-Identifier: Apache-2.0

"""VLDA episode-aware dataset strategy."""

from torchdata.stateful_dataloader.sampler import StatefulDistributedSampler

from loongforge.data.embodied.registry import register_sampler_builder



class VLDASampler(StatefulDistributedSampler):
    """Keep epoch-aware sample randomness independent of worker assignment."""

    def __iter__(self):
        epoch = self.epoch
        for index in super().__iter__():
            yield index, epoch

    def state_dict(self):
        return {**super().state_dict(), "epoch": self.epoch}

    def load_state_dict(self, state_dict):
        super().load_state_dict(state_dict)
        self.epoch = state_dict["epoch"]


@register_sampler_builder("vlda")
def build_vlda_sampler(context):
    if context.training_args.distributed_sampler_mode != "cyclic":
        raise ValueError("VLDA uses cyclic sampling to match the original training order")
    return VLDASampler(
        context.dataset,
        num_replicas=context.runtime.world_size,
        rank=context.runtime.rank,
        shuffle=context.shuffle,
        seed=context.seed,
        drop_last=context.training_args.batch_drop_last,
    )
