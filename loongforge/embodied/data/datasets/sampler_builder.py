# Copyright 2026 The LoongForge Authors.
# SPDX-License-Identifier: Apache-2.0

"""Registry for DataLoader samplers, with a default distributed sampler.

A *sampler builder* is a callable ``fn(SamplerBuilderContext) -> Sampler | None``
selected by ``model_cfg.model_type``. Models that need custom index ordering
(e.g. multi-frame grouping, curriculum sampling) register their own builder via
``@register_sampler_builder("<model_type>")``; models without a registered
builder fall back to :func:`default_sampler_builder`.

Returning ``None`` means "no sampler" — the DataLoader then falls back to plain
shuffling (map-style) or natural iteration (iterable-style).
"""

from __future__ import annotations

import importlib
import pkgutil
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Dict, Optional

import torch
from torch.utils.data import IterableDataset, Sampler
from torchdata.stateful_dataloader.sampler import StatefulDistributedSampler


@dataclass(frozen=True)
class SamplerBuilderContext:
    """Shared inputs available to sampler builders."""

    dataset: Any
    training_args: Any
    ctx: Any  # DistributedContext
    batch_size: int
    seed: int
    shuffle: bool


SamplerBuilder = Callable[[SamplerBuilderContext], Optional[Sampler]]

_SAMPLER_BUILDER_REGISTRY: Dict[str, SamplerBuilder] = {}
_DISCOVERED_SAMPLER_BUILDERS = False


class _BlockShardSampler(Sampler[int]):
    """Distribute contiguous batches across data-parallel ranks."""

    def __init__(
        self,
        dataset,
        *,
        batch_size: int,
        num_replicas: int,
        rank: int,
        shuffle: bool,
        seed: int,
        drop_last: bool = False,
    ) -> None:
        self.dataset = dataset
        self.batch_size = batch_size
        self.num_replicas = num_replicas
        self.rank = rank
        self.shuffle = shuffle
        self.seed = seed
        self.drop_last = drop_last
        self.epoch = 0

    def __iter__(self):
        length = len(self.dataset)
        if self.shuffle:
            generator = torch.Generator().manual_seed(self.seed + self.epoch)
            indices = torch.randperm(length, generator=generator).tolist()
        else:
            indices = list(range(length))

        if self.drop_last:
            usable = (len(indices) // self.batch_size) * self.batch_size
            indices = indices[:usable]

        batches = [
            indices[start : start + self.batch_size]
            for start in range(0, len(indices), self.batch_size)
            if len(indices[start : start + self.batch_size]) == self.batch_size or not self.drop_last
        ]
        for batch_idx, batch in enumerate(batches):
            if batch_idx % self.num_replicas == self.rank:
                yield from batch

    def __len__(self) -> int:
        full_batches, remainder = divmod(len(self.dataset), self.batch_size)
        total_batches = full_batches if self.drop_last or remainder == 0 else full_batches + 1
        local_batches = (total_batches + self.num_replicas - 1 - self.rank) // self.num_replicas
        if not local_batches:
            return 0
        if self.drop_last or remainder == 0 or (local_batches - 1) * self.num_replicas + self.rank < full_batches:
            return local_batches * self.batch_size
        return (local_batches - 1) * self.batch_size + remainder

    def set_epoch(self, epoch: int) -> None:
        """Set the epoch index for deterministic shuffling."""
        self.epoch = epoch

    def state_dict(self) -> dict:
        """Return checkpointable sampler state."""
        return {"epoch": self.epoch}

    def load_state_dict(self, state_dict: dict) -> None:
        """Load checkpointed sampler state."""
        self.epoch = int(state_dict.get("epoch", 0))


def register_sampler_builder(model_type: str):
    """Decorator to register a model-specific sampler builder."""
    def decorator(builder: SamplerBuilder) -> SamplerBuilder:
        _SAMPLER_BUILDER_REGISTRY[model_type] = builder
        return builder

    return decorator


def default_sampler_builder(context: SamplerBuilderContext) -> Optional[Sampler]:
    """Default distributed sampler selection (block vs stateful-distributed).

    Returns ``None`` for iterable datasets and for the single-process /
    non-distributed case, letting the DataLoader handle plain shuffling.
    """
    dataset = context.dataset
    if isinstance(dataset, IterableDataset):
        return None

    ctx = context.ctx
    if not (ctx.is_distributed and ctx.world_size > 1):
        return None

    drop_last = context.training_args.batch_drop_last
    if context.training_args.distributed_sampler_mode == "block":
        return _BlockShardSampler(
            dataset,
            batch_size=context.batch_size,
            num_replicas=ctx.world_size,
            rank=ctx.rank,
            shuffle=context.shuffle,
            seed=context.seed,
            drop_last=drop_last,
        )

    return StatefulDistributedSampler(
        dataset,
        num_replicas=ctx.world_size,
        rank=ctx.rank,
        shuffle=context.shuffle,
        seed=context.seed,
        drop_last=drop_last,
    )


def get_sampler_builder(model_type: str) -> SamplerBuilder:
    """Look up a sampler builder by model type, falling back to the default."""
    discover_sampler_builders()
    return _SAMPLER_BUILDER_REGISTRY.get(model_type, default_sampler_builder)


def build_sampler(
    model_type: str,
    *,
    dataset,
    training_args,
    ctx,
    batch_size: int,
    seed: int,
    shuffle: bool = True,
) -> Optional[Sampler]:
    """Build the sampler for ``model_type`` via its registered builder."""
    builder = get_sampler_builder(model_type)
    context = SamplerBuilderContext(
        dataset=dataset,
        training_args=training_args,
        ctx=ctx,
        batch_size=batch_size,
        seed=seed,
        shuffle=shuffle,
    )
    return builder(context)


def discover_sampler_builders() -> None:
    """Import model packages so sampler decorators can register.

    Walk the ``datasets`` package and import each ``<model>`` package (running
    its ``__init__``); if a dedicated ``<model>/samplers.py`` module exists,
    import that too. A model registers its custom sampler with
    ``@register_sampler_builder`` in either ``datasets/<model>/__init__.py`` or
    ``datasets/<model>/samplers.py`` — no ``transforms`` subpackage required.
    """
    global _DISCOVERED_SAMPLER_BUILDERS
    if _DISCOVERED_SAMPLER_BUILDERS:
        return

    _DISCOVERED_SAMPLER_BUILDERS = True
    datasets_dir = Path(__file__).resolve().parent
    datasets_pkg = "loongforge.embodied.data.datasets"
    for module_info in pkgutil.iter_modules([str(datasets_dir)]):
        if not module_info.ispkg or module_info.name.startswith("_"):
            continue
        importlib.import_module(f"{datasets_pkg}.{module_info.name}")
        if (datasets_dir / module_info.name / "samplers.py").is_file():
            importlib.import_module(f"{datasets_pkg}.{module_info.name}.samplers")
