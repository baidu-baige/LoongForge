# Copyright 2026 The LoongForge Authors.
# SPDX-License-Identifier: Apache-2.0

"""GigaBrain07 model-specific data config, transform, and collator.

Layer-A reshell: exposes the xvla/pi05-style transform-package surface
(``GigaBrain07DataConfig``, ``GigaBrain07Transform``, ``DefaultCollator``),
mirroring ``data/datasets/xvla/transforms`` and
``data/datasets/pi05/transforms``.

``GigaBrain07DataConfig`` is a lightweight, dependency-free dataclass and is
re-exported eagerly (unchanged from the original __init__, this is what
``train/config_map.py`` consumes). ``GigaBrain07Transform`` and
``DefaultCollator`` re-export the **real, untouched** vendored implementations
(``giga_brain_0.giga_brain_0_transforms`` / ``giga_datasets``) via the
naming-aligned ``giga_brain07_transform`` / ``giga_brain07_collator`` modules.
They are resolved **lazily** (PEP 562 ``__getattr__``) so importing the slim
data config does NOT pull the heavy vendored packages — keeping the training
import chain and numerics byte-identical. The custom trainer still builds the
transform/collator from the vendored packages directly.
"""

from __future__ import annotations

# Eager (unchanged): slim dataclass consumed by train/config_map.py.
from loongforge.embodied.data.datasets.giga_brain07.transforms.data_configuration_giga_brain07 import (
    GigaBrain07DataConfig,
)

__all__ = [
    "GigaBrain07DataConfig",
    "GigaBrain07Transform",
    "DefaultCollator",
]


def __getattr__(name):
    # Heavy vendored re-exports; require the giga PYTHONPATH (same as trainer).
    if name == "GigaBrain07Transform":
        from loongforge.embodied.data.datasets.giga_brain07.transforms.giga_brain07_transform import (
            GigaBrain07Transform,
        )

        return GigaBrain07Transform
    if name == "DefaultCollator":
        from loongforge.embodied.data.datasets.giga_brain07.transforms.giga_brain07_collator import (
            DefaultCollator,
        )

        return DefaultCollator
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
