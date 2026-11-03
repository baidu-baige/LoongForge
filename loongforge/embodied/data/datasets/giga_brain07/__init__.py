# Copyright 2026 The LoongForge Authors.
# SPDX-License-Identifier: Apache-2.0

"""giga_brain07 dataset package.

Route-B exposes ``build_giga_brain07_dataset`` at the package root (mirroring
``lingbot_vla_v2``), resolved lazily so importing this package does not pull the
shared lerobot reader / lerobot deps unless the builder is actually used.
"""

__all__ = ["build_giga_brain07_dataset"]


def __getattr__(name):
    if name == "build_giga_brain07_dataset":
        from loongforge.embodied.data.datasets.giga_brain07.transforms.giga_brain07_dataset import (
            build_giga_brain07_dataset,
        )

        return build_giga_brain07_dataset
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
