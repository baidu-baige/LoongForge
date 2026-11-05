# Copyright 2026 The LoongForge Authors.
# SPDX-License-Identifier: Apache-2.0

"""GigaBrain07 model-specific data config, transform, and dataset builder.

Layer-A reshell: exposes the xvla/pi05-style transform-package surface
(``GigaBrain07DataConfig``, ``GigaBrain07Transform``), mirroring
``data/datasets/xvla/transforms`` and ``data/datasets/pi05/transforms``.

``GigaBrain07DataConfig`` is a lightweight, dependency-free dataclass and is
re-exported eagerly (this is what ``train/config_map.py`` consumes).
``GigaBrain07Transform`` is the real transform implementation that lives in this
package (``giga_brain07_transform``); ``build_giga_brain07_dataset`` is the
route-B dataset builder on the shared LeRobot reader. Both are resolved
**lazily** (PEP 562 ``__getattr__``) so importing the slim data config does NOT
pull the heavy transform / lerobot dependencies.

The collator lives in the sibling ``collators`` package
(``giga_brain07.collators.GigaBrainCollator``), mirroring the xvla layout where
the collator is a dedicated module; it is re-exported here for backward
compatibility.
"""

from __future__ import annotations

# Eager (unchanged): slim dataclass consumed by train/config_map.py.
from loongforge.embodied.data.datasets.giga_brain07.transforms.data_configuration_giga_brain07 import (
    GigaBrain07DataConfig,
)

# Eager: importing this package triggers the preprocessor registry decorator
# (@register_preprocessor("giga_brain07")) so discover_preprocessors() finds it,
# exactly like xvla's transforms package. Pulls only collator + base preprocessor.
from loongforge.embodied.data.datasets.giga_brain07.collators import (  # noqa: E402,F401
    GigaBrain07Batch,
    GigaBrain07Preprocessor,
    GigaBrainCollator,
)

__all__ = [
    "GigaBrain07DataConfig",
    "GigaBrain07Transform",
    "GigaBrainCollator",
    "build_giga_brain07_dataset",
    "GigaBrain07Batch",
    "GigaBrain07Preprocessor",
]


def __getattr__(name):
    # Heavy vendored re-exports; require the giga PYTHONPATH (same as trainer).
    if name == "GigaBrain07Transform":
        from loongforge.embodied.data.datasets.giga_brain07.transforms.giga_brain07_transform import (
            GigaBrain07Transform,
        )

        return GigaBrain07Transform
    if name == "GigaBrainCollator":
        # Collator lives in the sibling ``collators`` package (xvla-style
        # dedicated module); re-exported here for backward compatibility.
        from loongforge.embodied.data.datasets.giga_brain07.collators import (
            GigaBrainCollator,
        )

        return GigaBrainCollator
    if name == "build_giga_brain07_dataset":
        # Route-B dataset builder on the shared LeRobot reader (pulls lerobot).
        from loongforge.embodied.data.datasets.giga_brain07.transforms.giga_brain07_dataset import (
            build_giga_brain07_dataset,
        )

        return build_giga_brain07_dataset
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


# Eager (light): register the per-sample transform builder so
# discover_transform_builders() finds giga_brain07, like xvla. Imports only the
# registry; GigaBrain07Transform is imported lazily inside the builder.
from loongforge.embodied.data.datasets.giga_brain07.transforms.giga_brain07_transform_builder import (  # noqa: E402,F401
    build_giga_brain07_transforms,
)
