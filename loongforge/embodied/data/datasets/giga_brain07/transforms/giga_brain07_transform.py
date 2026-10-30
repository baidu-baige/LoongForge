# Copyright 2026 The LoongForge Authors.
# SPDX-License-Identifier: Apache-2.0

"""GigaBrain07 data transform — xvla-style naming surface.

Layer-A reshell: thin **re-export** mirroring
``data/datasets/xvla/transforms/xvla_transform.py`` and
``data/datasets/pi05/transforms/pi05_transform.py`` so ``giga_brain07``
exposes its transform under the expected ``<name>_transform`` module.

The real transform is **unchanged** and lives in the vendored package
``giga_brain_0.giga_brain_0_transforms``. The LoongForge custom trainer
(``GigaBrain07Trainer._build_dataloaders``) still builds it via the vendored
package directly — this file only provides a naming-aligned import surface and
reimplements nothing.

Import prerequisite (identical to the trainer / run script PYTHONPATH):
    model/giga_brain07           -> top-level ``giga_brain_0`` package
"""

from __future__ import annotations

from giga_brain_0.giga_brain_0_transforms import GigaBrain07Transform

__all__ = ["GigaBrain07Transform"]
