# Copyright 2026 The LoongForge Authors.
# SPDX-License-Identifier: Apache-2.0

"""giga_brain07 collators (xvla-style dedicated collator package).

``DefaultCollator`` is the generic recursive structural collator used by the
giga pipeline. It is the canonical home for the collator (formerly the vendored
``giga_datasets.collators`` + the ``transforms/giga_brain07_collator`` naming
shim, now merged here). The implementation is unchanged.
"""

from loongforge.embodied.data.datasets.giga_brain07.collators.default_collator import (
    DefaultCollator,
)

__all__ = ["DefaultCollator"]
