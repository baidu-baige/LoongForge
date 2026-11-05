# Copyright 2026 The LoongForge Authors.
# SPDX-License-Identifier: Apache-2.0

"""giga_brain07 collators (xvla-style single-collator package).

Canonical home for giga_brain07's collation layer: ``GigaBrainCollator`` (the
generic recursive structural collator), plus the batch container
``GigaBrain07Batch`` and the registered ``GigaBrain07Preprocessor`` collate_fn
(merged here from the former ``transforms/giga_brain07_preprocessor`` module).
Importing this package triggers ``@register_preprocessor("giga_brain07")``.
"""

from loongforge.embodied.data.datasets.giga_brain07.collators.giga_brain_collator import (
    GigaBrain07Batch,
    GigaBrain07Preprocessor,
    GigaBrainCollator,
)

__all__ = ["GigaBrainCollator", "GigaBrain07Batch", "GigaBrain07Preprocessor"]
