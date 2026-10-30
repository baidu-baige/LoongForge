# Copyright 2026 The LoongForge Authors.
# SPDX-License-Identifier: Apache-2.0

"""GigaBrain07 model entry — xvla-style naming surface (``modeling_giga_brain07``).

Layer-A reshell: this module is a **thin re-export** that mirrors the
``model/xvla/modeling_xvla.py`` / ``model/pi05/modeling_pi05.py`` naming
convention so ``giga_brain07`` exposes its policy under the expected
``modeling_<name>`` path.

The real implementation is **unchanged** and still lives in the vendored
package ``giga_models.models.vla.giga_brain_0.modeling_giga_brain_0``; nothing
here moves, rewrites, or re-registers it. The LoongForge custom trainer
(``GigaBrain07Trainer``) continues to build the model via the vendored package
directly — this file only provides a naming-aligned import surface.

Import prerequisites (identical to the trainer / run script PYTHONPATH):
    model/giga_brain07           -> top-level ``giga_models`` package
"""

from __future__ import annotations

# Re-export the real policy class (diffusers ModelMixin/ConfigMixin based,
# so ``GigaBrain07Policy.from_pretrained(path)`` is provided by the vendored
# implementation — exactly what GigaBrain07Trainer._build_model calls).
from giga_models.models.vla.giga_brain_0.modeling_giga_brain_0 import (
    GigaBrain07Policy,
)


def from_pretrained(pretrained, *args, **kwargs) -> GigaBrain07Policy:
    """Naming-aligned convenience shim mirroring xvla's ``from_pretrained``.

    Pure delegation to ``GigaBrain07Policy.from_pretrained`` (diffusers
    ModelMixin). Adds no behavior and changes no numerics; provided only so the
    ``modeling_giga_brain07`` surface matches the xvla/pi05 entry style.
    """
    return GigaBrain07Policy.from_pretrained(pretrained, *args, **kwargs)


__all__ = ["GigaBrain07Policy", "from_pretrained"]
