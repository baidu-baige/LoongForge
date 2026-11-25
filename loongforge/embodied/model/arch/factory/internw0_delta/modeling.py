# Copyright 2026 The LoongForge Authors.
# SPDX-License-Identifier: Apache-2.0

"""InternW0-Delta's arch model: ``--model-name internw0_delta``.

Everything that runs lives in ``DualStreamWanBase``. What this module adds is
the three facts that are specific to InternW0-Delta and cannot be read off the
spec: the registry key, the config class ``from_pretrained`` will accept, and
the rename table ``load_pretrained`` will run.
"""

from __future__ import annotations

from loongforge.embodied.model.registry import register_model

from ..families import DualStreamWanBase
from .config import InternW0DeltaConfig
from .transfer import internw0_delta_to_arch

__all__ = ["InternW0DeltaArchModel"]


@register_model("internw0_delta")
class InternW0DeltaArchModel(DualStreamWanBase):
    """The InternW0-Delta world model."""

    config_cls = InternW0DeltaConfig
    transfer = staticmethod(internw0_delta_to_arch)
