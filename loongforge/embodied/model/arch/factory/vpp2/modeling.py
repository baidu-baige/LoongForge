# Copyright 2026 The LoongForge Authors.
# SPDX-License-Identifier: Apache-2.0

"""VPP2's arch model: ``--model-name vpp2``.

Everything that runs lives in ``DualStreamWanBase``. What this module adds is
the three facts that are specific to VPP2 and cannot be read off the spec: the
registry key, the config class ``from_pretrained`` will accept, and the rename
table ``load_pretrained`` will run.
"""

from __future__ import annotations

from loongforge.embodied.model.registry import register_model

from ..families import DualStreamWanBase
from .config import Vpp2Config
from .transfer import vpp2_to_arch

__all__ = ["Vpp2ArchModel"]


@register_model("vpp2")
class Vpp2ArchModel(DualStreamWanBase):
    """The VPP2 world model."""

    config_cls = Vpp2Config
    transfer = staticmethod(vpp2_to_arch)
