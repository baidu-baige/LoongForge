# Copyright 2026 The LoongForge Authors.
# SPDX-License-Identifier: Apache-2.0

"""One package per arch model, the families they sit in, and the shared base.

This module is the registration entry point the model registry falls back to
when a package has no ``modeling_<package>.py``: importing it imports every
``factory/<model_name>/modeling.py``, which is what runs the
``@register_model`` decorators. Adding a model therefore means adding a
directory and one import line here -- not editing a shell class.

``factory/families/`` holds the per-family layer (engine, batch path, frozen
towers) and registers nothing: a family is not nameable from YAML.

``fastwam`` has no model class on purpose: that registry key belongs to
``FastWAMPolicy``. Its package holds a checkpoint rename table only.
"""

from .base import (ArchModelBase, ArchModelConfig, PROPRIO_PAYLOAD_KEY,
                   RenameRules, attach_proprio, remap_state_dict,
                   resolve_dtype)
from .families import DualStreamWanBase, DualStreamWanConfig
from .fastwam import fastwam_to_arch
from .internw0_delta import (FUTURE_DELTA_QUERY_PREFIX,
                             INTERNW0_DELTA_REJECTED,
                             INTERNW0_DELTA_RENAME_RULES,
                             InternW0DeltaArchModel, InternW0DeltaConfig,
                             internw0_delta_to_arch)
from .vpp2 import (VPP2_RENAME_RULES, VPP2_VIDEO_TOKEN_PREFIXES,
                   Vpp2ArchModel, Vpp2Config, vpp2_to_arch)

__all__ = ["ArchModelBase", "ArchModelConfig", "DualStreamWanBase",
           "DualStreamWanConfig", "FUTURE_DELTA_QUERY_PREFIX",
           "INTERNW0_DELTA_REJECTED", "INTERNW0_DELTA_RENAME_RULES",
           "InternW0DeltaArchModel", "InternW0DeltaConfig",
           "PROPRIO_PAYLOAD_KEY", "RenameRules", "VPP2_RENAME_RULES",
           "VPP2_VIDEO_TOKEN_PREFIXES", "Vpp2ArchModel", "Vpp2Config",
           "attach_proprio", "fastwam_to_arch", "internw0_delta_to_arch",
           "remap_state_dict", "resolve_dtype", "vpp2_to_arch"]
