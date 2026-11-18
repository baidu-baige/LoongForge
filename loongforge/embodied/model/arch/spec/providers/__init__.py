# Copyright 2026 The LoongForge Authors.
# SPDX-License-Identifier: Apache-2.0

"""Model providers, one package per model.

The directory layout mirrors that split: ``<model_type>/spec.py`` holds that
model's visibility and its single ``@register_provider`` entry, while the
files at this level are named after the **topology families** of
``docs/design/mot_spec_model_matrix.zh-CN.md`` §1 and hold only what the
models of one family share.

The code in this directory is the **only place topology and algorithm live**
(rule R5): how streams are split, which streams enter the same joint attention,
how visibility is composed, how many loss terms there are. All of it is written
in the Python here, and none of it appears in YAML.

Two conventions:

1. **The registry key is ``model_type``**, one-to-one with that key in the
   model config, and one-to-one with the package name here.
2. The six topology families (``dual_stream_wan`` and friends) are
   **unregistered factory functions**, not a public contract; providers call
   them, but neither YAML nor the launch arguments can name them.
"""

from __future__ import annotations

# Importing is registering: ``@register_provider`` fires as an import-time side
# effect, so every new provider package must be listed here. The symptom of a
# missing line is ``get_provider('vpp2')`` reporting an unknown model_type
# while the package sits right next to this file.
from . import fastwam  # noqa: F401
from . import internw0_delta  # noqa: F401
from . import vpp2  # noqa: F401

__all__: list[str] = []
