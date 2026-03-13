# Copyright 2026 The OmniTraining Authors.
# SPDX-License-Identifier: Apache-2.0

"""omni custom models"""

from .wan.wan_config import WanConfig

# PI05 uses optional `lerobot`; import lazily.
try:
    from .pi05.configuration_pi05 import PI05Config
except ImportError:
    PI05Config = None
