# Copyright 2026 The BaigeOmni Authors.
# SPDX-License-Identifier: Apache-2.0

"""embodied ai models"""

# PI05 uses optional `lerobot`; import lazily.
try:
    from .pi05.configuration_pi05 import PI05Config
except ImportError:
    PI05Config = None
