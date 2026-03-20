# Copyright 2026 The BaigeOmni Authors.
# SPDX-License-Identifier: Apache-2.0

"""megatron core baige plugin init"""

import os


if os.getenv("XMLIR_MEGATRON_CORE_XPU_PLUGIN") in ("true", "1", "True"):
    try:
        from xpu_plugin import init_megatron_core_xpu_plugin

        init_megatron_core_xpu_plugin()
    except ImportError:
        print(
            "xpu_plugin module not installed, skip Megatron Core Baige Plugin initialization"
        )
