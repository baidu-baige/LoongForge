# Copyright 2026 The OmniTraining Authors.
# SPDX-License-Identifier: Apache-2.0

"""megatron core aiak plugin init"""

import os


if os.getenv("XMLIR_MEGATRON_CORE_AIAK_PLUGIN") in ("true", "1", "True"):
    try:
        from xpu_plugin import init_megatron_core_aiak_plugin

        init_megatron_core_aiak_plugin()
    except ImportError:
        print(
            "xpu_plugin module not installed, skip Megatron Core AIAK Plugin initialization"
        )
