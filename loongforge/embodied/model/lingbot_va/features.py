# Copyright 2026 The LoongForge Authors.
# SPDX-License-Identifier: Apache-2.0

"""Runtime-selectable LingBot features and fixed optimization constants."""

import os


# Only user-facing functional or compatibility choices remain switches.
FEATURE_DEFAULTS = {
    "LINGBOT_SAMPLE_META_EXPORT": False,
    "LINGBOT_SKIP_FINAL_CHECKPOINT": False,
    "LINGBOT_BASELINE_LOSS_LOG": True,
    "LINGBOT_BALANCED_SAMPLER": True,
    "LINGBOT_REPO_DISCOVERY_CACHE": True,
    "LINGBOT_LAYERWISE_COMPILE": True,
}

# Fixed values from the accepted performance baseline.
SELF_FLEX_FWD_CONFIG = (64, 64, 4, 3)
SELF_FLEX_BWD_CONFIG = (32, 32, 4, 1)
FLEX_MASK_CACHE_MAX_SIZE = 64
REPO_DISCOVERY_CACHE_WAIT_SECONDS = 1800.0
REPO_DISCOVERY_CACHE_POLL_SECONDS = 2.0


def feature_enabled(name: str) -> bool:
    """Return a supported LingBot user switch."""
    if name not in FEATURE_DEFAULTS:
        raise KeyError(f"Unknown LingBot feature switch: {name}")
    value = os.environ.get(name)
    if value is None:
        return FEATURE_DEFAULTS[name]
    return value.strip().lower() in {"1", "true", "yes", "on"}
