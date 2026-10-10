# Copyright 2026 The LoongForge Authors.
# SPDX-License-Identifier: Apache-2.0

"""Runtime-selectable LingBot features and fixed optimization constants."""

import os


# Only user-facing functional or compatibility choices remain switches.
FEATURE_DEFAULTS = {
    # Rank balancing implies cost-aligned microbatches; they are one choice.
    # Off falls back to the public DistributedSampler partitioning.
    "LINGBOT_BALANCED_SAMPLER": True,
    # Off keeps FSDP parameters unsharded between forward and backward, which is
    # the accepted configuration; on restores the framework default.
    "LINGBOT_FSDP_RESHARD": False,
    # Gradient reduce dtype: on reduces in the parameter dtype (BF16), off uses
    # FP32. Kept selectable because it is a numerics choice, not only a speed one.
    "LINGBOT_FSDP_BF16_REDUCE": True,
}

# Fixed values from the accepted performance baseline.
SELF_FLEX_OPTIMIZED_FWD_CONFIG = (64, 64, 4, 1)
SELF_FLEX_BWD_CONFIG = (32, 32, 4, 1)
# Manual GC keeps young-generation collection and suppresses generation 2.
GC_GENERATION2_THRESHOLD = 1_000_000_000
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
