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
    # Off by default: BF16 reduction rounds away small action-head gradients.
    "LINGBOT_FSDP_BF16_REDUCE": False,
    # On: reseed with 42 + microbatch index before drawing the model's
    # stochastic inputs, making timesteps, noise, chunk and window sizes a pure
    # function of the step index so a patched community trainer can be compared
    # step by step. Every rank then shares one draw, so a global batch sees a
    # single (timestep, noise, chunk, window) realisation.
    # Off (default): draw from the process-global streams like the unpatched
    # community trainer, giving each rank its own realisation. Pair it with
    # --set-seed-by-rank, or every rank starts from the same global seed and
    # still draws identical values.
    "LINGBOT_VA_FIXED_MICROBATCH_RNG": False,
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
