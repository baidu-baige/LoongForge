# Copyright 2026 The LoongForge Authors.
# SPDX-License-Identifier: Apache-2.0

"""DP (Data Parallel) load balancing for distributed training.

This package redistributes samples across DP ranks to equalize workload,
improving training throughput for variable-length inputs (VLM, InternVL, etc.).

Submodules:
    patches         - Runtime monkey-patches (pin_memory_loop, RerunDataIterator)
    pin_memory_hook - Resort hook injected into PyTorch's pin_memory thread
    train_hooks     - Training loop decorators for warmup profiling
    rerun_iterator  - Extended RerunDataIterator with __iter__ support
    vit_balance     - ViT encoder DP load balancing
    rebalance/      - Core rebalancing logic (balance, pack, reconstruct, warmup)
"""
