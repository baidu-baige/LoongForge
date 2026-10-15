# Copyright 2026 The LoongForge Authors.
# SPDX-License-Identifier: Apache-2.0

"""Shared contracts of the custom_dp (custom data parallel) training strategy.

These two modules sit between a model and the custom_dp trainer, which is why they are
neither model files nor trainer-internal:

    custom_dp_config.py     the *ModelConfig* mixin carrying the collective
                            precision and overlap knobs
    precision_policy.py     the per-parameter precision contract a model
                            implements and the strategy consumes

A model implementation imports them directly (``model/`` may depend on
``distributed/``); the trainer reads the resolved ``CollectiveConfig`` instead of
the raw model config.

Deliberately no re-exports here: ``custom_dp_config`` stays free of any torch import so
a model config module can inherit it cheaply.
"""
