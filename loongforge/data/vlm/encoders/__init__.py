# Copyright 2026 The LoongForge Authors.
# SPDX-License-Identifier: Apache-2.0

"""Task encoders, one module or package per encoder.

A task encoder that needs private preprocessing becomes a package of the same
name (``internvl/``, ``ernie/``); the rest stay single modules. Modules outside
this package (``samples.py``, ``cookers.py``, ``task_encoder_base.py``,
``packer.py``, ``length_sort_dataset.py``) are shared by every encoder.
"""
