# Copyright 2026 The LoongForge Authors.
# SPDX-License-Identifier: Apache-2.0

"""Code shared by every embodied model package.

``collator_base.py`` (``BaseCollator`` and batch-preparation types),
``sampler_factory.py`` (sampler builders and ``SamplerBuilderContext``),
``formats/`` (generic dataset readers: LeRobot, dummy, video backends, stats),
and ``transforms/`` (per-sample transforms shared by all models).

Model-specific code lives one level up in ``data/embodied/<model>/``.
"""
