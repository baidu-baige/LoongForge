# Copyright 2026 The LoongForge Authors.
# SPDX-License-Identifier: Apache-2.0

"""XPolicyLab policy plugin "LoongForge" for the RoboDojo official evaluator.

The robodojo runner links this directory as ``<RoboDojo>/XPolicyLab/policy/LoongForge``.
RoboDojo's sim client imports ``deploy`` from it; the XPolicyLab policy server imports
``model``. Keep this package import-free so the sim process never loads LoongForge.
"""
