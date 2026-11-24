# Copyright 2026 The LoongForge Authors.
# SPDX-License-Identifier: Apache-2.0

"""XPolicyLab policy plugin "LoongForgeBatch": the LoongForge bridge with ``eval_batch: true``.

Same bridge as ``robodojo_xpolicylab``, but RoboDojo's sim client runs several envs in one
Isaac Sim process (``--num_envs``) and calls ``eval_one_episode_batch``. RoboDojo reads
``eval_batch`` from the plugin's static deploy.yml, so the batched mode is a second plugin
directory. Keep this package import-free so the sim process never loads LoongForge.
"""
