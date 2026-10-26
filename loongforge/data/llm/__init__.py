# Copyright 2026 The LoongForge Authors.
# SPDX-License-Identifier: Apache-2.0

"""HuggingFace datasets pipeline: format, tokenize, pack, collate, iterate.

Shared by LLM SFT (``training/llm/sft.py``) and VLM offline-tokenized paths
(``training/vlm/pretrain_vlm.py``, ``training/vlm/sft_internvl.py`` with
--is-tokenized-data).
"""
