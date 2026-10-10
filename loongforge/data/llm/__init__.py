# Copyright 2026 The LoongForge Authors.
# SPDX-License-Identifier: Apache-2.0

"""HuggingFace datasets pipeline: format, tokenize, pack, collate, iterate.

Shared by LLM SFT (sft_llm) and VLM offline-tokenized paths (pretrain_vlm,
sft_internvl with --is-tokenized-data).  Future DPO/RL prompt datasets also
go here (pairwise_dataset.py, prompt_dataset.py).
"""
