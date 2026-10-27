# Copyright 2026 The LoongForge Authors.
# SPDX-License-Identifier: Apache-2.0

"""HuggingFace datasets pipeline: format, tokenize, pack, collate, iterate.

File map: ``sft/`` holds the pipeline (``format_utils.py`` raw -> unified
messages, ``preprocess.py`` encode/truncate/pack/chunkpipe, ``dataset.py``
dataset build, ``collator.py`` batch padding and the collator factory);
``dataloader.py`` holds ``SFTLoaderConfig`` and the iterators;
``blended_hf_dataset_*.py`` hold the dataset blend config and builder.

Shared by LLM SFT (``training/llm/sft.py``) and VLM offline-tokenized paths
(``training/vlm/pretrain_vlm.py``, ``training/vlm/sft_internvl.py`` with
--is-tokenized-data).
"""
