# Copyright 2026 The LoongForge Authors.
# SPDX-License-Identifier: Apache-2.0

"""SFT data pipeline: format, preprocess, dataset, and collation.

``format_utils.py`` (raw -> unified messages), ``preprocess.py`` (encode,
truncate, pack, chunkpipe), ``dataset.py`` (dataset build), ``collator.py``
(batch padding and the collator factory). Loading and iteration live one level
up in ``data/llm/dataloader.py``.
"""
