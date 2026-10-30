# Copyright 2026 The LoongForge Authors.
# SPDX-License-Identifier: Apache-2.0

"""Kimi image-placeholder contract for the K3 training template.

K3's reference processor renders ``<|media_begin|>image {W}x{H}...``, earlier Kimi
releases render the same block without the size. LoongForge pre-expands the
placeholder into one token per merged patch, so the token count also has to match
what the vision tower emits.
"""

import pytest

from loongforge.data.chat_template import MAPPING_NAME_TO_TEMPLATE


def plugin(name: str):
    return MAPPING_NAME_TO_TEMPLATE[name].mm_plugin


def test_k3_placeholder_carries_the_image_size():
    text = plugin("kimi-k3-hf")._build_image_placeholder(3, (448, 336))
    assert text == (
        "<|media_begin|>image 448x336"
        "<|media_content|><|media_content|><|media_content|>"
        "<|media_end|>"
    )


def test_k3_placeholder_needs_the_image_size():
    with pytest.raises(ValueError, match="image dimensions"):
        plugin("kimi-k3-hf")._build_image_placeholder(3)


@pytest.mark.parametrize("name", ["kimi-k2.5-hf", "kimi-k2.6-hf", "kimi-k2.7-code-hf"])
def test_earlier_kimi_placeholders_have_no_image_size(name):
    text = plugin(name)._build_image_placeholder(2, (448, 336))
    assert text == "<|media_begin|>image<|media_content|><|media_content|><|media_end|>"


def test_token_count_is_one_per_merged_patch():
    # 2x2 spatial merge with the time dimension pooled away.
    assert plugin("kimi-k3-hf")._compute_num_tokens_from_grid_thw([4, 8, 6]) == 12
