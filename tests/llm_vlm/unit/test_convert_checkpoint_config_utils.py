# Copyright 2026 The LoongForge Authors.
# SPDX-License-Identifier: Apache-2.0
# Purpose: Verify VLM checkpoint path scoping.
# Maintainer: dongximiao <dongximiao@baidu.com>

"""Checks for VLM checkpoint conversion configuration."""

import sys
from pathlib import Path

# tests/llm_vlm/unit -> tests/llm_vlm -> tests -> repo root, then repo-root tools/
sys.path.insert(0, str(Path(__file__).resolve().parents[3] / "tools"))

from convert_checkpoint.common.common_config import CommonConfig
from convert_checkpoint.utils.config_utils import convert_vlm_config


def test_vlm_conversion_accepts_root_relative_paths():
    """Root-relative, legacy, and scoped paths must all map to the foundation."""
    config = CommonConfig()
    config.update({
        "name_map": {
            "huggingface": {"word_embeddings": "model.embed_tokens"},
            "mcore": {
                "layer_prefix": "language_model.decoder.layers",
                "word_embeddings": "embedding.word_embeddings",
                "vision_word_embeddings": None,
                "final_layernorm": "foundation_model.decoder.final_layernorm",
                "word_embeddings_for_head": "output_layer",
            },
        },
    })

    name_map = convert_vlm_config(config, for_vlm=True).get("name_map")["mcore"]

    assert name_map["layer_prefix"] == "foundation_model.decoder.layers"
    assert name_map["word_embeddings"] == "foundation_model.embedding.word_embeddings"
    assert name_map["vision_word_embeddings"] == "encoder_model.text_encoder.word_embeddings"
    assert (
        config.get("name_map")["huggingface"]["vision_word_embeddings"]
        == "model.embed_tokens"
    )
    assert name_map["final_layernorm"] == "foundation_model.decoder.final_layernorm"
    assert name_map["word_embeddings_for_head"] == "foundation_model.output_layer"
