# Copyright 2026 The LoongForge Authors.
# SPDX-License-Identifier: Apache-2.0
# Purpose: Verify sharded HuggingFace safetensors loading.

"""Checks for loading sharded HuggingFace checkpoints in conversion tools."""

import sys
from pathlib import Path

import pytest
import torch
from safetensors.torch import save_file

# tests/llm_vlm/unit -> tests/llm_vlm -> tests -> repo root, then repo-root tools/
sys.path.insert(0, str(Path(__file__).resolve().parents[3] / "tools"))

from mcore_checkpoint_convert.utils.ckpt_util import load_huggingface_checkpoint


@pytest.mark.parametrize(
    "pattern",
    [
        "model-{:05d}-of-{:05d}.safetensors",
        "model-{:05d}-of-{:06d}.safetensors",  # e.g. DeepSeek-V3 / Kimi-K2 shards
        "model.safetensors-{:05d}-of-{:05d}.safetensors",
    ],
)
def test_load_sharded_checkpoint_keeps_filename_padding(tmp_path, pattern):
    save_file({"a": torch.ones(1)}, str(tmp_path / pattern.format(1, 2)))
    save_file({"b": torch.zeros(1)}, str(tmp_path / pattern.format(2, 2)))

    assert sorted(load_huggingface_checkpoint(str(tmp_path))) == ["a", "b"]
