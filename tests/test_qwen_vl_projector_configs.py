# Copyright 2026 The LoongForge Authors.
# SPDX-License-Identifier: Apache-2.0

from pathlib import Path

import yaml


REPO_ROOT = Path(__file__).resolve().parents[1]
PROJECTOR_CONFIG_DIR = REPO_ROOT / "configs" / "models" / "image_projector"


def _load_yaml(path: Path) -> dict:
    with path.open(encoding="utf-8") as stream:
        return yaml.safe_load(stream)


def test_qwen2_vl_and_qwen2_5_vl_use_separate_norm_mappings():
    qwen2_config = _load_yaml(PROJECTOR_CONFIG_DIR / "qwen2_vl_mlp_adapter.yaml")
    qwen2_convert = _load_yaml(
        PROJECTOR_CONFIG_DIR / "ckpt_convert" / "qwen2_vl_mlp_adapter_convert.yaml"
    )
    qwen2_5_config = _load_yaml(PROJECTOR_CONFIG_DIR / "qwen_mlp_adapter.yaml")
    qwen2_5_convert = _load_yaml(
        PROJECTOR_CONFIG_DIR / "ckpt_convert" / "qwen_mlp_adapter_convert.yaml"
    )

    assert qwen2_config["normalization"] == "LayerNorm"
    assert qwen2_config["model_type"] == "qwen2_vl_adapter"
    assert qwen2_config["convert_file"].endswith("qwen2_vl_mlp_adapter_convert.yaml")
    assert "adapter.layernorm.bias" in qwen2_convert

    assert qwen2_5_config["normalization"] == "RMSNorm"
    assert qwen2_5_config["model_type"] == "qwen2_5_vl_adapter"
    assert qwen2_5_config["convert_file"].endswith("qwen_mlp_adapter_convert.yaml")
    assert "adapter.layernorm.bias" not in qwen2_5_convert
