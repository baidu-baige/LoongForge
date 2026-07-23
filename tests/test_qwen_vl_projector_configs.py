# Copyright 2026 The LoongForge Authors.
# SPDX-License-Identifier: Apache-2.0

from pathlib import Path

import yaml


REPO_ROOT = Path(__file__).resolve().parents[1]
MODEL_CONFIG_DIR = REPO_ROOT / "configs" / "models"
PROJECTOR_CONFIG_DIR = MODEL_CONFIG_DIR / "image_projector"
PROJECTOR_DEFAULT_KEY = "../../models/image_projector@model.image_projector"

QWEN3_PROJECTOR_CONFIGS = {
    "qwen3_vl": ("qwen3_vl_30b_a3b.yaml", "qwen3_vl_235b_a22b.yaml"),
    "qwen3.5": (
        "qwen3_5_0_8b.yaml",
        "qwen3_5_2b.yaml",
        "qwen3_5_4b.yaml",
        "qwen3_5_9b.yaml",
        "qwen3_5_27b.yaml",
        "qwen3_5_35b_a3b.yaml",
        "qwen3_5_122b_a10b.yaml",
        "qwen3_5_397b_a17b.yaml",
    ),
    "qwen3.6": ("qwen3_6_27b.yaml", "qwen3_6_35b_a3b.yaml"),
}


def _load_yaml(path: Path) -> dict:
    with path.open(encoding="utf-8") as stream:
        return yaml.safe_load(stream)


def _get_projector_preset_name(config: dict):
    return next(
        (
            item[PROJECTOR_DEFAULT_KEY]
            for item in config.get("defaults", [])
            if isinstance(item, dict) and PROJECTOR_DEFAULT_KEY in item
        ),
        None,
    )


def _load_effective_projector(path: Path) -> dict:
    config = _load_yaml(path)
    preset_name = _get_projector_preset_name(config)
    assert preset_name is not None, path
    projector = _load_yaml(PROJECTOR_CONFIG_DIR / f"{preset_name}.yaml")
    projector.update(config["model"]["image_projector"])
    return projector


def _adapter_mapping(config: dict) -> dict:
    return {key: value for key, value in config.items() if key.startswith("adapter.")}


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


def test_normalization_overrides_explicitly_bind_projector_mapping():
    for config_path in sorted(MODEL_CONFIG_DIR.glob("*/*.yaml")):
        config = _load_yaml(config_path)
        preset_name = _get_projector_preset_name(config)
        projector = config.get("model", {}).get("image_projector", {})

        if preset_name is None or "normalization" not in projector:
            continue

        preset = _load_yaml(PROJECTOR_CONFIG_DIR / f"{preset_name}.yaml")
        if projector["normalization"] != preset["normalization"]:
            assert "convert_file" in projector, config_path.relative_to(REPO_ROOT)


def test_qwen3_family_configs_explicitly_use_qwen3_projector_mapping():
    qwen3_convert = _load_yaml(
        PROJECTOR_CONFIG_DIR / "ckpt_convert" / "qwen_3_mlp_adapter_convert.yaml"
    )

    for family, config_names in QWEN3_PROJECTOR_CONFIGS.items():
        for config_name in config_names:
            projector = _load_effective_projector(
                MODEL_CONFIG_DIR / family / config_name
            )

            assert projector["normalization"] == "LayerNorm", config_name
            assert projector["add_bias_linear"] is True, config_name
            assert projector["convert_file"].endswith(
                "qwen_3_mlp_adapter_convert.yaml"
            ), config_name

            if family == "qwen3_vl":
                assert projector["model_type"] == "qwen3_vl_adapter", config_name

    assert _adapter_mapping(qwen3_convert) == {
        "adapter.layernorm.weight": "model.visual.merger.norm.weight",
        "adapter.layernorm.bias": "model.visual.merger.norm.bias",
        "adapter.linear_fc1.weight": "model.visual.merger.linear_fc1.weight",
        "adapter.linear_fc1.bias": "model.visual.merger.linear_fc1.bias",
        "adapter.linear_fc2.weight": "model.visual.merger.linear_fc2.weight",
        "adapter.linear_fc2.bias": "model.visual.merger.linear_fc2.bias",
    }


def test_llava_onevision_config_explicitly_uses_layernorm_projector_mapping():
    projector = _load_effective_projector(
        MODEL_CONFIG_DIR / "llava_onevision" / "llava_onevision_1_5_4b.yaml"
    )
    llava_convert = _load_yaml(
        PROJECTOR_CONFIG_DIR / "ckpt_convert" / "llava_mlp_adapter_convert.yaml"
    )

    assert projector["normalization"] == "LayerNorm"
    assert projector["add_bias_linear"] is True
    assert projector["convert_file"].endswith("llava_mlp_adapter_convert.yaml")
    assert _adapter_mapping(llava_convert) == {
        "adapter.layernorm.weight": "visual.merger.ln_q.weight",
        "adapter.layernorm.bias": "visual.merger.ln_q.bias",
        "adapter.linear_fc1.weight": "visual.merger.mlp.0.weight",
        "adapter.linear_fc1.bias": "visual.merger.mlp.0.bias",
        "adapter.linear_fc2.weight": "visual.merger.mlp.2.weight",
        "adapter.linear_fc2.bias": "visual.merger.mlp.2.bias",
    }
