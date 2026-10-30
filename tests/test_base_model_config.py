# Copyright 2026 The LoongForge Authors.
# SPDX-License-Identifier: Apache-2.0

import importlib.util
import sys
import types
from pathlib import Path


def load_base_model_config(monkeypatch):
    transformers = types.ModuleType("transformers")

    class PretrainedConfig:
        pass

    transformers.PretrainedConfig = PretrainedConfig

    megatron = types.ModuleType("megatron")
    megatron_core = types.ModuleType("megatron.core")
    megatron_transformer = types.ModuleType("megatron.core.transformer")
    megatron_transformer_config = types.ModuleType(
        "megatron.core.transformer.transformer_config"
    )

    class TransformerConfig:
        pass

    class MLATransformerConfig:
        pass

    megatron_transformer.TransformerConfig = TransformerConfig
    megatron_transformer_config.MLATransformerConfig = MLATransformerConfig
    megatron_core.transformer = megatron_transformer
    megatron.core = megatron_core

    monkeypatch.setitem(sys.modules, "transformers", transformers)
    monkeypatch.setitem(sys.modules, "megatron", megatron)
    monkeypatch.setitem(sys.modules, "megatron.core", megatron_core)
    monkeypatch.setitem(sys.modules, "megatron.core.transformer", megatron_transformer)
    monkeypatch.setitem(
        sys.modules,
        "megatron.core.transformer.transformer_config",
        megatron_transformer_config,
    )

    module_path = (
        Path(__file__).parent.parent
        / "loongforge"
        / "models"
        / "common"
        / "base_model_config.py"
    )
    spec = importlib.util.spec_from_file_location("base_model_config", module_path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_pretrained_config_uses_post_init_on_modern_transformers(monkeypatch):
    module = load_base_model_config(monkeypatch)

    class ModernPretrainedConfig:
        def __post_init__(self, **kwargs):
            self.initializer = "__post_init__"
            self.kwargs = kwargs

    monkeypatch.setattr(module, "PretrainedConfig", ModernPretrainedConfig)
    config = types.SimpleNamespace()

    module._initialize_pretrained_config(config, dtype="bfloat16")

    assert config.initializer == "__post_init__"
    assert config.kwargs == {"dtype": "bfloat16"}


def test_pretrained_config_falls_back_to_init_on_transformers_5_3(monkeypatch):
    module = load_base_model_config(monkeypatch)

    class LegacyPretrainedConfig:
        def __init__(self, **kwargs):
            self.initializer = "__init__"
            self.kwargs = kwargs

    monkeypatch.setattr(module, "PretrainedConfig", LegacyPretrainedConfig)
    config = types.SimpleNamespace()

    module._initialize_pretrained_config(config, dtype="bfloat16")

    assert config.initializer == "__init__"
    assert config.kwargs == {"dtype": "bfloat16"}
