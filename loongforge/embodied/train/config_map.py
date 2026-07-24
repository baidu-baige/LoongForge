# Copyright 2026 The LoongForge Authors.
# SPDX-License-Identifier: Apache-2.0

"""model-name → (YAML path, ModelConfig class, DataConfig class) routing.

MODEL_SCHEMA is the single place that binds a model name to:
  - its shared YAML file (with ``model:`` / ``data:`` sections), and
  - the two typed config classes that the YAML sections are merged into.
"""

from dataclasses import dataclass
from pathlib import Path
from typing import Type

from loongforge.embodied.model.pi05.model_configuration_pi05 import Pi05ModelConfig
from loongforge.embodied.data.datasets.pi05.transforms.data_configuration_pi05 import Pi05DataConfig
from loongforge.embodied.model.groot_n1_6.model_configuration_groot_n1_6 import GrootN1d6ModelConfig
from loongforge.embodied.data.datasets.groot_n1_6.transforms.data_configuration_groot_n1_6 import GrootN1d6DataConfig
from loongforge.embodied.model.xvla.model_configuration_xvla import XvlaModelConfig
from loongforge.embodied.data.datasets.xvla.transforms.data_configuration_xvla import XvlaDataConfig
from loongforge.embodied.model.fastwam.modeling_configuration_fastwam import FastWAMModelConfig
from loongforge.embodied.data.datasets.fastwam.transforms.data_configuration_fastwam import FastWAMDataConfig
from loongforge.embodied.model.groot_n1_7.model_configuration_groot_n1_7 import GrootN1d7Config
from loongforge.embodied.data.datasets.groot_n1_7.transforms.data_configuration_groot_n1_7 import (
    GrootN1d7DataConfig,
)
from loongforge.embodied.model.cosmos3.modeling_configuration_cosmos3 import Cosmos3ModelConfig
from loongforge.embodied.data.datasets.cosmos3.data_configuration_cosmos3 import Cosmos3DroidConfig
from loongforge.embodied.model.dreamzero.model_configuration_dreamzero import DreamZeroConfig
from loongforge.embodied.data.datasets.dreamzero.transforms.data_configuration_dreamzero import DreamZeroDataConfig
from loongforge.embodied.model.lingbot_va.model_configuration_lingbot_va import LingBotVAModelConfig
from loongforge.embodied.data.datasets.lingbot_va.transforms.data_configuration_lingbot_va import LingBotVADataConfig

_CONFIGS_DIR = (
    Path(__file__).resolve().parent.parent.parent.parent
    / "configs" / "models" / "embodied"
)


@dataclass(frozen=True)
class ModelSchema:
    """Binding of a model name to its YAML file and typed config classes."""

    yaml_file: str
    model_config_cls: Type
    data_config_cls: Type


MODEL_SCHEMA = {
    "lingbot_va_robotwin": ModelSchema(
        "lingbot_va_robotwin.yaml", LingBotVAModelConfig, LingBotVADataConfig
    ),
    "lingbot_va_libero": ModelSchema(
        "lingbot_va_libero.yaml", LingBotVAModelConfig, LingBotVADataConfig
    ),
    "pi05": ModelSchema("pi05.yaml", Pi05ModelConfig, Pi05DataConfig),
    "groot_n1_6": ModelSchema("groot_n1_6.yaml", GrootN1d6ModelConfig, GrootN1d6DataConfig),
    "xvla": ModelSchema("xvla.yaml", XvlaModelConfig, XvlaDataConfig),
    "fastwam": ModelSchema("fastwam.yaml", FastWAMModelConfig, FastWAMDataConfig),
    "groot_n1_7": ModelSchema("groot_n1_7.yaml", GrootN1d7Config, GrootN1d7DataConfig),
    "cosmos3_nano": ModelSchema("cosmos3/nano.yaml", Cosmos3ModelConfig, Cosmos3DroidConfig),
    "dreamzero_lora_wan22_5b": ModelSchema(
        "dreamzero_wan22_5b.yaml", DreamZeroConfig, DreamZeroDataConfig
    ),
    "dreamzero_full_wan22_5b": ModelSchema(
        "dreamzero_wan22_5b.yaml", DreamZeroConfig, DreamZeroDataConfig
    ),
    "dreamzero_lora_wan21_14b": ModelSchema(
        "dreamzero_wan21_14b.yaml", DreamZeroConfig, DreamZeroDataConfig
    ),
    "dreamzero_full_wan21_14b": ModelSchema(
        "dreamzero_wan21_14b.yaml", DreamZeroConfig, DreamZeroDataConfig
    ),
    "dreamzero_libero_wan22_5b": ModelSchema(
        "dreamzero_libero_wan22_5b.yaml", DreamZeroConfig, DreamZeroDataConfig
    ),
    "dreamzero_agibot_wan21_14b": ModelSchema(
        "dreamzero_agibot_wan21_14b.yaml", DreamZeroConfig, DreamZeroDataConfig
    ),
    "dreamzero_yam_wan21_14b": ModelSchema(
        "dreamzero_yam_wan21_14b.yaml", DreamZeroConfig, DreamZeroDataConfig
    ),
}


def _normalize(model_name: str) -> str:
    return model_name.lower().replace("-", "_")


def get_model_schema(model_name: str) -> ModelSchema:
    """Look up the ModelSchema for a model name."""
    key = _normalize(model_name)
    if key not in MODEL_SCHEMA:
        available = ", ".join(sorted(MODEL_SCHEMA.keys()))
        raise ValueError(f"Unknown model-name '{model_name}'. Available: [{available}]")
    return MODEL_SCHEMA[key]


def get_config_path(model_name: str) -> str:
    """Look up config YAML path by model name."""
    schema = get_model_schema(model_name)
    path = _CONFIGS_DIR / schema.yaml_file
    if not path.exists():
        raise FileNotFoundError(f"Config file not found: {path}")
    return str(path)
