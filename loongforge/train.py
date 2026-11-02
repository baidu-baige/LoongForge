# Copyright 2026 The LoongForge Authors.
# SPDX-License-Identifier: Apache-2.0

"""Training entry for Megatron and Torch model families."""

from __future__ import annotations

import argparse
import importlib
import logging
from typing import Dict, Tuple

from loongforge.models.catalog import (
    DiffusionModelFamilies,
    LanguageModelFamilies,
    VisionLanguageModelFamilies,
    get_model_entry,
)

logging.basicConfig(level=logging.WARNING)

# ---------------------------------------------------------------------------
# Megatron: family -> group
# ---------------------------------------------------------------------------

FAMILY_GROUP: Dict[str, str] = {f: "llm" for f in LanguageModelFamilies.names()}
FAMILY_GROUP.update({f: "vlm" for f in VisionLanguageModelFamilies.names()})
FAMILY_GROUP[VisionLanguageModelFamilies.INTERN_VL] = "intern_vl"
FAMILY_GROUP[VisionLanguageModelFamilies.ERNIE4_5_VL] = "ernie4_5_vl"
FAMILY_GROUP[DiffusionModelFamilies.WAN2_1_I2V] = "wan"
FAMILY_GROUP[DiffusionModelFamilies.WAN2_2_I2V] = "wan"
FAMILY_GROUP[DiffusionModelFamilies.QWEN_IMAGE] = "qwen_image"

# (group, phase) -> method module path
METHOD_REGISTRY: Dict[Tuple[str, str], str] = {
    ("llm", "pretrain"): "loongforge.training.llm.pretrain_llm",
    ("llm", "sft"): "loongforge.training.llm.sft_llm",
    ("vlm", "pretrain"): "loongforge.training.vlm.pretrain_vlm",
    ("vlm", "sft"): "loongforge.training.vlm.sft_vlm",
    ("intern_vl", "pretrain"): "loongforge.training.vlm.pretrain_vlm",
    ("intern_vl", "sft"): "loongforge.training.vlm.sft_internvl",
    ("ernie4_5_vl", "pretrain"): "loongforge.training.vlm.pretrain_vlm",
    ("ernie4_5_vl", "sft"): "loongforge.training.vlm.sft_ernie",
    ("wan", "pretrain"): "loongforge.training.diffusion.pretrain_wan",
    ("qwen_image", "pretrain"): "loongforge.training.diffusion.pretrain_qwen_image",
}

# ---------------------------------------------------------------------------
# Torch: trainer type -> recipe class
# ---------------------------------------------------------------------------

TORCH_TRAINER_TYPES: Dict[str, str] = {
    "FinetuneTrainer": "loongforge.training.embodied.sft.FinetuneTrainer",
    "TensorWiseTrainer": "loongforge.training.embodied.custom.tensor_wise_trainer.TensorWiseTrainer",
    "LingbotVlaV2Trainer": (
        "loongforge.training.embodied.custom.lingbot_vla_v2.lingbot_vla_v2_trainer.LingbotVlaV2Trainer"
    ),
    "VLDAFinetuneTrainer": (
        "loongforge.training.embodied.custom.vlda.vlda_finetune_trainer.VLDAFinetuneTrainer"
    ),
    "GrootN1d6Trainer": (
        "loongforge.training.embodied.custom.gr00t_n1_6.gr00t_trainer.GrootN1d6Trainer"
    ),
    "GrootN1d7Trainer": (
        "loongforge.training.embodied.custom.gr00t_n1_7.gr00t_trainer.GrootN1d7Trainer"
    ),
    "LingBotFinetuneTrainer": (
        "loongforge.training.embodied.custom.lingbot_va.lingbot_finetune_trainer.LingBotFinetuneTrainer"
    ),
}


def resolve_method(model_family: str, training_phase: str):
    """Lazily import and return the Megatron method module for *model_family* + *phase*."""
    group = FAMILY_GROUP.get(model_family, model_family)
    key = (group, training_phase)
    module_path = METHOD_REGISTRY.get(key)
    if module_path is None:
        raise ValueError(
            f"No method registered for ({model_family!r}, {training_phase!r}) "
            f"[resolved group={group!r}]. "
            f"Available: {sorted(METHOD_REGISTRY.keys())}"
        )
    return importlib.import_module(module_path)


def resolve_torch_trainer(trainer_type: str):
    """Lazily import and return the Torch recipe class."""
    if not trainer_type:
        raise ValueError(
            "--trainer-type is required. "
            f"Available: {sorted(TORCH_TRAINER_TYPES.keys())}"
        )

    entry = TORCH_TRAINER_TYPES.get(trainer_type)
    if entry is None:
        raise ValueError(
            f"Unknown --trainer-type '{trainer_type}'. "
            f"Available: {sorted(TORCH_TRAINER_TYPES.keys())}"
        )

    module_path, class_name = entry.rsplit(".", 1)
    return getattr(importlib.import_module(module_path), class_name)


def _run_megatron():
    """Parse the Megatron arguments and run the recipe for their family and phase."""
    from loongforge.engines.megatron.parser import parse_train_args

    args = parse_train_args()
    return resolve_method(args.model_family, args.training_phase).train(args)


def _run_torch():
    """Parse the Torch arguments and run the trainer selected by ``--trainer-type``."""
    from loongforge.training.torch_args import parse_train_args

    training_args, model_cfg, data_cfg = parse_train_args()
    trainer_type = resolve_torch_trainer(training_args.trainer_type)
    return trainer_type(training_args, model_cfg, data_cfg).train()


def main():
    """Pick the engine for ``--model-name`` and run the selected recipe."""
    parser = argparse.ArgumentParser(add_help=False, allow_abbrev=False)
    parser.add_argument("--model-name")
    parser.add_argument("--config-file")
    args, _ = parser.parse_known_args()

    if args.model_name is None:
        if args.config_file is None:
            parser.error("one of --model-name or --config-file is required")
        engine = "megatron"
    else:
        try:
            engine = get_model_entry(args.model_name)["engine"]
        except ValueError as exc:
            parser.error(str(exc))

    if engine == "megatron":
        return _run_megatron()
    return _run_torch()


if __name__ == "__main__":
    main()
