# Copyright 2026 The LoongForge Authors.
# SPDX-License-Identifier: Apache-2.0

"""Unified argument parsing → three typed configs.

Flow:
  1. Parse CLI: TrainingArgs flags (--lr-base ...) + positional dotlist overrides.
  2. Build TrainingArgs DictConfig: structured(TrainingArgs) merged with CLI values.
  3. Route model_name → (ModelConfig, DataConfig) classes + shared YAML.
  4. Merge YAML model:/data: sections into structured schemas.
  5. Apply Shell dotlist overrides (model.* / data.*) — still DictConfig stage.
  6. Validate on DictConfig (missing-key / interpolation checks).
  7. to_object() → frozen TrainingArgs / ModelConfig / DataConfig instances.
  8. Store in global singletons; return the triple.

Usage:
    from loongforge.embodied.train.parser import parse_train_args
    training_args, model_cfg, data_cfg = parse_train_args()
    # also accessible via global_vars.get_training_args / get_model_config / get_data_config
"""

import os

from omegaconf import OmegaConf

from .training_args import build_arg_parser, TrainingArgs
from .config_map import get_config_path, get_model_schema
from .global_vars import set_data_config, set_model_config, set_training_args
from .validators import validate


def parse_train_args():
    """Parse CLI + YAML into (TrainingArgs, ModelConfig, DataConfig) instances."""
    parser = build_arg_parser()
    raw = parser.parse_args()
    raw_dict = vars(raw)
    overrides = raw_dict.pop("overrides", [])

    # ── 1. TrainingArgs: structured defaults ← CLI overrides ──
    base = OmegaConf.structured(TrainingArgs)
    cli = OmegaConf.create(dict(raw_dict))
    training_dc = OmegaConf.merge(base, cli)

    # Propagate tokenizer path to env var (model builder reads TOKENIZER_PATH)
    if training_dc.tokenizer_path:
        os.environ["TOKENIZER_PATH"] = training_dc.tokenizer_path

    # ── 2. Route model_name / config_file → schema + YAML ──
    if training_dc.config_file:
        config_path = training_dc.config_file
        if not training_dc.model_name:
            raise ValueError(
                "--model-name is required to select ModelConfig/DataConfig classes "
                "even when --config-file is provided."
            )
    elif training_dc.model_name:
        config_path = get_config_path(training_dc.model_name)
    else:
        raise ValueError("Must specify --model-name (and optionally --config-file).")

    schema = get_model_schema(training_dc.model_name)
    yaml_cfg = OmegaConf.load(config_path)

    # ── 3. ModelConfig / DataConfig: structured schema ← YAML section ──
    model_dc = OmegaConf.merge(
        OmegaConf.structured(schema.model_config_cls),
        yaml_cfg.get("model", {}),
    )
    data_dc = OmegaConf.merge(
        OmegaConf.structured(schema.data_config_cls),
        yaml_cfg.get("data", {}),
    )

    # ── 4. Shell dotlist overrides (model.* / data.*) ──
    if overrides:
        ov = OmegaConf.from_dotlist(list(overrides))
        unknown = set(ov.keys()) - {"model", "data"}
        if unknown:
            raise ValueError(
                "YAML dotlist overrides must be prefixed with 'model.' or 'data.'. "
                f"Unknown top-level override keys: {sorted(unknown)}"
            )
        if "model" in ov:
            model_dc = OmegaConf.merge(model_dc, ov.model)
        if "data" in ov:
            data_dc = OmegaConf.merge(data_dc, ov.data)

    # ── 5. Validate (DictConfig stage) ──
    validate(training_dc, model_dc, data_dc)

    # ── 6. Instantiate frozen dataclasses ──
    training_args = OmegaConf.to_object(training_dc)
    model_cfg = OmegaConf.to_object(model_dc)
    data_cfg = OmegaConf.to_object(data_dc)

    # ── 7. Store globally + return ──
    set_training_args(training_args)
    set_model_config(model_cfg)
    set_data_config(data_cfg)
    return training_args, model_cfg, data_cfg
