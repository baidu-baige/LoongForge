# Copyright 2026 The LoongForge Authors.
# SPDX-License-Identifier: Apache-2.0

"""Model YAML keys that the Megatron engine writes back onto args must not shadow LoongForge CLI options.

``validators._validate_extra_model_args`` copies every YAML key with a matching args attribute onto
args, so the YAML value silently replaces the CLI value. That is only safe when both name the same
model property.
"""

import argparse

from hydra import compose, initialize_config_dir
from hydra.core.global_hydra import GlobalHydra
from omegaconf import OmegaConf

from loongforge.models.catalog import (
    MODEL_CONFIG_REGISTRY,
    VisionLanguageModelFamilies,
    get_config_from_model_name,
    split_model_config,
)
from loongforge.training.megatron_args import add_training_args

# CLI options whose YAML key names the same setting; the YAML value wins.
SAME_MEANING = {
    # Megatron CLI-to-model-config translation for diffusion models.
    "latent_in_channels",
    "latent_out_channels",
    "latent_patch_size",
    "latent_space_scale",
    "latent_time_scale",
    "max_latent_height",
    "max_latent_width",
    "num_latent_frames",
    # Vocabulary size from the HF config.
    "vocab_size_in_config_file",
    # Execution option set in qwen3_235b_a22b.yaml; SFT and VLM validation recompute it afterwards.
    "variable_seq_lengths",
}


def _loongforge_cli_dests():
    parser = argparse.ArgumentParser(allow_abbrev=False)
    parser = add_training_args(parser)
    return {action.dest for action in parser._actions if action.dest != "help"}


def _compose(model_name):
    """Compose the catalog YAML the same way as ``parser.load_and_merge_config``, without overrides."""
    config_dir, config_name = get_config_from_model_name(model_name)
    GlobalHydra.instance().clear()
    with initialize_config_dir(config_dir=config_dir, version_base=None):
        return compose(config_name=config_name)


def _written_back_nodes(cfg):
    """Mirror ``parser.parse_arguments``: VLM submodules with ``_target_``, otherwise the whole config."""
    try:
        model_type, model_config = split_model_config(cfg)
    except ValueError:
        # The parser rejects this config before any write-back.
        return {}
    # Keys only: values may use resolvers that the parser registers at runtime.
    model_config = OmegaConf.to_container(model_config, resolve=False)
    if model_type not in VisionLanguageModelFamilies.names():
        return {"<top>": model_config}
    return {
        name: node
        for name, node in model_config.items()
        if isinstance(node, dict) and "_target_" in node
    }


def _overlaps():
    cli = _loongforge_cli_dests()
    found = {}
    for model_name, entry in MODEL_CONFIG_REGISTRY.items():
        if entry["engine"] != "megatron":
            continue
        for node_name, node in _written_back_nodes(_compose(model_name)).items():
            for key in node.keys():
                if key in cli:
                    found.setdefault(key, set()).add(f"{model_name}:{node_name}")
    return found


def test_yaml_keys_do_not_shadow_loongforge_cli():
    unregistered = {key: sorted(where) for key, where in _overlaps().items() if key not in SAME_MEANING}
    assert not unregistered, (
        "Model YAML keys overwrite LoongForge CLI options with a different meaning; move the "
        f"option into the model config or rename it: {unregistered}"
    )
