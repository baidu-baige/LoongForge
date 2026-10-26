# Copyright 2026 The LoongForge Authors.
# SPDX-License-Identifier: Apache-2.0

"""VLDA evaluation factory with checkpoint-owned normalization and data settings."""

import json
from dataclasses import asdict
from pathlib import Path

import torch

from loongforge.embodied.data.datasets.vlda.transforms.data_configuration_vlda import VLDADataConfig
from loongforge.embodied.data.datasets.vlda.vlda_dataset import load_norm_stats
from loongforge.embodied.eval.factories.registry import register_factory
from loongforge.embodied.eval.servers.loongforge_policy import PredictActionModelSpec
from loongforge.embodied.model.vlda.model_configuration_vlda import VLDAModelConfig


@register_factory("vlda")
class VLDAModelFactory:
    model_config_cls = VLDAModelConfig

    @classmethod
    def build(cls, model_cfg, server_args):
        from loongforge.embodied.model.vlda.modeling_vlda import VLDAPolicy

        if model_cfg.stage != 2:
            raise ValueError("VLDA evaluation requires model.stage=2")
        path = Path(server_args.ckpt_path)
        directory = path if path.is_dir() else path.parent
        data_cfg = VLDADataConfig()
        manifest_path = directory / "vlda_config.json"
        if manifest_path.is_file():
            with manifest_path.open() as stream:
                manifest = json.load(stream)
            actual = asdict(model_cfg)
            for name in (
                "stage",
                "action_dim",
                "state_dim",
                "action_horizon",
                "max_action_dim",
                "max_state_dim",
                "discrete_state_input",
                "delta_token_num",
                "delta_latent_dim",
                "contrast_dim",
                "vjepa",
            ):
                if actual[name] != manifest["model"][name]:
                    raise ValueError(f"VLDA eval model.{name} differs from the checkpoint configuration")
            data_cfg = VLDADataConfig(**manifest["data"])
        elif model_cfg.action_dim == 14:
            # Original RoboTwin exports have no LoongForge manifest.
            data_cfg = VLDADataConfig(adapt_to_pi=True, delta_joint_actions=True)
        stats_path = server_args.dataset_statistics_path or str(directory / "assets/norm_stats.json")
        model = VLDAPolicy(model_cfg)
        if not server_args.random_init:
            model.load_pretrained(path, resume=True)
        model._dataset_stats = load_norm_stats(stats_path)
        model._tokenizer_path = server_args.tokenizer_path
        model._inference_data_cfg = data_cfg
        model.model.drop_training_only_modules()
        device = torch.device(server_args.device)
        model.to(device).eval()
        # Keep authored mixed parameter dtypes; casting the entire policy would
        # change V-JEPA/projection/norm precision relative to training.
        metadata = {
            "framework": "loongforge",
            "model_type": "vlda",
            "ckpt_path": str(path),
            "action_dim": model_cfg.action_dim,
            "action_horizon": model_cfg.action_horizon,
            "dataset_statistics_path": stats_path,
            "tokenizer_path": server_args.tokenizer_path,
        }
        return PredictActionModelSpec(model=model, metadata=metadata)
