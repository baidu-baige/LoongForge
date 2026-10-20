# Copyright 2026 The LoongForge Authors.
# SPDX-License-Identifier: Apache-2.0

"""VLDA evaluation factory with checkpoint-owned normalization and data settings."""

import json
import types

import numpy as np
from dataclasses import asdict
from pathlib import Path

import torch

from loongforge.data.embodied.vlda.data_configuration_vlda import VLDADataConfig
from loongforge.data.embodied.vlda.vlda_dataset import load_norm_stats
from loongforge.evaluation.embodied.factories.registry import register_factory
from loongforge.evaluation.embodied.servers.loongforge_policy import PredictActionModelSpec
from loongforge.models.embodied.vlda.configuration_vlda import VLDAModelConfig


@torch.no_grad()
def _predict_action(self, images, instructions, state=None, dataset_stats=None):
    """Return unnormalized environment actions with shape [B,H,action_dim]."""
    from loongforge.data.embodied.vlda.data_configuration_vlda import VLDADataConfig
    from loongforge.data.embodied.vlda.vlda_collator import VLDACollator
    from loongforge.data.embodied.vlda.vlda_transform import (
        VLDATransform,
        decode_actions,
        encode_state,
    )

    if self._inference_transform is None:
        stats = dataset_stats if dataset_stats is not None else self._dataset_stats
        if stats is None:
            raise ValueError("VLDA inference requires the training norm_stats.json")
        stats = stats.get("norm_stats", stats)
        stats = {key: {name: np.asarray(value) for name, value in values.items()} for key, values in stats.items()}
        data_cfg = self._inference_data_cfg or VLDADataConfig()
        self._inference_transform = VLDATransform(self.config, data_cfg, stats, self._tokenizer_path)
    transform = self._inference_transform
    batch_size = len(instructions)
    if state is None:
        if self.config.discrete_state_input or transform.data_cfg.delta_joint_actions:
            raise ValueError("This VLDA configuration requires the current robot state")
        state = np.zeros((batch_size, self.config.state_dim), dtype=np.float32)
    state = np.asarray(state).reshape(batch_size, self.config.state_dim)
    examples = []
    for views, instruction, proprio in zip(images, instructions, state, strict=True):
        if not 1 <= len(views) <= 3:
            raise ValueError("VLDA expects one to three views in base/left/right order")
        examples.append(
            transform(
                {
                    "images": list(views) + [None] * (3 - len(views)),
                    "prompt": instruction,
                    "state": proprio,
                }
            )
        )
    batch = VLDACollator()(examples).to(next(self.parameters()).device)
    actions = transform.action_normalizer.inverse(self.predict_action_chunk(batch).cpu().float()).numpy()
    proprio = encode_state(state, transform.data_cfg.adapt_to_pi)
    return decode_actions(actions, proprio[:, None], transform.data_cfg).astype(np.float32)


@register_factory("vlda")
class VLDAModelFactory:
    model_config_cls = VLDAModelConfig

    @classmethod
    def build(cls, model_cfg, server_args):
        from loongforge.models.embodied.vlda.modeling_vlda import VLDAPolicy

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
        model.predict_action = types.MethodType(_predict_action, model)
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
