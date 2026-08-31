# Copyright 2026 The LoongForge Authors.
# SPDX-License-Identifier: Apache-2.0
#
# Modified from Wall-X under the Apache-2.0 License.

"""Batch collator for Wall-OSS-0.5."""

from __future__ import annotations

from typing import Any, Dict, List

import numpy as np
import torch
from transformers import AutoProcessor

from loongforge.embodied.data.datasets.transforms.collator import (
    BasePreprocessor,
    register_preprocessor,
)
from loongforge.embodied.data.datasets.wall_oss_0_5.transforms.wall_oss_0_5_utils import (
    load_norm_stats,
    preprocesser_call,
    replace_action_token,
)
from loongforge.embodied.model.wall_oss_0_5.core.moe_indices import (
    compute_moe_group_counts,
)


class WallOss05Batch(dict):
    """Dictionary batch with recursive tensor device transfer."""

    def to(self, device: torch.device) -> "WallOss05Batch":
        """To."""
        def move(value):
            """Move."""
            if isinstance(value, torch.Tensor):
                return value.to(device, non_blocking=True)
            if isinstance(value, dict):
                return {k: move(v) for k, v in value.items()}
            if isinstance(value, list):
                return [move(v) for v in value]
            if isinstance(value, tuple):
                return tuple(move(v) for v in value)
            return value

        for key, value in list(self.items()):
            self[key] = move(value)
        return self

    def to_dict(self):
        """To dict."""
        return dict(self)


def load_wall_oss_0_5_norm_stats(data_cfg):
    """Load wall oss 0 5 norm stats."""
    return load_norm_stats(data_cfg.norm_stats_path, data_cfg.key_mappings)


@register_preprocessor("wall_oss_0_5")
class WallOss05Preprocessor(BasePreprocessor):
    """DataLoader collate_fn for Wall-OSS-0.5."""

    _processor_cache: dict[str, Any] = {}
    _norm_stat_alignment_warnings: set[tuple] = set()

    def __init__(
        self,
        model_cfg,
        data_cfg,
        tokenizer_path: str,
        seed: int | None = None,
        dataset_path: str | None = None,
    ):
        """Initialize the instance."""
        self.model_cfg = model_cfg
        self.data_cfg = data_cfg
        self.tokenizer_path = tokenizer_path
        self.dataset_path = dataset_path
        self.stats = load_wall_oss_0_5_norm_stats(data_cfg)
        self.action_min_stat = self.stats["action"].min
        self.action_delta = self.stats["action"].delta
        self.state_min_stat = self.stats["state"].min
        self.state_delta = self.stats["state"].delta
        self.np_rng = np.random.default_rng(int(seed if seed is not None else 42))
        self.load_processor()

        noise_scheduler_config = data_cfg.noise_scheduler or {}
        self.beta_alpha = float(noise_scheduler_config.get("beta_alpha", 1.5))
        self.beta_beta = float(noise_scheduler_config.get("beta_beta", 1.0))
        self.s = float(noise_scheduler_config.get("s", 0.999))
        self.time_shift = float(noise_scheduler_config.get("time_shift", 1.0))

    @classmethod
    def from_config(
        cls,
        model_cfg,
        data_cfg,
        training_args=None,
        dataset_stats=None,
        dataset=None,
    ) -> "WallOss05Preprocessor":
        """From config."""
        del dataset_stats, dataset
        if training_args is None or not training_args.tokenizer_path:
            raise ValueError("--tokenizer-path is required for wall_oss_0_5.")
        seed = training_args.seed if training_args is not None else None
        dataset_path = training_args.dataset_path if training_args is not None else None
        return cls(
            model_cfg,
            data_cfg,
            tokenizer_path=training_args.tokenizer_path,
            seed=seed,
            dataset_path=dataset_path,
        )

    def load_processor(self):
        """Load processor."""
        processor_path = self.tokenizer_path
        if processor_path not in self._processor_cache:
            processor = AutoProcessor.from_pretrained(processor_path, use_fast=True)
            if self.data_cfg.padding_side == "left":
                processor.tokenizer.padding_side = "left"
            processor.tokenizer.add_tokens(["<|propri|>", "<|action|>"])
            self._processor_cache[processor_path] = processor
        self.processor = self._processor_cache[processor_path]

    @staticmethod
    def _normalize(value, min_stat, delta):
        """Normalize."""
        delta = torch.where(delta == 0, torch.ones_like(delta), delta)
        x = (value - min_stat) / delta
        x = x * 2 - 1
        return torch.clamp(x, -1, 1)

    @classmethod
    def _align_norm_stat(cls, stat, value, *, pad_value: float, name: str):
        """Align norm stat."""
        stat = stat.to(device=value.device, dtype=value.dtype)
        target_dim = value.shape[-1]
        stat_dim = stat.shape[-1]
        if stat_dim == target_dim:
            return stat
        if stat_dim > target_dim:
            return stat[..., :target_dim]
        pad_shape = (*stat.shape[:-1], target_dim - stat_dim)
        pad = torch.full(pad_shape, pad_value, device=value.device, dtype=value.dtype)
        return torch.cat([stat, pad], dim=-1)

    def sample_time(self, batch_size, device, dtype):
        """Sample time."""
        sample_np = self.np_rng.beta(self.beta_alpha, self.beta_beta, size=(batch_size,)).astype(np.float32)
        sample = torch.from_numpy(sample_np).to(device=device, dtype=dtype, non_blocking=True)
        time = 1 - sample
        if self.time_shift != 1.0:
            time = (self.time_shift * time) / (1 + (self.time_shift - 1) * time)
        return time * self.s

    def __call__(self, examples: List[Dict[str, Any]]) -> WallOss05Batch:
        """Call."""
        additional_inputs: dict[str, Any] = {}
        dof_total = int(self.model_cfg.action_dim)
        agent_pos_total = int(self.model_cfg.propri_dim)

        agent_pos = torch.stack([ex["agent_pos"] for ex in examples])
        if agent_pos.dim() == 2:
            agent_pos = agent_pos.unsqueeze(1)
        agent_pos_mask = (~torch.isnan(agent_pos)).float()
        agent_pos.nan_to_num_(nan=0.0)
        state_min_stat = self._align_norm_stat(self.state_min_stat, agent_pos, pad_value=0.0, name="state.min")
        state_delta = self._align_norm_stat(self.state_delta, agent_pos, pad_value=1.0, name="state.delta")
        agent_pos = self._normalize(agent_pos, state_min_stat, state_delta)
        if agent_pos.shape[-1] < agent_pos_total:
            pad_w = agent_pos_total - agent_pos.shape[-1]
            agent_pos = torch.nn.functional.pad(agent_pos, (0, pad_w))
            agent_pos_mask = torch.nn.functional.pad(agent_pos_mask, (0, pad_w))
        additional_inputs["proprioception"] = agent_pos
        additional_inputs["agent_pos_mask"] = agent_pos_mask

        action = torch.stack([ex["action"] for ex in examples])
        if action.dim() == 2:
            action = action.unsqueeze(1)
        dof_mask = (~torch.isnan(action)).float()
        action.nan_to_num_(nan=0.0)
        action_min_stat = self._align_norm_stat(self.action_min_stat, action, pad_value=0.0, name="action.min")
        action_delta = self._align_norm_stat(self.action_delta, action, pad_value=1.0, name="action.delta")
        action = self._normalize(action, action_min_stat, action_delta)
        if action.shape[-1] < dof_total:
            pad_w = dof_total - action.shape[-1]
            action = torch.nn.functional.pad(action, (0, pad_w))
            dof_mask = torch.nn.functional.pad(dof_mask, (0, pad_w))
        additional_inputs["action_chunk"] = action
        additional_inputs["dof_mask"] = dof_mask
        additional_inputs["sample_time"] = self.sample_time(action.shape[0], device=action.device, dtype=torch.float32)

        texts = replace_action_token(
            [ex["text"] for ex in examples],
            additional_inputs["action_chunk"],
            None,
            additional_inputs["dof_mask"],
        )
        image_inputs = [ex["image_inputs"] for ex in examples]
        inputs = preprocesser_call(
            processor=self.processor,
            text=texts,
            images=image_inputs,
            videos=None,
            padding=True,
            truncation=True,
            return_tensors="pt",
            max_length=self.data_cfg.max_length,
            norm_state=(
                additional_inputs["proprioception"]
                if self.model_cfg.use_state_string_representation
                else None
            ),
            agent_pos_mask=additional_inputs.get("agent_pos_mask"),
            state_bins=self.data_cfg.state_bins,
        )

        action_token_id = self.processor.tokenizer.convert_tokens_to_ids("<|action|>")
        additional_inputs["moe_token_types"] = inputs.input_ids == action_token_id
        additional_inputs["moe_group_counts"] = compute_moe_group_counts(
            additional_inputs["moe_token_types"]
        )
        for name in ("image_grid_thw", "video_grid_thw"):
            grid_thw = inputs.get(name)
            additional_inputs[f"{name}_cpu"] = (
                tuple(tuple(int(v) for v in row) for row in grid_thw.tolist())
                if grid_thw is not None
                else None
            )
        batch = WallOss05Batch(inputs)
        batch.update(additional_inputs)
        batch["dataset_names"] = [self.dataset_path] * batch["action_chunk"].shape[0]
        return batch
