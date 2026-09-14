# Copyright 2026 The LoongForge Authors.
# SPDX-License-Identifier: Apache-2.0

"""GigaBrainPolicy — LoongForge trainer entry point for GigaBrain-0.

Wraps the vendored ``GigaBrain0Policy`` (identical forward/loss numerics to
the reference giga-models / giga-brain-0 implementation) behind the interface
``FinetuneTrainer`` expects: ``forward(batch) -> (loss, log_loss_dict)``.

``from_pretrained`` is hand-written (not inherited from
``diffusers.ModelMixin``, since this wrapper is a plain ``nn.Module`` in
LoongForge's own registry): it reads the checkpoint's ``config.json`` and the
sharded ``diffusion_pytorch_model-*.safetensors`` files listed in
``diffusion_pytorch_model.safetensors.index.json`` (the format
``GigaBrain-0.1-3.5B-Base`` ships in), mirroring what
``ModelMixin.from_pretrained`` would have done.
"""

from __future__ import annotations

import json
import logging
import os
from pathlib import Path
from typing import Dict, Tuple

import torch
import torch.nn as nn
from safetensors.torch import load_file

from loongforge.embodied.model.registry import register_model

from .giga_brain_0_loss import GigaBrain0Loss
from .model_configuration_giga_brain import GigaBrainModelConfig
from .modeling_giga_brain_0 import GigaBrain0Policy

logger = logging.getLogger(__name__)

_INDEX_FILE = "diffusion_pytorch_model.safetensors.index.json"
_SINGLE_FILE = "diffusion_pytorch_model.safetensors"
_CONFIG_FILE = "config.json"

# Fields consumed by GigaBrain0Policy.__init__ (see model_configuration_giga_brain.py).
# robot_type/model_type are LoongForge-side routing fields, not model kwargs.
_POLICY_CONFIG_FIELDS = (
    "max_state_dim", "max_action_dim", "proj_width", "vlm_type",
    "vlm_hidden_size", "n_action_steps", "num_steps", "use_cache",
    "vision_in_channels", "enable_knowledge_insulation",
    "enable_next_token_prediction", "enable_learnable_traj_token",
    "num_traj_tokens", "max_traj_dim", "traj_hidden_dim", "num_embodiments",
)


def _load_sharded_state_dict(path: Path) -> Dict[str, torch.Tensor]:
    """Load a (possibly sharded) safetensors checkpoint from ``path``.

    Mirrors ``diffusers.ModelMixin.from_pretrained``'s shard-loading behavior
    for the two layouts it may ship in: a single ``model.safetensors`` file,
    or ``*.index.json`` + multiple ``*-NNNNN-of-NNNNN.safetensors`` shards.
    """
    index_path = path / _INDEX_FILE
    if index_path.exists():
        with open(index_path) as f:
            index = json.load(f)
        shard_files = sorted(set(index["weight_map"].values()))
        state_dict: Dict[str, torch.Tensor] = {}
        for shard_file in shard_files:
            state_dict.update(load_file(str(path / shard_file)))
        return state_dict

    single_file = path / _SINGLE_FILE
    if single_file.exists():
        return load_file(str(single_file))

    raise FileNotFoundError(
        f"Neither '{_INDEX_FILE}' nor '{_SINGLE_FILE}' found under {path}."
    )


@register_model("giga_brain")
class GigaBrainPolicy(nn.Module):
    """LoongForge policy wrapper around GigaBrain0Policy.

    Bridges the reference ``giga_train.Trainer.forward_step`` two-step call
    (``loss_func.add_noise`` then ``model(...)`` then ``loss_func(...)``) into
    a single ``forward(batch)`` matching ``FinetuneTrainer``'s expected
    ``model(batch) -> (loss, log_loss_dict)`` signature.
    """

    def __init__(self, config: GigaBrainModelConfig):
        super().__init__()
        self.config = config
        policy_kwargs = {k: getattr(config, k) for k in _POLICY_CONFIG_FIELDS}
        self.model = GigaBrain0Policy(**policy_kwargs)
        self.loss_func = GigaBrain0Loss()

    def forward(self, batch) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        """Training forward pass; returns (total_loss, log_loss_dict).

        ``batch`` carries the keys produced by the GigaBrain collator
        (``data/datasets/giga_brain/transforms/giga_brain_collator.py``),
        matching the reference ``forward_step``'s ``batch_dict`` keys.
        """
        actions = batch["action"]
        noisy_model_input, timesteps = self.loss_func.add_noise(actions)

        traj = batch.get("traj")
        traj_loss_mask = batch.get("traj_loss_mask")

        model_pred = self.model(
            batch["images"],
            batch["image_masks"],
            batch["lang_tokens"],
            batch["lang_masks"],
            noisy_model_input,
            timesteps,
            batch["embodiment_id"],
            lang_att_masks=batch["lang_att_masks"],
            fast_action_indicator=batch["fast_action_indicator"],
        )

        loss_dict = self.loss_func(
            model_pred,
            batch["lang_tokens"],
            batch["lang_loss_masks"],
            batch["action_loss_mask"],
            traj,
            traj_loss_mask,
        )

        total = sum(v for v in loss_dict.values() if torch.is_tensor(v))
        # Per-component losses are averaged over the batch for logging, same
        # as the reference ``giga_train`` step-print (mean over batch dim).
        log_loss_dict = {k: v.mean() for k, v in loss_dict.items()}
        log_loss_dict["action_loss"] = total.mean() if torch.is_tensor(total) else total
        return total.mean(), log_loss_dict

    @classmethod
    def from_pretrained(cls, config_or_path) -> "GigaBrainPolicy":
        """Create GigaBrainPolicy from a GigaBrainModelConfig, a pretrained
        checkpoint path string, or a config dict/OmegaConf object.

        Mirrors ``XVLAPolicy.from_pretrained``'s three-way dispatch (see
        ``model/xvla/modeling_xvla.py``): a bare path loads config.json from
        the checkpoint dir and then loads weights; a config object/dict is
        used as-is, optionally with a ``pretrained_path`` key.
        """
        if isinstance(config_or_path, GigaBrainModelConfig):
            cfg = config_or_path
            pretrained_path = None
        elif isinstance(config_or_path, (str, os.PathLike)):
            pretrained_path = str(config_or_path)
            cfg = cls._config_from_checkpoint(pretrained_path)
        else:
            outer = config_or_path if isinstance(config_or_path, dict) else vars(config_or_path)
            config_block = outer["model"] if "model" in outer else outer
            config_block = config_block if isinstance(config_block, dict) else vars(config_block)
            cfg = GigaBrainModelConfig(**{
                k: v for k, v in config_block.items() if k in _POLICY_CONFIG_FIELDS
                or k in ("model_type", "robot_type")
            })
            pretrained_path = outer.get("pretrained_path") if isinstance(outer, dict) else None

        policy = cls(cfg)
        if pretrained_path:
            policy.load_pretrained(pretrained_path)
        return policy

    @classmethod
    def _config_from_checkpoint(cls, pretrained_path: str) -> GigaBrainModelConfig:
        """Read ``config.json`` from a checkpoint directory into a GigaBrainModelConfig."""
        config_file = Path(pretrained_path) / _CONFIG_FILE
        with open(config_file) as f:
            raw = json.load(f)
        return GigaBrainModelConfig(**{k: v for k, v in raw.items() if k in _POLICY_CONFIG_FIELDS})

    def load_pretrained(self, pretrained_path: str, strict: bool = False, device=None) -> "GigaBrainPolicy":
        """Load GigaBrain-0 weights from a diffusers-style checkpoint directory
        (``config.json`` + sharded ``diffusion_pytorch_model-*.safetensors``).

        This policy wraps the core network as ``self.model`` (a
        ``GigaBrain0Policy`` instance), so state_dict keys are prefixed with
        ``model.`` — the reference checkpoint stores them unprefixed (it *is*
        the ``GigaBrain0Policy``). Re-add the prefix when missing, matching
        ``XVLAPolicy.load_pretrained``'s convention.

        ``device`` is intentionally NOT used to move the loaded tensors before
        ``load_state_dict``: this is called BEFORE FSDP wrapping (see
        ``BaseTrainer._setup``'s "5. Pretrained weights" step, which runs
        ahead of "6/7. wrap_model"), so every rank would otherwise materialize
        a full ~15GB copy of the checkpoint on its own GPU simultaneously —
        8 redundant full copies with no sharding yet applied, which reliably
        OOMs before FSDP2 gets a chance to shard anything. Loading on CPU and
        letting ``self.to(device)`` (called by the caller after this returns,
        or implicitly via FSDP's own placement during wrapping) move tensors
        avoids that redundant peak.
        """
        path = Path(pretrained_path)
        state_dict = _load_sharded_state_dict(path)

        own_keys = set(self.state_dict().keys())
        needs_prefix = any(
            (not k.startswith("model.")) and (f"model.{k}" in own_keys)
            for k in state_dict
        )
        if needs_prefix:
            state_dict = {
                (k if k.startswith("model.") else f"model.{k}"): v
                for k, v in state_dict.items()
            }

        missing, unexpected = self.load_state_dict(state_dict, strict=strict)
        if missing:
            logger.warning("[giga_brain] load_pretrained missing keys (%d): %s", len(missing), missing[:5])
        if unexpected:
            logger.warning("[giga_brain] load_pretrained unexpected keys (%d): %s", len(unexpected), unexpected[:5])

        # Re-tie lm_head <-> embed_tokens after loading (mirrors GigaBrain0Trainer.get_models'
        # assertion that the two share storage post-load).
        self.model.paligemma_with_expert.tie_weights()
        return self
