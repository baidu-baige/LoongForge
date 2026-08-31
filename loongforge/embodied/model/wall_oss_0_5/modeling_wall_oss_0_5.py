# Copyright 2026 The LoongForge Authors.
# SPDX-License-Identifier: Apache-2.0
#
# Modified from Wall-X under the Apache-2.0 License.

"""Embodied wrapper for the Wall-OSS-0.5 Qwen2.5 VLA model."""

from __future__ import annotations

import logging
import os
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Dict, Mapping

import torch
import torch.nn as nn
from safetensors.torch import load_file
from transformers import AutoProcessor

from loongforge.embodied.model.registry import register_model
from loongforge.embodied.model.wall_oss_0_5.model_configuration_wall_oss_0_5 import (
    WallOss05ModelConfig,
)
from loongforge.embodied.model.wall_oss_0_5.qwen2_5 import (
    Qwen25VLConfig,
    Qwen25VLMoEForAction,
)
from loongforge.embodied.train.global_vars import get_training_args

logger = logging.getLogger(__name__)


def _load_safetensors_dir(path: str | os.PathLike[str]) -> dict[str, torch.Tensor]:
    """Load safetensors dir."""
    ckpt_dir = Path(path)
    weight_files = sorted(ckpt_dir.glob("*.safetensors"))
    if not weight_files:
        raise FileNotFoundError(f"No .safetensors files found under {ckpt_dir}")
    merged: dict[str, torch.Tensor] = {}
    for weight_file in weight_files:
        merged.update(load_file(str(weight_file)))
    return merged


def _resolve_wall_oss_paths_from_training_args() -> tuple[str, str, str, str]:
    """Resolve wall oss paths from training args."""
    training_args = get_training_args()
    pretrained_checkpoint = training_args.pretrained_checkpoint
    tokenizer_path = training_args.tokenizer_path
    if not pretrained_checkpoint:
        raise ValueError("--pretrained-checkpoint is required for wall_oss_0_5.")
    if not tokenizer_path:
        raise ValueError("--tokenizer-path is required for wall_oss_0_5.")
    config_path, wall_checkpoint_path = _resolve_wall_checkpoint_paths(pretrained_checkpoint)
    return config_path, tokenizer_path, tokenizer_path, wall_checkpoint_path


def _resolve_wall_checkpoint_paths(pretrained_checkpoint: str) -> tuple[str, str]:
    """Return (config.json path, wall model.safetensors path) for a given checkpoint arg."""
    checkpoint = Path(pretrained_checkpoint)
    if checkpoint.is_dir() or not checkpoint.suffix:
        config_path = checkpoint / "config.json"
        wall_checkpoint_path = checkpoint / "model.safetensors"
    elif checkpoint.suffix == ".json":
        config_path = checkpoint
        wall_checkpoint_path = checkpoint.with_name("model.safetensors")
    else:
        config_path = checkpoint.parent / "config.json"
        wall_checkpoint_path = checkpoint
    return str(config_path), str(wall_checkpoint_path)


def _build_processor(processor_path: str, model_cfg: WallOss05ModelConfig):
    """Build processor."""
    processor = AutoProcessor.from_pretrained(processor_path, use_fast=True)
    processor.tokenizer.padding_side = "left"
    new_tokens = ["<|propri|>", "<|action|>"]
    if model_cfg.new_special_tokens is not None:
        new_tokens.extend(model_cfg.new_special_tokens)
    processor.tokenizer.add_tokens(new_tokens)
    return processor


def _build_train_config_dict(model_cfg: WallOss05ModelConfig) -> dict[str, Any]:
    """Build train config dict."""
    data = {
        "action_horizon_flow": model_cfg.action_horizon_flow,
        "use_state_string_representation": model_cfg.use_state_string_representation,
    }
    flat = {
        "model_type": "qwen2_5",
        "backbone": model_cfg.backbone,
        "attn_deterministic": model_cfg.attn_deterministic,
        "ar_loss_weight": model_cfg.ar_loss_weight,
        "flow_loss_weight": model_cfg.flow_loss_weight,
        "dof_config": dict(model_cfg.dof_config),
        "ar_dof_config": dict(model_cfg.ar_dof_config),
        "agent_pos_config": dict(model_cfg.agent_pos_config),
        "data": data,
    }
    if model_cfg.attn_implementation is not None:
        flat["_attn_implementation"] = model_cfg.attn_implementation
    return flat


def _load_qwen_pretrained(model: Qwen25VLMoEForAction, path: str | None) -> None:
    """Load qwen pretrained."""
    if not path:
        return
    weights = _load_safetensors_dir(path)
    renamed = model.rename_vlm_weights_for_vla(weights)
    renamed = {
        key: value
        for key, value in renamed.items()
        if "action_preprocessor.normalizer_" not in key
    }
    if (
        model.config.model_type == "qwen2_5_vl"
        and model.model.embed_tokens.weight.shape[0]
        != renamed["model.embed_tokens.weight"].shape[0]
    ):
        logger.info(
            "resize_token_embeddings from %d to %d to match Qwen pretrained weights",
            model.model.embed_tokens.weight.shape[0],
            renamed["model.embed_tokens.weight"].shape[0],
        )
        model.model.resize_token_embeddings(renamed["model.embed_tokens.weight"].shape[0])
    err = model.load_state_dict(renamed, strict=False)
    logger.info("Loaded Qwen pretrained weights from %s: %s", path, err)


def _load_wall_checkpoint(model: Qwen25VLMoEForAction, path: str | None) -> None:
    """Load wall checkpoint."""
    if not path:
        return
    logger.info("Loading Wall-OSS-0.5 checkpoint: %s", path)
    weights = load_file(path)
    weights = model.convert_to_fused(weights)
    weights = {
        key: value
        for key, value in weights.items()
        if "action_preprocessor.normalizer_" not in key
    }
    embed_key = "model.embed_tokens.weight"
    if embed_key in weights and model.model.embed_tokens.weight.shape[0] != weights[embed_key].shape[0]:
        logger.info(
            "resize_token_embeddings from %d to %d to match Wall checkpoint",
            model.model.embed_tokens.weight.shape[0],
            weights[embed_key].shape[0],
        )
        model.model.resize_token_embeddings(weights[embed_key].shape[0])
    err = model.load_state_dict(weights, strict=False)
    logger.info("Loaded Wall-OSS-0.5 checkpoint: %s", err)


@register_model("wall_oss_0_5")
class WallOss05Model(nn.Module):
    """Thin embodied training wrapper around Qwen25VLMoEForAction."""

    def __init__(
        self,
        model: Qwen25VLMoEForAction,
    ):
        """Initialize the instance."""
        super().__init__()
        self.model = model

    @classmethod
    def from_pretrained(cls, model_cfg: WallOss05ModelConfig) -> "WallOss05Model":
        """Build the model shell only; weights are loaded later via ``load_pretrained``.

        Config still comes from ``--pretrained-checkpoint`` because Qwen's
        backbone shape (layers/heads/dims) is defined by that ``config.json``
        and must be known to instantiate the module. Qwen VLM base weights
        and the Wall-OSS delta weights are deferred to ``load_pretrained`` so
        this class fits the standard trainer build → _load_pretrained flow.
        """
        config_path, processor_path, pretrained_path, _ = (
            _resolve_wall_oss_paths_from_training_args()
        )

        if config_path.endswith(".json"):
            qwen_config = Qwen25VLConfig.from_json_file(config_path)
        else:
            qwen_config = Qwen25VLConfig.from_pretrained(config_path)
        qwen_config.update_model_config(_build_train_config_dict(model_cfg))
        if model_cfg.attn_implementation is not None:
            qwen_config._attn_implementation = model_cfg.attn_implementation
            qwen_config.vision_config._attn_implementation = model_cfg.attn_implementation

        processor = _build_processor(processor_path, model_cfg)
        model = Qwen25VLMoEForAction(
            qwen_config,
            processor=processor,
            use_selective_recompute=model_cfg.use_selective_recompute,
        )
        model.resize_token_embeddings(len(processor.tokenizer))

        if hasattr(model, "enable_input_require_grads"):
            model.enable_input_require_grads()
        elif model.get_input_embeddings() is not None:
            model.get_input_embeddings().register_forward_hook(
                lambda _module, _inp, output: output.requires_grad_(True)
            )
        wrapper = cls(model)
        # Remember paths/sizes captured at build so ``load_pretrained`` can
        # perform the full two-stage load without re-reading training args.
        wrapper._qwen_pretrained_path = pretrained_path
        wrapper._tokenizer_len = len(processor.tokenizer)
        return wrapper

    def load_pretrained(self, path: str, device=None):
        """Load Qwen VLM base weights + Wall-OSS delta weights, then apply DDP mix precision.

        ``path`` points to the Wall-OSS-0.5 checkpoint (dir/file). Qwen VLM
        base weights are loaded from ``tokenizer_path`` captured at build.
        """
        del device
        _load_qwen_pretrained(self.model, self._qwen_pretrained_path)
        self.model.resize_token_embeddings(self._tokenizer_len)
        _, wall_checkpoint_path = _resolve_wall_checkpoint_paths(path)
        _load_wall_checkpoint(self.model, wall_checkpoint_path)
        self._maybe_apply_ddp_mix_precision()

    def _maybe_apply_ddp_mix_precision(self) -> None:
        """Apply model-owned mixed precision when training with DDP + bf16.

        Wall-OSS-0.5 keeps selected leaves (norms, action preprocessor) in
        fp32 while casting compute modules to bf16. Under FSDP the mixed
        precision policy owns dtypes, so this only runs for DDP.
        """
        training_args = get_training_args()
        if getattr(training_args, "distributed_strategy", None) != "ddp":
            return
        if str(getattr(training_args, "dtype", "")).lower() not in {"bfloat16", "bf16"}:
            return
        self.convert_to_mix_precision()
        logger.info(
            "DDP mixed precision: applied %s.convert_to_mix_precision().",
            self.__class__.__name__,
        )

    def convert_to_mix_precision(self) -> None:
        """Convert to mix precision."""
        self.model.convert_to_mix_precision()

    def convert_to_fsdp(
        self,
        *,
        mesh,
        mp_policy,
        offload_policy=None,
        reshard_after_forward: bool = True,
        use_dmuon: bool = False,
    ):
        """Convert to fsdp."""
        wrapped = self.model.convert_to_fsdp(
            mesh=mesh,
            mp_policy=mp_policy,
            offload_policy=offload_policy,
            reshard_after_forward=reshard_after_forward,
            use_dmuon=use_dmuon,
        )
        if use_dmuon and hasattr(wrapped, "_dedicated_comm_ctx"):
            self._dedicated_comm_ctx = wrapped._dedicated_comm_ctx
        return self

    def set_requires_gradient_sync(self, enabled: bool) -> None:
        """Set requires gradient sync."""
        if hasattr(self.model, "set_requires_gradient_sync"):
            self.model.set_requires_gradient_sync(enabled)

    @contextmanager
    def no_sync(self):
        """No sync."""
        if hasattr(self, "_dedicated_comm_ctx"):
            import dmuon

            with dmuon.no_sync(self):
                yield
            return

        if hasattr(self.model, "no_sync"):
            with self.model.no_sync():
                yield
            return

        toggled = hasattr(self.model, "set_requires_gradient_sync")
        if toggled:
            self.model.set_requires_gradient_sync(False)
        try:
            yield
        finally:
            if toggled:
                self.model.set_requires_gradient_sync(True)

    def forward(self, batch: Mapping[str, Any]):
        """Run the forward pass."""
        if hasattr(batch, "to_dict"):
            batch = batch.to_dict()
        outputs = self.model(mode="train", **dict(batch))
        loss = outputs.loss
        log_dict: Dict[str, torch.Tensor] = {
            "flow_loss": outputs.flow_loss.detach(),
            "loss": outputs.loss.detach(),
        }
        if outputs.cross_entropy_loss is not None:
            log_dict["cross_entropy_loss"] = outputs.cross_entropy_loss.detach()
        return loss, log_dict

    def on_train_begin(self, *, ctx) -> None:
        """Emit operator backend inventory before measured iterations."""
        from wall_oss_05_op import log_backend_inventory

        log_backend_inventory(rank=ctx.rank, world_size=ctx.world_size)
