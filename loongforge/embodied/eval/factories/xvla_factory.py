# Copyright 2026 The LoongForge Authors.
# SPDX-License-Identifier: Apache-2.0

"""XVLA model factory for the LoongForge eval server."""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any, Dict

from loongforge.embodied.eval.factories.registry import register_factory
from loongforge.embodied.eval.servers.eval_server_config import EvalServerArgs
from loongforge.embodied.eval.servers.loongforge_policy import (
    GenericPredictActionPolicy,
    PredictActionModelSpec,
)
from loongforge.embodied.model.xvla.model_configuration_xvla import XvlaModelConfig
from loongforge.embodied.model.registry import build_model


@register_factory("xvla")
class XVLAModelFactory:
    """Build an XVLA model instance that implements the common predict_action interface."""

    model_config_cls = XvlaModelConfig

    @classmethod
    def build(
        cls,
        model_cfg: XvlaModelConfig,
        server_args: EvalServerArgs,
    ) -> PredictActionModelSpec:
        """Create XVLAPolicy and return it with metadata for the generic eval policy.

        Args:
            model_cfg: Typed XvlaModelConfig resolved from the eval YAML model section.
            server_args: Typed EvalServerArgs with runtime/infra options.
        """
        import torch

        pretrained_path = str(Path(server_args.ckpt_path).expanduser()) if server_args.ckpt_path else ""
        tokenizer_path = server_args.tokenizer_path or pretrained_path
        resolved_device = torch.device(
            server_args.device
            if torch.cuda.is_available() or not server_args.device.startswith("cuda")
            else "cpu"
        )

        model = build_model(model_cfg)
        # XVLA lazily loads tokenizer/image-processor from _processor_path at first inference.
        model._processor_path = tokenizer_path
        model._num_image_views = int(getattr(model_cfg, "num_image_views", 3) or 3)
        if not server_args.random_init:
            model.load_pretrained(pretrained_path, device=str(resolved_device) if resolved_device is not None else None)
        model = model.to(resolved_device)
        model.eval()
        if server_args.use_bf16 and resolved_device.type == "cuda":
            model = model.to(dtype=torch.bfloat16)

        # Wrap predict_action to coerce domain_id (int from YAML) to tensor
        # and optionally truncate the action chunk (server.chunk_execute_steps).
        # 0 → default 10 (official X-VLA open-loop horizon); N>0 → truncate to N;
        # N<0 → no truncation.
        _orig_predict_action = model.predict_action
        _raw_chunk_steps = int(getattr(server_args, "chunk_execute_steps", 0) or 0)
        if _raw_chunk_steps == 0:
            _chunk_execute_steps = 10
        elif _raw_chunk_steps < 0:
            _chunk_execute_steps = 0  # disabled
        else:
            _chunk_execute_steps = _raw_chunk_steps

        def _predict_action_wrapper(images, instructions, state=None, dataset_stats=None, domain_id=None, **kwargs):
            # Extra eval payload keys (unnorm_key, cfg_scale, ...) are ignored:
            # XVLA's predict_action does not consume them.
            if domain_id is not None:
                if not isinstance(domain_id, torch.Tensor):
                    domain_id = torch.tensor([domain_id], dtype=torch.long)
                elif domain_id.dim() == 0:
                    domain_id = domain_id.unsqueeze(0)
                domain_id = domain_id.to(resolved_device)
            result = _orig_predict_action(
                images, instructions, state=state, dataset_stats=dataset_stats, domain_id=domain_id
            )
            # Truncate on the horizon axis. predict_action returns [B, H, D] (or [H, D]).
            # Index 0 is batch size (usually 1), not horizon — do not slice shape[0].
            if _chunk_execute_steps > 0 and hasattr(result, "shape") and getattr(result, "ndim", 0) >= 2:
                if result.ndim == 3 and result.shape[1] > _chunk_execute_steps:
                    result = result[:, :_chunk_execute_steps]
                elif result.ndim == 2 and result.shape[0] > _chunk_execute_steps:
                    result = result[:_chunk_execute_steps]
            return result

        model.predict_action = _predict_action_wrapper

        metadata: Dict[str, Any] = {
            "framework": "loongforge",
            "model_type": "xvla",
            "ckpt_path": pretrained_path if not server_args.random_init else "random_init://xvla",
            "random_init": bool(server_args.random_init),
            "loongforge_root": server_args.loongforge_root,
            "action_dim": model_cfg.real_action_dim,
            "action_horizon": model_cfg.action_horizon,
            "chunk_execute_steps": _chunk_execute_steps if _chunk_execute_steps > 0 else None,
            "dataset_statistics_path": server_args.dataset_statistics_path,
            "state_format": server_args.state_format,
        }
        return PredictActionModelSpec(model=model, metadata=metadata)


class LoongForgeXVLAPolicy(GenericPredictActionPolicy):
    """Backward-compatible XVLA policy built from the generic predict_action policy."""

    def __init__(
        self,
        ckpt_path: str,
        loongforge_root: str = "",
        device: str = "cuda",
        use_bf16: bool = True,
        dataset_statistics_path: str = "",
        action_dim: int = 7,
        state_dim: int = 8,
        action_horizon: int = 30,
        max_action_dim: int = 20,
        max_state_dim: int = 20,
        action_mode: str = "ee6d",
        real_action_dim: int = 20,
        num_actions: int = 30,
        num_image_views: int = 3,
        random_init: bool = False,
    ) -> None:
        """Build XVLA through its factory and attach it to the generic eval policy."""
        model_cfg = XvlaModelConfig(
            action_horizon=action_horizon,
            max_action_dim=max_action_dim,
            real_action_dim=real_action_dim,
            action_mode=action_mode,
            num_actions=num_actions,
        )
        server_args = EvalServerArgs(
            ckpt_path=ckpt_path,
            loongforge_root=loongforge_root,
            device=device,
            use_bf16=use_bf16,
            dataset_statistics_path=dataset_statistics_path,
            random_init=random_init,
        )
        spec = XVLAModelFactory.build(model_cfg, server_args)
        super().__init__(
            model=spec.model,
            metadata=spec.metadata,
            dataset_statistics_path=dataset_statistics_path,
            action_dim=action_dim,
            request_id_prefix="loongforge-xvla",
        )
