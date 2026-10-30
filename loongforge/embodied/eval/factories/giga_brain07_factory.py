# Copyright 2026 The LoongForge Authors.
# SPDX-License-Identifier: Apache-2.0

"""GigaBrain-0.7 model factory for the LoongForge eval server.

方案B: the GigaBrain-0.7 checkpoint carries its full architecture in its own
``config.json`` (proprio_anchor / num_embodiments=8 / max_action_dim=32), so the
eval model is built exactly like the trainer builds it —
``GigaBrain07Policy.from_pretrained(pretrained)`` — rather than from the slim
LoongForge ``model:`` dataclass. This factory then wires the eval-only
preprocessing via ``GigaBrain07Policy.configure_eval(...)`` (tokenizer path,
camera layout, prompt length) and wraps ``predict_action`` to coerce the
embodiment id and optionally truncate the open-loop action horizon, mirroring
``xvla_factory`` / ``pi05_factory``.
"""

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
from loongforge.embodied.model.giga_brain07.model_configuration_giga_brain07 import (
    GigaBrain07ModelConfig,
)

# Default GigaBrain camera layout (matches the training ImageTransform default:
# cam_high + two wrist cams) and 224x224 SigLIP resize.
_DEFAULT_PRESENT_IMG_KEYS = [
    "observation.images.cam_high",
    "observation.images.cam_left_wrist",
    "observation.images.cam_right_wrist",
]
_DEFAULT_RESIZE = (224, 224)
_DEFAULT_MAX_LENGTH = 120


def _build_giga_model(model_cfg: GigaBrain07ModelConfig, pretrained_path: str, random_init: bool, device):
    """Build a ``GigaBrain07Policy``, loading arch (+weights) from the checkpoint.

    The architecture always comes from the checkpoint's ``config.json`` (the
    0.7 base ckpt declares proprio_anchor / 8 embodiments / 32-dim action).
    ``random_init`` builds the same architecture with freshly initialized
    weights via diffusers ``from_config``; otherwise weights are loaded with
    ``from_pretrained``.
    """
    from giga_models.models.vla.giga_brain_0.modeling_giga_brain_0 import GigaBrain07Policy

    if not pretrained_path:
        raise ValueError(
            "giga_brain07 factory requires a checkpoint path (server.ckpt_path or "
            "model.pretrained) to read the model architecture from its config.json."
        )
    if random_init:
        # Same architecture, fresh weights: load only the config, skip weights.
        config_dict = GigaBrain07Policy.load_config(pretrained_path)
        return GigaBrain07Policy.from_config(config_dict)
    return GigaBrain07Policy.from_pretrained(pretrained_path)


@register_factory("giga_brain07")
class GigaBrain07ModelFactory:
    """Build a GigaBrain-0.7 model that implements the common predict_action interface."""

    model_config_cls = GigaBrain07ModelConfig

    @classmethod
    def build(
        cls,
        model_cfg: GigaBrain07ModelConfig,
        server_args: EvalServerArgs,
    ) -> PredictActionModelSpec:
        """Create GigaBrain07Policy and return it with metadata for the generic eval policy.

        Args:
            model_cfg: Typed GigaBrain07ModelConfig resolved from the eval YAML model section.
            server_args: Typed EvalServerArgs with runtime/infra options.
        """
        import torch

        # Checkpoint: prefer the explicit eval server ckpt_path, else the config default.
        pretrained_path = (
            str(Path(server_args.ckpt_path).expanduser())
            if server_args.ckpt_path
            else str(model_cfg.pretrained)
        )
        # Tokenizer / processor: required for proprio_anchor (the <|propri|> token
        # is registered on this tokenizer). Fall back to TOKENIZER_PATH, then the
        # checkpoint dir.
        tokenizer_path = (
            server_args.tokenizer_path
            or os.environ.get("TOKENIZER_PATH", "")
            or pretrained_path
        )

        resolved_device = torch.device(
            server_args.device
            if torch.cuda.is_available() or not server_args.device.startswith("cuda")
            else "cpu"
        )

        model = _build_giga_model(model_cfg, pretrained_path, server_args.random_init, resolved_device)

        if getattr(model, "state_input_mode", "prompt") == "proprio_anchor" and not tokenizer_path:
            raise ValueError(
                "giga_brain07 proprio_anchor mode requires a tokenizer path "
                "(server.tokenizer_path or env TOKENIZER_PATH) to register the <|propri|> token."
            )

        # Wire eval-only preprocessing. 'prompt' state mode folds the discretized
        # state into the text prompt; the continuous-state modes do not.
        model.configure_eval(
            processor_path=tokenizer_path,
            resize_imgs_with_padding=_DEFAULT_RESIZE,
            present_img_keys=_DEFAULT_PRESENT_IMG_KEYS,
            max_length=_DEFAULT_MAX_LENGTH,
            discrete_state_input=(getattr(model, "state_input_mode", "prompt") == "prompt"),
        )

        model = model.to(resolved_device)
        model.eval()
        if server_args.use_bf16 and resolved_device.type == "cuda":
            model = model.to(dtype=torch.bfloat16)

        # Wrap predict_action to coerce emb_id (int from YAML) to a tensor and
        # optionally truncate the action chunk (server.chunk_execute_steps):
        # 0 -> no truncation (use the model's full open-loop horizon),
        # N>0 -> truncate to N, N<0 -> no truncation.
        _orig_predict_action = model.predict_action
        _raw_chunk_steps = int(getattr(server_args, "chunk_execute_steps", 0) or 0)
        _chunk_execute_steps = _raw_chunk_steps if _raw_chunk_steps > 0 else 0

        def _predict_action_wrapper(images, instructions, state=None, dataset_stats=None, emb_id=None, **kwargs):
            if emb_id is not None and not isinstance(emb_id, torch.Tensor):
                emb_id = torch.tensor([int(emb_id)], dtype=torch.long)
            result = _orig_predict_action(
                images, instructions, state=state, dataset_stats=dataset_stats, emb_id=emb_id, **kwargs
            )
            # Truncate on the horizon axis. predict_action returns [B, H, D] (or [H, D]).
            if _chunk_execute_steps > 0 and hasattr(result, "shape") and getattr(result, "ndim", 0) >= 2:
                if result.ndim == 3 and result.shape[1] > _chunk_execute_steps:
                    result = result[:, :_chunk_execute_steps]
                elif result.ndim == 2 and result.shape[0] > _chunk_execute_steps:
                    result = result[:_chunk_execute_steps]
            return result

        model.predict_action = _predict_action_wrapper

        metadata: Dict[str, Any] = {
            "framework": "loongforge",
            "model_type": "giga_brain07",
            "ckpt_path": pretrained_path if not server_args.random_init else "random_init://giga_brain07",
            "random_init": bool(server_args.random_init),
            "loongforge_root": server_args.loongforge_root,
            "action_dim": int(getattr(model, "max_action_dim", model_cfg.max_action_dim)),
            "action_horizon": int(getattr(model, "n_action_steps", model_cfg.action_horizon)),
            "chunk_execute_steps": _chunk_execute_steps if _chunk_execute_steps > 0 else None,
            "dataset_statistics_path": server_args.dataset_statistics_path,
            "tokenizer_path": tokenizer_path,
        }
        return PredictActionModelSpec(model=model, metadata=metadata)


class LoongForgeGigaBrain07Policy(GenericPredictActionPolicy):
    """Backward-compatible GigaBrain-0.7 policy built from the generic predict_action policy."""

    def __init__(
        self,
        ckpt_path: str,
        loongforge_root: str = "",
        device: str = "cuda",
        use_bf16: bool = True,
        dataset_statistics_path: str = "",
        tokenizer_path: str = "",
        action_dim: int = 32,
        action_horizon: int = 50,
        random_init: bool = False,
    ) -> None:
        """Build GigaBrain-0.7 through its factory and attach it to the generic eval policy."""
        model_cfg = GigaBrain07ModelConfig(
            pretrained=ckpt_path,
            action_horizon=action_horizon,
            max_action_dim=action_dim,
        )
        server_args = EvalServerArgs(
            ckpt_path=ckpt_path,
            loongforge_root=loongforge_root,
            device=device,
            use_bf16=use_bf16,
            dataset_statistics_path=dataset_statistics_path,
            tokenizer_path=tokenizer_path,
            random_init=random_init,
        )
        spec = GigaBrain07ModelFactory.build(model_cfg, server_args)
        super().__init__(
            model=spec.model,
            metadata=spec.metadata,
            dataset_statistics_path=dataset_statistics_path,
            action_dim=action_dim,
            request_id_prefix="loongforge-giga-brain07",
        )
