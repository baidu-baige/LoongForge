# Copyright 2026 The LoongForge Authors.
# SPDX-License-Identifier: Apache-2.0

"""PI05 model factory for the LoongForge eval server."""

from __future__ import annotations

import os
import types
from pathlib import Path
from typing import Any, Dict, Optional

from loongforge.evaluation.embodied.factories.registry import register_factory
from loongforge.evaluation.embodied.servers.eval_server_config import EvalServerArgs
from loongforge.evaluation.embodied.servers.loongforge_policy import (
    GenericPredictActionPolicy,
    PredictActionModelSpec,
)
from loongforge.models.embodied.pi05.configuration_pi05 import Pi05ModelConfig
from loongforge.models.embodied.registry import build_model


def _q99_unnormalize_actions(actions, action_stats: Optional[dict]):
    """Unnormalize PI05 quantile-normalized actions with dataset action stats."""
    import torch

    if action_stats is None:
        return actions
    if "q01" not in action_stats or "q99" not in action_stats:
        raise ValueError(
            "PI05 is configured for quantile action unnormalization and requires "
            "dataset_stats['action'].q01 and q99"
        )
    q01 = torch.as_tensor(action_stats["q01"], dtype=actions.dtype, device=actions.device)
    q99 = torch.as_tensor(action_stats["q99"], dtype=actions.dtype, device=actions.device)
    dim = min(actions.shape[-1], q01.shape[-1])
    out = actions.clone()
    out[..., :dim] = (out[..., :dim] + 1.0) / 2.0 * (q99[:dim] - q01[:dim]) + q01[:dim]
    return out


def _predict_action(self, images, instructions, state=None, dataset_stats=None):
    """Inference: Euler ODE denoising, returns ndarray (B, action_horizon, max_action_dim).

    Bound to the PI05Policy instance as ``predict_action`` by :class:`PI05ModelFactory`;
    the image, prompt and state preprocessing comes from the data package.
    """
    import torch

    from loongforge.data.embodied.pi05.transforms.pi05_collator import tokenize_prompts
    from loongforge.data.embodied.pi05.transforms.pi05_transform import StateDiscretizationTransform
    from loongforge.data.embodied.transforms.builders import convert_stats
    from loongforge.data.embodied.transforms.image_transform import ImageTransform

    with torch.no_grad():
        device = next(self.parameters()).device
        if not isinstance(images[0], list):
            images = [[img] for img in images]
        B = len(images)
        image_transform = ImageTransform(apply_to=[], image_size=self._image_size)
        # Use the number of views provided by the eval client so LIBERO (2) and
        # RoboTwin (3) both work without hard-coding a single camera count.
        num_images = max(len(images[0]), 1)

        images_list, img_masks = [], []
        for v in range(num_images):
            view = []
            mask_vals = []
            for im_list in images:
                if v < len(im_list):
                    view.append(im_list[v])
                    mask_vals.append(True)
                else:
                    view.append(torch.zeros(3, self._image_size, self._image_size))
                    mask_vals.append(False)
            images_list.append(image_transform.process_batch(view).to(device))
            img_masks.append(torch.tensor(mask_vals, dtype=torch.bool, device=device))

        if state is not None:
            if not isinstance(state, torch.Tensor):
                state = torch.as_tensor(state, dtype=torch.float32)
            if state.dim() == 1:
                state = state.unsqueeze(0)
            state_stats = convert_stats(dataset_stats.get("observation.state")) if dataset_stats else None
            state_transform = StateDiscretizationTransform(
                apply_to=["prompt"], state_key="observation.state", task_key="task",
                num_bins=256, normalization_mode="q99", statistics=state_stats,
            )
            prompts = [state_transform.apply({"observation.state": state[i], "task": instructions[i]})["prompt"]
                       for i in range(B)]
        else:
            prompts = [f"Task: {t.strip()};\nAction: " for t in instructions]

        tok = tokenize_prompts(prompts, self.tokenizer, max_length=200)
        tokens = tok["input_ids"].to(device)
        masks = tok["attention_mask"].bool().to(device)

        actions = self.model.sample_actions(images_list, img_masks, tokens, masks)
        action_stats = convert_stats(dataset_stats.get("action")) if dataset_stats else None
        actions = _q99_unnormalize_actions(actions, action_stats)
        return actions.cpu().numpy()


def _load_pi05_pretrained(pi05_pytorch, pretrained_path: str, device=None):
    """Load PI05 weights, handling the extra 'model.' wrapper prefix.

    Some checkpoints store PI05Pytorch weights with an extra ``model.``
    prefix (i.e. ``model.action_in_proj.bias`` instead of
    ``action_in_proj.bias``).  The training-side ``load_pretrained`` only
    strips ``model.architecture.pi05_model.`` and
    ``architecture.pi05_model.`` prefixes, so it fails on this format.

    This wrapper pre-processes the state dict on the eval side to strip
    the bare ``model.`` prefix before delegating to the standard method.
    """
    from pathlib import Path

    from safetensors.torch import load_file

    path = Path(pretrained_path)
    safetensors_file = path / "model.safetensors" if path.is_dir() else path
    load_kwargs = {"device": str(device)} if device is not None else {}
    raw_sd = load_file(str(safetensors_file), **load_kwargs)

    # Detect whether this checkpoint uses the bare 'model.' wrapper prefix
    # by checking if stripping it produces keys that exist in the model.
    model_sd_keys = set(pi05_pytorch.state_dict().keys())
    needs_strip = any(
        k.startswith("model.") and k[len("model."):] in model_sd_keys
        for k in raw_sd
    )
    if needs_strip:
        stripped = {}
        for k, v in raw_sd.items():
            stripped[k[len("model."):] if k.startswith("model.") else k] = v
        # Write back so load_pretrained receives the corrected state dict via
        # a temporary safetensors file would be heavy; instead call
        # load_state_dict directly using the same remapping logic.
        _RENAME = {
            "action_time_mlp_in.": "time_mlp_in.",
            "action_time_mlp_out.": "time_mlp_out.",
        }
        fixed = {}
        for key, value in stripped.items():
            new_key = key
            for old, new in _RENAME.items():
                new_key = new_key.replace(old, new)
            if new_key.startswith("state_proj."):
                continue
            if new_key in ("paligemma_with_expert.paligemma.lm_head.weight",):
                fixed["paligemma_with_expert.paligemma.model.language_model.embed_tokens.weight"] = value.clone()
            fixed[new_key] = value
        missing, unexpected = pi05_pytorch.load_state_dict(fixed, strict=True)
        del fixed
        if missing:
            raise RuntimeError(f"Missing keys after strip: {missing[:5]}")
        if unexpected:
            raise RuntimeError(f"Unexpected keys after strip: {unexpected[:5]}")
    else:
        pi05_pytorch.load_pretrained(pretrained_path, device=str(device) if device is not None else None)


@register_factory("pi05")
class PI05ModelFactory:
    """Build a PI05 model instance that implements the common predict_action interface."""

    model_config_cls = Pi05ModelConfig

    @classmethod
    def build(
        cls,
        model_cfg: Pi05ModelConfig,
        server_args: EvalServerArgs,
    ) -> PredictActionModelSpec:
        """Create PI05 and return it with metadata for the generic eval policy.

        Args:
            model_cfg: Typed Pi05ModelConfig resolved from the eval YAML model section.
            server_args: Typed EvalServerArgs with runtime/infra options.
        """
        import torch

        pretrained_path = str(Path(server_args.ckpt_path).expanduser()) if server_args.ckpt_path else ""
        resolved_device = torch.device(
            server_args.device
            if torch.cuda.is_available() or not server_args.device.startswith("cuda")
            else "cpu"
        )

        model = build_model(model_cfg)
        model.predict_action = types.MethodType(_predict_action, model)
        model._tokenizer_path = server_args.tokenizer_path or os.environ.get("TOKENIZER_PATH", "")
        if not server_args.random_init:
            _load_pi05_pretrained(model.model, pretrained_path, device=resolved_device)
        model = model.to(resolved_device)
        model.eval()
        if server_args.use_bf16 and resolved_device.type == "cuda":
            model = model.to(dtype=torch.bfloat16)

        metadata: Dict[str, Any] = {
            "framework": "loongforge",
            "model_type": "pi05",
            "ckpt_path": pretrained_path if not server_args.random_init else "random_init://pi05",
            "random_init": bool(server_args.random_init),
            "loongforge_root": server_args.loongforge_root,
            "action_dim": model_cfg.action_dim,
            "action_horizon": model_cfg.action_horizon,
            "dataset_statistics_path": server_args.dataset_statistics_path,
            "tokenizer_path": model._tokenizer_path,
        }
        return PredictActionModelSpec(model=model, metadata=metadata)


class LoongForgePI05Policy(GenericPredictActionPolicy):
    """Backward-compatible PI05 policy built from the generic predict_action policy."""

    def __init__(
        self,
        ckpt_path: str,
        loongforge_root: str = "",
        device: str = "cuda",
        use_bf16: bool = True,
        dataset_statistics_path: str = "",
        tokenizer_path: str = "",
        action_dim: int = 7,
        state_dim: int = 7,
        action_horizon: int = 50,
        max_action_dim: int = 32,
        max_state_dim: int = 32,
        compile_model: bool = False,
        compile_mode: str = "max-autotune",
        random_init: bool = False,
    ) -> None:
        """Build PI05 through its factory and attach it to the generic eval policy."""
        model_cfg = Pi05ModelConfig(
            action_dim=action_dim,
            state_dim=state_dim,
            action_horizon=action_horizon,
            max_action_dim=max_action_dim,
            max_state_dim=max_state_dim,
            compile_model=compile_model,
            compile_mode=compile_mode,
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
        spec = PI05ModelFactory.build(model_cfg, server_args)
        super().__init__(
            model=spec.model,
            metadata=spec.metadata,
            dataset_statistics_path=dataset_statistics_path,
            action_dim=action_dim,
            request_id_prefix="loongforge-pi05",
        )
