# Copyright 2026 The LoongForge Authors.
# SPDX-License-Identifier: Apache-2.0

"""GigaBrain-0 per-sample transform — bridges the vendored
``GigaBrain0Transform`` (unmodified reference preprocessing pipeline) into
LoongForge's :class:`BaseTransform` / transform-builder registry.

Unlike XVLA's several small composable transforms, GigaBrain-0's reference
pipeline is a single ``GigaBrain0Transform.__call__`` that runs delta-action,
normalization, prompt tokenization, image encoding and (optional) trajectory
transforms in one fixed order — see
``data/datasets/giga_brain/giga_brain_0_transforms.py``. Wrapping it as one
:class:`BaseTransform` keeps that order and its numerics byte-for-byte
identical to the reference instead of re-decomposing it into separate steps.
"""

from __future__ import annotations

from dataclasses import asdict, is_dataclass
from typing import Any, Dict

from loongforge.embodied.data.datasets.giga_brain.giga_brain_0_transforms import (
    GigaBrain0Transform,
)
from loongforge.embodied.data.datasets.transforms.base import BaseTransform
from loongforge.embodied.data.datasets.transforms.registry import (
    TransformBuilderContext,
    register_transform_builder,
)


def _coerce_to_dict(value: Any) -> Any:
    """Convert (possibly nested) dataclass config sections to plain dicts.

    ``GigaBrain0Transform`` expects plain dict kwargs (see
    ``delta_action_cfg['mask']`` / ``norm_cfg['norm_stats_path']`` etc.),
    matching the reference dict-based configs. OmegaConf-merged DataConfig
    sections are frozen dataclasses, so convert them here.
    """
    if is_dataclass(value) and not isinstance(value, type):
        return {k: _coerce_to_dict(v) for k, v in asdict(value).items()}
    if isinstance(value, dict):
        return {k: _coerce_to_dict(v) for k, v in value.items()}
    return value


class GigaBrainTransform(BaseTransform):
    """Adapts :class:`GigaBrain0Transform` to the :class:`BaseTransform` interface."""

    def __init__(self, is_train: bool, delta_action_cfg, norm_cfg, image_cfg, prompt_cfg, traj_cfg=None):
        super().__init__(apply_to=[], training=is_train)
        self._impl = GigaBrain0Transform(
            delta_action_cfg=delta_action_cfg,
            norm_cfg=norm_cfg,
            traj_cfg=traj_cfg,
            image_cfg=image_cfg,
            prompt_cfg=prompt_cfg,
            is_train=is_train,
        )

    def apply(self, data: Dict[str, Any]) -> Dict[str, Any]:
        return self._impl(data)


@register_transform_builder("giga_brain")
def build_giga_brain_transforms(ctx: TransformBuilderContext):
    """Build the single GigaBrain-0 per-sample transform from ``data_cfg``.

    Config sections (``delta_action_cfg`` / ``norm_cfg`` / ``image_cfg`` /
    ``prompt_cfg``) come straight from ``GigaBrainDataConfig`` (YAML ``data:``
    section), mirroring the reference ``configs/giga_brain_0_*_finetune*.py``
    dict-config fields 1:1.
    """
    model_cfg = ctx.model_cfg
    if model_cfg.model_type != "giga_brain":
        return []

    data_cfg = ctx.data_cfg
    # No generic "is_train" flag on TrainingArgs; FinetuneTrainer only builds
    # the "vla" dataloader for training (see FinetuneTrainer._build_dataloaders),
    # so this transform is always constructed in training mode.
    is_train = True

    return [
        GigaBrainTransform(
            is_train=is_train,
            delta_action_cfg=_coerce_to_dict(data_cfg.delta_action_cfg),
            norm_cfg=_coerce_to_dict(data_cfg.norm_cfg),
            image_cfg=_coerce_to_dict(data_cfg.image_cfg),
            prompt_cfg=_coerce_to_dict(data_cfg.prompt_cfg),
        )
    ]
