# Copyright 2026 The LoongForge Authors.
# SPDX-License-Identifier: Apache-2.0

"""Dataset strategy for lingbot_vla_v2 — MultiVLADataset (LeRobot-based).

Bridges the LoongForge (model_cfg, data_cfg, training_args) triple onto the
vendored benchmark dataset stack so the sample pipeline is numerically
identical to upstream lingbot-vla-v2.
"""

import logging
from dataclasses import dataclass, field
from typing import List, Optional

logger = logging.getLogger(__name__)


@dataclass
class _VendorDatasetConfig:
    """Attribute-compatible shim for the vendored ``build_vla_dataset``."""

    data_name: str
    train_path: str
    robot_config_root: str
    chunk_size: int
    prompt_type: str
    img_size: int
    image_augment: bool
    use_future_image: bool
    joints: List[str] = field(default_factory=list)
    cameras: List[str] = field(default_factory=list)
    norm_type: List[str] = field(default_factory=list)
    state_norm_type: Optional[str] = None
    video_backend: str = "torchcodec"


def _to_list(value):
    """OmegaConf ListConfig → plain list (vendored code uses list semantics)."""
    return list(value) if value is not None else []


def build_lingbot_vla_v2_dataset(model_cfg, data_cfg, training_args):
    """Build the vendored MultiVLADataset from typed LoongForge configs."""
    from transformers import AutoProcessor

    from loongforge.embodied.model.lingbot_vla_v2.modeling_lingbot_vla_v2 import (
        build_internal_config,
    )
    from loongforge.embodied.model.lingbot_vla_v2.vendor.data.dataset import (
        build_vla_dataset,
    )

    tokenizer_path = training_args.tokenizer_path or model_cfg.tokenizer_path
    processor = AutoProcessor.from_pretrained(
        tokenizer_path, padding_side="right", trust_remote_code=True
    )

    internal_cfg = build_internal_config(model_cfg)

    vendor_data_cfg = _VendorDatasetConfig(
        data_name=data_cfg.data_name,
        train_path=training_args.dataset_path,
        robot_config_root=data_cfg.robot_config_root,
        chunk_size=model_cfg.chunk_size,
        prompt_type=data_cfg.prompt_type,
        img_size=data_cfg.img_size,
        image_augment=data_cfg.image_augment,
        use_future_image=data_cfg.use_future_image,
        joints=_to_list(data_cfg.joints),
        cameras=_to_list(data_cfg.cameras),
        norm_type=_to_list(data_cfg.norm_type),
        state_norm_type=data_cfg.state_norm_type,
        video_backend=data_cfg.video_backend,
    )

    use_depth_align = bool(model_cfg.align_params)

    dataset = build_vla_dataset(
        dataset_config=vendor_data_cfg,
        model_config=internal_cfg,
        config=internal_cfg,
        processor=processor,
        use_depth_align=use_depth_align,
    )
    logger.info(
        "lingbot_vla_v2 dataset built: %d frames, use_depth_align=%s, "
        "use_future_image=%s",
        len(dataset),
        use_depth_align,
        data_cfg.use_future_image,
    )
    return dataset
