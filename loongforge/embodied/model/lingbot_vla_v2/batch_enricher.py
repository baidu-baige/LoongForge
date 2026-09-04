# Copyright 2026 The LoongForge Authors.
# SPDX-License-Identifier: Apache-2.0

"""Online teacher target generation for LingBot VLA v2."""

from __future__ import annotations

import logging

import torch

logger = logging.getLogger(__name__)


class LingbotVlaV2BatchEnricher:
    """Keep frozen depth/video teachers outside model and optimizer state."""

    def __init__(self, model_cfg, ctx=None):
        self.model_cfg = model_cfg
        self.ctx = ctx
        self._moge_model = None
        self._morgbd_model = None
        self._video_teacher = None
        self._runner = None
        self._use_depth_align = bool(model_cfg.align_params)
        self._use_future_depth = False
        self._use_future_video = False
        if self._use_depth_align:
            align = self.align_params()
            self._use_future_depth = bool(
                align.get("depth", {}).get("use_future_depth", False)
            )
            self._use_future_video = bool(align.get("use_future_video", False))

    @property
    def has_runner(self) -> bool:
        """True when the teachers can be started without blocking the caller."""
        return self._runner is not None

    def align_params(self) -> dict:
        """Return ``model.align_params`` as a plain resolved dict."""
        align = self.model_cfg.align_params or {}
        if not isinstance(align, dict):
            from omegaconf import OmegaConf

            align = OmegaConf.to_container(align, resolve=True)
        return align

    def setup(self, ctx=None, model_cfg=None, training_args=None) -> None:
        """Build the depth/video teachers and their side-stream runner."""
        if ctx is not None:
            self.ctx = ctx
        if model_cfg is not None and model_cfg is not self.model_cfg:
            self.__init__(model_cfg, self.ctx)
        if not self._use_depth_align:
            return
        from loongforge.embodied.model.lingbot_vla_v2.vendor.vision_models.module_utils import (
            build_depth_model,
            build_video_model,
        )

        align = self.align_params()
        if self.ctx is None or self.ctx.is_main:
            logger.info("Building frozen depth teacher (MoGE + MoRGBD)")
        self._moge_model, self._morgbd_model = build_depth_model(align)
        if self.model_cfg.use_compile:
            self._moge_model = torch.compile(self._moge_model)
            self._morgbd_model = torch.compile(self._morgbd_model)
        if self._use_future_video:
            if self.ctx is None or self.ctx.is_main:
                logger.info("Building frozen DINO video teacher")
            self._video_teacher = build_video_model(align["video"])
        if (
            bool(getattr(self.model_cfg, "async_teacher", True))
            and torch.cuda.is_available()
            and self._runner is None
        ):
            from loongforge.embodied.model.lingbot_vla_v2.async_teacher import (
                AsyncTeacherRunner,
            )

            self._runner = AsyncTeacherRunner()
            if self.ctx is None or self.ctx.is_main:
                logger.info("Teacher targets run on a side stream / worker thread")
        elif self.ctx is None or self.ctx.is_main:
            logger.info("Teacher targets run synchronously in the training thread")

    def _pop_inputs(self, batch) -> dict:
        """Detach the teacher's inputs from ``batch.data`` on the calling thread.

        Done here rather than in the worker so the batch dict is never mutated
        concurrently with the student forward reading it.
        """
        data = batch.data
        inputs = {
            "pil_images": data.pop("pil_images", None),
            "future_pil_images": (
                data.pop("future_pil_images", None)
                if (self._use_future_depth or self._use_future_video)
                else None
            ),
            "future_video_effective_fps": data.pop("future_video_effective_fps", None),
        }
        return inputs

    def _compute_targets(self, inputs: dict) -> dict:
        """Pure teacher forward. No access to ``batch``, safe on a worker thread."""
        from loongforge.embodied.model.lingbot_vla_v2.vendor.vision_models.module_utils import (
            get_depth_target,
            get_video_target,
        )

        align = self.align_params()
        pil_images = inputs["pil_images"]
        future_pil_images = inputs["future_pil_images"]
        out = {}
        with torch.no_grad():
            with torch.autocast("cuda", dtype=torch.bfloat16):
                depth_targets, _ = get_depth_target(
                    "MoRGBD", (self._moge_model, self._morgbd_model), pil_images
                )
                if self._use_future_depth:
                    future_depth_targets, _ = get_depth_target(
                        "MoRGBD",
                        (self._moge_model, self._morgbd_model),
                        future_pil_images,
                    )
                else:
                    future_depth_targets = None
            if self._use_future_video:
                with torch.autocast("cuda", dtype=torch.bfloat16):
                    bundle = get_video_target(
                        self._video_teacher,
                        pil_images,
                        future_pil_images,
                        align["video"],
                        effective_fps=inputs["future_video_effective_fps"],
                    )
                if isinstance(bundle, dict):
                    out["future_video_targets"] = bundle["patch"]
                    out["future_video_cls_targets"] = bundle.get("cls")
                    out["future_video_current_patch"] = bundle.get("current_patch")
                elif isinstance(bundle, tuple):
                    out["future_video_targets"], out["future_video_cls_targets"] = bundle
                else:
                    out["future_video_targets"] = bundle
        out["depth_targets"] = depth_targets
        out["future_depth_targets"] = future_depth_targets
        return out

    def enrich(self, batch) -> None:
        """Synchronous path: compute targets and write them into ``batch.data``."""
        if not self._use_depth_align:
            return
        targets = self._compute_targets(self._pop_inputs(batch))
        batch.data.update(targets)

    def enrich_async(self, batch):
        """Submit the teacher to the side stream; returns a handle or ``None``.

        ``None`` means the caller should fall back to ``enrich`` (feature disabled
        or no runner available).
        """
        if not self._use_depth_align or self._runner is None:
            return None
        return self._runner.submit(self._compute_targets, self._pop_inputs(batch))

    def close(self) -> None:
        """Stop the async runner and drop the teacher models."""
        if self._runner is not None:
            self._runner.close()
        self._runner = None
        self._moge_model = None
        self._morgbd_model = None
        self._video_teacher = None


__all__ = ["LingbotVlaV2BatchEnricher"]
