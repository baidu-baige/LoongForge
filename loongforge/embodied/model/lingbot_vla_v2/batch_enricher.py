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
        self._executor = None
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
    def has_executor(self) -> bool:
        """True when the teachers can be started without blocking the caller."""
        return self._executor is not None

    def align_params(self) -> dict:
        """Return ``model.align_params`` as a plain resolved dict."""
        return self.model_cfg.resolved_align_params()

    def setup(self, ctx=None, model_cfg=None, executor=None) -> None:
        """Build the depth/video teachers and attach their side-stream executor.

        ``executor`` is built by the trainer (see
        ``LingbotVlaV2ReplicatedShardedTrainer._build_teacher_executor``)
        and is ``None`` when the teachers run synchronously in the training
        thread.
        """
        if ctx is not None:
            self.ctx = ctx
        if model_cfg is not None and model_cfg is not self.model_cfg:
            self.__init__(model_cfg, self.ctx)
        if executor is not None:
            self._executor = executor
        if not self._use_depth_align:
            return
        from loongforge.embodied.model.lingbot_vla_v2.upstream_adapter import (
            resolve_build_depth_model,
            resolve_build_video_model,
        )

        build_depth_model = resolve_build_depth_model()
        build_video_model = resolve_build_video_model()

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
        if self.ctx is None or self.ctx.is_main:
            logger.info(
                "Teacher targets run on a side stream / worker thread"
                if self._executor is not None
                else "Teacher targets run synchronously in the training thread"
            )

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
        from loongforge.embodied.model.lingbot_vla_v2.upstream_adapter import (
            resolve_get_depth_target,
            resolve_get_video_target,
        )

        get_depth_target = resolve_get_depth_target()
        get_video_target = resolve_get_video_target()

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
        or no executor available).
        """
        if not self._use_depth_align or self._executor is None:
            return None
        return self._executor.submit(self._compute_targets, self._pop_inputs(batch))

    def close(self) -> None:
        """Stop the async executor and drop the teacher models."""
        if self._executor is not None:
            self._executor.close()
        self._executor = None
        self._moge_model = None
        self._morgbd_model = None
        self._video_teacher = None

    # -- look-ahead -----------------------------------------------------------
    #
    # The trainer starts the teachers a step early through ``submit``; everything
    # LingBot-specific about the teachers stays above.

    def submit(self, batch):
        """Start the teachers for ``batch``, or compute them inline.

        ``None`` means the targets are already in ``batch.data``: either the executor
        is unavailable, in which case this blocked and computed them, or there is
        nothing to compute.
        """
        handle = self.enrich_async(batch)
        if handle is None:
            self.enrich(batch)
        return handle

    def prepare_batch(self, policy, batch, teacher=None) -> None:
        """Downcast float inputs, drop unused keys, and attach teacher targets.

        ``teacher`` is the handle of teachers the trainer already started a step
        ago; without one they are started here, which also covers the case where
        the targets are computed inline into ``batch.data``.
        """
        data = batch.data
        data.pop("rep_id", None)
        if policy.training:
            for name in ("state", "actions"):
                value = data.get(name)
                if isinstance(value, torch.Tensor) and value.is_floating_point():
                    data[name] = value.to(torch.bfloat16)

        # Teacher targets are only read by the auxiliary heads, which run after the
        # backbone, so they can be computed concurrently with the student forward.
        # The handle is resolved by the first auxiliary head that reads its targets
        # (see the async-teacher shim in ``upstream_adapter``).
        if teacher is None:
            teacher = self.submit(batch)
        if teacher is not None:
            policy.set_pending_teacher(teacher)


__all__ = ["LingbotVlaV2BatchEnricher"]
