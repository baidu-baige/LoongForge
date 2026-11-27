# Copyright 2026 The LoongForge Authors.
# SPDX-License-Identifier: Apache-2.0

"""``dual_stream_wan`` family shell: pixels in, joint attention, Wan towers.

Three models sit in this family (``vpp2``, ``fastwam``, ``internw0_delta``),
and everything they share above the spec layer is here:

* the engine, ``JointAttentionEngine``, because equal-depth streams attending
  jointly per layer *is* this family's topology;
* the batch path, which VAE-encodes pixel video into the latents the video
  stream reads and materialises the three conditions that depend on the
  model's own geometry;
* the two frozen Wan towers, built from the ``source_encoder`` the spec
  declares next to the key each one feeds.

The weight path lives on this family's config rather than on
``ArchModelConfig``: "where do the frozen towers come from" is only a question
for a family that has frozen towers, and a shared field would have to be
filled in by families that have none.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Mapping, Optional, Tuple

import torch
from torch import Tensor, nn

from ...builder import WorldModel
from ...spec.core import build
from ...spec.world import WorldModelSpec
from ...streams.conditions import InjectionKind
from ...streams.engine import JointAttentionEngine
from ..base import ArchModelBase, ArchModelConfig

__all__ = ["DualStreamWanBase", "DualStreamWanConfig"]


@dataclass(frozen=True)
class DualStreamWanConfig(ArchModelConfig):
    """Shared ``model:`` section for the ``dual_stream_wan`` family.

    Attributes:
        model_id: Repo or directory holding the VAE and CLIP image encoder.
            The streams are built from the spec, so this is the *tower* stack
            only; fine-tuned DiT weights arrive via ``--pretrained-checkpoint``.
        compile_vae_encode: Wrap the VAE encode call in ``torch.compile``.
    """

    model_id: str = "Wan-AI/Wan2.1-I2V-14B-480P"
    compile_vae_encode: bool = False


class DualStreamWanBase(ArchModelBase):
    """Shell for the dual-stream Wan family.

    Attributes:
        vae: Frozen video VAE turning pixel frames into latents.
        image_encoder: Frozen CLIP encoder feeding the ``clip_image`` slot.
            ``None`` when the spec declares no such condition.
    """

    engine_cls: Optional[type] = JointAttentionEngine
    config_cls: type = DualStreamWanConfig

    def __init__(self, config: DualStreamWanConfig, core: WorldModel, *,
                 vae: nn.Module, image_encoder: Optional[nn.Module]) -> None:
        """Wire a built core and its frozen towers into one module.

        Args:
            config: Typed config instance for this model.
            core: Assembled ``WorldModel``.
            vae: Frozen video VAE.
            image_encoder: Frozen CLIP encoder, or ``None``.
        """
        super().__init__(config, core)
        self.vae = vae
        self.image_encoder = image_encoder

    @classmethod
    def build_towers(cls, cfg: Any, spec: WorldModelSpec) -> Dict[str, Any]:
        """Build the two frozen Wan towers the spec declares.

        Which towers exist is read off the spec rather than decided here: the
        video stream's ``source_encoder`` is the VAE and ``clip_image``'s is
        the CLIP encoder, and a model without the CLIP condition gets ``None``
        without this method knowing which model that is.

        Args:
            cfg: The typed config; ``model_id`` is the weight root.
            spec: The validated spec.

        Returns:
            ``{"vae": ..., "image_encoder": ...}``.

        Raises:
            ValueError: The video stream declares no ``source_encoder``, which
                would leave the family with no way to reach latents.
        """
        video = next((s for s in spec.streams if s.name == "video"), None)
        if video is None or video.source_encoder is None:
            raise ValueError(
                f"model {spec.model_type!r} is in the dual_stream_wan family "
                "but its 'video' stream declares no source_encoder; the "
                "family reads pixels and cannot reach latents without one")
        clip = next((c for c in spec.conditions
                     if c.name == "clip_image" and c.enabled), None)
        return {
            "vae": build(video.source_encoder, spec=spec, path=cfg.model_id),
            "image_encoder": (
                None if clip is None or clip.source_encoder is None
                else build(clip.source_encoder, path=cfg.model_id)),
        }

    def forward(self, batch: Any, iteration: Optional[int] = None
                ) -> Tuple[Tensor, Dict[str, Any]]:
        """Run one training step. Interface I4: ``(loss, metrics)``.

        Args:
            batch: Either a ``FastWAMPreparedBatch`` or a plain dict with the
                same keys (``video``, ``action``, ``context``, ``proprio``).
            iteration: Optimizer steps completed before this one, from the
                trainer; it becomes the core's ``global_step``
                (``ArchModelBase._sample_at_step``).

        Returns:
            The weighted total loss and the per-term metrics dict.

        Raises:
            ValueError: The sample already carries a different
                ``global_step``.
        """
        return self.core(self.encode_batch(self._sample_at_step(batch, iteration)))

    def encode_batch(self, sample: Mapping[str, Any]) -> Dict[str, Any]:
        """Map a pixel-space sample onto the keys the spec declares.

        Three kinds of keys come out: one raw tensor per stream, named by
        ``StreamSpec.batch_key``; one per enabled ``ConditionSpec``, named by
        its ``source_key``; and the masks the spec declares, passed through
        as bool when the sample has them (see ``ArchModelBase._mask_keys``).
        All three lists are read off the spec, so adding a condition to a
        provider does not require editing this method unless the new
        condition needs an encoder that does not exist yet.

        Args:
            sample: Collated batch in pixel space. ``video`` is
                ``[B, 3, T, H, W]`` in ``[-1, 1]``; ``action`` is ``[B, T, D]``.

        Returns:
            The batch ``WorldModel.forward`` consumes.

        Raises:
            KeyError: The sample lacks a key some stream or condition needs.
        """
        video = self._require(sample, "video").to(dtype=self._dtype)
        latents = self.encode_video(video)
        out: Dict[str, Any] = {}
        # Passed on only when given: the core decides what a missing step
        # means (0 in evaluation, an error when training a ramped loss).
        if "global_step" in sample:
            out["global_step"] = int(sample["global_step"])
        for stream in self.spec.streams:
            if stream.name == "video":
                out[stream.batch_key] = latents
            else:
                out[stream.batch_key] = self._require(
                    sample, stream.name).to(device=latents.device,
                                            dtype=self._dtype)
        for cond in self.spec.conditions:
            if not cond.enabled:
                continue
            out[cond.source_key] = self._condition_value(cond, sample, video,
                                                         latents)
        for key in self._mask_keys():
            mask = sample.get(key)
            if mask is not None:
                out[key] = mask.to(device=latents.device, dtype=torch.bool,
                                   non_blocking=True)
        for key in self._target_keys():
            target = sample.get(key)
            if target is not None:
                dtype = self._dtype if target.is_floating_point() else target.dtype
                out[key] = target.to(device=latents.device, dtype=dtype,
                                     non_blocking=True)
        return out

    @torch.no_grad()
    def encode_video(self, video: Tensor) -> Tensor:
        """VAE-encode pixel frames into the clean latents the video stream reads.

        Runs under ``no_grad`` and stays out of the autograd graph: the VAE is
        frozen, and the video stream's loss is computed on latents, so no
        gradient needs to reach the pixels.

        Args:
            video: ``[B, 3, T, H, W]`` in ``[-1, 1]``.

        Returns:
            ``[B, latent_channels, f, h, w]`` matching ``geometry.latent_grid``.

        Raises:
            ValueError: The encoded grid disagrees with the spec's geometry,
                which means the data pipeline and the config describe different
                videos.
        """
        latents = self.vae.encode(video, tiled=False).to(dtype=self._dtype)
        expected = self.spec.geometry.latent_grid
        if tuple(latents.shape[2:]) != expected:
            raise ValueError(
                f"VAE produced a {tuple(latents.shape[2:])} latent grid while "
                f"geometry says {expected}. Check num_frames / frame_height / "
                f"frame_width / num_views against the dataset, not the VAE")
        return latents

    def _condition_value(self, cond: Any, sample: Mapping[str, Any],
                         video: Tensor, latents: Tensor) -> Tensor:
        """Materialise the raw input of one condition slot.

        Three conditions are produced here instead of by the data pipeline,
        each because it depends on the model's own geometry: the CLIP feature
        (the tower is a model weight), the i2v channel-concat block (its
        channel count is ``4 + latent_channels``) and the proprioception vector
        (sliced to the single step the token encoder expects). Everything else
        is taken from the sample as-is.

        Args:
            cond: The ``ConditionSpec`` being materialised.
            sample: Collated batch in pixel space.
            video: Pixel video, already cast to the model dtype.
            latents: Clean video latents from :meth:`encode_video`.

        Returns:
            A tensor shaped as the condition's consumer expects.

        Raises:
            KeyError: A pass-through condition has no value in the sample.
        """
        if cond.name == "clip_image":
            return self.encode_first_frame(video)
        if cond.kind is InjectionKind.CHANNEL_CONCAT:
            return self._i2v_channels(cond, latents)
        value = self._require(sample, cond.source_key).to(
            device=latents.device, dtype=self._dtype)
        if cond.encoder is not None and value.ndim == 3:
            # The token encoder takes ``[B, source_width]``; the collator
            # repeats proprioception over the action horizon, so take the step
            # the official implementation takes — the first one.
            value = value[:, 0, :]
        return value

    @torch.no_grad()
    def encode_first_frame(self, video: Tensor) -> Tensor:
        """CLIP-encode the first pixel frame into the ``clip_image`` slot.

        The encoder iterates its argument and interpolates each element, so it
        has to be handed ``[B, 1, 3, H, W]`` — one ``[1, 3, H, W]`` image per
        sample — and not a ``[B, 3, T, H, W]`` video.

        Args:
            video: ``[B, 3, T, H, W]`` in ``[-1, 1]``.

        Returns:
            ``[B, 257, 1280]`` CLIP patch features.

        Raises:
            RuntimeError: The spec declares ``clip_image`` but no encoder was
                built, which can only happen through direct construction.
        """
        if self.image_encoder is None:
            raise RuntimeError(
                "the spec declares a 'clip_image' condition but this instance "
                "was built without an image encoder")
        first = video[:, :, :1].transpose(1, 2)
        return self.image_encoder.encode_image(first).to(dtype=self._dtype)

    def _i2v_channels(self, cond: Any, latents: Tensor) -> Tensor:
        """Build the image-to-video channel-concat block.

        Layout, matching the upstream Wan i2v input: four mask channels that
        are 1 on the conditioned leading latent frames and 0 elsewhere,
        followed by ``latent_channels`` channels holding the clean latents of
        those same frames and zeros after them. The mask is *generated* from
        the frame count rather than carried in the batch, which is why this is
        one condition of width ``4 + latent_channels`` and not two.

        Args:
            cond: The channel-concat ``ConditionSpec``; its ``width`` is the
                authority on the total channel count.
            latents: Clean video latents ``[B, C, f, h, w]``.

        Returns:
            ``[B, cond.width, f, h, w]``.

        Raises:
            ValueError: ``cond.width`` is not ``4 + latent_channels``, or more
                frames are conditioned than exist.
        """
        channels = latents.shape[1]
        if cond.width != 4 + channels:
            raise ValueError(
                f"condition {cond.name!r} declares width={cond.width}, but the "
                f"i2v block is 4 mask channels + {channels} latent channels = "
                f"{4 + channels}")
        frames = getattr(self.config, "condition_frames", None) or 1
        if frames > latents.shape[2]:
            raise ValueError(
                f"condition_frames={frames} exceeds the {latents.shape[2]} "
                "latent frames the geometry produces")
        mask = latents.new_zeros((latents.shape[0], 4, *latents.shape[2:]))
        mask[:, :, :frames] = 1.0
        clean = torch.zeros_like(latents)
        clean[:, :, :frames] = latents[:, :, :frames]
        return torch.cat([mask, clean], dim=1)
