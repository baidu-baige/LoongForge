# Copyright 2026 The LoongForge Authors.
# SPDX-License-Identifier: Apache-2.0

"""Training gate for the five-layer ``arch`` world models.

``WorldModel`` is pure assembly: it takes a batch of *latents* and returns
``(loss, metrics)``. The embodied trainer, on the other hand, hands the model a
batch of *pixels* produced by ``FastWAMPreprocessor``. ``ArchWorldModel`` is the
one object that bridges those two: it owns the frozen encoder stack (video VAE
and CLIP image encoder), turns the pixel batch into the keys each
``ConditionSpec`` declares, and forwards to ``WorldModel``.

Why one shell for every arch model instead of one per model: nothing in here is
model-specific. Which conditions exist, how wide they are and which batch key
feeds them are all read off the built ``WorldModelSpec``, so a new provider
becomes a new registry row and a new YAML file, not a new ``nn.Module``.

The registry keys are the ``model_type`` values of the config subclasses, which
is also what ``--model-name`` resolves to. ``fastwam`` is deliberately absent:
that name already belongs to ``FastWAMPolicy``, and two classes claiming one
registry key would make the winner depend on import order.
"""

from __future__ import annotations

import logging
import os
from typing import Any, Callable, Dict, Mapping, Optional, Tuple

import torch
from torch import Tensor, nn

from loongforge.embodied.model.registry import register_model

from .builder import WorldModel
from .layout import TokenLayout
from .model_configuration_arch import WorldModelConfig
from .spec.world import WorldModelSpec
from .streams.conditions import InjectionKind
from .streams.engine import JointAttentionEngine
from .transfer import vpp2_to_arch

logger = logging.getLogger(__name__)

__all__ = ["ArchWorldModel"]

_DTYPES: Dict[str, torch.dtype] = {
    "bf16": torch.bfloat16, "bfloat16": torch.bfloat16,
    "fp16": torch.float16, "float16": torch.float16, "half": torch.float16,
    "fp32": torch.float32, "float32": torch.float32, "full": torch.float32,
}

# The VAE generation is selected by latent width rather than by a YAML string.
# The two generations differ in channel count *and* spatial stride, and
# ``GeometrySpec`` already carries both numbers, so a separate preset name in
# the config would be a second place for the same fact to live.
_VAE_PRESETS: Dict[int, Tuple[str, int]] = {
    16: ("Wan2.1_VAE.pth", 8),      # Wan2.1: 16 latent channels, /8 spatially
    48: ("Wan2.2_VAE.pth", 16),     # Wan2.2: 48 latent channels, /16
}

_CLIP_FILENAME = "models_clip_open-clip-xlm-roberta-large-vit-huge-14.pth"

# ``vpp2_to_arch`` is a per-model rename table, not a generic loader: it encodes
# where that model's checkpoint put the output heads and the condition
# projectors. A model without a table must say so rather than silently load
# nothing, which would look like "trained from scratch with a warm start".
_TRANSFERS: Dict[str, Callable[[Mapping[str, Any]], Dict[str, Any]]] = {
    "vpp2": vpp2_to_arch,
}


def _resolve_dtype(name: str) -> torch.dtype:
    """Map a config dtype string to a ``torch.dtype``.

    Args:
        name: Value of ``WorldModelConfig.dtype``.

    Returns:
        The matching torch dtype.

    Raises:
        ValueError: The alias is not one of the recognised spellings.
    """
    try:
        return _DTYPES[name.lower()]
    except KeyError:
        raise ValueError(
            f"unsupported arch dtype {name!r}; expected one of "
            f"{sorted(_DTYPES)}") from None


def _optional_file(root: str, filename: str) -> Optional[str]:
    """Resolve ``root`` to a concrete weight file, or ``None`` if absent.

    ``model_id`` may be a HuggingFace repo id, a local Wan checkpoint directory
    or a direct path to one weight file. Only the last two can be loaded from
    disk; a repo id that has not been downloaded yet leaves the encoder
    randomly initialised, which is what ``--dry_run`` and unit tests need.

    Args:
        root: Directory or file path from the config.
        filename: File looked for when ``root`` is a directory.

    Returns:
        An existing file path, or ``None`` when nothing was found.
    """
    if not root:
        return None
    if os.path.isfile(root):
        return root
    candidate = os.path.join(root, filename)
    return candidate if os.path.isfile(candidate) else None


def _build_vae(spec: WorldModelSpec, model_id: str) -> nn.Module:
    """Construct the frozen video VAE matching the spec's latent geometry.

    Args:
        spec: Built world-model spec; its geometry fixes both the latent
            channel count and the spatial stride the VAE must have.
        model_id: Directory (or file) holding the Wan encoder weights.

    Returns:
        A frozen, eval-mode VAE module.

    Raises:
        ValueError: The geometry asks for a latent width no Wan VAE produces,
            or the declared spatial stride disagrees with that generation's.
    """
    from loongforge.embodied.model.dreamzero.modules.wan_video_vae import (
        WanVideoVAE, WanVideoVAE38)

    channels = spec.geometry.latent_channels
    if channels not in _VAE_PRESETS:
        raise ValueError(
            f"geometry.latent_channels={channels} matches no Wan VAE; "
            f"known generations produce {sorted(_VAE_PRESETS)} channels")
    filename, stride = _VAE_PRESETS[channels]
    if spec.geometry.vae_spatial_stride != stride:
        raise ValueError(
            f"geometry says vae_spatial_stride="
            f"{spec.geometry.vae_spatial_stride}, but a {channels}-channel "
            f"Wan VAE compresses by {stride}. The latent grid would be off by "
            f"{stride / spec.geometry.vae_spatial_stride:g}x in each spatial "
            "axis, which shows up as a token-count mismatch, not as a "
            "shape error")
    weights = _optional_file(model_id, filename)
    factory = WanVideoVAE if channels == 16 else WanVideoVAE38
    vae = factory(z_dim=channels, vae_pretrained_path=weights)
    _load_into(getattr(vae, "model", vae), weights, "VAE")
    return vae.eval().requires_grad_(False)


def _build_image_encoder(model_id: str) -> nn.Module:
    """Construct the frozen CLIP image encoder feeding the ``clip_image`` slot.

    Args:
        model_id: Directory (or file) holding the Wan encoder weights.

    Returns:
        A frozen, eval-mode ``WanImageEncoder``.
    """
    from loongforge.embodied.model.dreamzero.modules.wan_video_image_encoder \
        import WanImageEncoder

    weights = _optional_file(model_id, _CLIP_FILENAME)
    encoder = WanImageEncoder(image_encoder_pretrained_path=weights)
    _load_into(getattr(encoder, "model", encoder), weights, "CLIP")
    return encoder.eval().requires_grad_(False)


def _load_into(module: nn.Module, weights: Optional[str], what: str) -> None:
    """Load a plain state dict into a frozen encoder, logging the key counts.

    Load the exact VAE trunk or CLIP visual tower strictly. Published CLIP files
    also carry an unused text tower, which is removed by namespace first.

    Args:
        module: Destination module (the wrapper's inner trunk).
        weights: Path to a ``torch.save``d state dict, or ``None`` to skip.
        what: Component name used in the log line.
    """
    if weights is None:
        return
    state = torch.load(weights, map_location="cpu", mmap=True, weights_only=True)
    for key in ("state_dict", "model_state"):
        if isinstance(state, dict) and key in state:
            state = state[key]
    if what == "CLIP":
        state = {key: value for key, value in state.items()
                 if key.startswith("visual.") or key == "log_scale"}
    missing, unexpected = module.load_state_dict(state, strict=True)
    logger.info("[arch] %s loaded from %s; missing=%d unexpected=%d",
                what, weights, len(missing), len(unexpected))


@register_model("vpp2")
@register_model("internw0_delta")
class ArchWorldModel(nn.Module):
    """Trainer-facing wrapper around one ``WorldModel``.

    Attributes:
        config: The typed config the model was built from. Kept because
            ``load_pretrained`` needs ``model_type`` to pick a rename table and
            because the condition builders read ``condition_frames``.
        core: The assembled ``WorldModel``. Every trainable parameter lives
            here or in nothing at all; the encoders below are frozen.
        vae: Frozen video VAE turning pixel frames into latents.
        image_encoder: Frozen CLIP encoder feeding the ``clip_image`` slot.
            ``None`` when the spec declares no such condition.
    """

    def __init__(self, config: WorldModelConfig, core: WorldModel, *,
                 vae: nn.Module, image_encoder: Optional[nn.Module]) -> None:
        """Wire a built core and its frozen encoders into one module.

        The encoders are injected rather than built here so that tests (and
        any caller without CUDA) can pass stubs: ``WanVideoVAE.__init__``
        allocates its normalisation statistics straight onto ``cuda``.

        Args:
            config: Typed config instance for this model.
            core: Assembled ``WorldModel``.
            vae: Frozen video VAE.
            image_encoder: Frozen CLIP encoder, or ``None``.
        """
        super().__init__()
        self.config = config
        self.core = core
        self.vae = vae
        self.image_encoder = image_encoder
        self._dtype = _resolve_dtype(config.dtype)

    @property
    def spec(self) -> WorldModelSpec:
        """The built spec. Single source for geometry, conditions and losses."""
        return self.core.spec

    @classmethod
    def from_pretrained(cls, cfg: Any) -> "ArchWorldModel":
        """Build the model from a typed config (the ``build_model`` entry point).

        Four steps, in order: build the spec, assemble the core on the spec's
        own attention backend, build the frozen encoders the spec's conditions
        call for, then cast the whole thing to the configured dtype.

        The attention backend comes from the spec rather than from a separate
        argument because ``WorldModelSpec`` already validated the
        backend/block-size pair; re-deciding it here would allow a run whose
        engine and whose masks disagree.

        Args:
            cfg: A ``WorldModelConfig`` subclass instance.

        Returns:
            A ready-to-train ``ArchWorldModel``.

        Raises:
            TypeError: ``cfg`` is not a ``WorldModelConfig``.
        """
        if not isinstance(cfg, WorldModelConfig):
            raise TypeError(
                "ArchWorldModel.from_pretrained expects a WorldModelConfig "
                f"subclass instance; got {type(cfg).__name__}")
        spec = cfg.build_spec()
        action_stage = (
            cfg.model_type == "vpp2"
            and getattr(cfg, "training_stage", "joint") == "action"
        )
        core_cls = WorldModel
        if action_stage:
            from .vpp2_training import VPP2TrainingWorldModel

            core_cls = VPP2TrainingWorldModel
        core = core_cls(
            spec,
            engine=JointAttentionEngine(backend=spec.attention_backend,
                                        block_size=spec.attention_block_size,
                                        frozen_streams={"video"} if action_stage else (),
                                        checkpoint_layers=getattr(
                                            cfg, "gradient_checkpointing", False)),
            layout=TokenLayout.from_geometry(spec.geometry, spec.streams))
        core.to(dtype=_resolve_dtype(cfg.dtype))
        if getattr(cfg, "video_checkpoint_path", None):
            from .transfer import load_vpp2_initializers

            report = load_vpp2_initializers(
                core, cfg.video_checkpoint_path, cfg.action_dit_pretrained_path)
            logger.info("[arch] VPP2 split initialization: %s", report)
        names = {cond.name for cond in spec.conditions if cond.enabled}
        model = cls(cfg, core,
                    vae=_build_vae(spec, cfg.model_id),
                    image_encoder=(_build_image_encoder(cfg.model_id)
                                   if "clip_image" in names else None))
        if action_stage:
            # Match master: VAE fp32, CLIP fp16, trainable DiT bf16.
            model.vae.to(dtype=torch.float32)
            if model.image_encoder is not None:
                model.image_encoder.to(dtype=torch.float16)
            return model
        return model.to(dtype=_resolve_dtype(cfg.dtype))

    def train(self, mode: bool = True) -> "ArchWorldModel":
        super().train(mode)
        self.vae.eval()
        if self.image_encoder is not None:
            self.image_encoder.eval()
        if getattr(self.config, "training_stage", None) == "action":
            self.core.blocks["video"].eval()
        return self

    def forward(self, batch: Any) -> Tuple[Tensor, Dict[str, Any]]:
        """Run one training step. Interface I4: ``(loss, metrics)``.

        Args:
            batch: Either a ``FastWAMPreparedBatch`` or a plain dict with the
                same keys (``video``, ``action``, ``context``, ``proprio``).

        Returns:
            The weighted total loss and the per-term metrics dict.
        """
        sample = batch.to_sample() if hasattr(batch, "to_sample") else dict(batch)
        return self.core(self.encode_batch(sample))

    def encode_batch(self, sample: Mapping[str, Any]) -> Dict[str, Any]:
        """Map a pixel-space sample onto the keys the spec declares.

        Two kinds of keys come out: one raw tensor per stream, named by
        ``StreamSpec.batch_key``, and one per enabled ``ConditionSpec``, named
        by its ``source_key``. Both lists are read off the spec, so adding a
        condition to a provider does not require editing this method unless the
        new condition needs an encoder that does not exist yet.

        Args:
            sample: Collated batch in pixel space. ``video`` is
                ``[B, 3, T, H, W]`` in ``[-1, 1]``; ``action`` is ``[B, T, D]``.

        Returns:
            The batch ``WorldModel.forward`` consumes.

        Raises:
            KeyError: The sample lacks a key some stream or condition needs.
        """
        video = self._require(sample, "video").to(dtype=self._dtype)
        action_stage = (
            self.config.model_type == "vpp2"
            and getattr(self.config, "training_stage", "joint") == "action"
        )
        if action_stage and self.config.action_only_first_frame_fast_path:
            expected = self.spec.geometry
            if tuple(video.shape[2:]) != (
                    expected.num_frames, expected.frame_height,
                    expected.frame_width * expected.num_views):
                raise ValueError("VPP2 pixel video disagrees with configured geometry")
            with torch.no_grad(), torch.autocast(video.device.type, enabled=False):
                first = self.vae.encode(video[:, :, :1].float(), tiled=False)
            if tuple(first.shape[2:]) != (1, *expected.latent_grid[1:]):
                raise ValueError("VPP2 first-frame VAE grid disagrees with geometry")
            latents = first.new_zeros(
                (first.shape[0], first.shape[1], *expected.latent_grid))
            latents[:, :, :1] = first
            latents = latents.to(dtype=self._dtype)
        else:
            latents = self.encode_video(video)
        out: Dict[str, Any] = {
            "global_step": int(sample.get("global_step", 0)),
        }
        if action_stage:
            for name in ("action_is_pad", "action_terminal_mask", "image_is_pad"):
                if sample.get(name) is not None:
                    out[name] = sample[name].to(latents.device)
        # These optional values are already in joint-token order. Pixel/latent
        # padding conversion belongs to the adapter that creates them.
        if "attention_key_valid" in sample:
            out["attention_key_valid"] = sample["attention_key_valid"].to(latents.device)
        if "attention_labels" in sample:
            out["attention_labels"] = {
                name: value.to(latents.device)
                for name, value in sample["attention_labels"].items()
            }
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
        return out

    @staticmethod
    def _require(sample: Mapping[str, Any], key: str) -> Tensor:
        """Fetch a required tensor, naming the available keys when it is absent.

        Args:
            sample: Collated batch.
            key: Key the caller needs.

        Returns:
            The tensor at ``key``.

        Raises:
            KeyError: ``key`` is missing or ``None``.
        """
        value = sample.get(key)
        if value is None:
            raise KeyError(
                f"arch batch needs {key!r}, but the sample only has "
                f"{sorted(k for k, v in sample.items() if v is not None)}")
        return value

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
        (the encoder is a model weight), the i2v channel-concat block (its
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

    def load_pretrained(self, path: str, device: Any = None) -> Optional[int]:
        """Load upstream weights into the core (the trainer's warm-start hook).

        The checkpoint is remapped before loading: upstream keeps the output
        heads inside the experts and names the condition projectors after the
        modality they happened to carry, while this architecture keeps heads in
        ``heads.*`` and projectors under their slot name. The rename table is
        per-model and lives in ``transfer.py``.

        ``strict=True``: a warm start that silently leaves half the backbone
        randomly initialised trains, converges slowly and looks like a
        hyper-parameter problem.

        Args:
            path: Checkpoint file written by the upstream implementation.
            device: Optional device to move the model to afterwards.

        Returns:
            The optimizer step recorded in the checkpoint, if it has one.

        Raises:
            NotImplementedError: No rename table is registered for this model.
        """
        transfer = _TRANSFERS.get(self.config.model_type)
        if transfer is None:
            raise NotImplementedError(
                f"no checkpoint rename table for model_type="
                f"{self.config.model_type!r}; add one to "
                f"loongforge/embodied/model/arch/transfer.py next to "
                f"{sorted(_TRANSFERS)}")
        payload = torch.load(path, map_location="cpu", weights_only=True)
        self.core.load_state_dict(transfer(payload), strict=True)
        logger.info("[arch] core weights loaded from %s", path)
        if device is not None:
            self.to(device)
        step = payload.get("step") if isinstance(payload, dict) else None
        return int(step) if isinstance(step, int) else None
