# Copyright 2026 The LoongForge Authors.
# SPDX-License-Identifier: Apache-2.0
#
# Modified from LingBot-VA under the Apache-2.0 License.
# Copyright 2024-2025 The Robbyant Team Authors. All rights reserved.
#
# ``WanVAEStreamingWrapper`` and ``_vae_patchify`` are transcribed from upstream's
# ``wan_va/modules/utils.py``; the causal chunk grouping enforced by ``encode_stream`` and
# the latent normalization follow ``AutoencoderKLWan._encode`` in diffusers (Apache-2.0,
# Copyright 2024 The HuggingFace Inc. team).

"""Frozen Wan2.2 codec helpers for LingBot-VA: VAE latents and UMT5 text embeddings.

LingBot-VA training consumes an *offline* preprocessed dataset whose VAE latents and
T5 text embeddings are already materialized, so the training forward never touches
these sub-models. Online inference has to reproduce that preprocessing, which is what
this module provides.

Everything here is a plain function or a thin stateful wrapper taking explicit
arguments (no model config object), so the same code serves both the offline
preprocessing scripts and the online ``predict_action`` path.

The heavyweight ``diffusers`` / ``transformers`` imports are deferred into the loader
functions: they are inference-only dependencies and must not be paid for by training,
which imports this package's siblings.
"""

from __future__ import annotations

import html
import re

import torch
import torch.nn.functional as F


# ── frozen sub-model loading ─────────────────────────────────────────
# The ~20GB LingBot-VA checkpoint does not bundle the VAE / text encoder; they are
# loaded from ``wan_pretrained_path`` subfolders (``vae``, ``text_encoder``, ``tokenizer``).
def load_vae(vae_path: str, torch_dtype: torch.dtype, torch_device, subfolder: str | None = None):
    """Load the frozen Wan2.2 VAE (``AutoencoderKLWan``, ``z_dim=48``)."""
    from diffusers import AutoencoderKLWan

    vae = AutoencoderKLWan.from_pretrained(vae_path, subfolder=subfolder, torch_dtype=torch_dtype)
    return vae.to(torch_device).eval()


def _retie_input_embeddings(text_encoder) -> None:
    """Alias ``encoder.embed_tokens`` to ``shared`` when the checkpoint only ships the latter.

    The Wan2.2 ``text_encoder`` checkpoint stores a single ``shared.weight`` and declares
    ``tie_word_embeddings=False``. ``transformers`` used to alias ``encoder.embed_tokens``
    onto ``shared`` regardless; newer versions do not, so ``encoder.embed_tokens.weight``
    loads as zeros and the encoder returns an all-zero hidden state without raising.
    """
    shared = getattr(text_encoder, "shared", None)
    embed_tokens = getattr(getattr(text_encoder, "encoder", None), "embed_tokens", None)
    if shared is None or embed_tokens is None:
        return
    if embed_tokens.weight.abs().sum() == 0 and shared.weight.abs().sum() != 0:
        embed_tokens.weight = shared.weight


def load_text_encoder(
    text_encoder_path: str, torch_dtype: torch.dtype, torch_device, subfolder: str | None = None
):
    """Load the frozen UMT5 text encoder (``d_model=4096``)."""
    from transformers import UMT5EncoderModel

    text_encoder = UMT5EncoderModel.from_pretrained(
        text_encoder_path, subfolder=subfolder, torch_dtype=torch_dtype
    )
    _retie_input_embeddings(text_encoder)
    return text_encoder.to(torch_device).eval()


def load_tokenizer(tokenizer_path: str, subfolder: str | None = None):
    """Load the ``T5TokenizerFast`` paired with the UMT5 encoder."""
    from transformers import T5TokenizerFast

    return T5TokenizerFast.from_pretrained(tokenizer_path, subfolder=subfolder)


# ── VAE latent normalization ─────────────────────────────────────────
def normalize_vae_latent(enc_out: torch.Tensor, latents_mean, latents_std) -> torch.Tensor:
    """Take the mean of a VAE encoder output and channel-normalize it.

    ``enc_out`` is the raw ``quant_conv`` output holding ``[mu, logvar]`` stacked on the
    channel axis; only ``mu`` is used (deterministic encoding, matching upstream).
    """
    mu, _logvar = torch.chunk(enc_out, 2, dim=1)
    mean = torch.as_tensor(latents_mean, device=mu.device).view(1, -1, 1, 1, 1)
    inv_std = 1.0 / torch.as_tensor(latents_std, device=mu.device).view(1, -1, 1, 1, 1)
    return ((mu.float() - mean) * inv_std).to(mu)


def denormalize_latents(
    latents: torch.Tensor, latents_mean, latents_std, z_dim: int
) -> torch.Tensor:
    """Inverse of :func:`normalize_vae_latent`, for VAE-decoding predicted latents."""
    mean = torch.as_tensor(latents_mean).view(1, z_dim, 1, 1, 1).to(latents.device, latents.dtype)
    inv_std = (
        1.0 / torch.as_tensor(latents_std).view(1, z_dim, 1, 1, 1).to(latents.device, latents.dtype)
    )
    return latents / inv_std + mean


def _vae_patchify(x: torch.Tensor, patch_size: int | None) -> torch.Tensor:
    """Space-to-channel patchify required by VAE variants that set ``config.patch_size``."""
    if patch_size is None or patch_size == 1:
        return x
    batch_size, channels, frames, height, width = x.shape
    x = x.view(
        batch_size, channels, frames, height // patch_size, patch_size, width // patch_size, patch_size
    )
    x = x.permute(0, 1, 6, 4, 2, 3, 5).contiguous()
    return x.view(
        batch_size, channels * patch_size * patch_size, frames, height // patch_size, width // patch_size
    )


class WanVAEStreamingWrapper:
    """Causal streaming encoder over an ``AutoencoderKLWan``.

    The VAE's temporal downsample is x4, and its ``WanCausalConv3d`` layers carry a
    ``feat_cache`` so consecutive chunks encode as if they were one continuous clip.
    One instance therefore owns the temporal state of *one* video stream and must be
    reset via :meth:`clear_cache` at every episode boundary — a stale cache silently
    conditions the first chunk of a new episode on the tail of the previous one.
    """

    def __init__(self, vae_model):
        """Wrap a loaded VAE, count its causal convs and start with an empty cache.

        The conv count decides the ``feat_cache`` length, so it is read from the
        model's precomputed counts when available and otherwise derived by walking
        the encoder modules.
        """
        self.vae = vae_model
        self.encoder = vae_model.encoder
        self.quant_conv = vae_model.quant_conv

        cached_counts = getattr(self.vae, "_cached_conv_counts", None)
        if cached_counts is not None:
            self.enc_conv_num = cached_counts["encoder"]
        else:
            self.enc_conv_num = sum(
                1 for m in self.encoder.modules() if m.__class__.__name__ == "WanCausalConv3d"
            )
        self.clear_cache()

    def clear_cache(self) -> None:
        """Drop the causal conv state; call once per episode."""
        self.feat_cache = [None] * self.enc_conv_num
        self.frames_seen = 0

    @torch.no_grad()
    def encode_chunk(self, x_chunk: torch.Tensor) -> torch.Tensor:
        """Encode ``[B, C, F, H, W]`` in ``[-1, 1]`` to a raw ``[mu, logvar]`` latent chunk.

        ``F`` must be one causal group (see :meth:`encode_stream`); prefer that method.
        """
        x_chunk = _vae_patchify(x_chunk, getattr(self.vae.config, "patch_size", None))
        feat_idx = [0]
        out = self.encoder(x_chunk, feat_cache=self.feat_cache, feat_idx=feat_idx)
        return self.quant_conv(out)

    @torch.no_grad()
    def encode_stream(self, x: torch.Tensor) -> torch.Tensor:
        """Encode ``[B, C, F, H, W]`` as the next ``F`` frames of this stream.

        Wan's causal encoder only accepts one *group* per :meth:`encode_chunk` call: the
        first group of a stream is exactly 1 frame, every later group exactly 4 (the x4
        temporal downsample, so each group yields one latent frame). Deviating either
        crashes inside ``WanResidualDownBlock``'s ``avg_shortcut`` or, for lengths that
        happen to broadcast, silently returns latents that differ from the reference. This
        method splits ``x`` along those boundaries, tracking the stream position across
        calls so consecutive observation chunks stay aligned.
        """
        total = x.shape[2]
        groups = []
        position = self.frames_seen
        offset = 0
        while offset < total:
            take = 1 if position == 0 else 4 - (position - 1) % 4
            if offset + take > total:
                raise ValueError(
                    f"cannot encode {total} frames from stream position {self.frames_seen}: "
                    f"the trailing group needs {take} frames but only {total - offset} remain "
                    "(the first group of a stream is 1 frame, later groups are 4)"
                )
            groups.append((offset, take))
            offset += take
            position += take
        outputs = [self.encode_chunk(x[:, :, start : start + size]) for start, size in groups]
        self.frames_seen = position
        return outputs[0] if len(outputs) == 1 else torch.cat(outputs, dim=2)


# ── image preprocessing ──────────────────────────────────────────────
def prepare_camera_frame(
    image: torch.Tensor,
    size: tuple[int, int],
    dtype: torch.dtype,
    device,
    hflip: bool = False,
) -> torch.Tensor:
    """Resize and rescale one camera image to a single-frame VAE clip ``[1, C, 1, H, W]``.

    ``image`` is ``[C, H, W]`` or ``[1, C, H, W]`` float in ``[0, 1]``; the output is
    scaled to ``[-1, 1]`` as the VAE expects. ``hflip`` undoes an env-side horizontal
    flip when the eval pipeline applies one.
    """
    if image.dim() == 3:
        image = image.unsqueeze(0)
    image = image.to(device, torch.float32)
    if hflip:
        image = torch.flip(image, dims=[-1])
    image = F.interpolate(image, size=size, mode="bilinear", align_corners=False)
    image = image * 2.0 - 1.0
    return image.unsqueeze(2).to(dtype)  # [1, C, F=1, H, W]


@torch.no_grad()
def encode_frames_width_concat(
    streaming_vae: WanVAEStreamingWrapper,
    frames_per_camera: list[list[torch.Tensor]],
    device,
) -> torch.Tensor:
    """``camera_layout="width_concat"``: encode each camera, then concat latents on width.

    ``frames_per_camera[c]`` is the temporal list of ``[1, C, 1, H, W]`` clips for camera
    ``c`` (all cameras must share ``H``/``W`` and length). Cameras are batched into a
    single :meth:`WanVAEStreamingWrapper.encode_stream` call so the x4 temporal downsample
    collapses ``F`` input frames into ``F / 4`` latent frames under one shared
    ``feat_cache``.
    """
    per_cam_videos = [torch.cat(frames, dim=2) for frames in frames_per_camera]
    videos = torch.cat(per_cam_videos, dim=0)  # [num_cam, C, F, H, W]
    vae_device = next(streaming_vae.vae.parameters()).device
    enc_out = streaming_vae.encode_stream(videos.to(vae_device))
    mu_norm = normalize_vae_latent(
        enc_out, streaming_vae.vae.config.latents_mean, streaming_vae.vae.config.latents_std
    )
    video_latent = torch.cat(mu_norm.split(1, dim=0), dim=-1)
    return video_latent.to(device)


@torch.no_grad()
def encode_frames_tshape(
    streaming_vae: WanVAEStreamingWrapper,
    streaming_vae_half: WanVAEStreamingWrapper,
    head_frames: list[torch.Tensor],
    left_frames: list[torch.Tensor],
    right_frames: list[torch.Tensor],
    device,
) -> torch.Tensor:
    """``camera_layout="robotwin_tshape"``: full-res head latent with half-res wrists above it.

    The two wrist latents go side by side on the width axis and that strip is stacked on
    the height axis on top of the head latent. The wrists are half resolution, so they
    need their *own* streaming wrapper (``streaming_vae_half``) — sharing one
    ``feat_cache`` across two different spatial shapes is invalid.
    """
    vae_device = next(streaming_vae.vae.parameters()).device
    head = torch.cat(head_frames, dim=2)
    wrists = torch.cat([torch.cat(left_frames, dim=2), torch.cat(right_frames, dim=2)], dim=0)
    enc_high = streaming_vae.encode_stream(head.to(vae_device))
    enc_lr = streaming_vae_half.encode_stream(wrists.to(vae_device))
    enc_out = torch.cat([torch.cat(enc_lr.split(1, dim=0), dim=-1), enc_high], dim=-2)
    video_latent = normalize_vae_latent(
        enc_out, streaming_vae.vae.config.latents_mean, streaming_vae.vae.config.latents_std
    )
    return video_latent.to(device)


# ── text encoding ────────────────────────────────────────────────────
def clean_prompt(text: str) -> str:
    """Normalize a task prompt (HTML-unescape + whitespace collapse).

    Mirrors diffusers' Wan ``prompt_clean`` minus ``ftfy.fix_text``, which is a no-op for
    the ASCII task strings used here, so the extra ``ftfy`` dependency is avoided.
    """
    text = html.unescape(html.unescape(text)).strip()
    return re.sub(r"\s+", " ", text).strip()


@torch.no_grad()
def encode_text(
    tokenizer,
    text_encoder,
    prompts: str | list[str],
    max_sequence_length: int,
    dtype: torch.dtype,
    device,
) -> torch.Tensor:
    """Encode task prompts to UMT5 embeddings ``[B, max_sequence_length, 4096]``.

    Padding tokens are zeroed rather than left as encoder output: the embeddings are
    truncated to each prompt's true length and right-padded with zeros, matching the
    offline preprocessing so inference sees the same conditioning as training.
    """
    if isinstance(prompts, str):
        prompts = [prompts]
    prompts = [clean_prompt(p) for p in prompts]

    text_inputs = tokenizer(
        prompts,
        padding="max_length",
        max_length=max_sequence_length,
        truncation=True,
        add_special_tokens=True,
        return_attention_mask=True,
        return_tensors="pt",
    )
    input_ids, mask = text_inputs.input_ids, text_inputs.attention_mask
    seq_lens = mask.gt(0).sum(dim=1).long()

    te_device = next(text_encoder.parameters()).device
    embeds = text_encoder(input_ids.to(te_device), mask.to(te_device)).last_hidden_state
    if not embeds.any():
        raise RuntimeError(
            "the text encoder returned an all-zero hidden state; its token embedding was "
            "most likely left uninitialized (see _retie_input_embeddings). Conditioning on "
            "zeros would run to completion while ignoring the task instruction."
        )
    embeds = embeds.to(dtype=dtype, device=device)
    embeds = [e[:n] for e, n in zip(embeds, seq_lens, strict=False)]
    embeds = torch.stack(
        [torch.cat([e, e.new_zeros(max_sequence_length - e.size(0), e.size(1))]) for e in embeds],
        dim=0,
    )
    return embeds.to(device)
