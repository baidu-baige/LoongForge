# Copyright 2026 The LoongForge Authors.
# SPDX-License-Identifier: Apache-2.0

"""L0 geometry: the single source of token-grid arithmetic (Lane A).

Signatures and invariants follow ``docs/design/mot_layer_interfaces.zh-CN.md``
§3.1; the field list follows ``mot_spec_config.zh-CN.md`` §3.3 (the two have
already been reconciled).
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import torch
from torch import Tensor

__all__ = ["GeometrySpec"]


def _exact_div(numerator: int, denominator: int, *, what: str) -> int:
    """Exact division; a non-exact division raises at construction time.

    Silently truncating would push a wrong grid size into every downstream
    shape instead of failing where the bad config was written.

    Args:
        numerator: Dividend from the pixel-to-token chain.
        denominator: Divisor, must be positive.
        what: Human-readable name of the division, used in the error message.

    Returns:
        The exact quotient.

    Raises:
        ValueError: If ``denominator`` is not positive or the division is not
            exact.
    """
    if denominator <= 0:
        raise ValueError(
            f"GeometrySpec.{what}: divisor must be positive, got {denominator}"
        )
    if numerator % denominator != 0:
        raise ValueError(
            f"GeometrySpec.{what}: {numerator} is not divisible by "
            f"{denominator}. Every division along the chain "
            f"`pixel -> VAE -> latent -> patch` must be exact"
        )
    return numerator // denominator


@dataclass(frozen=True)
class GeometrySpec:
    """The single source of token-grid arithmetic.

    Responsibility: collapse the chain "pixel frame -> VAE compression ->
    latent grid -> patch -> tokens per frame -> sequence length -> alignment
    padding" into one object. No arithmetic expression from that chain may
    appear anywhere outside this class.

    Invariants:
      - every derived quantity is a property, never a stored field, so there
        is never a second copy of the truth to drift;
      - every division along the chain must be exact; a non-exact one raises
        at construction time;
      - the class holds no tensors, so it is cheap to rebuild, hashable, and
        safe to put into a config fingerprint.
    """

    frame_height: int                   # Pixel frame height.
    frame_width: int                    # Pixel frame width (kept separate
                                        # from height: inputs may be
                                        # non-square).
    num_frames: int                     # Video frame count (time axis,
                                        # before compression).
    num_views: int = 1                  # Number of cameras. VPP2's
                                        # concat_multi_camera=horizontal => 2
    vae_spatial_stride: int = 16        # VAE spatial compression ratio; must
                                        # stay separate from the temporal one
    vae_temporal_stride: int = 4        # VAE temporal compression ratio
    latent_channels: int = 16           # VAE latent channel count. The video
                                        # stream's **own** share of the input
                                        # and output channels both take this
                                        # value --
                                        # ``HeadSpec("video_flow").output_width``
                                        # takes it as well, so both ends have
                                        # one source.
                                        # The current ``in_dim=36`` is not an
                                        # independent number:
                                        # 36 = 16 + 4(mask) + 16(condition
                                        # frame latent); the last two are the
                                        # channel counts of CHANNEL_CONCAT
                                        # conditions, declared by
                                        # ``ConditionSpec.width`` => this
                                        # field covers only the latent
                                        # itself, and the concatenated width
                                        # is a derived quantity
    patch_size: tuple[int, int, int] = (1, 2, 2)   # (t, h, w)
    action_horizon: int = 0             # Predicted action steps. 0 => this
                                        # model has no action stream
    action_video_freq_ratio: int = 1    # Action steps per video latent frame
    action_dim: int = 0                 # **Packed** width of the action
                                        # vector
                                        # = raw DoF x action_chunk_size
                                        # (VPP2 libero: 7 x 1 = 7).
                                        # Same role as latent_channels: the
                                        # action stream entry
                                        # ``action_encoder`` input width and
                                        # ``HeadSpec("action_flow")
                                        # .output_width`` both take this
                                        # value, one source for both ends.
                                        # 0 => this model has no action
                                        # stream (tied to action_horizon,
                                        # see __post_init__)
    align_to: int = 1                   # Sequence-length alignment
                                        # granularity (ownership still
                                        # undecided, master plan §10-1)

    def __post_init__(self) -> None:
        positive = (
            "frame_height", "frame_width", "num_frames", "num_views",
            "vae_spatial_stride", "vae_temporal_stride",
            "action_video_freq_ratio", "align_to", "latent_channels",
        )
        for field_name in positive:
            value = getattr(self, field_name)
            if value <= 0:
                raise ValueError(
                    f"GeometrySpec.{field_name} must be positive, got {value}"
                )
        if self.action_horizon < 0:
            raise ValueError(
                f"GeometrySpec.action_horizon must not be negative, got "
                f"{self.action_horizon}"
            )
        if self.action_dim < 0:
            raise ValueError(
                f"GeometrySpec.action_dim must not be negative, got "
                f"{self.action_dim}"
            )
        # Both action fields must be set or unset together: horizon without
        # dim leaves the action stream with no entry and no head; dim without
        # horizon builds a stream with zero tokens.
        if (self.action_horizon == 0) != (self.action_dim == 0):
            raise ValueError(
                f"GeometrySpec: action_horizon={self.action_horizon} and "
                f"action_dim={self.action_dim} must both be 0 or both be "
                "positive. The first is the action stream's token count, the "
                "second is its entry/exit width"
            )
        if len(self.patch_size) != 3 or any(p <= 0 for p in self.patch_size):
            raise ValueError(
                f"GeometrySpec.patch_size must be three positive integers "
                f"(t, h, w), got {self.patch_size!r}"
            )
        # Force the whole chain's exactness checks now: errors must surface at
        # construction time, not at the first property access.
        _ = self.patch_grid
        _ = self.num_chunks

    # ---- Derived quantities: all properties, never recomputed elsewhere ----

    @property
    def latent_grid(self) -> tuple[int, int, int]:
        """The latent grid (t, h, w) after VAE compression.

        The time axis is compressed the **causal VAE** way:
        ``1 + (num_frames - 1) / stride``. The first frame becomes a latent
        frame on its own, then every ``stride`` frames collapse into one. The
        current code writes the same formula in five places
        (``model/fastwam/mot/fastwam.py:873``, ``mot/idm.py:353``,
        ``mot/joint.py:155``, ``wan/core.py:358``, ``wan/vae.py:1141``); this
        property is where they converge.

        A plain division would reject 21 frames with stride 4, yet that is the
        actual setting of VPP2 and EWAM (21 => 6 latent frames). With
        ``stride=1`` the two formulas agree.

        ``num_views`` acts on the width axis: multiple cameras are
        concatenated along width (VPP2's ``concat_multi_camera: horizontal``).

        Returns:
            The (t, h, w) latent grid sizes.
        """
        return (
            1 + _exact_div(self.num_frames - 1, self.vae_temporal_stride,
                           what="(num_frames - 1) / vae_temporal_stride"),
            _exact_div(self.frame_height, self.vae_spatial_stride,
                       what="frame_height / vae_spatial_stride"),
            _exact_div(self.frame_width * self.num_views, self.vae_spatial_stride,
                       what="frame_width × num_views / vae_spatial_stride"),
        )

    @property
    def patch_grid(self) -> tuple[int, int, int]:
        """The grid (t, h, w) after patchification, i.e. tokens per axis."""
        latent = self.latent_grid
        axis_names = ("t", "h", "w")
        return tuple(  # type: ignore[return-value]
            _exact_div(latent[i], self.patch_size[i],
                       what=f"latent_grid[{axis_names[i]}] / "
                            f"patch_size[{axis_names[i]}]")
            for i in range(3)
        )

    @property
    def tokens_per_frame(self) -> int:
        """Tokens in one latent frame = patch_grid[1] x patch_grid[2]."""
        _, h, w = self.patch_grid
        return h * w

    @property
    def video_seq_len(self) -> int:
        """Total tokens of the video stream (unaligned) = prod(patch_grid)."""
        t, h, w = self.patch_grid
        return t * h * w

    @property
    def padded_seq_len(self) -> int:
        """Length rounded up to ``align_to``; padded slots get pad=True.

        This is the **single source of static shape** (master plan INV-2):
        CUDA graph capture requires the sequence length to be identical across
        steps, so the length may only come from here, never from the data.
        """
        return math.ceil(self.video_seq_len / self.align_to) * self.align_to

    @property
    def num_chunks(self) -> int:
        """Action chunk count = action_horizon / action_video_freq_ratio.

        The single source of the chunk label and one piece of interface I2. A
        non-exact division raises at construction time.
        """
        if self.action_horizon == 0:
            return 0
        return _exact_div(self.action_horizon, self.action_video_freq_ratio,
                          what="action_horizon / action_video_freq_ratio")

    def build_labels(self) -> dict[str, Tensor]:
        """Build the token labels for this geometry, consumed by ``TokenLayout``.

        Returns:
            A dict with these keys:
              frame  [L] int   which latent frame the token belongs to
              row    [L] int   patch row index (used by spatial-locality rules)
              col    [L] int   patch column index
              pad    [L] bool  whether the slot exists only for alignment

        Note that stream / chunk / noise_level are **not** produced here: the
        stream label is attached by each stream, chunk is attached by the
        action stream using ``num_chunks``, and noise_level is attached by the
        L3 noising scheme. The geometry layer only supplies the **number** of
        chunks; it does not map chunks onto tokens.
        """
        _, h, w = self.patch_grid
        real = self.video_seq_len
        total = self.padded_seq_len

        index = torch.arange(real, dtype=torch.long)
        frame = torch.full((total,), -1, dtype=torch.long)
        row = torch.full((total,), -1, dtype=torch.long)
        col = torch.full((total,), -1, dtype=torch.long)
        pad = torch.ones(total, dtype=torch.bool)

        # Flatten order = C order over (t, h, w), matching patchify's flatten
        frame[:real] = index // (h * w)
        row[:real] = (index // w) % h
        col[:real] = index % w
        pad[:real] = False
        return {"frame": frame, "row": row, "col": col, "pad": pad}
