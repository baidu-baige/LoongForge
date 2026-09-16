# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This software may be used and distributed in accordance with
# the terms of the DINOv3 License Agreement.

import math
from typing import Literal

import numpy as np
import torch
from torch import Tensor, nn


class RopePositionEmbedding(nn.Module):
    """2D Rotary Position Embedding for spatial (H, W) grids."""

    def __init__(
        self,
        embed_dim: int,
        *,
        num_heads: int,
        base: float | None = 100.0,
        min_period: float | None = None,
        max_period: float | None = None,
        normalize_coords: Literal["min", "max", "separate"] = "separate",
        shift_coords: float | None = None,
        jitter_coords: float | None = None,
        rescale_coords: float | None = None,
        dtype: torch.dtype | None = None,
        device: torch.device | None = None,
    ):
        super().__init__()
        assert embed_dim % (4 * num_heads) == 0
        both_periods = min_period is not None and max_period is not None
        if (base is None and not both_periods) or (base is not None and both_periods):
            raise ValueError("Either `base` or `min_period`+`max_period` must be provided.")

        D_head = embed_dim // num_heads
        self.base = base
        self.min_period = min_period
        self.max_period = max_period
        self.D_head = D_head
        self.normalize_coords = normalize_coords
        self.shift_coords = shift_coords
        self.jitter_coords = jitter_coords
        self.rescale_coords = rescale_coords

        self.dtype = dtype
        self.register_buffer(
            "periods",
            torch.empty(D_head // 4, device=device, dtype=dtype),
            persistent=True,
        )
        self._init_weights()

    def forward(self, *, H: int, W: int) -> tuple[Tensor, Tensor]:
        device = self.periods.device
        dtype = self.dtype
        dd = {"device": device, "dtype": dtype}

        if self.normalize_coords == "max":
            max_HW = max(H, W)
            coords_h = torch.arange(0.5, H, **dd) / max_HW
            coords_w = torch.arange(0.5, W, **dd) / max_HW
        elif self.normalize_coords == "min":
            min_HW = min(H, W)
            coords_h = torch.arange(0.5, H, **dd) / min_HW
            coords_w = torch.arange(0.5, W, **dd) / min_HW
        elif self.normalize_coords == "separate":
            coords_h = torch.arange(0.5, H, **dd) / H
            coords_w = torch.arange(0.5, W, **dd) / W
        else:
            raise ValueError(f"Unknown normalize_coords: {self.normalize_coords}")
        coords = torch.stack(torch.meshgrid(coords_h, coords_w, indexing="ij"), dim=-1)
        coords = coords.flatten(0, 1)
        coords = 2.0 * coords - 1.0

        if self.training and self.shift_coords is not None:
            shift_coords = float(self.shift_coords)
            shift_hw = torch.empty(2, **dd).uniform_(-shift_coords, shift_coords)
            coords += shift_hw[None, :]

        if self.training and self.jitter_coords is not None:
            jitter_max = np.log(self.jitter_coords)
            jitter_min = -jitter_max
            jitter_hw = torch.empty(2, **dd).uniform_(jitter_min, jitter_max).exp()
            coords *= jitter_hw[None, :]

        if self.training and self.rescale_coords is not None:
            rescale_max = np.log(self.rescale_coords)
            rescale_min = -rescale_max
            rescale_hw = torch.empty(1, **dd).uniform_(rescale_min, rescale_max).exp()
            coords *= rescale_hw

        angles = 2 * math.pi * coords[:, :, None] / self.periods[None, None, :]
        angles = angles.flatten(1, 2)
        angles = angles.tile(2)
        cos = torch.cos(angles)
        sin = torch.sin(angles)

        return (sin, cos)

    def _init_weights(self):
        device = self.periods.device
        dtype = self.dtype
        if self.base is not None:
            periods = self.base ** (
                2 * torch.arange(self.D_head // 4, device=device, dtype=dtype) / (self.D_head // 2)
            )
        else:
            base = self.max_period / self.min_period
            exponents = torch.linspace(0, 1, self.D_head // 4, device=device, dtype=dtype)
            periods = base**exponents
            periods = periods / base
            periods = periods * self.max_period
        self.periods.data = periods


class RopePositionEmbedding3D(RopePositionEmbedding):
    """3D Rotary Position Embedding with interleaved temporal encoding.

    Extends 2D RoPE by inserting temporal encodings at positions where
    index % 4 == 3 in D_head.

    Spatial (h, w): normalized to [-1, 1], uses parent's `periods` buffer
    with `angle = 2π * coord / period` (same as 2D RoPE).

    Temporal (t): absolute integer frame indices (no normalization), uses
    a separate `periods_t` buffer with a larger base (default 10000,
    matching standard LLM RoPE convention). The angle formula is
    `angle = frame_idx / periods_t` (no 2π factor), so that adjacent
    frames always have a fixed angular difference regardless of video
    length.

    This design gives the model a unified sense of physical time:
    - Spatial positions are relative (within-frame structure)
    - Temporal positions are absolute (physical time progression)
    """

    def __init__(
        self,
        embed_dim: int,
        *,
        num_heads: int,
        base: float | None = 100.0,
        temporal_base: float = 10000.0,
        min_period: float | None = None,
        max_period: float | None = None,
        normalize_coords: Literal["min", "max", "separate"] = "separate",
        shift_coords: float | None = None,
        jitter_coords: float | None = None,
        rescale_coords: float | None = None,
        fps: float | None = None,
        base_fps: float = 24.0,
        dtype: torch.dtype | None = None,
        device: torch.device | None = None,
    ):
        super().__init__(
            embed_dim,
            num_heads=num_heads,
            base=base,
            min_period=min_period,
            max_period=max_period,
            normalize_coords=normalize_coords,
            shift_coords=shift_coords,
            jitter_coords=jitter_coords,
            rescale_coords=rescale_coords,
            dtype=dtype,
            device=device,
        )
        self.temporal_base = temporal_base
        self.fps = fps
        self.base_fps = base_fps

        D_head = self.D_head
        n_t_freqs = D_head // 8
        periods_t = temporal_base ** (
            2 * torch.arange(n_t_freqs, device=device, dtype=dtype) / (2 * n_t_freqs)
        )
        self.register_buffer("periods_t", periods_t, persistent=True)

    def _init_weights(self):
        super()._init_weights()
        # The parent __init__ calls _init_weights() before our temporal state
        # exists; skip then (the periods_t buffer is materialised right after in
        # our __init__). On the real FSDP2 path (meta -> to_empty -> init_weights)
        # to_empty leaves periods_t as uninitialised storage and the external
        # init_weights() reaches us here — so we MUST re-fill periods_t, else
        # angles_t = grid_t / periods_t = x/0 = NaN.
        if not hasattr(self, "periods_t"):
            return
        device = self.periods.device
        dtype = self.dtype
        n_t_freqs = self.D_head // 8
        periods_t = self.temporal_base ** (
            2 * torch.arange(n_t_freqs, device=device, dtype=dtype) / (2 * n_t_freqs)
        )
        self.periods_t.data = periods_t

    @property
    def temporal_channel_mask(self) -> Tensor:
        """Bool mask over D_head: True where the channel carries temporal
        (idx % 4 == 3) encoding. Single source of truth for the interleave
        layout used by forward() and by prefix-rope construction."""
        return torch.arange(self.D_head, device=self.periods.device) % 4 == 3

    def forward(
        self, *, H: int, W: int, T: int = 1, fps: float | None = None
    ) -> tuple[Tensor, Tensor]:
        """Compute 3D RoPE sin/cos for a (T, H, W) spatiotemporal grid.

        Args:
            H, W: spatial grid size (in patches).
            T: number of frames.
            fps: source video FPS. If provided, overrides self.fps for
                FPS modulation (adjacent frames differ by base_fps/fps
                in position space). If None and self.fps is None,
                frame step = 1 (no FPS scaling).

        Returns (sin, cos) each of shape (T*H*W, D_head).
        """
        device = self.periods.device
        dtype = self.dtype
        dd = {"device": device, "dtype": dtype}

        # --- spatial coordinates (normalized, same as 2D parent) ---
        if self.normalize_coords == "max":
            max_HW = max(H, W)
            coords_h = torch.arange(0.5, H, **dd) / max_HW
            coords_w = torch.arange(0.5, W, **dd) / max_HW
        elif self.normalize_coords == "min":
            min_HW = min(H, W)
            coords_h = torch.arange(0.5, H, **dd) / min_HW
            coords_w = torch.arange(0.5, W, **dd) / min_HW
        elif self.normalize_coords == "separate":
            coords_h = torch.arange(0.5, H, **dd) / H
            coords_w = torch.arange(0.5, W, **dd) / W
        else:
            raise ValueError(f"Unknown normalize_coords: {self.normalize_coords}")

        # Shift spatial to [-1, +1]
        coords_h = 2.0 * coords_h - 1.0
        coords_w = 2.0 * coords_w - 1.0

        # --- temporal coordinates (absolute, integer frame indices) ---
        active_fps = fps if fps is not None else self.fps
        if active_fps is not None:
            step = self.base_fps / active_fps
        else:
            step = 1.0
        coords_t = torch.arange(T, **dd) * step  # [0, step, 2*step, ...]

        # --- train-time spatial augmentations ---
        if self.training and self.shift_coords is not None:
            shift_coords = float(self.shift_coords)
            shift_hw = torch.empty(2, **dd).uniform_(-shift_coords, shift_coords)
            coords_h = coords_h + shift_hw[0]
            coords_w = coords_w + shift_hw[1]

        if self.training and self.jitter_coords is not None:
            jitter_max = np.log(self.jitter_coords)
            jitter_min = -jitter_max
            jitter_hw = torch.empty(2, **dd).uniform_(jitter_min, jitter_max).exp()
            coords_h = coords_h * jitter_hw[0]
            coords_w = coords_w * jitter_hw[1]

        if self.training and self.rescale_coords is not None:
            rescale_max = np.log(self.rescale_coords)
            rescale_min = -rescale_max
            rescale_hw = torch.empty(1, **dd).uniform_(rescale_min, rescale_max).exp()
            coords_h = coords_h * rescale_hw
            coords_w = coords_w * rescale_hw

        # --- compute spatial angles: 2π * coord / period ---
        # Expand to (T*H*W, ...) via meshgrid
        grid_t, grid_h, grid_w = torch.meshgrid(coords_t, coords_h, coords_w, indexing="ij")
        grid_t = grid_t.reshape(-1, 1)  # (T*H*W, 1)
        grid_h = grid_h.reshape(-1, 1)
        grid_w = grid_w.reshape(-1, 1)

        periods = self.periods  # (D_head//4,)
        D_head = self.D_head

        angles_h = 2 * math.pi * grid_h / periods[None, :]  # (T*H*W, D_head//4)
        angles_h = angles_h.repeat(1, 2)  # (T*H*W, D_head//2)

        angles_w = 2 * math.pi * grid_w / periods[None, :]
        angles_w = angles_w.repeat(1, 2)  # (T*H*W, D_head//2)

        # --- compute temporal angles: frame_idx / periods_t (no 2π) ---
        periods_t = self.periods_t  # (D_head//8,)
        angles_t = grid_t / periods_t[None, :]  # (T*H*W, D_head//8)
        angles_t = angles_t.repeat(1, 2)  # (T*H*W, D_head//4)

        # --- interleave: replace idx % 4 == 3 with temporal ---
        angles = torch.cat([angles_h, angles_w], dim=1)  # (T*H*W, D_head)
        t_mask = self.temporal_channel_mask  # single source of truth (idx % 4 == 3)
        angles[:, t_mask] = angles_t

        sin = torch.sin(angles)
        cos = torch.cos(angles)
        return (sin, cos)
