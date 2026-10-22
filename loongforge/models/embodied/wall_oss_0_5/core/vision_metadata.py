# Copyright 2026 The LoongForge Authors.
# SPDX-License-Identifier: Apache-2.0

"""CPU-owned metadata for dynamic Qwen2.5-VL vision shapes."""

from dataclasses import dataclass
from typing import Iterable, Optional


@dataclass(frozen=True)
class VisionGridMetadata:
    """Shape values needed by vision kernels without reading CUDA scalars."""

    total_tokens: int
    total_elements: int
    total_windows: int
    max_grid_t: int
    max_grid_size: int
    total_grid_t: int
    max_seqlen_full: int
    max_seqlen_window: int


def compute_vision_grid_metadata(
    grid_thw: Optional[Iterable[Iterable[int]]],
    spatial_merge_size: int,
    window_size: int,
    patch_size: int,
    spatial_merge_unit: int = 1,
) -> VisionGridMetadata:
    """Compute dynamic vision allocation sizes from CPU grid metadata."""
    device = getattr(grid_thw, "device", None)
    if device is not None and device.type != "cpu":
        raise ValueError("vision grid metadata must be computed from CPU values")

    if grid_thw is None:
        rows = ()
    else:
        rows = tuple(tuple(int(v) for v in row) for row in grid_thw)

    if not rows:
        return VisionGridMetadata(0, 0, 0, 0, 0, 0, 0, 0)

    if spatial_merge_size <= 0 or patch_size <= 0 or spatial_merge_unit <= 0:
        raise ValueError("vision shape factors must be positive")

    vit_window = window_size // spatial_merge_size // patch_size
    if vit_window <= 0:
        raise ValueError("window_size must cover at least one merged patch")

    total_tokens = 0
    total_elements = 0
    total_windows = 0
    max_grid_t = 0
    max_grid_size = 0
    total_grid_t = 0
    max_seqlen_full = 0
    max_seqlen_window = 0

    for row in rows:
        if len(row) != 3:
            raise ValueError(f"grid_thw rows must have three values, got {row}")
        grid_t, grid_h, grid_w = row
        if min(row) < 0:
            raise ValueError(f"grid_thw values must be non-negative, got {row}")

        llm_h = grid_h // spatial_merge_size
        llm_w = grid_w // spatial_merge_size
        num_windows_h = (llm_h + vit_window - 1) // vit_window
        num_windows_w = (llm_w + vit_window - 1) // vit_window

        grid_elements = grid_t * llm_h * llm_w
        total_elements += grid_elements
        total_tokens += grid_elements * spatial_merge_size * spatial_merge_size
        total_windows += grid_t * num_windows_h * num_windows_w
        max_grid_t = max(max_grid_t, grid_t)
        max_grid_size = max(max_grid_size, grid_h, grid_w)
        total_grid_t += grid_t
        max_seqlen_full = max(max_seqlen_full, grid_h * grid_w)
        max_seqlen_window = max(
            max_seqlen_window,
            min(vit_window, llm_h)
            * min(vit_window, llm_w)
            * spatial_merge_unit,
        )

    return VisionGridMetadata(
        total_tokens=total_tokens,
        total_elements=total_elements,
        total_windows=total_windows,
        max_grid_t=max_grid_t,
        max_grid_size=max_grid_size,
        total_grid_t=total_grid_t,
        max_seqlen_full=max_seqlen_full,
        max_seqlen_window=max_seqlen_window,
    )
