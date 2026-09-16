# Copyright 2026 The LoongForge Authors.
# SPDX-License-Identifier: Apache-2.0

"""ErnieImagePreprocess — pure-PyTorch image normalisation for ERNIE-VL.
  1. Rescale uint8 patches: pixel = pixel / 255.0
  2. Normalize: pixel = (pixel - mean) / std

where mean/std are the per-channel OPENAI CLIP constants expanded over the
flattened patch dimension (C * patch_size^2), matching the original
`add_image_preprocess()` logic.

Input contract (same as before):
  images : torch.Tensor  shape [S, C * patch_size^2], dtype uint8
  grid_thw : torch.Tensor | None

Output: (images_bf16, grid_thw_processed)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

# OPENAI CLIP normalisation constants — identical to image_preprocessor.py
_OPENAI_CLIP_MEAN = [0.48145466, 0.4578275, 0.40821073]
_OPENAI_CLIP_STD = [0.26862954, 0.26130258, 0.27577711]


class ErnieImagePreprocess(nn.Module):
    """Pure-PyTorch drop-in replacement for the AutoProcessor-based preprocess.

    Mean/std constants are kept as plain Python lists (not nn.Parameter /
    register_buffer) so that module-level dtype casts (.half() / .bfloat16())
    never touch them.  As a consequence, the corresponding tensors must be
    constructed anew on every forward() call — this is intentional: it keeps
    the constants in float64 regardless of the module's current dtype, matching
    the original numpy-based precision.
    """

    def __init__(self, patch_size: int = 14, in_channels: int = 3) -> None:
        super().__init__()
        self.patch_size = patch_size
        self.in_channels = in_channels

        # _preprocess in image_preprocessor.py normalizes the full [C,H,W] image
        # first, then reshapes via transpose([0,2,5,3,6,1,4,7]) and flattens to
        # [S, C*P*P].  After the transpose the layout is (..., C, P, P), so the
        # flat order per patch is channel-first: R_p0..R_pN, G_p0..G_pN, B_p0..B_pN.
        # → expand each channel constant P*P times with repeat (numpy.repeat).
        pp = patch_size * patch_size
        self._mean_flat = [v for v in _OPENAI_CLIP_MEAN for _ in range(pp)]  # len C*P*P
        self._std_flat  = [v for v in _OPENAI_CLIP_STD  for _ in range(pp)]  # len C*P*P

    def forward(
        self,
        images: torch.Tensor,
        grid_thw: torch.Tensor,
    ):
        """Apply rescale + normalize, then process grid_thw.

        Args:
            images   : uint8 tensor, shape [S, C*P*P]
            grid_thw : int tensor, shape [B, 3] or with padding zeros, or None

        Returns:
            images   : bfloat16 tensor, shape [S, C*P*P]
            grid_thw : processed int tensor, shape [total_frames, 3]
        """

        # ---- rescale + normalize in float64 to match original numpy precision ----
        # Intentionally rebuilt every forward(): _mean_flat/_std_flat are plain
        # Python lists, so .to(device) here is the only way to get a tensor on
        # the right device without registering buffers (which would be downcast
        # by .half()/.bfloat16()).
        device = images.device
        mean = torch.tensor(self._mean_flat, dtype=torch.float64, device=device)
        std  = torch.tensor(self._std_flat,  dtype=torch.float64, device=device)

        x = images.to(torch.float64)           # [S, C*P*P]
        x = x * (1.0 / 255.0)                 # rescale
        x = (x - mean) / std                   # normalize
        images = x.to(torch.bfloat16)

        # ---- grid_thw post-processing (identical to original preprocess()) ----
        if grid_thw is not None:
            grid_thw = grid_thw[grid_thw > 0].reshape([-1, 3])
            grid_thw = F.pad(
                torch.repeat_interleave(grid_thw[:, 1:], grid_thw[:, 0], dim=0),
                [1, 0, 0, 0],
                value=1,
            )

        return images, grid_thw
