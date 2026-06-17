# Copyright 2026 The LoongForge Authors.
# SPDX-License-Identifier: Apache-2.0

"""Packed dataset for sequence packing in diffusion model training.

Patchify variable-length latents, pad text to fixed
max length, apply CP padding, then concatenate across samples in each bin.
"""

from typing import List, Dict, Any

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import IterableDataset
from megatron.core import parallel_state

from loongforge.utils import get_args, print_rank_0
from loongforge.data.video.latent_dataset import TensorDataset
from loongforge.data.video.sequence_packing_utils import first_fit



def _get_cp_sample_alignment(cp_world_size: int) -> int:
    """Return per-sample sequence alignment required by CP split."""
    return cp_world_size * 2 if cp_world_size > 1 else 1


# ---------------------------------------------------------------------------
# Patchify utilities (Wan: patch_size=(1,2,2))
# ---------------------------------------------------------------------------

def _patchify_latent(latent: torch.Tensor, patch_size=(1, 2, 2)):
    """Convert [C, T, H, W] → [num_patches, pT * pH * pW * C].

    Output is in **(pT, pH, pW, C)** order within each patch.
    This matches the ``unpatchify`` / ``Head`` output layout used for
    training-target comparison in the loss function.
    """
    c, t, h, w = latent.shape
    pT, pH, pW = patch_size
    assert t % pT == 0 and h % pH == 0 and w % pW == 0, (
        f"Spatial dims ({t},{h},{w}) must be divisible by patch_size {patch_size}"
    )
    fP, hP, wP = t // pT, h // pH, w // pW
    # (c, fP, pT, hP, pH, wP, pW) → (fP, hP, wP, pT, pH, pW, c)
    x = latent.reshape(c, fP, pT, hP, pH, wP, pW)
    x = x.permute(1, 3, 5, 2, 4, 6, 0).contiguous()
    num_patches = fP * hP * wP
    return x.reshape(num_patches, c * pT * pH * pW)


def _patchify_for_conv3d(latent: torch.Tensor, patch_size=(1, 2, 2)):
    """Convert [C, T, H, W] → [num_patches, C * pT * pH * pW].

    Output is in **(C, pT, pH, pW)** order within each patch — the same
    layout as ``nn.Conv3d(weight).view(out_channels, -1)``.  Use this for
    model-input patchification so that ``F.linear(x, conv_weight, conv_bias)``
    is numerically identical to the Conv3d forward pass.
    """
    c, t, h, w = latent.shape
    pT, pH, pW = patch_size
    assert t % pT == 0 and h % pH == 0 and w % pW == 0, (
        f"Spatial dims ({t},{h},{w}) must be divisible by patch_size {patch_size}"
    )
    fP, hP, wP = t // pT, h // pH, w // pW
    # (c, fP, pT, hP, pH, wP, pW) → (fP, hP, wP, c, pT, pH, pW)
    x = latent.reshape(c, fP, pT, hP, pH, wP, pW)
    x = x.permute(1, 3, 5, 0, 2, 4, 6).contiguous()
    num_patches = fP * hP * wP
    return x.reshape(num_patches, c * pT * pH * pW)


class PackedDataset(IterableDataset):
    """A dataset that packs variable-length diffusion samples.

    Each sample's latent is patchified into a 2-D token sequence, text is
    padded to ``context_max_len``, and optional CP padding is applied.
    Samples in each bin are concatenated along the sequence dimension.
    """

    # Wan patch size
    PATCH_SIZE = (1, 2, 2)

    def __init__(
        self,
        data_path: str,
        steps_per_epoch: int,
        args,
        scheduler=None,
        packing_buffer_size: int = 512,
        seq_length: int = 8192,
        dp_rank: int = 0,
        dp_world_size: int = 1,
    ):
        super().__init__()
        self.data_path = data_path
        self.steps_per_epoch = steps_per_epoch
        self.args = args
        self.scheduler = scheduler
        self.packing_buffer_size = packing_buffer_size
        self.seq_length = seq_length
        self.dp_rank = dp_rank
        self.dp_world_size = dp_world_size

        self.context_max_len = getattr(args, "context_max_len", 512)

        cp_world_size = parallel_state.get_context_parallel_world_size()
        self.cp_world_size = cp_world_size
        self.cp_chunk_factor = _get_cp_sample_alignment(cp_world_size)
        self.sharding_factor = self.cp_chunk_factor if cp_world_size > 1 else 0

        # Initialize the base dataset
        keep_keys = None
        if getattr(args, "model_name", None) in ("wan2-1-i2v", "wan2-2-i2v"):
            keep_keys = {
                "context", "input_latents", "y", "clip_feature",
                "height", "width", "num_frames",
                "max_timestep_boundary", "min_timestep_boundary",
            }
        self.base_dataset = TensorDataset(
            data_path, steps_per_epoch, seed=args.seed, keep_keys=keep_keys,
        )

        # Validate timestep boundaries
        max_ts = args.max_timestep_boundary
        min_ts = args.min_timestep_boundary
        assert 0 <= max_ts <= 1, "max_timestep should range from 0 to 1"
        assert 0 <= min_ts <= 1, "min_timestep should range from 0 to 1"
        assert min_ts <= max_ts, f"min_timestep: {min_ts} should <= max_timestep: {max_ts}"

    # ------------------------------------------------------------------
    # Sequence length helpers
    # ------------------------------------------------------------------

    def _compute_seq_length(self, sample: Dict[str, Any]) -> int:
        """Compute video sequence length (num_patches) from input_latents."""
        input_latents = sample["input_latents"]
        _, _, latent_frames, latent_height, latent_width = input_latents.shape
        patch_frames, patch_height, patch_width = self.PATCH_SIZE
        return (
            (latent_frames // patch_frames)
            * (latent_height // patch_height)
            * (latent_width // patch_width)
        )

    @staticmethod
    def _ceil_to_multiple(n: int, m: int) -> int:
        return ((n + m - 1) // m) * m

    # ------------------------------------------------------------------
    # Bin packing
    # ------------------------------------------------------------------

    def _pack_samples(self, samples: List[Dict[str, Any]]) -> List[List[Dict[str, Any]]]:
        """Pack samples into bins using First-Fit while preserving sample order."""
        seq_lengths = [self._compute_seq_length(s) for s in samples]
        if self.sharding_factor > 0:
            seq_lengths = [self._ceil_to_multiple(length, self.sharding_factor) for length in seq_lengths]

        packed_indices = first_fit(seq_lengths, self.seq_length)

        bins = [[samples[i] for i in idxs] for idxs in packed_indices]
        return bins

    # ------------------------------------------------------------------
    # Main iteration
    # ------------------------------------------------------------------

    def __iter__(self):
        from loongforge.models.diffusion.wan.wan_flow_match import FlowMatchScheduler

        scheduler = self.scheduler
        if scheduler is None:
            scheduler = FlowMatchScheduler(shift=5, sigma_min=0.0, extra_one_step=True)
            scheduler.set_timesteps(1000, training=True)

        required_bins = self.steps_per_epoch
        buffer = []
        sample_idx = self.dp_rank
        batch_count = 0

        while batch_count < required_bins:
            # Fill buffer
            while len(buffer) < self.packing_buffer_size:
                wrapped_idx = sample_idx % len(self.base_dataset)
                buffer.append(self.base_dataset[wrapped_idx])
                sample_idx += self.dp_world_size

            bins = self._pack_samples(buffer)

            for bin_samples in bins:
                if batch_count >= required_bins:
                    return
                processed = self._process_bin(bin_samples, scheduler)
                merged = self._merge_samples(processed)
                batch_count += 1
                yield merged

            buffer = []

        return

    # ------------------------------------------------------------------
    # Per-bin processing
    # ------------------------------------------------------------------

    def _process_bin(self, bin_samples, scheduler):
        """Process all samples in a bin: patchify, pad, gen noise."""
        processed = []
        max_ts = self.args.max_timestep_boundary
        min_ts = self.args.min_timestep_boundary
        max_ts_boundary = int(max_ts * scheduler.num_train_timesteps)
        min_ts_boundary = int(min_ts * scheduler.num_train_timesteps)

        for sample in bin_samples:
            out = self._process_sample(sample, scheduler, min_ts_boundary, max_ts_boundary)
            processed.append(out)
        return processed

    def _process_sample(self, sample, scheduler, min_ts_boundary, max_ts_boundary):
        """Process a single sample: patchify, pad text, CP pad, gen noise."""
        result = {}

        # --- 1. Extract latents and y ---
        input_latents = sample["input_latents"]
        if input_latents.size(0) == 1:
            input_latents = input_latents.squeeze(0)

        y = sample.get("y")
        if y is not None:
            if y.dim() == 5 and y.size(0) == 1:
                y = y.squeeze(0)

        # grid_sizes from .pth or computed from latent shape (without y)
        if "grid_sizes" in sample:
            grid_sizes = sample["grid_sizes"]
        else:
            _, latent_frames, latent_height, latent_width = input_latents.shape
            patch_frames, patch_height, patch_width = self.PATCH_SIZE
            grid_sizes = torch.tensor(
                [
                    latent_frames // patch_frames,
                    latent_height // patch_height,
                    latent_width // patch_width,
                ],
                dtype=torch.int32,
            )

        seq_len_q = int(torch.prod(torch.tensor(grid_sizes.tolist())).item())

        # --- 2. Text padding ---
        context = sample["context"]  # [1, actual_len, dim]
        if context.dim() == 3:
            context = context.squeeze(0)  # [actual_len, dim]
        actual_text_len = context.shape[0]
        if actual_text_len < self.context_max_len:
            context = F.pad(context, (0, 0, 0, self.context_max_len - actual_text_len))
        elif actual_text_len > self.context_max_len:
            context = context[:self.context_max_len]
        seq_len_kv = self.context_max_len

        # --- 3. CP padding ---
        # Compute padded lengths for CP split planning. Padding is deferred
        if self.sharding_factor > 0:
            seq_len_q_padded = self._ceil_to_multiple(seq_len_q, self.sharding_factor)
            seq_len_kv_padded = self._ceil_to_multiple(seq_len_kv, self.sharding_factor)
        else:
            seq_len_q_padded = seq_len_q
            seq_len_kv_padded = seq_len_kv

        loss_mask = torch.ones(seq_len_q, dtype=torch.float32)

        seed = sample["seed"]
        rng = np.random.RandomState(seed=seed)
        noise_np = rng.randn(*input_latents.shape)
        noise = torch.tensor(noise_np, dtype=input_latents.dtype, device=input_latents.device)
        rand_int = rng.randint(min_ts_boundary, max_ts_boundary)
        timestep_id = torch.tensor([rand_int], dtype=torch.long)

        timestep_for_scale = scheduler.timesteps[timestep_id].to(dtype=input_latents.dtype)
        scale = scheduler.training_weight(timestep_for_scale)

        result["input_latents_raw"] = input_latents
        result["y_raw"] = y.to(dtype=input_latents.dtype) if y is not None else None
        result["noise_raw"] = noise
        result["loss_mask"] = loss_mask.unsqueeze(1)
        result["context"] = context.unsqueeze(1)
        result["grid_sizes"] = grid_sizes
        result["seq_len_q"] = torch.tensor([seq_len_q], dtype=torch.int32)
        result["seq_len_q_padded"] = torch.tensor([seq_len_q_padded], dtype=torch.int32)
        result["seq_len_kv"] = torch.tensor([seq_len_kv], dtype=torch.int32)
        result["seq_len_kv_padded"] = torch.tensor([seq_len_kv_padded], dtype=torch.int32)
        result["timestep_id"] = timestep_id
        result["scale"] = scale.reshape(-1)
        result["seed"] = torch.tensor([seed], dtype=torch.long)

        return result

    def _pad_packed_sequence(self, chunks: List[torch.Tensor], padded_lengths: List[int]) -> torch.Tensor:
        padded_chunks = []
        for chunk, padded_length in zip(chunks, padded_lengths):
            pad_len = padded_length - chunk.shape[0]
            if pad_len > 0:
                pad_shape = (pad_len, *chunk.shape[1:])
                pad = torch.zeros(pad_shape, dtype=chunk.dtype, device=chunk.device)
                chunk = torch.cat([chunk, pad], dim=0)
            padded_chunks.append(chunk)
        return torch.cat(padded_chunks, dim=0)

    # ------------------------------------------------------------------
    # Merge samples within a bin
    # ------------------------------------------------------------------

    def _merge_samples(self, bin_samples: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Merge processed samples into one packed bin."""
        merged = {}

        merged["input_latents_raw"] = [s["input_latents_raw"] for s in bin_samples]
        merged["y_raw"] = [s["y_raw"] for s in bin_samples]
        merged["noise_raw"] = [s["noise_raw"] for s in bin_samples]

        if self.sharding_factor > 0:
            merged["loss_mask"] = self._pad_packed_sequence(
                [s["loss_mask"] for s in bin_samples],
                [s["seq_len_q_padded"].item() for s in bin_samples],
            )
            merged["context"] = self._pad_packed_sequence(
                [s["context"] for s in bin_samples],
                [s["seq_len_kv_padded"].item() for s in bin_samples],
            )
        else:
            merged["loss_mask"] = torch.cat([s["loss_mask"] for s in bin_samples], dim=0)
            merged["context"] = torch.cat([s["context"] for s in bin_samples], dim=0)

        merged["grid_sizes"] = torch.stack([s["grid_sizes"] for s in bin_samples], dim=0)
        merged["seq_len_q"] = torch.cat([s["seq_len_q"] for s in bin_samples])
        merged["seq_len_q_padded"] = torch.cat([s["seq_len_q_padded"] for s in bin_samples])
        merged["seq_len_kv"] = torch.cat([s["seq_len_kv"] for s in bin_samples])
        merged["seq_len_kv_padded"] = torch.cat([s["seq_len_kv_padded"] for s in bin_samples])
        merged["timestep_id"] = torch.cat([s["timestep_id"] for s in bin_samples])
        merged["scale"] = torch.cat([s["scale"] for s in bin_samples])
        merged["seed"] = torch.stack([s["seed"] for s in bin_samples])

        return merged
