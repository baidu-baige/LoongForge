# Copyright 2026 The LoongForge Authors.
# SPDX-License-Identifier: Apache-2.0
#
# Modified from https://github.com/thu-ml/Motus under the Apache-2.0 License.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Motus batch collator.

Transplants the source Motus ``data/dataset.py`` collation numerics
(``collate_fn`` + ``_process_vlm_inputs_batch`` + ``_process_language_embeddings_batch``)
into loongforge's :class:`BasePreprocessor` / :class:`PreparedBatch` framework.

The per-sample dict produced by :class:`LeRobotMotusDataset` carries::

    first_frame        [C, H, W]
    video_frames       [F, C, H, W]
    action_sequence    [action_chunk_size, action_dim]
    initial_state      [state_dim]                 (optional)
    language_embedding [S, D]                       (T5, ragged -> padded here)
    vlm_inputs         {input_ids, attention_mask, pixel_values, image_grid_thw}

The collator stacks fixed-shape tensors and pads the ragged VLM / T5 fields to
fixed lengths, matching the source exactly.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Optional

import torch

from loongforge.embodied.data.datasets.transforms.collator import (
    BasePreprocessor,
    PreparedBatch,
    register_preprocessor,
)


@dataclass
class MotusPreparedBatch(PreparedBatch):
    """Model-ready Motus batch. Field names match ``MotusPolicy.forward``."""

    first_frame: Optional[torch.Tensor] = None          # [B, C, H, W]
    video_frames: Optional[torch.Tensor] = None         # [B, F, C, H, W]
    action_sequence: Optional[torch.Tensor] = None      # [B, F, D]
    initial_state: Optional[torch.Tensor] = None        # [B, state_dim] or None
    language_embedding: Optional[torch.Tensor] = None   # [B, text_len, D] or None
    vlm_inputs: Optional[Dict[str, torch.Tensor]] = None
    # Precomputed fp32 VAE latent [B, 48, T', H', W'], present only when the
    # offline latent cache is enabled (--latent-cache-dir); lets the trainer
    # skip the online encode. None on the normal (encode-online) path.
    clean_full_latent: Optional[torch.Tensor] = None


def _process_vlm_inputs_batch(
    vlm_inputs: List[Dict[str, Any]]
) -> Dict[str, torch.Tensor]:
    """Pad + batch VLM inputs (verbatim from source ``data/dataset.py``).

    Matches base exactly: pad each sample's ``input_ids``/``attention_mask`` to
    the **dynamic per-batch max** length with zeros; no fixed length, no
    truncation. Widths therefore vary batch-to-batch (incompatible with a fixed
    CUDA-graph capture shape — this is the base-aligned behaviour).
    """
    input_ids_list = [vlm_input["input_ids"] for vlm_input in vlm_inputs]
    pixel_values_list = [vlm_input.get("pixel_values") for vlm_input in vlm_inputs]
    image_grid_thw_list = [vlm_input.get("image_grid_thw") for vlm_input in vlm_inputs]
    attention_mask_list = [vlm_input.get("attention_mask") for vlm_input in vlm_inputs]

    # Dynamic per-batch max (base data/dataset.py:246), no truncation.
    max_seq_len = max(ids.shape[1] for ids in input_ids_list)

    padded_input_ids = []
    padded_attention_masks = []

    for ids, mask in zip(input_ids_list, attention_mask_list):
        if ids.shape[1] < max_seq_len:
            padding_size = max_seq_len - ids.shape[1]
            padding = torch.zeros(ids.shape[0], padding_size, dtype=ids.dtype, device=ids.device)
            padded_ids = torch.cat([ids, padding], dim=1)
            if mask is not None:
                mask_padding = torch.zeros(
                    mask.shape[0], padding_size, dtype=mask.dtype, device=mask.device
                )
                padded_mask = torch.cat([mask, mask_padding], dim=1)
            else:
                padded_mask = None
        else:
            padded_ids = ids
            padded_mask = mask

        padded_input_ids.append(padded_ids)
        padded_attention_masks.append(padded_mask)

    return {
        "input_ids": torch.cat(padded_input_ids, dim=0),
        "pixel_values": torch.cat([pv for pv in pixel_values_list if pv is not None], dim=0)
        if pixel_values_list and any(pv is not None for pv in pixel_values_list)
        else None,
        "image_grid_thw": torch.cat([igt for igt in image_grid_thw_list if igt is not None], dim=0)
        if image_grid_thw_list and any(igt is not None for igt in image_grid_thw_list)
        else None,
        "attention_mask": torch.cat([m for m in padded_attention_masks if m is not None], dim=0)
        if any(m is not None for m in padded_attention_masks)
        else None,
    }


def _process_language_embeddings_batch(
    language_embeddings: List[torch.Tensor], text_len: int
) -> torch.Tensor:
    """Pad/truncate T5 embeddings to ``text_len`` then stack (verbatim from source)."""
    padded_embeddings = []
    for emb in language_embeddings:
        if emb.shape[0] <= text_len:
            padded = torch.cat([emb, emb.new_zeros(text_len - emb.shape[0], emb.shape[1])])
        else:
            padded = emb[:text_len]
        padded_embeddings.append(padded)
    return torch.stack(padded_embeddings, dim=0)


@register_preprocessor("motus")
class MotusPreprocessor(BasePreprocessor):
    """Collate Motus samples into a :class:`MotusPreparedBatch`."""

    def __init__(self, t5_text_len: int) -> None:
        """Store the T5 text length used for padding language embeddings."""
        self.t5_text_len = int(t5_text_len)

    @classmethod
    def from_config(
        cls,
        model_cfg,
        data_cfg,
        training_args=None,
        dataset_stats=None,
        dataset=None,
    ) -> "MotusPreprocessor":
        """Build the preprocessor, resolving the T5 text length from the data config."""
        t5_text_len = data_cfg.t5_text_len
        return cls(t5_text_len=t5_text_len)

    def __call__(self, examples: List[Optional[Dict[str, Any]]]) -> MotusPreparedBatch:
        """Collate a list of (nullable) samples into a padded :class:`MotusPreparedBatch`."""
        batch = [s for s in examples if s is not None]
        if len(batch) == 0:
            return MotusPreparedBatch()

        first_frames = torch.stack([s["first_frame"] for s in batch])
        video_frames = torch.stack([s["video_frames"] for s in batch])
        action_sequences = torch.stack([s["action_sequence"] for s in batch])

        has_initial_state = all(
            ("initial_state" in s and s["initial_state"] is not None) for s in batch
        )
        initial_states = (
            torch.stack([s["initial_state"] for s in batch]) if has_initial_state else None
        )

        vlm_inputs = [s.get("vlm_inputs") for s in batch]
        processed_vlm_inputs = None
        if vlm_inputs and all(v is not None for v in vlm_inputs):
            processed_vlm_inputs = _process_vlm_inputs_batch(vlm_inputs)

        language_embeddings = [
            s.get("language_embedding") for s in batch if "language_embedding" in s
        ]
        processed_language_embeddings = None
        if language_embeddings and any(e is not None for e in language_embeddings):
            processed_language_embeddings = _process_language_embeddings_batch(
                language_embeddings, self.t5_text_len
            )

        # Offline latent cache (--latent-cache-dir): if every sample carries a
        # precomputed latent, stack it (fp32, unchanged dtype) so the trainer can
        # skip the online VAE encode. Absent on the normal path -> None.
        has_latent = all(
            ("clean_full_latent" in s and s["clean_full_latent"] is not None)
            for s in batch
        )
        clean_full_latent = (
            torch.stack([s["clean_full_latent"] for s in batch]) if has_latent else None
        )

        return MotusPreparedBatch(
            first_frame=first_frames,
            video_frames=video_frames,
            action_sequence=action_sequences,
            initial_state=initial_states,
            language_embedding=processed_language_embeddings,
            vlm_inputs=processed_vlm_inputs,
            clean_full_latent=clean_full_latent,
        )
