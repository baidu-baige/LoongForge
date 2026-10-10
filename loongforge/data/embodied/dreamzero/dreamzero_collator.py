# Copyright 2026 The LoongForge Authors.
# SPDX-License-Identifier: Apache-2.0
#
# Modified from DreamZero under the Apache-2.0 License.

"""DreamZero collation utilities.

Provides text cleaning helpers, a Hugging Face tokenizer wrapper, a batch
collate function that expands per-embodiment language prompts, and the
``DreamZeroCollator`` used to prepare DreamZero batches (video, state, action,
language) for training and inference.
"""

import ast
import html
import os
from typing import Any, Dict, List

import ftfy
import numpy as np
import regex as re
import torch
from transformers import AutoTokenizer
from transformers.data.data_collator import DataCollatorMixin

from loongforge.data.embodied.collator import _move_to_device, _pin_memory
from loongforge.data.embodied.registry import register_collator

from .schema import EmbodimentTag


def basic_clean(text):
    """Fix mojibake/encoding issues and unescape HTML entities in ``text``."""
    text = ftfy.fix_text(text)
    text = html.unescape(html.unescape(text))
    return text.strip()


def whitespace_clean(text):
    """Collapse repeated whitespace in ``text`` into single spaces."""
    text = re.sub(r'\s+', ' ', text)
    text = text.strip()
    return text


class HuggingfaceTokenizer:
    """Thin wrapper around a Hugging Face tokenizer with optional cleaning."""

    def __init__(self, name, seq_len=None, clean=None, **kwargs):
        """Load the tokenizer by name/path and store padding/cleaning options."""
        assert clean in (None, 'whitespace')
        self.name = name
        self.seq_len = seq_len
        self.clean = clean

        # When loading from a local checkpoint path (e.g. from training runs), pass
        # local_files_only=True to avoid HFValidationError from validate_repo_id.
        load_kwargs = dict(kwargs)
        if os.path.isdir(name):
            load_kwargs.setdefault("local_files_only", True)
        # init tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(name, **load_kwargs)
        self.vocab_size = self.tokenizer.vocab_size

    def __call__(self, sequence, **kwargs):
        """Tokenize ``sequence``, optionally returning the attention mask."""
        return_mask = kwargs.pop('return_mask', False)

        # arguments
        _kwargs = {'return_tensors': 'pt'}
        if self.seq_len is not None:
            _kwargs.update({
                'padding': 'max_length',
                'truncation': True,
                'max_length': self.seq_len
            })
        _kwargs.update(**kwargs)


        # tokenization
        if isinstance(sequence, str):
            sequence = [sequence]
        if self.clean:
            sequence = [self._clean(u) for u in sequence]
        ids = self.tokenizer(sequence, **_kwargs)

        # output
        if return_mask:
            return ids.input_ids, ids.attention_mask
        else:
            return ids.input_ids

    def _clean(self, text):
        """Apply the configured text-cleaning strategy to ``text``."""
        if self.clean == 'whitespace':
            text = whitespace_clean(basic_clean(text))
        # elif self.clean == 'lower':
        #     text = whitespace_clean(basic_clean(text)).lower()
        # elif self.clean == 'canonicalize':
        #     text = canonicalize(basic_clean(text))
        return text


def collate(features: List[dict], tokenizer: AutoTokenizer, num_views=3, embodiment_tag_mapping=None) -> dict:
    """Collate a list of per-sample dicts into a batched dict, tokenizing text fields."""
    batch = {}
    keys = features[0].keys()

    for key in keys:
        if key == "text":
            output_values = []
            for elem in features:
                item = elem[key]
                try:
                    parsed_item = ast.literal_eval(item)
                    # Handle different return types from ast.literal_eval
                    if isinstance(parsed_item, (list, tuple)):
                        processed_item = str(parsed_item[0])
                    else:
                        # If it's already a scalar (string, float, int, etc.), convert to string
                        processed_item = str(parsed_item)

                    if num_views > 1 and elem["embodiment_id"] == embodiment_tag_mapping[EmbodimentTag.AGIBOT.value]:
                        processed_item = (
                            "A multi-view video shows that a robot "
                            + processed_item.lower()
                            + " The video is split into four views: The top-left view shows the camera view "
                            "from the robot's head, the top-right view shows the camera view from the right "
                            "hand, the bottom-left view shows the camera view from the left hand, and the "
                            "bottom-right view is a black screen (inactive view). The robot "
                            + processed_item.lower()
                        )
                    elif elem["embodiment_id"] == embodiment_tag_mapping[EmbodimentTag.OXE_DROID.value]:
                        processed_item = (
                            "A multi-view video shows that a robot "
                            + processed_item.lower()
                            + " The video is split into three views: The top view shows the camera view "
                            "from the robot's wrist, the bottom-left view shows the camera view from the "
                            "left exterior camera, and the bottom-right view shows the camera view from the "
                            "right exterior camera. During training, one of the two bottom exterior views "
                            "may be a black screen (dropped view). The robot "
                            + processed_item.lower()
                        )
                    elif elem["embodiment_id"] == embodiment_tag_mapping[EmbodimentTag.LIBERO_SIM.value]:
                        processed_item = (
                            "A multi-view video shows that a robot "
                            + processed_item.lower()
                            + " The video is split into two horizontal views: the left view shows the "
                            "exterior camera and the right view shows the wrist camera. The robot "
                            + processed_item.lower()
                        )
                    elif elem["embodiment_id"] == embodiment_tag_mapping[EmbodimentTag.GR1_UNIFIED.value]:
                        processed_item = "A single view video shows that a human " + processed_item.lower()
                    elif elem["embodiment_id"] == embodiment_tag_mapping[EmbodimentTag.MECKA_HANDS.value]:
                        processed_item = "A single view video shows that a human " + processed_item.lower()
                    elif elem["embodiment_id"] == embodiment_tag_mapping[EmbodimentTag.XDOF.value]:
                        processed_item = (
                            "A multi-view video shows that a robot "
                            + processed_item.lower()
                            + " The video is split into four views: The top-left view shows the camera view "
                            "from the robot's head, the top-right view shows the camera view from the right "
                            "hand, the bottom-left view shows the camera view from the left hand, and the "
                            "bottom-right view is a black screen (inactive view). The robot "
                            + processed_item.lower()
                        )
                    elif elem["embodiment_id"] == embodiment_tag_mapping[EmbodimentTag.YAM.value]:
                        processed_item = (
                            "A multi-view video shows that a robot "
                            + processed_item.lower()
                            + " The video is split into four views: The top-left view shows the top camera, "
                            "the top-right view shows the right camera, the bottom-left view shows the left "
                            "camera, and the bottom-right view is a black screen. The robot "
                            + processed_item.lower()
                        )
                    else:
                        raise ValueError(f"Embodiment ID {elem['embodiment_id']} not supported.")
                    output_values.append(processed_item)
                except (ValueError, SyntaxError, TypeError):
                    # If parsing fails or item is already a string, use it directly
                    if num_views > 1 and elem["embodiment_id"] == embodiment_tag_mapping[EmbodimentTag.AGIBOT.value]:
                        item = (
                            "A multi-view video shows that a robot "
                            + str(item).lower()
                            + " The video is split into four views: The top-left view shows the camera view "
                            "from the robot's head, the top-right view shows the camera view from the right "
                            "hand, the bottom-left view shows the camera view from the left hand, and the "
                            "bottom-right view is a black screen (inactive view). The robot "
                            + str(item).lower()
                        )
                    elif elem["embodiment_id"] == embodiment_tag_mapping[EmbodimentTag.OXE_DROID.value]:
                        item = (
                            "A multi-view video shows that a robot "
                            + str(item).lower()
                            + " The video is split into three views: The top view shows the camera view "
                            "from the robot's wrist, the bottom-left view shows the camera view from the "
                            "left exterior camera, and the bottom-right view shows the camera view from the "
                            "right exterior camera. During training, one of the two bottom exterior views "
                            "may be a black screen (dropped view). The robot "
                            + str(item).lower()
                        )
                    elif elem["embodiment_id"] == embodiment_tag_mapping[EmbodimentTag.LIBERO_SIM.value]:
                        item = (
                            "A multi-view video shows that a robot "
                            + str(item).lower()
                            + " The video is split into two horizontal views: the left view shows the "
                            "exterior camera and the right view shows the wrist camera. The robot "
                            + str(item).lower()
                        )
                    elif elem["embodiment_id"] == embodiment_tag_mapping[EmbodimentTag.GR1_UNIFIED.value]:
                        item = "A single view video shows that a human " + str(item).lower()
                    elif elem["embodiment_id"] == embodiment_tag_mapping[EmbodimentTag.MECKA_HANDS.value]:
                        item = "A single view video shows that a human " + str(item).lower()
                    elif elem["embodiment_id"] == embodiment_tag_mapping[EmbodimentTag.XDOF.value]:
                        item = (
                            "A multi-view video shows that a robot "
                            + str(item).lower()
                            + " The video is split into four views: The top-left view shows the camera view "
                            "from the robot's head, the top-right view shows the camera view from the right "
                            "hand, the bottom-left view shows the camera view from the left hand, and the "
                            "bottom-right view is a black screen (inactive view). The robot "
                            + str(item).lower()
                        )
                    elif elem["embodiment_id"] == embodiment_tag_mapping[EmbodimentTag.YAM.value]:
                        item = (
                            "A multi-view video shows that a robot "
                            + str(item).lower()
                            + " The video is split into four views: The top-left view shows the top camera, "
                            "the top-right view shows the right camera, the bottom-left view shows the left "
                            "camera, and the bottom-right view is a black screen. The robot "
                            + str(item).lower()
                        )
                    else:
                        raise ValueError(f"Embodiment ID {elem['embodiment_id']} not supported.")
                    output_values.append(item)
            ids, mask = tokenizer(output_values, return_mask=True, add_special_tokens=True)
            batch[key] = ids
            batch['text_attention_mask'] = mask
        elif key == "text_negative":
            values = [elem[key] for elem in features]
            ids, mask = tokenizer(values, return_mask=True, add_special_tokens=True)
            batch[key] = ids
            batch['text_attention_mask_negative'] = mask
        else:
            values = [elem[key] for elem in features]
            try:
                if all(torch.is_tensor(value) for value in values):
                    batch[key] = torch.stack(values)
                else:
                    batch[key] = torch.from_numpy(np.stack(values))
            except (RuntimeError, ValueError) as exc:
                shape_info = [
                    {
                        "index": i,
                        "shape": getattr(value, "shape", None),
                        "dtype": str(getattr(value, "dtype", type(value))),
                    }
                    for i, value in enumerate(values)
                ]
                raise ValueError(
                    f"DreamZero collate failed for key={key!r}; "
                    f"batch_size={len(values)}; shapes={shape_info}"
                ) from exc
    return batch



class DreamZeroPreparedBatch(dict):
    """Dict-backed batch honoring the framework ``PreparedBatch`` contract.

    DreamZero batches have dynamic, embodiment-dependent keys (video/state/action/
    text/... vary by embodiment + language mode), so a typed ``PreparedBatch``
    dataclass (pi05/groot style) does not fit. This ``dict`` subclass keeps
    ``batch["k"]`` and ``isinstance(batch, dict)`` working for
    ``DreamZeroPolicy.forward`` / ``_prepare_action_input`` unchanged, while adding
    ``.to(device)`` / ``pin_memory()`` so the generic trainer's
    ``_move_batch_to_device`` moves it uniformly like other models' PreparedBatch.

    Values are identical to the legacy plain-dict output; only the container type
    changes. Device movement now happens in the trainer (idempotent w.r.t. the
    per-tensor move still done inside ``_prepare_action_input``).
    """

    def to(self, device) -> "DreamZeroPreparedBatch":
        """Move all tensor-like values in the batch to ``device``."""
        for key in list(self.keys()):
            self[key] = _move_to_device(self[key], device)
        return self

    def pin_memory(self) -> "DreamZeroPreparedBatch":
        """Pin all tensor-like values in the batch for asynchronous host-to-device copies."""
        for key in list(self.keys()):
            self[key] = _pin_memory(self[key])
        return self


@register_collator("dreamzero")
class DreamZeroCollator(DataCollatorMixin):
    """Data collator registry entry for DreamZero batches."""

    @classmethod
    def from_config(cls, model_cfg, data_cfg, training_args=None, dataset_stats=None, dataset=None):
        """Build the DreamZero collator from typed configs (collator registry entry).

        Produces a ``DreamZeroPreparedBatch`` (dict-backed, ``.to(device)``-capable);
        tensor values are unchanged relative to the legacy dataloader-builder path.
        """
        from loongforge.data.embodied.dreamzero.modality_configs import (
            EMBODIMENT_TAG_TO_ID,
        )

        base_dataset = dataset._dreamzero_base_dataset
        num_views = len(base_dataset.modality_configs["video"].modality_keys)
        return cls(
            tokenizer_path=training_args.tokenizer_path,
            max_length=int(data_cfg.max_text_length),
            num_views=num_views,
            embodiment_tag_mapping=EMBODIMENT_TAG_TO_ID,
        )

    def __init__(
        self,
        tokenizer_path: str = "google/umt5-xxl",
        max_length: int = 512,
        num_views: int = 1,
        embodiment_tag_mapping=None,
    ):
        """Build the tokenizer used to collate text fields in a batch."""
        super().__init__()
        self.tokenizer = HuggingfaceTokenizer(name=tokenizer_path, seq_len=max_length, clean='whitespace')
        self.num_views = num_views
        self.embodiment_tag_mapping = embodiment_tag_mapping

    def __call__(self, features: List[Dict[str, Any]]) -> "DreamZeroPreparedBatch":
        """Collate a list of feature dicts into a training batch (PreparedBatch-equivalent)."""
        return DreamZeroPreparedBatch(
            collate(features, self.tokenizer, self.num_views, self.embodiment_tag_mapping)
        )
