# Copyright 2026 The LoongForge Authors.
# SPDX-License-Identifier: Apache-2.0

"""GigaBrain07 collator + batch preprocessor (single module, xvla-style).

Holds the whole collation layer for giga_brain07:

* ``GigaBrainCollator`` -- the generic recursive structural collator (field-wise
  stacking, padding to the max shape via ``giga_datasets.utils.stack_data`` when
  samples differ). Renamed from ``DefaultCollator`` so it does not clash with
  other models' collator names.
* ``GigaBrain07Batch`` -- a ``dict`` batch with tensor-recursive ``.to`` /
  ``.pin_memory`` (mirrors xvla's ``XVLABatch``).
* ``GigaBrain07Preprocessor`` -- the ``@register_preprocessor("giga_brain07")``
  DataLoader ``collate_fn`` (reachable through ``build_preprocessor`` /
  ``discover_preprocessors``, like ``XVLAPreprocessor``). It delegates the actual
  stacking to ``GigaBrainCollator`` and wraps the result in ``GigaBrain07Batch``.

The ``giga_brain07_preprocessor`` module was merged into this file so the model
keeps a single collator source, matching xvla's one-collator layout. Collation
logic is unchanged (tensors are byte-identical to the previous implementation).
"""

from typing import Any, Dict, List

import numpy as np
import torch

from loongforge.embodied.data.datasets.transforms.collator import (
    BasePreprocessor,
    register_preprocessor,
)
from loongforge.embodied.data.datasets.giga_brain07.giga_datasets import utils


class GigaBrainCollator:
    """Collator that stacks common array/tensor fields.

    Args:
        is_equal: If True, enforce equal-sized tensors/arrays when stacking.
    """

    def __init__(self, is_equal: bool = False) -> None:
        """Initialize the collator.

        Args:
            is_equal (bool): If True, enforce equal-sized tensors/arrays when stacking.
        """
        self.is_equal = is_equal

    def __call__(self, batch: dict | list[dict]) -> dict:
        """Collate a list/dict batch into a dict of stacked tensors/arrays.

        Args:
            batch (dict | list[dict]): A batch either as a list of sample dicts or already as a dict of lists.

        Returns:
            dict: A dict whose values are collated tensors/arrays following field-wise stacking.
        """
        batch_dict = dict()
        if isinstance(batch, list):
            for key in batch[0]:
                # Column-wise collate across list of dicts
                batch_dict[key] = self._collate([d[key] for d in batch])
        elif isinstance(batch, dict):
            for key in batch:
                # Already column-oriented: collate each value container
                batch_dict[key] = self._collate(batch[key])
        else:
            assert False
        return batch_dict

    def _collate(self, batch: list | tuple | np.ndarray | torch.Tensor | dict):
        """Recursively collate nested containers into tensors where possible.

        Args:
            batch: A nested container of tensors/arrays/lists/dicts to be collated.

        Returns:
            Collated structure with tensors/arrays stacked where applicable.
        """
        if isinstance(batch, (list, tuple)):
            if isinstance(batch[0], torch.Tensor):
                # Stack tensors directly, with optional equal-shape enforcement
                batch = utils.stack_data(batch, is_equal=self.is_equal)
            elif isinstance(batch[0], np.ndarray):
                batch = utils.stack_data(batch, is_equal=self.is_equal)
                batch = torch.from_numpy(batch)
            elif isinstance(batch[0], (np.bool_, np.number, np.object_)):
                # Scalars to tensor
                batch = torch.as_tensor(batch)
            elif isinstance(batch[0], dict):
                # Dict of lists -> dict of tensors (recursive)
                batch = {key: self._collate([d[key] for d in batch]) for key in batch[0]}
            elif isinstance(batch[0], (list, tuple)):
                # Transpose list-of-lists/tuples, then collate each field
                batch = type(batch[0])([self._collate(d) for d in zip(*batch)])
        elif isinstance(batch, np.ndarray):
            batch = torch.from_numpy(batch)
        return batch


class GigaBrain07Batch(dict):
    """Dict batch with tensor-recursive ``to`` / ``pin_memory`` helpers."""

    def to(self, device: torch.device) -> "GigaBrain07Batch":
        """Move all tensor values (recursing into dict/list/tuple) to device."""
        def move(value):
            if isinstance(value, torch.Tensor):
                return value.to(device, non_blocking=True)
            if isinstance(value, dict):
                return {k: move(v) for k, v in value.items()}
            if isinstance(value, list):
                return [move(v) for v in value]
            if isinstance(value, tuple):
                return tuple(move(v) for v in value)
            return value

        for key, value in list(self.items()):
            self[key] = move(value)
        return self

    def pin_memory(self) -> "GigaBrain07Batch":
        """Pin CPU tensors (recursing) so H2D copies can be non-blocking."""
        def pin(value):
            if isinstance(value, torch.Tensor):
                if value.device.type != "cpu":
                    return value
                try:
                    return value.pin_memory()
                except RuntimeError:
                    return value
            if isinstance(value, dict):
                return {k: pin(v) for k, v in value.items()}
            if isinstance(value, list):
                return [pin(v) for v in value]
            if isinstance(value, tuple):
                return tuple(pin(v) for v in value)
            return value

        for key, value in list(self.items()):
            self[key] = pin(value)
        return self


@register_preprocessor("giga_brain07")
class GigaBrain07Preprocessor(BasePreprocessor):
    """DataLoader collate_fn for giga_brain07.

    Delegates to :class:`GigaBrainCollator` (unchanged collation / padding), then
    wraps the resulting dict in :class:`GigaBrain07Batch`.
    """

    def __init__(self, is_equal: bool = False) -> None:
        self._collator = GigaBrainCollator(is_equal=is_equal)

    @classmethod
    def from_config(
        cls, model_cfg, data_cfg=None, training_args=None, dataset_stats=None, dataset=None,
    ) -> "GigaBrain07Preprocessor":
        """Construct from typed configs (is_equal defaults to False, matching
        the former ``GigaBrainCollator()`` call-site)."""
        is_equal = bool(getattr(data_cfg, "collator_is_equal", False)) if data_cfg is not None else False
        return cls(is_equal=is_equal)

    def __call__(self, examples: List[Dict[str, Any]]) -> GigaBrain07Batch:
        """Collate a list of sample dicts into a GigaBrain07Batch."""
        return GigaBrain07Batch(self._collator(examples))
