# Copyright 2026 The LoongForge Authors.
# SPDX-License-Identifier: Apache-2.0
#
# Modified from Megatron-LM under the BSD 3-Clause License.
# Copyright (c) 2024, NVIDIA CORPORATION. All rights reserved.

"""Rerun state wrapper with proper iterator protocol support.

Extends Megatron's RerunDataIterator to add __iter__ support and
properly distinguish between iterables and iterators.
"""

from typing import Any

from megatron.core.rerun_state_machine import (
    RerunDataIterator as _OrigRerunDataIterator,
    RerunMode,
    get_rerun_state_machine,
)


class RerunDataIterator(_OrigRerunDataIterator):
    """Extended RerunDataIterator with __iter__ and proper iterator tracking.

    Differences from the base class:
    - Stores a separate ``_iterator`` object so that the original ``iterable``
      can be re-iterated via ``__iter__``.
    - Adds ``__iter__`` to allow resetting and re-iterating.
    - ``__next__`` reads from ``_iterator`` instead of ``self.iterable``
      directly, which is necessary when the iterable is not itself an iterator.
    """

    def __init__(self, iterable: Any) -> None:
        super().__init__(iterable)
        self._iterator = iterable if hasattr(iterable, "__iter__") else iter(iterable)

    def __iter__(self) -> "RerunDataIterator":
        self._iterator = (
            self.iterable if hasattr(self.iterable, "__iter__") else iter(self.iterable)
        )
        self.saved_microbatches = []
        self.replaying = False
        self.replay_pos = 0
        return self

    def __next__(self) -> Any:
        if self.replaying:
            assert (
                len(self.saved_microbatches) > self.replay_pos
            ), "No more batches to replay"
            n = self.saved_microbatches[self.replay_pos]
            self.replay_pos += 1
            return n
        n: Any = next(self._iterator)
        if get_rerun_state_machine().get_mode() != RerunMode.DISABLED:
            self.saved_microbatches.append(n)
        return n
