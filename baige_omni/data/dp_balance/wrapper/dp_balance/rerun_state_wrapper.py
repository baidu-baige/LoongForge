# Copyright 2026 The BaigeOmni Authors.
# SPDX-License-Identifier: Apache-2.0
#
# Modified from Megatron-LM under the BSD 3-Clause License.
# Copyright (c) 2024, NVIDIA CORPORATION. All rights reserved.

"""Rerun state wrapper."""

from typing import Any, Iterable

from megatron.core.rerun_state_machine import (
    RerunStateMachine,
    RerunMode,
    SerializableStateType,
    get_rerun_state_machine,
)


class RerunDataIterator:
    """A wrapper class for data iterators that adds replay capability.

    Args:
        iterable: data iterator that needs the replay capability.
        make_iterable: if set, iterator is created by calling iter() on iterable.

    The RerunState class below uses the rewind capability to replay all the microbatches
    fetched during an iteration.

    Example usage:

        class MyDataIterator:
            ...

        data_iterator = MyDataIterator(...)
        replay_data_iterator = RerunDataIterator(data_iterator)
    """

    def __init__(self, iterable: Iterable[Any]) -> None:
        """__init__ method."""
        self.iterable: Iterable[Any] = iterable
        self.saved_microbatches: list[Any] = []
        self.replaying: bool = False
        self.replay_pos: int = 0
        self._iterator = iterable if hasattr(iterable, "__iter__") else iter(iterable)

    def __iter__(self) -> "RerunDataIterator":
        """__iter__"""

        self._iterator = (
            self.iterable if hasattr(self.iterable, "__iter__") else iter(self.iterable)
        )
        self.saved_microbatches = []
        self.replaying = False
        self.replay_pos = 0
        return self

    def __next__(self) -> Any:
        """__next__ method override adding replay capability."""

        if self.replaying:
            # we should not read past the saved batches if execution is deterministic,
            # as the number of calls to get_batch() should remain the same across reruns
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

    def rewind(self) -> None:
        """Method to rewind the data iterator to the first microbatch of the iteration."""

        self.replaying = True
        self.replay_pos = 0

    def advance(self) -> None:
        """Method to drop all the buffered microbatches and jump to the next iteration."""

        self.replaying = False
        self.saved_microbatches = []

    def state_dict(self) -> SerializableStateType:
        """Method to capture the state of the iterator as a serializable dict."""

        return {
            "saved_microbatches": self.saved_microbatches,
            "replaying": self.replaying,
            "replay_pos": self.replay_pos,
        }

    def load_state_dict(self, state_dict: SerializableStateType) -> None:
        """Method to restore the state saved as a serializable dict."""

        self.saved_microbatches = state_dict["saved_microbatches"]
        self.replaying = state_dict["replaying"]
        self.replay_pos = state_dict["replay_pos"]

