# Copyright 2026 The LoongForge Authors.
# SPDX-License-Identifier: Apache-2.0

"""Python interface for GR00T-N1.7 DDP reducer bucket control."""

from . import _ddp_reducer_bucket_control


def _extension():
    return _ddp_reducer_bucket_control


def initialize_buckets(reducer, bucket_indices: list[list[int]]) -> None:
    """Initialize reducer buckets using parameter-index groups."""
    _extension().ddp_reducer_initialize_buckets(reducer, bucket_indices)


def get_buckets(reducer):
    """Return the reducer's current GradBucket objects."""
    return _extension().ddp_reducer_get_buckets(reducer)
