# Copyright 2026 The LoongForge Authors.
# SPDX-License-Identifier: Apache-2.0

"""Msgpack helpers with NumPy support.

Kept self-contained so this eval module does not depend on model internals.
"""

from __future__ import annotations

import msgpack
import numpy as np


def encode(obj):
    """Run encode."""
    if isinstance(obj, np.ndarray):
        return {
            b"__ndarray__": True,
            b"dtype": str(obj.dtype),
            b"shape": obj.shape,
            b"data": obj.tobytes(),
        }
    if isinstance(obj, np.generic):
        return obj.item()
    raise TypeError(f"Cannot msgpack encode object of type {type(obj)}")


def decode(obj):
    """Run decode."""
    if b"__ndarray__" in obj:
        array = np.frombuffer(
            obj[b"data"], dtype=np.dtype(obj[b"dtype"].decode() if isinstance(obj[b"dtype"], bytes) else obj[b"dtype"])
        )
        return array.reshape(obj[b"shape"])
    return obj


class Packer:
    """Provide Packer behavior."""

    def pack(self, obj):
        """Run pack."""
        return msgpack.packb(obj, default=encode, use_bin_type=True)


def packb(obj):
    """Run packb."""
    return msgpack.packb(obj, default=encode, use_bin_type=True)


def unpackb(payload):
    """Run unpackb."""
    return msgpack.unpackb(payload, object_hook=decode, raw=False)
