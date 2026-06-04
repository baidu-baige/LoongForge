# Copyright 2026 The LoongForge Authors.
# SPDX-License-Identifier: Apache-2.0

"""Typed records shared by the WDS-native offline packing stages."""

from dataclasses import dataclass
from typing import Tuple

HASH_BUCKET_WEIGHT_KEY = "w"
HASH_BUCKET_TOKEN_LEN_KEY = "l"
HASH_BUCKET_SAMPLE_ID_KEY = "name"


@dataclass(frozen=True)
class PackItem:
    """Minimal structured item consumed by the hash-bucket algorithm."""

    sample_id: str
    token_len: int
    weight: int = 0


def pack_item_from_hashbucket_record(record) -> PackItem:
    """Convert one legacy hashbucket ndarray record into a structured item."""
    return PackItem(
        sample_id=str(record[HASH_BUCKET_SAMPLE_ID_KEY]),
        token_len=int(record[HASH_BUCKET_TOKEN_LEN_KEY]),
        weight=int(record[HASH_BUCKET_WEIGHT_KEY]),
    )


def pack_item_to_hashbucket_tuple(item: PackItem) -> Tuple[int, int, str]:
    """Convert a structured item to the legacy ndarray tuple layout."""
    return (int(item.weight), int(item.token_len), str(item.sample_id))


@dataclass(frozen=True)
class SampleTokenInfo:
    """Slim manifest view used when only packing metadata is needed."""

    sample_id: str
    media_type: str
    token_len: int


@dataclass(frozen=True)
class ManifestMember:
    """Byte location for one media member inside a source tar shard."""

    part: str
    member: str
    offset_data: int
    size: int


@dataclass(frozen=True)
class ManifestSample:
    """One valid source sample recorded in the offline-packing manifest."""

    sample_id: str
    media_type: str
    token_len: int
    shard: str
    base_key: str
    prompt: str
    caption: str
    media_files: Tuple[str, ...]

    def pack_item(self) -> PackItem:
        return PackItem(sample_id=self.sample_id, token_len=self.token_len)
