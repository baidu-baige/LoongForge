# Copyright 2026 The LoongForge Authors.
# SPDX-License-Identifier: Apache-2.0

"""SQLite manifest helpers for WDS-native offline packing."""

import json
import sqlite3
from pathlib import Path
from typing import Dict, List, Mapping, Optional, Tuple

from wds_pack.core.types import ManifestMember, ManifestSample, PackItem, SampleTokenInfo


def _connect(path: Path) -> sqlite3.Connection:
    conn = sqlite3.connect(path)
    conn.row_factory = sqlite3.Row
    return conn


def create_manifest(path: Path) -> sqlite3.Connection:
    """Create a new manifest database and return an open connection."""
    if path.exists():
        path.unlink()
    conn = _connect(path)
    conn.execute(
        """
        CREATE TABLE samples (
            sample_id TEXT PRIMARY KEY,
            media_type TEXT NOT NULL,
            token_len INTEGER NOT NULL,
            shard TEXT NOT NULL,
            base_key TEXT NOT NULL,
            prompt TEXT NOT NULL,
            caption TEXT NOT NULL,
            media_files_json TEXT NOT NULL
        )
        """
    )
    conn.execute(
        """
        CREATE TABLE members (
            sample_id TEXT NOT NULL,
            part TEXT NOT NULL,
            member TEXT NOT NULL,
            offset_data INTEGER NOT NULL,
            size INTEGER NOT NULL,
            PRIMARY KEY (sample_id, part)
        )
        """
    )
    conn.execute("CREATE INDEX idx_samples_media_type ON samples(media_type)")
    return conn


def insert_manifest_row(conn: sqlite3.Connection, row: Mapping[str, object]) -> None:
    """Insert one scanner row and its media byte references."""
    conn.execute(
        """
        INSERT INTO samples
        (sample_id, media_type, token_len, shard, base_key, prompt, caption, media_files_json)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?)
        """,
        (
            row["sample_id"],
            row["media_type"],
            int(row["token_len"]),
            row["shard"],
            row["base_key"],
            row["prompt"],
            row["caption"],
            json.dumps(row["media_files"], ensure_ascii=False),
        ),
    )
    for member in row["members"]:
        conn.execute(
            """
            INSERT INTO members
            (sample_id, part, member, offset_data, size)
            VALUES (?, ?, ?, ?, ?)
            """,
            (
                row["sample_id"],
                member["part"],
                member["member"],
                int(member["offset_data"]),
                int(member["size"]),
            ),
        )


def _sample_from_row(row: sqlite3.Row) -> ManifestSample:
    return ManifestSample(
        sample_id=row["sample_id"],
        media_type=row["media_type"],
        token_len=int(row["token_len"]),
        shard=row["shard"],
        base_key=row["base_key"],
        prompt=row["prompt"],
        caption=row["caption"],
        media_files=tuple(json.loads(row["media_files_json"])),
    )


def load_manifest_samples(
    manifest_sqlite: Path,
    media_type: Optional[str] = None,
) -> Dict[str, ManifestSample]:
    """Load manifest samples indexed by sample_id."""
    if not manifest_sqlite.exists():
        raise FileNotFoundError(f"Manifest sqlite not found: {manifest_sqlite}")

    conn = _connect(manifest_sqlite)
    try:
        if media_type is None:
            rows = conn.execute(
                "SELECT * FROM samples ORDER BY sample_id"
            ).fetchall()
        else:
            rows = conn.execute(
                "SELECT * FROM samples WHERE media_type = ? ORDER BY sample_id",
                (media_type,),
            ).fetchall()
    finally:
        conn.close()

    return {row["sample_id"]: _sample_from_row(row) for row in rows}


def load_pack_items(manifest_sqlite: Path, media_type: str) -> List[PackItem]:
    """Load the structured items consumed by hash-bucket packing."""
    if not manifest_sqlite.exists():
        raise FileNotFoundError(f"Manifest sqlite not found: {manifest_sqlite}")

    conn = _connect(manifest_sqlite)
    try:
        rows = conn.execute(
            """
            SELECT sample_id, token_len
            FROM samples
            WHERE media_type = ?
            ORDER BY sample_id
            """,
            (media_type,),
        ).fetchall()
    finally:
        conn.close()

    return [
        PackItem(sample_id=row["sample_id"], token_len=int(row["token_len"]))
        for row in rows
    ]


def load_sample_token_index(manifest_sqlite: Path) -> Dict[str, SampleTokenInfo]:
    """Load only the fields needed to validate/build pack plans."""
    if not manifest_sqlite.exists():
        raise FileNotFoundError(f"Manifest sqlite not found: {manifest_sqlite}")

    conn = _connect(manifest_sqlite)
    try:
        rows = conn.execute(
            """
            SELECT sample_id, media_type, token_len
            FROM samples
            ORDER BY sample_id
            """
        ).fetchall()
    finally:
        conn.close()

    return {
        row["sample_id"]: SampleTokenInfo(
            sample_id=row["sample_id"],
            media_type=row["media_type"],
            token_len=int(row["token_len"]),
        )
        for row in rows
    }


def load_manifest_members(manifest_sqlite: Path) -> Dict[str, Dict[str, ManifestMember]]:
    """Load media members indexed by sample_id and logical member part."""
    if not manifest_sqlite.exists():
        raise FileNotFoundError(f"Manifest sqlite not found: {manifest_sqlite}")

    conn = _connect(manifest_sqlite)
    try:
        rows = conn.execute("SELECT * FROM members ORDER BY sample_id, part").fetchall()
    finally:
        conn.close()

    members: Dict[str, Dict[str, ManifestMember]] = {}
    for row in rows:
        members.setdefault(row["sample_id"], {})[row["part"]] = ManifestMember(
            part=row["part"],
            member=row["member"],
            offset_data=int(row["offset_data"]),
            size=int(row["size"]),
        )
    return members


def load_manifest_for_packing(
    manifest_sqlite: Path,
) -> Tuple[Dict[str, ManifestSample], Dict[str, Dict[str, ManifestMember]]]:
    """Load both sample metadata and tar byte locators for WDS writing."""
    return load_manifest_samples(manifest_sqlite), load_manifest_members(manifest_sqlite)
