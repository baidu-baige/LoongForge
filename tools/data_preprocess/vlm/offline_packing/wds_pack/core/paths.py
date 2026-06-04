# Copyright 2026 The LoongForge Authors.
# SPDX-License-Identifier: Apache-2.0

"""Filesystem path helpers for offline packing artifacts."""

import os
from pathlib import Path
from typing import Dict, Union


def get_temp_dir(wds_dir: Union[str, Path]) -> Path:
    """.temp folder."""
    wds_dir = Path(wds_dir)
    temp_dir = wds_dir / ".temp"
    temp_dir.mkdir(parents=True, exist_ok=True)
    return temp_dir


def get_sample_record_path(wds_dir: Union[str, Path]) -> Path:
    """.temp/sample_record.txt."""
    return get_temp_dir(wds_dir) / "sample_record.txt"


def get_token_info_report_path(wds_dir: Union[str, Path]) -> Path:
    """.temp/sample_len_report.txt."""
    return get_temp_dir(wds_dir) / "sample_len_report.txt"


def get_log_file_path(wds_dir: Union[str, Path]) -> Path:
    """.temp/log.txt."""
    return get_temp_dir(wds_dir) / "log.txt"


def get_packed_output_dir(cfg: Dict) -> Path:
    """Return the work/output directory for packing artifacts."""
    wds_dir = Path(cfg["data"]["wds_dir"])
    custom_output = cfg["data"].get("work_dir") or cfg["data"].get("packed_json_dir")

    if custom_output:
        packed_dir = Path(custom_output)
    else:
        packed_dir = wds_dir / "packed_json"

    packed_dir.mkdir(parents=True, exist_ok=True)
    return packed_dir


def get_work_dir(cfg: Dict) -> Path:
    """Return the working directory for WDS-native offline packing."""
    return get_packed_output_dir(cfg)


def get_manifest_jsonl_path(cfg: Dict) -> Path:
    """sample_manifest.jsonl path."""
    return get_work_dir(cfg) / "sample_manifest.jsonl"


def get_manifest_sqlite_path(cfg: Dict) -> Path:
    """sample_manifest.sqlite path."""
    return get_work_dir(cfg) / "sample_manifest.sqlite"


def get_token_report_dir(cfg: Dict) -> Path:
    """Directory for per-media token reports."""
    path = get_work_dir(cfg) / "token_len"
    path.mkdir(parents=True, exist_ok=True)
    return path


def get_token_report_path(cfg: Dict, media_type: str) -> Path:
    """Per-media token report path."""
    return get_token_report_dir(cfg) / f"sample_len_report_{media_type}.txt"


def get_combined_token_report_path(cfg: Dict) -> Path:
    """Combined token report path for diagnostics."""
    return get_work_dir(cfg) / "sample_len_report.txt"


def get_bins_dir(cfg: Dict) -> Path:
    """Directory for per-media hashbucket outputs."""
    path = get_work_dir(cfg) / "bins"
    path.mkdir(parents=True, exist_ok=True)
    return path


def get_bins_path(cfg: Dict, media_type: str) -> Path:
    """Per-media bins pkl path."""
    return get_bins_dir(cfg) / f"bins_boxs_{media_type}.pkl"


def get_bins_plan_path(cfg: Dict, media_type: str) -> Path:
    """Compact per-media bins plan path."""
    return get_bins_dir(cfg) / f"bins_plan_{media_type}.jsonl"


def get_pack_plan_path(cfg: Dict) -> Path:
    """pack_plan.jsonl path."""
    return get_work_dir(cfg) / "pack_plan.jsonl"


def get_init_file(cfg):
    max_token_len = cfg["sample"]["max_token_len"]
    wds_dir = Path(cfg["data"]["wds_dir"])
    packed_files_dir = get_packed_output_dir(cfg)
    os.makedirs(packed_files_dir, exist_ok=True)
    token_info_file = get_token_info_report_path(wds_dir)
    return str(token_info_file), max_token_len, str(packed_files_dir), str(wds_dir)
