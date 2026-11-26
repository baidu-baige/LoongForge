# Copyright 2026 The LoongForge Authors.
# SPDX-License-Identifier: Apache-2.0

"""Unit and CLI coverage for the ERNIE Preview MXFP4 reference check."""

import importlib.util
import json
import os
from pathlib import Path
import subprocess
import sys

import pytest
import torch
from safetensors.torch import save_file

ROOT = Path(__file__).resolve().parents[3]
SCRIPT = ROOT / "tests/mcore_bridge_roundtrip/ernie5.5/mxfp4_reference_check.py"
SPEC = importlib.util.spec_from_file_location("mxfp4_reference_check", SCRIPT)
mxfp4_reference_check = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(mxfp4_reference_check)

WEIGHT_BASES = (
    "model.layers.3.block_sparse_moe.experts.0.w1",
    "model.layers.3.block_sparse_moe.experts.0.w2",
    "model.layers.3.block_sparse_moe.experts.0.w3",
)
WEIGHT_KEY = [f"{base}.weight" for base in WEIGHT_BASES]

# Reference-side and Preview-side shard names deliberately differ: the check
# must locate every tensor through its own index instead of assuming a layout.
PREVIEW_SHARDS = (
    "preview-00049-of-00128.safetensors",
    "preview-00008-of-00128.safetensors",
    "preview-00009-of-00128.safetensors",
)
REFERENCE_SHARDS = (
    "reference-00049-of-00074.safetensors",
    "reference-00008-of-00074.safetensors",
    "reference-00009-of-00074.safetensors",
)

# Two distinct E8M0 block scales: 2 ** (128 - 127) = 2.0, 2 ** (126 - 127) = 0.5.
SCALE_RAW = (128, 126)
SCALE_VALUES = (2.0, 0.5)

# 32 ratios per block: positive and negative grid points, saturation at +-6,
# exact midpoints (0.25, 0.75, 1.25, 1.75, 2.5, 3.5, 5.0) and zeros. The second
# item is the E2M1 magnitude index the production decode is expected to emit;
# at a midpoint either neighbouring index is accepted, this pattern uses the
# lower one.
_BLOCK_ELEMENTS = (
    (6.0, 7), (-6.0, 7), (4.0, 6), (-4.0, 6), (3.0, 5), (-3.0, 5),
    (2.0, 4), (-2.0, 4), (1.5, 3), (-1.5, 3), (1.0, 2), (-1.0, 2),
    (0.5, 1), (-0.5, 1), (0.0, 0), (-0.0, 0),
    (0.25, 0), (-0.25, 0),  # midpoint 0.0 / 0.5
    (0.75, 1), (-0.75, 1),  # midpoint 0.5 / 1.0
    (1.25, 2), (-1.25, 2),  # midpoint 1.0 / 1.5
    (1.75, 3), (-1.75, 3),  # midpoint 1.5 / 2.0
    (2.5, 4), (-2.5, 4),    # midpoint 2.0 / 3.0
    (3.5, 5), (-3.5, 5),    # midpoint 3.0 / 4.0
    (5.0, 6), (-5.0, 6),    # midpoint 4.0 / 6.0
    (6.0, 7), (-6.0, 7),
)


def _nibble(ratio, magnitude):
    return magnitude | (0x8 if ratio < 0 else 0)


def _packed_weight(swap_nibbles=False):
    packed = bytearray(32)
    for block in range(2):
        for byte_index in range(16):
            element = 2 * byte_index
            low = _nibble(*_BLOCK_ELEMENTS[element])
            high = _nibble(*_BLOCK_ELEMENTS[element + 1])
            packed[block * 16 + byte_index] = low | (high << 4)
    weight = torch.tensor(list(packed), dtype=torch.uint8).view(torch.int8).reshape(1, 32)
    if swap_nibbles:
        weight = (((weight.view(torch.uint8) & 0x0F) << 4)
                  | ((weight.view(torch.uint8) >> 4) & 0x0F)).view(torch.int8)
    return weight


def _triplet(scale_raw=SCALE_RAW, swap_nibbles=False):
    """Return one Preview (weight, scale) pair and its official BF16 reference."""
    scale = torch.tensor([list(scale_raw)], dtype=torch.uint8).view(torch.float8_e8m0fnu)
    reference = torch.tensor(
        [
            ratio * SCALE_VALUES[block]
            for block in range(2)
            for ratio, _ in _BLOCK_ELEMENTS
        ],
        dtype=torch.bfloat16,
    ).reshape(1, 64)
    return _packed_weight(swap_nibbles=swap_nibbles), scale, reference


def _write_sources(tmp_path, corrupt=None):
    preview_dir = tmp_path / "preview"
    reference_dir = tmp_path / "reference"
    preview_dir.mkdir()
    reference_dir.mkdir()
    preview_map = {}
    reference_map = {}
    for index, base in enumerate(WEIGHT_BASES):
        swap_nibbles = corrupt == "nibble_swap" and index == 0
        scale_raw = SCALE_RAW
        if corrupt == "scale" and index == 1:
            scale_raw = (SCALE_RAW[0] + 1,) + SCALE_RAW[1:]
        weight, scale, reference = _triplet(scale_raw=scale_raw, swap_nibbles=swap_nibbles)
        preview_shard = PREVIEW_SHARDS[index]
        reference_shard = REFERENCE_SHARDS[index]
        save_file({f"{base}.weight": weight, f"{base}.weight_scale_inv": scale}, preview_dir / preview_shard)
        save_file({f"{base}.weight": reference}, reference_dir / reference_shard)
        preview_map[f"{base}.weight"] = preview_shard
        preview_map[f"{base}.weight_scale_inv"] = preview_shard
        reference_map[f"{base}.weight"] = reference_shard
    (preview_dir / "model.safetensors.index.json").write_text(
        json.dumps({"weight_map": preview_map})
    )
    (reference_dir / "model.safetensors.index.json").write_text(
        json.dumps({"weight_map": reference_map})
    )
    return preview_dir, reference_dir


def _run_cli(preview_dir, reference_dir, report_path):
    env = os.environ.copy()
    # The documented command runs from the repository root with no PYTHONPATH;
    # the script is responsible for finding its own tools/ package.
    env.pop("PYTHONPATH", None)
    return subprocess.run(
        [
            sys.executable, str(SCRIPT),
            "--preview", str(preview_dir),
            "--reference", str(reference_dir),
            "--report", str(report_path),
        ],
        capture_output=True,
        text=True,
        env=env,
        timeout=300,
    )


def _mismatch_count(reference, decoded):
    """Run the script's pure mismatch counter at unit scale (scale = 1)."""
    reference = torch.tensor(reference, dtype=torch.float32)
    decoded = torch.tensor(decoded, dtype=torch.float32)
    return mxfp4_reference_check.grid_mismatches(
        reference, decoded, torch.ones_like(reference)
    )


class _StubSource:
    """Just enough Source surface for read_tensor's header/offset validation."""

    location = "stub"

    def __init__(self, data=b""):
        self.data = data

    def size(self, name):
        return len(self.data)

    def read_range(self, shard, start, length):
        return self.data[start:start + length]


def _one_tensor_header(offsets, dtype="BF16", shape=(1,)):
    return {
        "tensor": {
            "dtype": dtype,
            "shape": list(shape),
            "data_offsets": list(offsets),
        }
    }


@pytest.mark.parametrize(
    "reference, decoded, expected",
    [
        pytest.param([1.0], [1.1], 1, id="non-grid-magnitude"),
        pytest.param([0.0], [0.1], 1, id="zero-reference-nonzero-decoded"),
        pytest.param([6.0], [100.0], 1, id="clamped-above-max-grid"),
        pytest.param([1.0], [-1.0], 1, id="opposite-sign"),
        pytest.param([-1.5], [1.5], 1, id="opposite-sign-negative-reference"),
        pytest.param([0.5, 0.5], [0.5, 1.1], 1, id="one-off-grid-among-correct"),
        # Element 0 decodes to 1.1, which is off the E2M1 magnitude grid, and
        # element 1 flips the sign of -1.0; element 2 is a correct control.
        # The exact count of 2 keeps the check from collapsing to "a mismatch
        # exists", which would still report 1.
        pytest.param(
            [1.0, -1.0, 2.0], [1.1, 1.0, 2.0], 2,
            id="off-grid-magnitude-and-opposite-sign",
        ),
    ],
)
def test_grid_mismatches_rejects_non_exact_decode(reference, decoded, expected):
    assert _mismatch_count(reference, decoded) == expected


def test_grid_mismatches_accepts_exact_grid_points():
    grid = [
        6.0, 4.0, 3.0, 2.0, 1.5, 1.0, 0.5, 0.0,
        -0.0, -0.5, -1.0, -1.5, -2.0, -3.0, -4.0, -6.0,
    ]
    decoded = [-x if x == 0.0 else x for x in grid]
    assert _mismatch_count(grid, grid) == 0
    assert _mismatch_count(grid, decoded) == 0


def test_grid_mismatches_accepts_both_midpoint_neighbours():
    reference = [0.25, 0.75, 1.25, 1.75, 2.5, 3.5, 5.0]
    lower = [0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0]
    upper = [0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0]
    assert _mismatch_count(reference, lower) == 0
    assert _mismatch_count(reference, upper) == 0


@pytest.mark.parametrize(
    "offsets, shard_data, message",
    [
        pytest.param((-1, 2), b"\x00\x00", "invalid data_offsets", id="negative"),
        pytest.param((2, 1), b"\x00\x00", "invalid data_offsets", id="reversed"),
        pytest.param(("0", 2), b"\x00\x00", "invalid data_offsets", id="non-integer"),
        pytest.param((0, 2), b"\x00", "exceeds", id="past-shard-end"),
    ],
)
def test_read_tensor_rejects_invalid_data_offsets(offsets, shard_data, message):
    with pytest.raises(mxfp4_reference_check.CheckFailure, match=message):
        mxfp4_reference_check.read_tensor(
            _StubSource(shard_data),
            "shard.safetensors",
            _one_tensor_header(offsets),
            0,
            "tensor",
            "BF16",
        )


def test_source_size_failure_is_a_check_failure(tmp_path):
    source = mxfp4_reference_check.Source(str(tmp_path))
    with pytest.raises(
        mxfp4_reference_check.CheckFailure, match="cannot determine the size"
    ):
        source.size("missing.safetensors")


def test_reference_check_reports_unavailable_remote_filesystem(tmp_path):
    preview_dir, _ = _write_sources(tmp_path)
    report_path = tmp_path / "report.json"

    result = _run_cli(preview_dir, "unknown-protocol://bucket/path", report_path)

    assert result.returncode != 0
    assert "cannot initialise source" in result.stderr
    report = json.loads(report_path.read_text())
    assert report["passed"] is False
    assert report["tensors"] == []
    assert any("cannot initialise source" in failure for failure in report["failures"])


def test_reference_check_passes_with_distinct_shard_layouts(tmp_path):
    preview_dir, reference_dir = _write_sources(tmp_path)
    report_path = tmp_path / "report.json"

    result = _run_cli(preview_dir, reference_dir, report_path)

    assert result.returncode == 0, result.stderr
    report = json.loads(report_path.read_text())
    assert report["passed"] is True
    assert report["failures"] == []
    assert [entry["name"] for entry in report["tensors"]] == WEIGHT_KEY
    for entry in report["tensors"]:
        assert entry["num_elements"] == 64
        assert entry["grid_mismatches"] == 0
        assert entry["max_abs_diff"] >= 0
        # BF16 -> FP4 rounding at midpoints is recorded, not a pass criterion.
        assert entry["relative_l2"] > 0


@pytest.mark.parametrize("corrupt", ["nibble_swap", "scale"])
def test_reference_check_reports_grid_mismatches(tmp_path, corrupt):
    preview_dir, reference_dir = _write_sources(tmp_path, corrupt=corrupt)
    report_path = tmp_path / "report.json"

    result = _run_cli(preview_dir, reference_dir, report_path)

    assert result.returncode != 0
    report = json.loads(report_path.read_text())
    assert report["passed"] is False
    assert report["failures"]
    assert any(entry["grid_mismatches"] > 0 for entry in report["tensors"])
    assert WEIGHT_BASES[0 if corrupt == "nibble_swap" else 1] in result.stderr


def test_reference_check_reports_missing_weight(tmp_path):
    preview_dir, reference_dir = _write_sources(tmp_path)
    missing_base = WEIGHT_BASES[2]
    preview_map = json.loads(
        (preview_dir / "model.safetensors.index.json").read_text()
    )["weight_map"]
    del preview_map[f"{missing_base}.weight"]
    (preview_dir / "model.safetensors.index.json").write_text(
        json.dumps({"weight_map": preview_map})
    )
    report_path = tmp_path / "report.json"

    result = _run_cli(preview_dir, reference_dir, report_path)

    assert result.returncode != 0
    report = json.loads(report_path.read_text())
    assert report["passed"] is False
    assert any(missing_base in failure for failure in report["failures"])
    assert missing_base in result.stderr
