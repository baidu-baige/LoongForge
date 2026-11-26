# Copyright 2026 The LoongForge Authors.
# SPDX-License-Identifier: Apache-2.0

"""Check Preview's packed MXFP4 experts against the official BF16 checkpoint.

ERNIE-5.5-Flash-Preview stores routed experts as packed E2M1 values (safetensors
``I8``, two values per byte) with one E8M0 block scale per 32 columns
(``F8_E8M0``, ``.weight_scale_inv``). This script re-derives the official BF16
weights' E2M1 grid point and sign and requires the production
``dequantize_mxfp4`` decode to land on the exact same grid point and sign for
every element of the three layer-3 expert projections. The decoded magnitude
must equal the grid point value exactly (``+0`` and ``-0`` are equivalent);
the scales are powers of two and the E2M1 magnitudes are exactly representable,
so this equality does not need a tolerance. It is a strict packing/grid check,
not a tolerance comparison: the BF16 -> FP4 quantization error recorded as
``relative_l2`` is expected and is not a pass criterion.

Neither side is loaded whole. Both read ``model.safetensors.index.json`` to
locate the shard holding each key, then parse the safetensors header and read
only the key's ``data_offsets`` byte range. The Preview header declares
``F8_E8M0``, which old safetensors releases cannot parse; this script reads the
JSON header itself and only needs a torch that has ``float8_e8m0fnu``, so the
safetensors version on the host does not matter.

Usage:
    python3 tests/mcore_bridge_roundtrip/ernie5.5/mxfp4_reference_check.py \
        --preview /path/to/ERNIE-5.5-Flash-Preview \
        --report report.json

Exit code 0 means the three tensors were read, all values were finite, all
scale factors were positive and every element matched the official grid point
and sign. Anything else exits non-zero and names the tensor and reason on
stderr.
"""

import argparse
import json
import math
import os
import posixpath
import struct
import sys
from pathlib import Path

# Add the repository root and its tools/ directory to sys.path so the command
# in docs/source/features/mcore_bridge.md works from the repository root without
# PYTHONPATH. This file lives at tests/mcore_bridge_roundtrip/ernie5.5/, so the
# repository root is three levels above its directory.
_file_dir = os.path.dirname(os.path.abspath(__file__))
_repo_root = os.path.dirname(os.path.dirname(os.path.dirname(_file_dir)))
for _path in (_repo_root, os.path.join(_repo_root, "tools")):
    if _path not in sys.path:
        sys.path.insert(0, _path)

import torch

from mcore_checkpoint_convert.huggingface.mxfp4_dequant import dequantize_mxfp4

DEFAULT_REFERENCE = "hf://baidu/ERNIE-5.5-Flash-Preview-BF16"
DEFAULT_REVISION = "33a9925b17506c5ff57accac2b31a6520c951c4e"

# The three routed-expert projections this check covers.
WEIGHT_BASES = (
    "model.layers.3.block_sparse_moe.experts.0.w1",
    "model.layers.3.block_sparse_moe.experts.0.w2",
    "model.layers.3.block_sparse_moe.experts.0.w3",
)

MXFP4_BLOCK_SIZE = 32
E2M1_MAGNITUDES = (0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0)
# An absolute ratio exactly on a midpoint is a tie: both neighbours are valid.
E2M1_MIDPOINTS = tuple(
    (low + high) / 2 for low, high in zip(E2M1_MAGNITUDES, E2M1_MAGNITUDES[1:])
)

# Read a remote shard in 256 KiB range requests instead of one whole-file read.
RANGE_BLOCK_SIZE = 262144

_SAFETENSORS_DTYPES = {
    "BF16": torch.bfloat16,
    "F16": torch.float16,
    "F32": torch.float32,
    "I8": torch.int8,
    "U8": torch.uint8,
}

try:
    _E8M0_DTYPE = torch.float8_e8m0fnu
except AttributeError:  # torch < 2.1 has no E8M0 dtype
    _E8M0_DTYPE = None


class CheckFailure(Exception):
    """A source, tensor or value failed validation; reported, not raised."""


class Source:
    """Read individual safetensors byte ranges from a directory or fsspec URI."""

    def __init__(self, location, revision=None):
        self.location = location
        self.revision = revision
        self._remote = "://" in location
        if self._remote:
            from fsspec.core import url_to_fs

            self._fs, self._root = url_to_fs(location)
        else:
            self._fs = None
            self._root = str(Path(location).expanduser())

    def _path(self, name):
        if self._remote:
            return posixpath.join(self._root, name)
        return os.path.join(self._root, name)

    def _open(self, name):
        if self._remote:
            return self._fs.open(
                self._path(name), "rb", block_size=RANGE_BLOCK_SIZE, revision=self.revision
            )
        return open(self._path(name), "rb")

    def exists(self, name):
        if self._remote:
            return self._fs.exists(self._path(name), revision=self.revision)
        return os.path.exists(self._path(name))

    def size(self, name):
        """Return the file's byte size, or raise CheckFailure when unavailable."""
        try:
            if self._remote:
                # HfFileSystem.size() does not accept a revision, so read the
                # size from info(), which resolves the pinned revision.
                info = self._fs.info(self._path(name), revision=self.revision)
                size = info.get("size") if isinstance(info, dict) else None
            else:
                size = os.path.getsize(self._path(name))
        except Exception as exc:
            raise CheckFailure(
                f"cannot determine the size of {name} from {self.location}: {exc}"
            ) from exc
        if not isinstance(size, int) or size < 0:
            raise CheckFailure(
                f"cannot determine the size of {name} from {self.location}: "
                f"got {size!r}"
            )
        return size

    def read_file(self, name):
        try:
            with self._open(name) as stream:
                return stream.read()
        except Exception as exc:
            raise CheckFailure(f"cannot read {name} from {self.location}: {exc}") from exc

    def read_range(self, shard, start, length):
        try:
            with self._open(shard) as stream:
                stream.seek(start)
                data = stream.read(length)
        except Exception as exc:
            raise CheckFailure(f"cannot read {shard} from {self.location}: {exc}") from exc
        if len(data) != length:
            raise CheckFailure(
                f"{shard} returned {len(data)} of {length} requested byte(s) "
                f"at offset {start}"
            )
        return data


def read_weight_map(source, keys):
    """Locate every key's shard through the source's own safetensors index."""
    index_name = "model.safetensors.index.json"
    try:
        has_index = source.exists(index_name)
    except Exception as exc:
        raise CheckFailure(f"{source.location}: cannot access {index_name}: {exc}") from exc
    if has_index:
        try:
            payload = json.loads(source.read_file(index_name))
        except json.JSONDecodeError as exc:
            raise CheckFailure(f"{source.location}/{index_name}: invalid JSON: {exc}") from exc
        weight_map = payload.get("weight_map")
        if not isinstance(weight_map, dict):
            raise CheckFailure(f"{source.location}/{index_name}: missing weight_map")
        return weight_map
    single_name = "model.safetensors"
    try:
        has_single = source.exists(single_name)
    except Exception as exc:
        raise CheckFailure(f"{source.location}: cannot access {single_name}: {exc}") from exc
    if has_single:
        return {key: single_name for key in keys}
    raise CheckFailure(
        f"{source.location}: neither {index_name} nor {single_name} exists"
    )


def read_shard_header(source, shard):
    """Return the shard's JSON header and the byte offset where tensor data starts."""
    prefix = source.read_range(shard, 0, 8)
    header_length = struct.unpack("<Q", prefix)[0]
    if not 0 < header_length < 100 * 1024 * 1024:
        raise CheckFailure(f"{shard}: implausible safetensors header length {header_length}")
    try:
        header = json.loads(source.read_range(shard, 8, header_length))
    except json.JSONDecodeError as exc:
        raise CheckFailure(f"{shard}: invalid safetensors header JSON: {exc}") from exc
    return header, 8 + header_length


def read_tensor(source, shard, header, data_start, key, expected_dtype):
    """Read one tensor's exact byte range and rebuild it with its declared dtype."""
    info = header.get(key)
    if info is None:
        raise CheckFailure(f"{key} is not present in {shard}")
    dtype_name = info.get("dtype")
    if dtype_name != expected_dtype:
        raise CheckFailure(f"expected dtype {expected_dtype}, got {dtype_name}")
    shape = tuple(info.get("shape", ()))
    if not shape or any(dimension <= 0 for dimension in shape):
        raise CheckFailure(f"invalid shape {shape}")
    begin, end = info.get("data_offsets", (-1, -1))
    if (
        not isinstance(begin, int)
        or not isinstance(end, int)
        or begin < 0
        or end < begin
    ):
        raise CheckFailure(
            f"{key}: invalid data_offsets ({begin!r}, {end!r}) in {shard}"
        )
    if dtype_name == "F8_E8M0":
        if _E8M0_DTYPE is None:
            raise CheckFailure("torch has no float8_e8m0fnu dtype")
        dtype = _E8M0_DTYPE
    else:
        dtype = _SAFETENSORS_DTYPES.get(dtype_name)
        if dtype is None:
            raise CheckFailure(f"unsupported safetensors dtype {dtype_name}")
    expected_bytes = math.prod(shape) * dtype.itemsize
    if end - begin != expected_bytes:
        raise CheckFailure(
            f"{key}: data_offsets span {end - begin} byte(s), "
            f"{dtype_name} {list(shape)} needs {expected_bytes}"
        )
    shard_size = source.size(shard)
    if data_start + end > shard_size:
        raise CheckFailure(
            f"{key}: data_offsets end {end} plus header {data_start} exceeds "
            f"{shard}'s {shard_size} byte(s)"
        )
    raw = source.read_range(shard, data_start + begin, expected_bytes)
    # F8_E8M0 has no torch storage type of its own in the safetensors header;
    # rebuild it from raw bytes and view the result as the float8 dtype.
    storage_dtype = torch.uint8 if dtype_name == "F8_E8M0" else dtype
    tensor = torch.frombuffer(bytearray(raw), dtype=storage_dtype).reshape(shape)
    if dtype_name == "F8_E8M0":
        tensor = tensor.view(_E8M0_DTYPE)
    return tensor


def _grid_index(value, scale):
    """Nearest E2M1 magnitude index for ``value / scale``, with midpoint ties allowed."""
    ratio = value.float() / scale.float()
    magnitude = ratio.abs()
    boundaries = torch.tensor(E2M1_MIDPOINTS, dtype=torch.float32)
    index = torch.searchsorted(boundaries, magnitude.contiguous())
    index.clamp_(max=len(E2M1_MAGNITUDES) - 1)
    tie_upper = torch.zeros_like(index, dtype=torch.bool)
    inside = index < len(E2M1_MIDPOINTS)
    tie_upper[inside] = magnitude[inside] == boundaries[index[inside]]
    return ratio, index, tie_upper


def grid_mismatches(reference, decoded, scale):
    """Count elements whose decoded value leaves the BF16 element's grid point.

    The BF16 reference is mapped to its nearest E2M1 grid point (either
    neighbour is valid exactly on a midpoint). The decoded value must land on
    that point exactly: its magnitude is compared with the E2M1 magnitude of
    its index without tolerance. E8M0 scales are powers of two and all eight
    E2M1 magnitudes are exactly representable in BF16 and FP32, so this
    equality cannot fail because of floating-point rounding; a difference is a
    decode error.
    """
    reference_ratio, reference_index, tie_upper = _grid_index(reference, scale)
    decoded_ratio, decoded_index, _ = _grid_index(decoded, scale)
    index_ok = (decoded_index == reference_index) | (
        tie_upper & (decoded_index == reference_index + 1)
    )
    e2m1_magnitudes = torch.tensor(E2M1_MAGNITUDES, dtype=torch.float32)
    magnitude_ok = decoded_ratio.abs() == e2m1_magnitudes[decoded_index]
    # A decoded zero carries no sign, so +0 and -0 both match; every non-zero
    # decoded value must keep the reference element's sign.
    decoded_zero = decoded_ratio == 0
    sign_ok = decoded_zero | (decoded_ratio.sign() == reference_ratio.sign())
    return int((~(index_ok & magnitude_ok & sign_ok)).sum().item())


def check_tensor(preview, reference, preview_map, reference_map, base):
    """Validate one expert projection and return its report entry."""
    weight_key = f"{base}.weight"
    scale_key = f"{base}.weight_scale_inv"
    preview_shard = preview_map.get(weight_key)
    preview_scale_shard = preview_map.get(scale_key)
    reference_shard = reference_map.get(weight_key)
    for key, shard, source in (
        (weight_key, preview_shard, preview),
        (scale_key, preview_scale_shard, preview),
        (weight_key, reference_shard, reference),
    ):
        if shard is None:
            raise CheckFailure(f"{key} is absent from {source.location}'s weight map")

    preview_header, preview_start = read_shard_header(preview, preview_shard)
    scale_header, scale_start = read_shard_header(preview, preview_scale_shard)
    reference_header, reference_start = read_shard_header(reference, reference_shard)
    weight = read_tensor(preview, preview_shard, preview_header, preview_start, weight_key, "I8")
    scale = read_tensor(
        preview, preview_scale_shard, scale_header, scale_start, scale_key, "F8_E8M0"
    )
    reference = read_tensor(
        reference, reference_shard, reference_header, reference_start, weight_key, "BF16"
    )

    packed_features = weight.shape[-1] * 2
    if scale.shape[:-1] != weight.shape[:-1]:
        raise CheckFailure(
            f"scale shape {list(scale.shape)} does not match packed weight "
            f"shape {list(weight.shape)}"
        )
    if scale.shape[-1] * MXFP4_BLOCK_SIZE != packed_features:
        raise CheckFailure(
            f"scale block width {scale.shape[-1]} * {MXFP4_BLOCK_SIZE} does not cover "
            f"{packed_features} unpacked columns"
        )
    if tuple(reference.shape) != tuple(weight.shape[:-1]) + (packed_features,):
        raise CheckFailure(
            f"reference shape {list(reference.shape)} does not match packed weight "
            f"shape {list(weight.shape)}"
        )

    scale_values = scale.float()
    if not torch.isfinite(scale_values).all():
        raise CheckFailure("scale has non-finite values")
    if not (scale_values > 0).all():
        raise CheckFailure(f"scale has {(scale_values <= 0).sum().item()} non-positive value(s)")
    if not torch.isfinite(reference).all():
        raise CheckFailure(
            f"reference has {torch.isnan(reference).sum().item()} NaN and "
            f"{torch.isinf(reference).sum().item()} infinite value(s)"
        )

    expanded_scale = scale_values.repeat_interleave(MXFP4_BLOCK_SIZE, dim=-1)
    decoded = dequantize_mxfp4(
        weight, scale, block_size=MXFP4_BLOCK_SIZE, dtype=torch.bfloat16
    ).float()
    if not torch.isfinite(decoded).all():
        raise CheckFailure("production dequantization produced non-finite values")

    mismatches = grid_mismatches(reference, decoded, expanded_scale)
    difference = decoded - reference.float()
    reference_norm = reference.float().norm()
    decoded_norm = decoded.norm()
    if reference_norm > 0:
        relative_l2 = float(difference.norm() / reference_norm)
    else:
        relative_l2 = 0.0 if decoded_norm == 0 else float("inf")
    return {
        "name": weight_key,
        "num_elements": int(reference.numel()),
        "grid_mismatches": mismatches,
        "relative_l2": relative_l2,
        "max_abs_diff": float(difference.abs().max()),
    }


def _failure_detail(exc):
    """Readable one-line cause for a CheckFailure or an unexpected exception."""
    if isinstance(exc, CheckFailure):
        return str(exc)
    return f"{type(exc).__name__}: {exc}"


def _source_or_failure(location, failures, revision=None):
    """Create a Source, recording a readable failure instead of raising."""
    try:
        return Source(location, revision=revision)
    except Exception as exc:
        detail = f"cannot initialise source {location}: {_failure_detail(exc)}"
        failures.append(detail)
        print(f"[mxfp4-reference] FAILED {detail}", file=sys.stderr)
        return None


def _check_all_tensors(preview, reference, failures, tensors):
    """Read the weight maps and run all three tensor checks, collecting failures."""
    preview_keys = [f"{base}.weight" for base in WEIGHT_BASES] + [
        f"{base}.weight_scale_inv" for base in WEIGHT_BASES
    ]
    reference_keys = [f"{base}.weight" for base in WEIGHT_BASES]
    try:
        preview_map = read_weight_map(preview, preview_keys)
        reference_map = read_weight_map(reference, reference_keys)
    except CheckFailure as exc:
        failures.append(str(exc))
        print(f"[mxfp4-reference] FAILED {exc}", file=sys.stderr)
        return

    for base in WEIGHT_BASES:
        weight_key = f"{base}.weight"
        try:
            entry = check_tensor(preview, reference, preview_map, reference_map, base)
        except Exception as exc:
            detail = _failure_detail(exc)
            failures.append(f"{weight_key}: {detail}")
            print(f"[mxfp4-reference] FAILED {weight_key}: {detail}", file=sys.stderr)
            continue
        print(
            f"[mxfp4-reference] {weight_key}: {entry['num_elements']} element(s), "
            f"{entry['grid_mismatches']} grid mismatch(es), "
            f"relative_l2 {entry['relative_l2']:.4f}, "
            f"max_abs_diff {entry['max_abs_diff']}",
            file=sys.stderr,
        )
        if entry["grid_mismatches"]:
            detail = (
                f"{entry['grid_mismatches']} of {entry['num_elements']} element(s) "
                "mismatch the official E2M1 grid point or sign"
            )
            failures.append(f"{weight_key}: {detail}")
            print(f"[mxfp4-reference] FAILED {weight_key}: {detail}", file=sys.stderr)
        tensors.append(entry)


def build_report(preview_location, reference_location, revision):
    """Run all three tensor checks and assemble the report, collecting failures."""
    failures = []
    tensors = []
    preview = _source_or_failure(preview_location, failures)
    reference = _source_or_failure(reference_location, failures, revision=revision)
    if preview is not None and reference is not None:
        _check_all_tensors(preview, reference, failures, tensors)

    passed = not failures and len(tensors) == len(WEIGHT_BASES)
    return {
        "passed": passed,
        "reference": {"uri": reference_location, "revision": revision},
        "tensors": tensors,
        "failures": failures,
    }


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Cross-check Preview packed MXFP4 experts against official BF16 weights."
    )
    parser.add_argument(
        "--preview", required=True, metavar="DIR",
        help="local ERNIE-5.5-Flash-Preview checkpoint directory",
    )
    parser.add_argument(
        "--reference", default=DEFAULT_REFERENCE, metavar="PATH_OR_URI",
        help=f"official BF16 checkpoint directory or fsspec URI (default: {DEFAULT_REFERENCE})",
    )
    parser.add_argument(
        "--revision", default=DEFAULT_REVISION,
        help=f"reference revision (default: {DEFAULT_REVISION})",
    )
    parser.add_argument(
        "--report", metavar="PATH",
        help="write the JSON report here instead of stdout",
    )
    args = parser.parse_args()

    report = build_report(args.preview, args.reference, args.revision)
    payload = json.dumps(report, indent=2, ensure_ascii=False)
    if args.report:
        report_path = Path(args.report)
        try:
            report_path.parent.mkdir(parents=True, exist_ok=True)
            report_path.write_text(payload + "\n", encoding="utf-8")
        except OSError as exc:
            print(f"ERROR: cannot write report {report_path}: {exc}", file=sys.stderr)
            return 1
    else:
        print(payload)
    return 0 if report["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
