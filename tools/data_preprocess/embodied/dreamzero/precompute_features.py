#!/usr/bin/env python3
# Copyright 2026 The LoongForge Authors.
# SPDX-License-Identifier: Apache-2.0

"""Precompute DreamZero frozen-condition features for structured cache artifacts.

This tool writes per-sample tensors that can be consumed by
DreamZero's structured ``precomputed_cache`` model-config overrides. It can
cache main-video VAE latents, optional i2v first-frame VAE latents, and optional
raw prompt embeddings from the frozen text encoder; per-sample transform seed
settings must match between cache generation and training.
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

_REPO_ROOT = Path(__file__).resolve().parents[4]
if str(_REPO_ROOT) in sys.path:
    sys.path.remove(str(_REPO_ROOT))
sys.path.insert(0, str(_REPO_ROOT))
_TOOL_DIR = Path(__file__).resolve().parent
if str(_TOOL_DIR) not in sys.path:
    sys.path.insert(0, str(_TOOL_DIR))
os.environ.setdefault("NO_ALBUMENTATIONS_UPDATE", "1")
os.environ.setdefault("ALBUMENTATIONS_DISABLE_VERSION_CHECK", "1")
_DEFAULT_MEGATRON_ROOT = _REPO_ROOT.parent / "AIAK-Megatron"
_MEGATRON_ROOT = Path(os.getenv("MEGATRON_PATH", str(_DEFAULT_MEGATRON_ROOT)))
if _MEGATRON_ROOT.exists() and str(_MEGATRON_ROOT) not in sys.path:
    sys.path.insert(0, str(_MEGATRON_ROOT))

import torch
from cache_precompute.config import _load_config
from cache_precompute.features import (
    _build_dataset,
    _build_precompute_loader,
    _build_text_encoder_for_cache,
    _build_vae_for_cache,
    _default_include_first_frame_latents,
    _default_include_prompt_embs,
    _encode_prompt_embs_for_cache,
    _prepare_videos,
    _rank_indices_for_sampler_order,
)
from cache_precompute.storage import (
    _CSV_FIELDNAMES,
    _cache_path,
    _merge_compare_summaries,
    _merge_tensor_shard_storage,
    _PrecomputeStats,
    _process_cache_sample,
    _rank_manifest_path,
    _rank_summary_path,
    _read_jsonl_records,
    _record_for_csv,
    _tensor_storage_files,
    _TensorShardWriter,
    _write_json_atomic,
    _write_manifest_files,
    _write_tensor_shard_index,
)

from loongforge.embodied.model.dreamzero.precomputed_cache.artifact import (
    DREAMZERO_PRECOMPUTED_FEATURES_KIND,
    TENSOR_SHARDS_FORMAT,
    dataset_fingerprint,
    json_sha256,
    sha256_file,
)

SAMPLE_FILES_FORMAT = "sample_files"


@dataclass(frozen=True)
class _DistributedContext:
    rank: int
    world_size: int
    local_rank: int
    backend: str

    @property
    def enabled(self) -> bool:
        return self.world_size > 1

    @property
    def is_rank0(self) -> bool:
        return self.rank == 0


def _env_int(name: str, default: int) -> int:
    raw = os.getenv(name)
    if raw is None or raw == "":
        return default
    return int(raw)


def _load_indices_file(path: Path) -> list[int]:
    """Load selected dataset indices from a JSON list or {"indices": [...]} file."""
    with path.open("r", encoding="utf-8") as f:
        payload = json.load(f)
    raw_indices = payload.get("indices") if isinstance(payload, dict) else payload
    if not isinstance(raw_indices, list):
        raise ValueError(
            f"indices file must be a JSON list or object with an 'indices' list: {path}"
        )

    indices: list[int] = []
    for position, raw_index in enumerate(raw_indices):
        try:
            index = int(raw_index)
        except (TypeError, ValueError) as exc:
            raise ValueError(
                f"indices file {path} has non-integer value at position {position}: "
                f"{raw_index!r}"
            ) from exc
        if index < 0:
            raise ValueError(
                f"indices file {path} has negative index at position {position}: {index}"
            )
        indices.append(index)
    return indices


def _select_dataset_indices(
    *,
    dataset_len: int,
    start_index: int,
    num_samples: int,
    indices_file: Path | None,
) -> list[int]:
    if start_index < 0:
        raise ValueError(f"--start-index must be non-negative, got {start_index}")
    if num_samples < 0:
        raise ValueError(f"--num-samples must be non-negative, got {num_samples}")

    if indices_file is None:
        stop = min(dataset_len, start_index + num_samples)
        return list(range(start_index, stop))

    indices = _load_indices_file(indices_file)
    stop = min(len(indices), start_index + num_samples)
    selected = indices[start_index:stop]
    if len(set(selected)) != len(selected):
        raise ValueError(
            f"indices file contains duplicate indices in selected slice: {indices_file}"
        )
    out_of_range = [index for index in selected if index >= dataset_len]
    if out_of_range:
        preview = ", ".join(str(index) for index in out_of_range[:8])
        raise ValueError(
            f"indices file contains indices outside dataset length {dataset_len}: {preview}"
        )
    return selected


def _read_distributed_context(
    dist_backend: str, device: torch.device
) -> _DistributedContext:
    rank = _env_int("RANK", 0)
    world_size = _env_int("WORLD_SIZE", 1)
    local_rank = _env_int("LOCAL_RANK", 0)
    if world_size < 1:
        raise ValueError(f"WORLD_SIZE must be >= 1, got {world_size}")
    if rank < 0 or rank >= world_size:
        raise ValueError(f"RANK must be in [0, {world_size}), got {rank}")
    if local_rank < 0:
        raise ValueError(f"LOCAL_RANK must be >= 0, got {local_rank}")
    backend = dist_backend
    if backend == "auto":
        backend = "nccl" if device.type == "cuda" else "gloo"
    return _DistributedContext(
        rank=rank,
        world_size=world_size,
        local_rank=local_rank,
        backend=backend,
    )


def _resolve_device(
    device_arg: str | None, *, local_rank: int, world_size: int
) -> torch.device:
    if device_arg is None:
        if torch.cuda.is_available():
            return torch.device(f"cuda:{local_rank}" if world_size > 1 else "cuda")
        return torch.device("cpu")
    if device_arg == "cuda" and torch.cuda.is_available() and world_size > 1:
        return torch.device(f"cuda:{local_rank}")
    return torch.device(device_arg)


def _init_distributed(context: _DistributedContext, device: torch.device) -> None:
    if device.type == "cuda" and device.index is not None:
        torch.cuda.set_device(device)
    if not context.enabled:
        return
    if device.type == "cuda" and device.index is None:
        torch.cuda.set_device(context.local_rank)
    if not torch.distributed.is_available():
        raise RuntimeError("torch.distributed is not available")
    if not torch.distributed.is_initialized():
        init_kwargs: dict[str, Any] = {"backend": context.backend}
        if context.backend == "nccl" and device.type == "cuda":
            init_kwargs["device_id"] = (
                device
                if device.index is not None
                else torch.device(f"cuda:{context.local_rank}")
            )
        torch.distributed.init_process_group(**init_kwargs)


def _barrier(context: _DistributedContext) -> None:
    if context.enabled:
        torch.distributed.barrier()


def _sync_rank0_finalization(context: _DistributedContext) -> None:
    """Keep worker ranks in the process group until rank0 finishes final cache I/O."""
    _barrier(context)


def _destroy_distributed(context: _DistributedContext) -> None:
    if context.enabled and torch.distributed.is_initialized():
        torch.distributed.destroy_process_group()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config-file",
        type=Path,
        default=Path("configs/models/embodied/dreamzero_wan22_5b.yaml"),
    )
    parser.add_argument("--data-path", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--start-index", type=int, default=0)
    parser.add_argument("--num-samples", type=int, default=8)
    parser.add_argument(
        "--indices-file",
        type=Path,
        default=None,
        help=(
            "Optional JSON list or {'indices': [...]} file of dataset indices. "
            "--start-index/--num-samples slice positions in this file when set."
        ),
    )
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument(
        "--pin-memory", action=argparse.BooleanOptionalAction, default=True
    )
    parser.add_argument("--prefetch-factor", type=int, default=None)
    parser.add_argument("--cache-template", default="index_{index:08d}.pt")
    parser.add_argument(
        "--storage-format",
        choices=(SAMPLE_FILES_FORMAT, TENSOR_SHARDS_FORMAT),
        default=SAMPLE_FILES_FORMAT,
        help=(
            "Cache storage backend. sample_files preserves one .pt payload per "
            "sample; tensor_shards writes large mmap-friendly tensor shards plus "
            "a global index for full-dataset caches."
        ),
    )
    parser.add_argument(
        "--tensor-shard-size",
        type=int,
        default=4096,
        help=(
            "Maximum samples per tensor shard when --storage-format=tensor_shards. "
            "The effective size is capped by the rank-local assignment to avoid "
            "large sparse shards for small selected-cache runs."
        ),
    )
    parser.add_argument(
        "--tensor-shard-file-hash",
        action=argparse.BooleanOptionalAction,
        default=False,
        help=(
            "Compute sha256 for large tensor shard files. Default false avoids "
            "rereading full-dataset shards during generation; bytes/shape/count "
            "and hashed index files are still recorded."
        ),
    )
    parser.add_argument("--tokenizer-path", default=None)
    parser.add_argument("--device", default=None)
    parser.add_argument("--dtype", choices=("bf16", "fp32"), default="bf16")
    parser.add_argument(
        "--dist-backend", choices=("auto", "nccl", "gloo"), default="auto"
    )
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--compare-existing", action="store_true")
    parser.add_argument("--compare-atol", type=float, default=0.0)
    parser.add_argument(
        "--include-video-latents", action=argparse.BooleanOptionalAction, default=True
    )
    parser.add_argument(
        "--include-first-frame-latents",
        action=argparse.BooleanOptionalAction,
        default=None,
        help=(
            "Precompute first-frame VAE latents. Default auto: enabled for "
            "I2V/concat-first-frame DreamZero backbones, disabled otherwise."
        ),
    )
    parser.add_argument(
        "--include-prompt-embs",
        action=argparse.BooleanOptionalAction,
        default=None,
        help=(
            "Precompute raw frozen text-encoder prompt embeddings. Default auto: "
            "enabled for 5B/TI2V and disabled for 14B/I2V."
        ),
    )
    parser.add_argument(
        "--use-sample-transform-seed",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    parser.add_argument("--sample-transform-seed", type=int, default=0)
    parser.add_argument(
        "--require-full-language-chunks",
        action=argparse.BooleanOptionalAction,
        default=False,
    )
    args = parser.parse_args()
    if args.storage_format == TENSOR_SHARDS_FORMAT and args.compare_existing:
        raise ValueError(
            "--compare-existing is only supported with --storage-format=sample_files"
        )
    if args.tensor_shard_size <= 0:
        raise ValueError(
            f"--tensor-shard-size must be positive, got {args.tensor_shard_size}"
        )
    env_world_size = _env_int("WORLD_SIZE", 1)
    env_local_rank = _env_int("LOCAL_RANK", 0)
    device = _resolve_device(
        args.device, local_rank=env_local_rank, world_size=env_world_size
    )
    dist_context = _read_distributed_context(args.dist_backend, device)
    _init_distributed(dist_context, device)

    config = _load_config(args.config_file)
    if args.tokenizer_path:
        config.tokenizer_path = args.tokenizer_path
    if not config.tokenizer_path:
        raise ValueError(
            "DreamZero VAE precompute requires --tokenizer-path or config.tokenizer_path"
        )
    include_first_frame_latents = (
        _default_include_first_frame_latents(config)
        if args.include_first_frame_latents is None
        else bool(args.include_first_frame_latents)
    )
    args.include_first_frame_latents = include_first_frame_latents
    include_prompt_embs = (
        _default_include_prompt_embs(config)
        if args.include_prompt_embs is None
        else bool(args.include_prompt_embs)
    )
    args.include_prompt_embs = include_prompt_embs
    if not (
        args.include_video_latents
        or args.include_first_frame_latents
        or args.include_prompt_embs
    ):
        raise ValueError(
            "at least one feature must be enabled: --include-video-latents, "
            "--include-first-frame-latents, or --include-prompt-embs"
        )
    dataset = _build_dataset(
        config,
        args.data_path,
        bool(args.use_sample_transform_seed),
        int(args.sample_transform_seed),
    )
    indices = _select_dataset_indices(
        dataset_len=len(dataset),
        start_index=int(args.start_index),
        num_samples=int(args.num_samples),
        indices_file=args.indices_file,
    )
    if not indices:
        raise ValueError("no dataset indices selected")
    rank_indices = _rank_indices_for_sampler_order(
        indices,
        rank=dist_context.rank,
        world_size=dist_context.world_size,
    )
    tensor_shard_writer = None
    effective_tensor_shard_size = None
    if args.storage_format == TENSOR_SHARDS_FORMAT:
        effective_tensor_shard_size = min(
            int(args.tensor_shard_size),
            max(1, len(rank_indices)),
        )
        tensor_shard_writer = _TensorShardWriter(
            output_dir=args.output_dir,
            shard_size=effective_tensor_shard_size,
            rank=dist_context.rank,
            overwrite=bool(args.overwrite),
            hash_files=bool(args.tensor_shard_file_hash),
        )
    require_full_chunks = bool(args.require_full_language_chunks)
    expected_frames = (
        8 * int(config.max_chunk_size or 1) + 1
        if require_full_chunks and bool(config.language_chunk_sampling)
        else None
    )

    args.output_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = args.output_dir / "manifest.jsonl"
    csv_path = args.output_dir / "manifest.csv"
    artifact_manifest_path = args.output_dir / "manifest.json"
    success_path = args.output_dir / "_SUCCESS"
    if dist_context.is_rank0:
        if success_path.exists():
            success_path.unlink()
        if dist_context.enabled:
            for path in args.output_dir.glob("manifest.rank*.*"):
                path.unlink()
            for path in args.output_dir.glob("precompute.rank*.summary.json"):
                path.unlink()
    _barrier(dist_context)
    if dist_context.is_rank0:
        print(
            "[dreamzero-cache] starting precompute "
            f"world_size={dist_context.world_size} backend={dist_context.backend} "
            f"device={device} selected={len(indices)} batch_size={args.batch_size} "
            f"num_workers={args.num_workers} pin_memory={bool(args.pin_memory) and device.type == 'cuda'} "
            f"prefetch_factor={args.prefetch_factor} "
            f"storage_format={args.storage_format} "
            f"tensor_shard_size={effective_tensor_shard_size}",
            flush=True,
        )

    loader = _build_precompute_loader(
        dataset=dataset,
        indices=rank_indices,
        config=config,
        args=args,
        device=device,
    )

    dtype = torch.bfloat16 if args.dtype == "bf16" else torch.float32
    vae = None
    if args.include_video_latents or args.include_first_frame_latents:
        vae = _build_vae_for_cache(config).to(device=device, dtype=dtype).eval()
        vae.requires_grad_(False)
    text_encoder = None
    if args.include_prompt_embs:
        text_encoder = (
            _build_text_encoder_for_cache(config).to(device=device, dtype=dtype).eval()
        )
        text_encoder.requires_grad_(False)

    local_manifest_path = (
        _rank_manifest_path(args.output_dir, dist_context.rank, "jsonl")
        if dist_context.enabled
        else manifest_path
    )
    local_csv_path = (
        _rank_manifest_path(args.output_dir, dist_context.rank, "csv")
        if dist_context.enabled
        else csv_path
    )

    stats = _PrecomputeStats()
    local_start_time = time.perf_counter()
    with (
        local_manifest_path.open("w", encoding="utf-8") as mf,
        local_csv_path.open(
            "w",
            newline="",
            encoding="utf-8",
        ) as cf,
    ):
        writer = csv.DictWriter(cf, fieldnames=_CSV_FIELDNAMES)
        writer.writeheader()

        def _record_write_result(result: dict[str, Any]) -> None:
            record = result["record"]
            mf.write(json.dumps(record, sort_keys=True) + "\n")
            writer.writerow(_record_for_csv(record))
            stats.record(result)

        def _process_write(job: dict[str, Any]) -> None:
            if tensor_shard_writer is not None:
                _record_write_result(
                    tensor_shard_writer.write_sample(
                        dataset_index=int(job["dataset_index"]),
                        trajectory_id=int(job["trajectory_id"]),
                        base_index=int(job["base_index"]),
                        sample_latents=job["sample_latents"],
                        sample_first_frame_latents=job["sample_first_frame_latents"],
                        sample_prompt_embs=job["sample_prompt_embs"],
                    )
                )
                return
            _record_write_result(_process_cache_sample(**job))

        loader_iter = iter(loader)
        while True:
            try:
                batch = next(loader_iter)
            except StopIteration:
                break
            image_batch = batch["images"]
            batch_indices = batch["dataset_indices"]
            if expected_frames is not None:
                original_batch_size = len(batch_indices)
                kept_images = []
                kept_indices = []
                kept_positions = []
                for position, (image, dataset_index) in enumerate(
                    zip(image_batch, batch_indices)
                ):
                    frame_count = int(image.shape[0])
                    if frame_count == expected_frames:
                        kept_images.append(image)
                        kept_indices.append(dataset_index)
                        kept_positions.append(position)
                    else:
                        stats.skipped_partial += 1
                image_batch = kept_images
                batch_indices = kept_indices
                if not batch_indices:
                    continue
                if len(kept_positions) != original_batch_size:
                    positions = torch.as_tensor(kept_positions, dtype=torch.long)
                    if "text" in batch:
                        batch["text"] = batch["text"].index_select(0, positions)
                    if "text_attention_mask" in batch:
                        batch["text_attention_mask"] = batch[
                            "text_attention_mask"
                        ].index_select(0, positions)
            videos = _prepare_videos(image_batch, config, device, dtype)
            with torch.no_grad():
                latents = None
                if args.include_video_latents:
                    if vae is None:
                        raise RuntimeError("VAE is not initialized")
                    latents = vae.encode(
                        videos,
                        tiled=bool(config.vae_tiled),
                        tile_size=(
                            int(config.vae_tile_size_height),
                            int(config.vae_tile_size_width),
                        ),
                        tile_stride=(
                            int(config.vae_tile_stride_height),
                            int(config.vae_tile_stride_width),
                        ),
                    )
                first_frame_latents = None
                if args.include_first_frame_latents:
                    if vae is None:
                        raise RuntimeError("VAE is not initialized")
                    first_frame_zeros = torch.zeros(
                        videos.shape[0],
                        videos.shape[1],
                        max(int(videos.shape[2]) - 1, 0),
                        videos.shape[3],
                        videos.shape[4],
                        dtype=videos.dtype,
                        device=videos.device,
                    )
                    first_frame_input = torch.cat(
                        [videos[:, :, :1], first_frame_zeros],
                        dim=2,
                    )
                    # Match the training image-condition bf16 autocast path.
                    with torch.amp.autocast(
                        dtype=torch.bfloat16, device_type=device.type
                    ):
                        first_frame_latents = vae.encode(first_frame_input)
                prompt_embs = None
                if args.include_prompt_embs:
                    if text_encoder is None:
                        raise RuntimeError("text encoder is not initialized")
                    if "text" not in batch or "text_attention_mask" not in batch:
                        raise KeyError(
                            "batch is missing text/text_attention_mask for prompt precompute"
                        )
                    prompt_embs = _encode_prompt_embs_for_cache(
                        text_encoder,
                        batch["text"],
                        batch["text_attention_mask"],
                        device=device,
                    )
            for local_idx, dataset_index in enumerate(batch_indices):
                trajectory_id, base_index = dataset.all_steps[dataset_index]
                out_path = _cache_path(
                    args.output_dir,
                    args.cache_template,
                    dataset_index,
                    trajectory_id,
                    base_index,
                )
                sample_latents = (
                    latents[local_idx].detach().cpu() if latents is not None else None
                )
                sample_first_frame_latents = (
                    first_frame_latents[local_idx].detach().cpu()
                    if first_frame_latents is not None
                    else None
                )
                sample_prompt_embs = (
                    prompt_embs[local_idx].detach().cpu()
                    if prompt_embs is not None
                    else None
                )
                _process_write(
                    {
                        "output_dir": args.output_dir,
                        "out_path": out_path,
                        "dataset_index": int(dataset_index),
                        "trajectory_id": int(trajectory_id),
                        "base_index": int(base_index),
                        "sample_latents": sample_latents,
                        "sample_first_frame_latents": sample_first_frame_latents,
                        "sample_prompt_embs": sample_prompt_embs,
                        "overwrite": bool(args.overwrite),
                        "compare_existing": bool(args.compare_existing),
                        "compare_atol": float(args.compare_atol),
                    }
                )

    local_storage = (
        tensor_shard_writer.close() if tensor_shard_writer is not None else None
    )
    local_elapsed = time.perf_counter() - local_start_time
    selected = len(indices)
    local_compare_summary = stats.compare_summary(args)
    local_summary = {
        "rank": int(dist_context.rank),
        "world_size": int(dist_context.world_size),
        "local_rank": int(dist_context.local_rank),
        "device": str(device),
        "backend": dist_context.backend,
        "selected": int(selected),
        "assigned": int(len(rank_indices)),
        "processed": int(stats.processed),
        "saved": int(stats.saved),
        "compared": int(stats.compared),
        "skipped_partial": int(stats.skipped_partial),
        "elapsed_sec": float(local_elapsed),
        "samples_per_sec": (
            float(stats.processed / local_elapsed) if local_elapsed > 0 else 0.0
        ),
        "manifest_jsonl": str(local_manifest_path.name),
        "manifest_csv": str(local_csv_path.name),
        "compare": local_compare_summary,
    }
    if local_storage is not None:
        local_summary["storage"] = local_storage
    processed = stats.processed
    saved = stats.saved
    compared = stats.compared
    skipped_partial = stats.skipped_partial
    cache_files = stats.cache_files
    storage: dict[str, Any] | None = None
    if dist_context.enabled:
        _write_json_atomic(
            _rank_summary_path(args.output_dir, dist_context.rank), local_summary
        )
        # Rank0 can only merge once every worker has flushed its local manifest and summary.
        _barrier(dist_context)
        if not dist_context.is_rank0:
            _sync_rank0_finalization(dist_context)
            _destroy_distributed(dist_context)
            return

        rank_summaries = []
        records: list[dict[str, Any]] = []
        for rank in range(dist_context.world_size):
            with _rank_summary_path(args.output_dir, rank).open(
                "r", encoding="utf-8"
            ) as f:
                rank_summaries.append(json.load(f))
            records.extend(
                _read_jsonl_records(_rank_manifest_path(args.output_dir, rank, "jsonl"))
            )
        records.sort(
            key=lambda item: (
                int(item["index"]),
                int(item["trajectory_id"]),
                int(item["base_index"]),
            )
        )
        processed_from_summaries = sum(
            int(item["processed"]) for item in rank_summaries
        )
        if len(records) != processed_from_summaries:
            raise ValueError(
                "distributed precompute manifest row count mismatch: "
                f"records={len(records)}, summaries={processed_from_summaries}"
            )
        if args.storage_format == TENSOR_SHARDS_FORMAT:
            merged_storage = _merge_tensor_shard_storage(rank_summaries)
            if merged_storage is None:
                raise ValueError(
                    "tensor_shards precompute produced no shard storage metadata"
                )
            storage = _write_tensor_shard_index(
                output_dir=args.output_dir,
                records=records,
                storage=merged_storage,
            )
        _write_manifest_files(manifest_path, csv_path, records)
        processed = len(records)
        saved = sum(int(item["saved"]) for item in rank_summaries)
        compared = sum(int(item["compared"]) for item in rank_summaries)
        skipped_partial = sum(int(item["skipped_partial"]) for item in rank_summaries)
        cache_files = [
            {
                "index": int(record["index"]),
                "trajectory_id": int(record["trajectory_id"]),
                "base_index": int(record["base_index"]),
                "path": str(record["path"]),
                "bytes": int(record["bytes"]),
                "sha256": str(record["sha256"]),
            }
            for record in records
        ]
    else:
        rank_summaries = [local_summary]
        if args.storage_format == TENSOR_SHARDS_FORMAT:
            records = _read_jsonl_records(manifest_path)
            records.sort(
                key=lambda item: (
                    int(item["index"]),
                    int(item["trajectory_id"]),
                    int(item["base_index"]),
                )
            )
            if len(records) != processed:
                raise ValueError(
                    "single-rank precompute manifest row count mismatch: "
                    f"records={len(records)}, processed={processed}"
                )
            _write_manifest_files(manifest_path, csv_path, records)
            cache_files = [
                {
                    "index": int(record["index"]),
                    "trajectory_id": int(record["trajectory_id"]),
                    "base_index": int(record["base_index"]),
                    "path": str(record["path"]),
                    "bytes": int(record["bytes"]),
                    "sha256": str(record["sha256"]),
                }
                for record in records
            ]
        if args.storage_format == TENSOR_SHARDS_FORMAT:
            merged_storage = _merge_tensor_shard_storage(rank_summaries)
            if merged_storage is None:
                raise ValueError(
                    "tensor_shards precompute produced no shard storage metadata"
                )
            storage = _write_tensor_shard_index(
                output_dir=args.output_dir,
                records=records,
                storage=merged_storage,
            )

    compare_summary = _merge_compare_summaries(rank_summaries, args)
    coverage = {
        "dataset_len": int(len(dataset)),
        "selected": int(selected),
        "assigned": int(sum(int(item["assigned"]) for item in rank_summaries)),
        "processed": int(processed),
        "saved": int(saved),
        "compared": int(compared),
        "skipped_partial": int(skipped_partial),
        "coverage_ratio": float(processed / selected) if selected else 0.0,
    }
    precompute_elapsed_sec = max(float(item["elapsed_sec"]) for item in rank_summaries)
    precompute_summary = {
        "distributed": bool(dist_context.enabled),
        "rank": int(dist_context.rank),
        "world_size": int(dist_context.world_size),
        "backend": dist_context.backend,
        "batch_size": int(args.batch_size),
        "storage_format": str(args.storage_format),
        "tensor_shard_size": (
            int(effective_tensor_shard_size)
            if effective_tensor_shard_size is not None
            else None
        ),
        "tensor_shard_file_hash": bool(args.tensor_shard_file_hash),
        "num_workers": int(args.num_workers),
        "pin_memory": bool(args.pin_memory) and device.type == "cuda",
        "prefetch_factor": (
            int(args.prefetch_factor) if args.prefetch_factor is not None else None
        ),
        "elapsed_sec": precompute_elapsed_sec,
        "samples_per_sec": (
            float(processed / precompute_elapsed_sec)
            if precompute_elapsed_sec > 0
            else 0.0
        ),
        "rank_summaries": rank_summaries,
    }
    storage_files = _tensor_storage_files(storage)
    artifact = {
        "version": 3 if args.storage_format == TENSOR_SHARDS_FORMAT else 2,
        "created_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "kind": DREAMZERO_PRECOMPUTED_FEATURES_KIND,
        "features": {
            "video_latents": {
                "enabled": bool(args.include_video_latents),
                "batch_key": "video_latents",
                "payload_keys": ["video_latents", "latents"],
                "layout": "bcthw",
            },
            "first_frame_latents": {
                "enabled": bool(args.include_first_frame_latents),
                "batch_key": "first_frame_latents",
                "payload_keys": ["first_frame_latents", "image_latents", "y_latents"],
            },
            "prompt_embs": {
                "enabled": bool(args.include_prompt_embs),
                "batch_key": "prompt_embs",
                "payload_keys": ["prompt_embs", "prompt_embeddings", "text_embs"],
                "layout": "blc",
            },
        },
        "config": {
            "config_file": str(args.config_file),
            "config_sha256": (
                sha256_file(args.config_file) if args.config_file.exists() else None
            ),
            "backbone_variant": config.backbone_variant,
            "text_encoder_pretrained_path": config.text_encoder_pretrained_path,
            "text_encoder_pretrained_sha256": (
                sha256_file(Path(config.text_encoder_pretrained_path))
                if config.text_encoder_pretrained_path
                and Path(config.text_encoder_pretrained_path).exists()
                else None
            ),
            "vae_class": config.vae_class,
            "vae_pretrained_path": config.vae_pretrained_path,
            "vae_pretrained_sha256": (
                sha256_file(Path(config.vae_pretrained_path))
                if config.vae_pretrained_path
                and Path(config.vae_pretrained_path).exists()
                else None
            ),
            "image_height": int(config.image_height),
            "image_width": int(config.image_width),
            "crop_scale": float(config.crop_scale),
            "enable_color_jitter": bool(config.enable_color_jitter),
            "resize_interpolation": "linear",
            "color_jitter": (
                {
                    "brightness": 0.3,
                    "contrast": 0.4,
                    "saturation": 0.5,
                    "hue": 0.08,
                }
                if bool(config.enable_color_jitter)
                else None
            ),
            "target_video_height": config.target_video_height,
            "target_video_width": config.target_video_width,
            "num_frames": int(config.num_frames),
            "max_chunk_size": int(config.max_chunk_size),
            "language_chunk_sampling": bool(config.language_chunk_sampling),
            "tokenizer_path": str(config.tokenizer_path),
        },
        "dataset": dataset_fingerprint(args.data_path),
        "cache": {
            "output_dir": str(args.output_dir),
            "cache_template": args.cache_template,
            "storage_format": str(args.storage_format),
            "tensor_shard_size": (
                int(effective_tensor_shard_size)
                if effective_tensor_shard_size is not None
                else None
            ),
            "tensor_shard_file_hash": bool(args.tensor_shard_file_hash),
            "dtype": args.dtype,
            "start_index": int(args.start_index),
            "num_samples": int(args.num_samples),
            "indices_file": str(args.indices_file) if args.indices_file else None,
            "indices_file_sha256": (
                sha256_file(args.indices_file)
                if args.indices_file is not None and args.indices_file.exists()
                else None
            ),
            "indices_sha256": json_sha256(indices),
            "use_sample_transform_seed": bool(args.use_sample_transform_seed),
            "sample_transform_seed": int(args.sample_transform_seed),
            "require_full_language_chunks": bool(require_full_chunks),
            "expected_frames": expected_frames,
        },
        "precompute": precompute_summary,
        "coverage": coverage,
        "compare": compare_summary,
        "storage": storage if storage is not None else {"format": SAMPLE_FILES_FORMAT},
        "files": {
            "manifest_jsonl": {
                "path": "manifest.jsonl",
                "bytes": int(manifest_path.stat().st_size),
                "sha256": sha256_file(manifest_path),
            },
            "manifest_csv": {
                "path": "manifest.csv",
                "bytes": int(csv_path.stat().st_size),
                "sha256": sha256_file(csv_path),
            },
            "cache_files_count": len(cache_files),
            "cache_files_total_bytes": int(sum(item["bytes"] for item in cache_files)),
            "cache_files_sha256": json_sha256(cache_files),
            "storage_files_count": len(storage_files),
            "storage_files_total_bytes": int(
                sum(item["bytes"] for item in storage_files)
            ),
            "storage_files_sha256": (
                json_sha256(storage_files) if storage_files else None
            ),
        },
    }
    tmp_manifest_path = artifact_manifest_path.with_suffix(".json.tmp")
    tmp_manifest_path.write_text(
        json.dumps(artifact, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    os.replace(tmp_manifest_path, artifact_manifest_path)
    artifact_sha256 = sha256_file(artifact_manifest_path)
    success_payload = {
        "ok": True,
        "manifest": "manifest.json",
        "manifest_sha256": artifact_sha256,
        "coverage": coverage,
        "compare": compare_summary,
    }
    tmp_success_path = success_path.with_suffix(".tmp")
    tmp_success_path.write_text(
        json.dumps(success_payload, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    os.replace(tmp_success_path, success_path)
    _sync_rank0_finalization(dist_context)

    print(
        "[dreamzero-cache] processed "
        f"{processed} latents (saved={saved}, compared={compared}, "
        f"skipped_partial={skipped_partial}) in {args.output_dir}; "
        f"template={args.cache_template!r}; use_sample_transform_seed={args.use_sample_transform_seed}; "
        f"require_full_language_chunks={require_full_chunks}",
        flush=True,
    )
    training_overrides = [
        "model.precomputed_cache.enabled=true",
        f"model.precomputed_cache.cache_dir={args.output_dir}",
        f"model.precomputed_cache.manifest={artifact_manifest_path}",
        f"model.precomputed_cache.cache_template={args.cache_template}",
        "model.precomputed_cache.strict=true",
        "model.precomputed_cache.validation.validate_artifact=true",
        "model.precomputed_cache.validation.require_success=true",
    ]
    print(
        "[dreamzero-cache] training overrides: " + " ".join(training_overrides),
        flush=True,
    )
    print(
        "[dreamzero-cache] feature defaults: omit model.precomputed_cache.features "
        "for full caches; training enables the model-relevant full feature set "
        "(14B/I2V: video_latents+first_frame_latents; "
        "5B/TI2V: video_latents+raw prompt_embs). Add explicit features only "
        "for partial/custom caches.",
        flush=True,
    )
    print(
        "[dreamzero-cache] data transform seed: "
        f"data.use_sample_transform_seed={bool(args.use_sample_transform_seed)} "
        f"data.sample_transform_seed={args.sample_transform_seed}",
        flush=True,
    )
    print(
        "[dreamzero-cache] artifact manifest: "
        f"{artifact_manifest_path} sha256={artifact_sha256} "
        f"coverage={coverage['processed']}/{coverage['selected']} "
        f"cache_files_sha256={artifact['files']['cache_files_sha256']}",
        flush=True,
    )
    _destroy_distributed(dist_context)


if __name__ == "__main__":
    main()
