# Copyright 2026 The LoongForge Authors.
# SPDX-License-Identifier: Apache-2.0

"""Build a WDS-native pack plan from per-media hashbucket results."""

import json
import pickle

from wds_pack.core.artifacts import debug_artifacts_enabled
from wds_pack.core.config import get_cfg, parse_args
from wds_pack.core.paths import (
    get_bins_plan_path,
    get_bins_path,
    get_manifest_sqlite_path,
    get_pack_plan_path,
    get_work_dir,
)
from wds_pack.core.types import pack_item_from_hashbucket_record
from wds_pack.manifest.sqlite import load_sample_token_index

MEDIA_TYPES = ("text", "image", "video")


def box_sample_ids(box) -> list[str]:
    return [item.sample_id for item in box_pack_items(box)]


def box_token_lens(box) -> list[int]:
    return [item.token_len for item in box_pack_items(box)]


def box_pack_items(box):
    return [pack_item_from_hashbucket_record(record) for record in box]


def iter_media_bins(cfg: dict, media_type: str):
    compact_plan = get_bins_plan_path(cfg, media_type)
    if compact_plan.exists():
        with compact_plan.open("r", encoding="utf-8") as f:
            for line in f:
                if not line.strip():
                    continue
                row = json.loads(line)
                yield (
                    row["sample_ids"],
                    [int(value) for value in row["sample_token_lens"]],
                    int(row["total_token_len"]),
                )
        return

    bins_path = get_bins_path(cfg, media_type)
    if not bins_path.exists():
        return
    with bins_path.open("rb") as f:
        bins_boxs = pickle.load(f)
    for box in bins_boxs:
        sample_ids = box_sample_ids(box)
        sample_token_lens = box_token_lens(box)
        yield sample_ids, sample_token_lens, sum(sample_token_lens)


def main() -> None:
    args = parse_args()
    cfg = get_cfg(args.config)
    manifest_sqlite = get_manifest_sqlite_path(cfg)
    if not manifest_sqlite.exists():
        raise FileNotFoundError(f"Manifest sqlite not found: {manifest_sqlite}")

    packing_cfg = cfg.get("packing", {})
    algorithm = packing_cfg.get("algorithm", "hashbucket")
    validate_pack_plan = packing_cfg.get("validate_pack_plan")
    if validate_pack_plan is None:
        validate_pack_plan = algorithm not in ("best_fit_decreasing", "bfd")
    samples = load_sample_token_index(manifest_sqlite) if validate_pack_plan else None
    max_token_len = int(cfg["sample"]["max_token_len"])
    used = set()
    pack_plan = get_pack_plan_path(cfg)
    unpacked_path = get_work_dir(cfg) / "unpacked_samples.jsonl"
    keep_debug = debug_artifacts_enabled(cfg)

    with pack_plan.open("w", encoding="utf-8") as out:
        for media_type in MEDIA_TYPES:
            for index, (sample_ids, sample_token_lens, total_token_len) in enumerate(
                iter_media_bins(cfg, media_type)
            ):
                if not sample_ids:
                    continue
                if len(sample_ids) != len(sample_token_lens):
                    raise ValueError(
                        f"{media_type} bin {index} has mismatched sample_ids/token_lens: "
                        f"{len(sample_ids)} vs {len(sample_token_lens)}"
                    )
                if sum(sample_token_lens) != total_token_len:
                    raise ValueError(
                        f"{media_type} bin {index} total_token_len mismatch: "
                        f"{sum(sample_token_lens)} vs {total_token_len}"
                    )
                if total_token_len > max_token_len:
                    raise ValueError(
                        f"{media_type} bin {index} total_token_len exceeds max_token_len: "
                        f"{total_token_len} > {max_token_len}"
                    )

                if samples is not None:
                    for sample_id, token_len in zip(sample_ids, sample_token_lens):
                        if sample_id not in samples:
                            raise KeyError(
                                f"Sample {sample_id} from {media_type} bins not found in manifest"
                            )
                        sample_media_type = samples[sample_id].media_type
                        if sample_media_type != media_type:
                            raise ValueError(
                                f"Mixed media pack detected in {media_type} bins: "
                                f"{sample_id} is {sample_media_type}, expected {media_type}"
                            )
                        manifest_token_len = samples[sample_id].token_len
                        if manifest_token_len != token_len:
                            raise ValueError(
                                f"Token length mismatch for {sample_id}: "
                                f"bin={token_len}, manifest={manifest_token_len}"
                            )
                pack_id = f"{media_type}_pack_{index:08d}"
                out.write(
                    json.dumps(
                        {
                            "pack_id": pack_id,
                            "media_type": media_type,
                            "sample_ids": sample_ids,
                            "sample_token_lens": sample_token_lens,
                            "total_token_len": total_token_len,
                        },
                        ensure_ascii=False,
                    )
                    + "\n"
                )
                used.update(sample_ids)

    if keep_debug:
        with unpacked_path.open("w", encoding="utf-8") as out:
            if samples is None:
                pass
            else:
                for sample_id, sample in samples.items():
                    if sample_id in used:
                        continue
                    out.write(
                        json.dumps(
                            {
                                "sample_id": sample_id,
                                "media_type": sample.media_type,
                                "token_len": sample.token_len,
                                "reason": "not_selected_by_hashbucket",
                            },
                            ensure_ascii=False,
                        )
                        + "\n"
                    )
    elif unpacked_path.exists():
        unpacked_path.unlink()

    print(f"pack_plan saved to {pack_plan}")
    if keep_debug:
        print(f"unpacked sample report saved to {unpacked_path}")
    else:
        print("unpacked sample report skipped because artifacts.debug_artifacts=false")


if __name__ == "__main__":
    main()
