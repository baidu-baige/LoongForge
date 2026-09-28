# Copyright 2026 The LoongForge Authors.
# SPDX-License-Identifier: Apache-2.0

"""Script for executing hash bucket packing process."""

from pprint import pprint
import json
import os
import random
import pickle
from wds_pack.algorithms.hashbucket import HashBucketProcessor, PackingTracker
from wds_pack.core.artifacts import keep_intermediate_artifacts
from wds_pack.core.config import get_cfg, parse_args
from wds_pack.core.paths import (
    get_bins_plan_path,
    get_bins_path,
    get_init_file,
    get_manifest_sqlite_path,
)
from wds_pack.core.types import pack_item_from_hashbucket_record
from wds_pack.manifest.sqlite import load_pack_items

random.seed(100)


def update_info(update_stats):
    print("update analysis:")
    print(f"The number of deleted keys: {update_stats['changes']['keys_removed']}")
    print(f"The number of deleted items: {update_stats['changes']['items_removed']}")
    print(f"Remaining key number: {update_stats['after']['total_keys']}")
    print(f"Remaining items number: {update_stats['after']['total_items']}")
    return update_stats["after"]["total_items"]


def get_hs(hs):
    a = list(hs.keys())
    if not a:
        return 0, 0, 0, 0
    mean = sum(a) / (len(a) + 1)
    min_ = min(a)
    max_ = max(a)
    num = sum(len(arr) for arr in hs.values())

    return mean, min_, max_, num


MEDIA_TYPES = ("text", "image", "video")


def init_from_file(token_info_file, max_token_len):
    if not os.path.exists(token_info_file):
        print(f" file {token_info_file} does not exist!")
        processor = None
        tracker = None
        assert token_info_file, f"File {token_info_file} does not exist!"
        raise FileNotFoundError(f"File not found: {token_info_file}")
    else:
        processor = HashBucketProcessor(token_info_file)
        processor.build_buckets()
        processor.summary()
        capacity = max_token_len
        processor.find_items(capacity)
        processor.summary()
        initial_summary = processor.get_hash_buckets_summary()
        print("-------------------- initial_summary ----------------------")
        pprint(initial_summary)
        tracker = PackingTracker(processor)
    return processor, tracker


def init_from_items(items, max_token_len):
    processor = HashBucketProcessor.from_items(items)
    processor.summary()
    processor.find_items(max_token_len)
    processor.summary()
    initial_summary = processor.get_hash_buckets_summary()
    print("-------------------- initial_summary ----------------------")
    pprint(initial_summary)
    tracker = PackingTracker(processor)
    return processor, tracker


def init(cfg):
    token_info_file, max_token_len, packed_files_dir, _ = get_init_file(cfg)
    processor, tracker = init_from_file(token_info_file, max_token_len)
    return processor, tracker, token_info_file, max_token_len, packed_files_dir


def run_hashbucket(token_info_file, max_token_len):
    bins_boxs = []
    processor, tracker = init_from_file(token_info_file, max_token_len)
    return run_hashbucket_processor(processor, tracker, max_token_len)


def run_hashbucket_items(items, max_token_len):
    processor, tracker = init_from_items(items, max_token_len)
    return run_hashbucket_processor(processor, tracker, max_token_len)


def run_best_fit_decreasing_items(items, max_token_len):
    processor = HashBucketProcessor.from_items(items)
    processor.summary()
    return processor.pack_best_fit_decreasing(box_capacity=max_token_len)


def write_bins_plan(cfg, media_type, bins_boxs):
    plan_path = get_bins_plan_path(cfg, media_type)
    with plan_path.open("w", encoding="utf-8") as out:
        for box in bins_boxs:
            sample_ids = []
            token_lens = []
            for record in box:
                item = pack_item_from_hashbucket_record(record)
                sample_ids.append(item.sample_id)
                token_lens.append(item.token_len)
            out.write(
                json.dumps(
                    {
                        "sample_ids": sample_ids,
                        "sample_token_lens": token_lens,
                        "total_token_len": sum(token_lens),
                    },
                    ensure_ascii=False,
                )
                + "\n"
            )
    return plan_path


def run_hashbucket_processor(processor, tracker, max_token_len):
    bins_boxs = []
    mean, min_, max_, tmp_num = get_hs(processor.hash_buckets)
    turn = 1
    bin_boxs_001 = tracker.track_packing(
        "pack_with_deletion", box_capacity=max_token_len
    )
    update_stats = processor.update_hash_buckets(remove_empty=True, verbose=True)
    rest_items = update_info(update_stats)
    bins_boxs.extend(bin_boxs_001)
    mean, min_, max_, num = get_hs(processor.hash_buckets)
    if num == 0:
        return bins_boxs
    scale = int(mean - (mean - min_) * 0.1)
    print(
        f"in the first round of end ----------the current processing box number: {len (bin_boxs_001)}, total box {len (bins_boxs)}, handle the {num - tmp_num} items, remaining {num} the items"
    )
    # tmp_items=[(1,0.96),(5,0.96),(5,0.94),(6,0.92),(4,0.92),(4,0.92)]
    tmp_items = [(1, 0.96), (1, 0.95), (8, 0.9), (4, 0.92), (6, 0.92), (4, 0.9)]
    # tmp_items=[(1,0.96),(1,0.95),(8,0.9),(4,0.92),(6,0.92)]
    for i, item in enumerate(tmp_items):
        if num == 0:
            break
        min_items, min_ratio = item
        bin_boxs_turn = tracker.track_packing(
            "pack_with_flexible_seeds",
            box_capacity=max_token_len,
            seed_strategy="custom_half",
            seed_params={"half": int(mean)},
            min_items=min_items,
            min_ratio=min_ratio,
            max_workers=os.cpu_count(),
        )
        update_stats = processor.update_hash_buckets(remove_empty=True, verbose=True)
        update_info(update_stats)
        bins_boxs.extend(bin_boxs_turn)
        mean, min_, max_, tmp_num = get_hs(processor.hash_buckets)
        print(
            f" the {turn+i+1} th round ends ---------- current number of processed boxes: {len(bin_boxs_turn)}, total box {len(bins_boxs)}, processed {num-tmp_num}items, remaining {tmp_num}items"
        )
        num = tmp_num

    if num > 10000:
        for i in range(5):
            topn = 20
            bin_boxs_TOP = tracker.track_packing(
                "pack_with_flexible_seeds",
                box_capacity=max_token_len,
                seed_strategy="top_n",
                seed_params={"n": 20},  # 1 for 16384
                min_items=4,
                min_ratio=0.90,
                max_workers=os.cpu_count(),
            )
            update_stats = processor.update_hash_buckets(
                remove_empty=True, verbose=True
            )
            update_info(update_stats)
            bins_boxs.extend(bin_boxs_TOP)
            mean, min_, max_, tmp_num = get_hs(processor.hash_buckets)
            print(
                f"top{i} end----------the current processing box number: {len(bin_boxs_TOP)}, total box{len(bins_boxs)},handle the{num-tmp_num}items,remaining {tmp_num} items"
            )
            if num - tmp_num < 3000:
                break
            topn += 20
            num = tmp_num

    for i in range(2):
        if num > 10000:
            scale = int(mean - (mean - min_) * 0.1)
            min_items, min_ratio = 4, 0.90
            bin_boxs_turn = tracker.track_packing(
                "pack_with_flexible_seeds",
                box_capacity=max_token_len,
                seed_strategy="custom_half",
                seed_params={"half": int(mean)},
                min_items=min_items,
                min_ratio=min_ratio,
                max_workers=os.cpu_count(),
            )
            update_stats = processor.update_hash_buckets(
                remove_empty=True, verbose=True
            )
            update_info(update_stats)
            bins_boxs.extend(bin_boxs_turn)
            mean, min_, max_, tmp_num = get_hs(processor.hash_buckets)
            print(
                f" the final pack_with_flexible_seeds end ---------- current number of processed boxes: {len(bin_boxs_turn)}, total box {len(bins_boxs)}, processed {num-tmp_num}items, remaining {tmp_num}items"
            )
            num = tmp_num

    if num > 100:
        keys = list(processor.hash_buckets.keys())[::5][-20:]
        if keys:
            bin_boxs_simplest = tracker.track_packing(
                "pack_simplest_strategy",
                keys=keys,
                m=10,
                box_capacity=max_token_len,
                min_ratio=0.95,
                max_workers=os.cpu_count(),
            )
            update_stats = processor.update_hash_buckets(remove_empty=True, verbose=True)
            update_info(update_stats)
            print(len(bin_boxs_simplest))
            bins_boxs.extend(bin_boxs_simplest)
            print(
                f"finally----------the current processing box number: {len(bin_boxs_simplest)}, total box{len(bins_boxs)}"
            )
    return bins_boxs


def load_manifest_pack_items(cfg, media_type):
    manifest_sqlite = get_manifest_sqlite_path(cfg)
    return load_pack_items(manifest_sqlite, media_type)


def run_wds_native(cfg):
    max_token_len = int(cfg["sample"]["max_token_len"])
    packing_cfg = cfg.get("packing", {})
    algorithm = packing_cfg.get("algorithm", "hashbucket")
    keep_intermediate = keep_intermediate_artifacts(cfg)
    total_boxes = 0
    for media_type in MEDIA_TYPES:
        bins_path = get_bins_path(cfg, media_type)
        if bins_path.exists():
            bins_path.unlink()
        bins_plan_path = get_bins_plan_path(cfg, media_type)
        if bins_plan_path.exists():
            bins_plan_path.unlink()

    for media_type in MEDIA_TYPES:
        items = load_manifest_pack_items(cfg, media_type)
        if not items:
            print(f"Skip {media_type}: no samples in manifest")
            continue
        print(f"=== Packing media_type={media_type}, algorithm={algorithm} ===")
        if algorithm in ("best_fit_decreasing", "bfd"):
            bins_boxs = run_best_fit_decreasing_items(items, max_token_len)
        elif algorithm == "hashbucket":
            bins_boxs = run_hashbucket_items(items, max_token_len)
        else:
            raise ValueError(f"Unsupported packing algorithm: {algorithm}")
        file_path = get_bins_path(cfg, media_type)
        if keep_intermediate:
            with file_path.open("wb") as f:
                pickle.dump(bins_boxs, f)
        elif file_path.exists():
            file_path.unlink()
        plan_path = write_bins_plan(cfg, media_type, bins_boxs)
        total_boxes += len(bins_boxs)
        if keep_intermediate:
            print(
                f"{media_type} bins saved to {file_path}, compact_plan={plan_path}, "
                f"boxes={len(bins_boxs)}"
            )
        else:
            print(f"{media_type} compact bins plan saved to {plan_path}, boxes={len(bins_boxs)}")
    print(f"WDS-native hashbucket done, total boxes={total_boxes}")


def run_legacy(cfg):
    processor, tracker, token_info_file, max_token_len, packed_files_dir = init(cfg)
    bins_boxs = run_hashbucket(token_info_file, max_token_len)
    file_path = os.path.join(packed_files_dir, "bins_boxs.pkl")
    with open(file_path, "wb") as f:
        pickle.dump(bins_boxs, f)
    print(f"bins_boxs.pkl saved to {file_path}")


def main():
    args = parse_args()
    cfg = get_cfg(args.config)
    if cfg.get("data", {}).get("input_format", "wds") == "wds":
        run_wds_native(cfg)
    else:
        run_legacy(cfg)


if __name__ == "__main__":
    main()
