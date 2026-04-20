# Copyright 2026 The LoongForge Authors.
# SPDX-License-Identifier: Apache-2.0

"""Script for executing hash bucket packing process."""

from hashbacket import *
from pprint import pprint
import os
import yaml
import random
from utils import get_init_file, parse_args, get_cfg

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
    mean = sum(a) / (len(a) + 1)
    min_ = min(a)
    max_ = max(a)
    num = sum(len(arr) for arr in hs.values())

    return mean, min_, max_, num


def init(cfg):
    token_info_file, max_token_len, packed_files_dir, _ = get_init_file(cfg)
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
    return processor, tracker, token_info_file, max_token_len, packed_files_dir


def main():
    args = parse_args()
    cfg = get_cfg(args.config)
    bins_boxs = []
    processor, tracker, token_info_file, max_token_len, packed_files_dir = init(cfg)
    mean, min_, max_, tmp_num = get_hs(processor.hash_buckets)
    turn = 1
    bin_boxs_001 = tracker.track_packing(
        "pack_with_deletion", box_capacity=max_token_len
    )
    update_stats = processor.update_hash_buckets(remove_empty=True, verbose=True)
    rest_items = update_info(update_stats)
    bins_boxs.extend(bin_boxs_001)
    mean, min_, max_, num = get_hs(processor.hash_buckets)
    scale = int(mean - (mean - min_) * 0.1)
    print(
        f"in the first round of end ----------the current processing box number: {len (bin_boxs_001)}, total box {len (bins_boxs)}, handle the {num - tmp_num} items, remaining {num} the items"
    )
    # tmp_items=[(1,0.96),(5,0.96),(5,0.94),(6,0.92),(4,0.92),(4,0.92)]
    tmp_items = [(1, 0.96), (1, 0.95), (8, 0.9), (4, 0.92), (6, 0.92), (4, 0.9)]
    # tmp_items=[(1,0.96),(1,0.95),(8,0.9),(4,0.92),(6,0.92)]
    for i, item in enumerate(tmp_items):
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
    file_path = os.path.join(packed_files_dir, "bins_boxs.pkl")
    with open(file_path, "wb") as f:
        pickle.dump(bins_boxs, f)
    print(f"bins_boxs.pkl saved to {file_path}")


if __name__ == "__main__":
    main()