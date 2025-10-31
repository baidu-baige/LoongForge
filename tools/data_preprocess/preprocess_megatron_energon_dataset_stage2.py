import os
import re
import glob
import shutil
import argparse
from datasets import load_from_disk, concatenate_datasets


# ========== Argument Parsing ==========
def parse_args():
    """
    Parse command-line arguments for dataset merging.
    
    Returns:
        argparse.Namespace: Parsed arguments.
    """
    parser = argparse.ArgumentParser(
        description="DP dataset merging tool",
        epilog=(
            "Example:\n"
            "  python preprocess_megatron_energon_dataset_stage2.py "
            "--data-path /mnt/cluster/aiak-training-llm/dataset/mllm/demo/data "
            "--world-size 16 --tp 2 --pp 2 --clean-temp"
        ),
        formatter_class=argparse.RawTextHelpFormatter
    )

    parser.add_argument("--data-path", type=str, required=True,
                        help="Root dataset path containing temp_rank_xxx directories.")
    parser.add_argument("--world-size", type=int, required=True,
                        help="Training world size.")
    parser.add_argument("--tp", type=int, required=True,
                        help="Tensor parallel size.")
    parser.add_argument("--pp", type=int, required=True,
                        help="Pipeline parallel size.")
    parser.add_argument("--clean-temp", action="store_true",
                        help="Remove temp_rank_xxx directories after merging.")
    return parser.parse_args()


def _natural_key_for_batch(path: str) -> int:
    """Extract batch index for natural sorting."""
    name = os.path.basename(path.rstrip("/"))
    m = re.search(r"batch_(\d+)$", name)
    return int(m.group(1)) if m else 10**9


def find_all_ranks(data_path):
    """Find all available rank directories under data_path."""
    temp_dirs = glob.glob(os.path.join(data_path, "temp_rank_*"))
    ranks = []
    for temp_dir in temp_dirs:
        if os.path.isdir(temp_dir):
            try:
                rank = int(temp_dir.split("_")[-1])
                ranks.append(rank)
            except ValueError:
                print(f"[WARN] Skipped invalid directory name: {temp_dir}")
    ranks.sort()
    return ranks


def merge_rank_group(dp_rank: int, rank_list, data_path, clean_temp=False):
    """Merge all rank datasets belonging to one DP group."""
    print(f"[DP {dp_rank}] Start merging, ranks={rank_list}")
    datasets_list = []
    total_samples = 0
    used_ranks = []

    for rank in rank_list:
        temp_dir = os.path.join(data_path, f"temp_rank_{rank}")
        if not os.path.exists(temp_dir):
            print(f"[DP {dp_rank}] [WARN] Missing temp directory: {temp_dir}, skipped")
            continue

        batch_candidates = glob.glob(os.path.join(temp_dir, "batch_*"))
        batch_dirs = [p for p in batch_candidates if os.path.isdir(p)]
        batch_files = sorted(batch_dirs, key=_natural_key_for_batch)

        if not batch_files:
            print(f"[DP {dp_rank}] [WARN] No batch dirs found for rank {rank}, skipped")
            continue

        print(f"[DP {dp_rank}] Rank {rank} has {len(batch_files)} batches")
        for bf in batch_files:
            try:
                ds = load_from_disk(bf)
                datasets_list.append(ds)
                total_samples += len(ds)
            except Exception as e:
                print(f"[DP {dp_rank}] [ERROR] Failed to load {bf}: {e}")
                continue

        used_ranks.append(rank)

    if not datasets_list:
        print(f"[DP {dp_rank}] No valid datasets found, skipped")
        return False

    print(f"[DP {dp_rank}] Concatenating {len(datasets_list)} datasets...")
    full_dataset = concatenate_datasets(datasets_list)
    final_samples = len(full_dataset)
    print(f"[DP {dp_rank}] Done: total {total_samples}, merged {final_samples}")

    save_path = os.path.join(data_path, "preprocess", str(dp_rank))
    os.makedirs(os.path.dirname(save_path), exist_ok=True)

    if os.path.exists(save_path):
        print(f"[DP {dp_rank}] Target dir already exists, removing: {save_path}")
        shutil.rmtree(save_path, ignore_errors=True)

    full_dataset.save_to_disk(save_path, max_shard_size="1GB")
    print(f"[DP {dp_rank}] Saved -> {save_path}")

    try:
        test_load = load_from_disk(save_path)
        print(f"[DP {dp_rank}] Verified reload, samples: {len(test_load)}")
    except Exception as e:
        print(f"[DP {dp_rank}] [WARN] Verification failed: {e}")

    if clean_temp:
        for rank in used_ranks:
            temp_dir = os.path.join(data_path, f"temp_rank_{rank}")
            shutil.rmtree(temp_dir, ignore_errors=True)
        print(f"[DP {dp_rank}] Cleaned temp dirs: {used_ranks}")

    return True


def build_expected_groups(world_size: int, tp: int, pp: int):
    """Compute expected DP groups based on world_size, tp, and pp."""
    group_size = tp * pp
    if group_size <= 0:
        raise ValueError("tp and pp must be positive integers.")

    if world_size % group_size != 0:
        print(f"[WARN] world_size={world_size} not divisible by tp*pp={group_size}. "
              f"Extra ranks will be ignored.")

    dp = world_size // group_size
    groups = []
    for i in range(dp):
        start = i * group_size
        end = start + group_size
        groups.append(list(range(start, end)))
    return dp, groups


def split_ranks_to_dp(found_ranks, dp):
    """Evenly split found ranks into DP groups."""
    found_ranks = sorted(found_ranks)
    n = len(found_ranks)
    groups = []
    per_group = n // dp
    extra = n % dp
    start = 0
    for i in range(dp):
        end = start + per_group + (1 if i < extra else 0)
        groups.append(found_ranks[start:end])
        start = end
    return groups


def main():
    """Main execution flow."""
    args = parse_args()
    data_path = args.data_path

    print("=" * 60)
    print("DP Dataset Merging Tool")
    print(f"Dataset path: {data_path}")
    print("=" * 60)

    world_size = args.world_size
    tp = args.tp
    pp = args.pp

    found_ranks = find_all_ranks(data_path)
    print(f"\nFound {len(found_ranks)} temp ranks")
    print(f"Available ranks: {found_ranks}")

    try:
        dp, expected_groups = build_expected_groups(world_size, tp, pp)
    except Exception as e:
        print(f"[ERROR] {e}")
        return

    if dp > len(found_ranks):
        print(f"[ERROR] dp={dp} is larger than found ranks {len(found_ranks)}")
        return

    if dp == 0:
        print("[ERROR] Computed dp=0. Please check world_size/tp/pp values.")
        return

    print(f"Computed: tp={tp}, pp={pp}, world_size={world_size}, dp={dp}")

    actual_groups = split_ranks_to_dp(found_ranks, dp)
    print("\nPlanned DP groups:")
    for i, g in enumerate(actual_groups):
        print(f"  DP {i}: {g}")

    success = 0
    for dp_rank, rank_list in enumerate(actual_groups):
        print("\n" + "=" * 40)
        ok = merge_rank_group(dp_rank, rank_list, data_path, clean_temp=args.clean_temp)
        if ok:
            success += 1
        print("=" * 40)

    print(f"\nFinished: success {success}/{dp}")
    print(f"Output directory: {os.path.join(data_path, 'preprocess')}")


if __name__ == '__main__':
    main()
