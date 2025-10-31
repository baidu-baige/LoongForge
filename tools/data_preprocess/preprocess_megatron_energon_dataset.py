"""preprocess sft data"""

import os, gc
import glob
import time
import sys
import torch
import shutil
import math
import fcntl
import argparse
from datasets import Dataset, concatenate_datasets

from aiak_training_omni.utils import initialize_aiak_megatron
from aiak_training_omni.train import parse_train_args
from aiak_training_omni.train.pretrain.pretrain_qwen2_vl import (
    train_valid_test_dataset_provider,
    get_dataset_size,
)

from megatron.core import parallel_state
from megatron.training.training import build_train_valid_test_data_iterators


def process_batch(samples, num_proc=None):
    """直接构建 Dataset 并用内部 map 并行"""
    dataset = Dataset.from_list(samples)

    def process_example(batch):
        batch["input_ids"] = batch["tokens"]
        return batch

    return dataset.map(
        process_example,
        batched=True,
        num_proc=1,
        desc="Processing examples"
    )


def acquire_slot(rank, data_path, world_size, local_world_size, pp, tp, max_concurrent):
    """
    获取节点 slot，确保：
      1. 每个节点最多运行 max_concurrent rank
      2. 节点内部参与 rank ≤ max_rank_node（dp 均匀分布到节点）
    返回 slot_path
    """
    node_rank = int(os.environ["RANK"]) // local_world_size
    node_dir = os.path.join(data_path, f"node{node_rank}_slots")
    os.makedirs(node_dir, exist_ok=True)

    # dp 均匀分布到节点
    dp = world_size // pp // tp
    num_nodes = max(1, world_size // local_world_size)
    max_rank_node = (dp + num_nodes - 1) // num_nodes  # ceil(dp / num_nodes)

    # 检查 rank 是否在节点允许范围
    node_rank_start = node_rank * max_rank_node
    node_rank_end = min(node_rank_start + max_rank_node, world_size)
    if not (node_rank_start <= rank < node_rank_end):
        print(f"[RANK {rank}] exceeds node{node_rank} allowed range [{node_rank_start},{node_rank_end}), exiting.")
        #sys.exit(0)
        return None

    slot_path = os.path.join(node_dir, f"slot_{rank}")
    counter_file = os.path.join(node_dir, "counter.txt")
    lock_file = os.path.join(node_dir, "counter.lock")
    os.makedirs(node_dir, exist_ok=True)

    while True:
        with open(lock_file, "w") as lf:
            fcntl.flock(lf, fcntl.LOCK_EX)
            with open(counter_file, "a+") as f:
                f.seek(0)
                content = f.read().strip()
                count = int(content) if content else 0

                if count < max_concurrent:
                    try:
                        fd = os.open(slot_path, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
                        os.close(fd)
                    except FileExistsError:
                        print(f"[RANK {rank}] Slot already exists, exiting.")
                        #sys.exit(0)
                        return None

                    count += 1
                    f.seek(0)
                    f.truncate()
                    f.write(str(count))
                    f.flush()
                    os.fsync(f.fileno())

                    fcntl.flock(lf, fcntl.LOCK_UN)
                    print(f"[RANK {rank}] Acquired slot {slot_path}, current_count={count}")
                    return slot_path

            fcntl.flock(lf, fcntl.LOCK_UN)

        time.sleep(1)

def release_slot(slot_path, rank):
    """释放 slot 并更新节点计数"""
    node_dir = os.path.dirname(slot_path)
    counter_file = os.path.join(node_dir, "counter.txt")

    try:
        os.remove(slot_path)
        print(f"[RANK {rank}] Released slot {slot_path}")
    except FileNotFoundError:
        pass

    with open(counter_file, "r+") as f:
        fcntl.flock(f, fcntl.LOCK_EX)
        f.seek(0)
        content = f.read().strip()
        count = int(content) if content else 0
        count = max(0, count - 1)
        f.seek(0)
        f.truncate()
        f.write(str(count))
        f.flush()
        os.fsync(f.fileno())
        fcntl.flock(f, fcntl.LOCK_UN)

def wait_for_previous_ranks(args, rank, data_path, max_concurrent=4):
    """阻塞直到当前 rank 可以执行"""
    world_size = int(os.environ["WORLD_SIZE"])
    local_world_size = int(os.environ.get("LOCAL_WORLD_SIZE", 8))
    pp = args.pipeline_model_parallel_size
    tp = args.tensor_model_parallel_size

    slot_path = acquire_slot(
        rank=rank,
        data_path=data_path,
        world_size=world_size,
        local_world_size=local_world_size,
        pp=pp,
        tp=tp,
        max_concurrent=max_concurrent
    )
    setattr(args, "_slot_path", slot_path)


def mark_rank_done(rank, args):
    """标记 rank 完成并释放 slot"""
    print("go to mark rank done")
    slot_path = getattr(args, "_slot_path", None)
    if slot_path:
        release_slot(slot_path, rank)
        print(f"[RANK {rank}] Processing done.")
    
    print(f"[RANK {rank}] Finished preprocessing.")
    torch.distributed.barrier()

def save_dataset(dataset, path):
    """Save dataset with optimized format"""
    print(f"save to disk")
    dataset.save_to_disk(path)


def normalize_grid_thw(v):
    if isinstance(v, torch.Tensor):
        v = v.tolist()
    if v is None:
        return [[0, 0, 0]]
    elif isinstance(v, list):
        if len(v) == 0:
            return [[0, 0, 0]]
        if all(isinstance(x, (int, float)) for x in v):
            return [[int(x) for x in v]]
        if all(isinstance(x, list) for x in v):
            return [[int(y) for y in x] for x in v]
    elif isinstance(v, (int, float)):
        return [[int(v)]]


def build_pack_dataset(args):
    """build sft dataset"""
    print("[build_pack_dataset] Initializing Megatron...")
    
    sys.modules['aiak_accelerator.multiacc_engine'] = None
    
    initialize_aiak_megatron(
        args=args,
        allow_no_cuda=True,
        get_embedding_ranks=None,
        get_position_embedding_ranks=None
    )

    if not hasattr(args, "iteration"):
        args.iteration = 50000
    if not hasattr(args, "consumed_train_samples"):
        args.consumed_train_samples = 0
    if not hasattr(args, "consumed_valid_samples"):
        args.consumed_valid_samples = 0

    print("[build_pack_dataset] Building data iterators...")
    train_data_iterator, _, _ = build_train_valid_test_data_iterators(train_valid_test_dataset_provider)
    
    rank = parallel_state.get_data_parallel_rank()
    total_samples = get_dataset_size()
    if train_data_iterator is None:
        print("[INFO] train_data_iterator is None, exiting current process.")
        #sys.exit(0)
        mark_rank_done(rank, args)
        return

    wait_for_previous_ranks(args, rank, args.output_path)
    
    if getattr(args, "_slot_path", None) is None:
        mark_rank_done(rank, args)
        return
    
    print(f"[RANK {rank}] Starting dataset processing...") 
    
    interested_keys = [
        'tokens','labels','max_lengths','cu_lengths',
        'attn_mask','imgs','pixel_values_videos',
        'image_grid_thw','video_grid_thw'
    ]
    
    temp_dir = os.path.join(args.output_path, f"temp_rank_{rank}")
    os.makedirs(temp_dir, exist_ok=True)

    total_samples = get_dataset_size()
    #total_samples =200
    batch_size = 100
    total_packed_samples = 0
    total_raw_samples = 0
    num_batches = math.ceil(total_samples / batch_size)
    print(f"[build_pack_dataset] Collecting {total_samples} samples ...")

    for batch_idx in range(num_batches):
        batch = []
        for sample_idx in range(batch_size):
            if total_raw_samples >= total_samples:
                print(f"[DEBUG] Reached total_samples={total_samples}, stop collecting more samples.")
                break  # 跳出 sample 循环，直接进入 process_batch
            
            try:
                data = next(train_data_iterator)
            except StopIteration:
                break

            cu_lengths = data["cu_lengths"]
            if isinstance(cu_lengths, torch.Tensor):
                cu_lengths = cu_lengths.tolist()
            if isinstance(cu_lengths[0], list):
                cu_lengths = cu_lengths[0]
            num_raw_samples = len(cu_lengths) - 1
            #print(f"[DEBUG] Packed sample contains {num_raw_samples} original samples.")
            total_raw_samples += num_raw_samples
            if total_raw_samples >= total_samples:
                total_packed_samples = batch_idx * batch_size + sample_idx
            
            item = {}
            for k in interested_keys:
                v = data[k]
                if k in ("video_grid_thw",'image_grid_thw'):
                    v = normalize_grid_thw(v)
                elif isinstance(v, torch.Tensor):
                    v = v.tolist()
                item[k] = v 

            batch.append(item)

            global_sample_idx = batch_idx * batch_size + sample_idx
            if global_sample_idx % 10 == 0:
                print(f"  - Sample {global_sample_idx} collected and packed.")

        if not batch:
            break

        # 处理并保存每个 batch
        processed_dataset = process_batch(batch)
        batch_file = os.path.join(temp_dir, f"batch_{batch_idx:04d}")
        save_dataset(processed_dataset, batch_file)
        print(f"[build_pack_dataset] Saved batch {batch_idx+1}/{num_batches} to {batch_file}")

        del batch, processed_dataset
        gc.collect()
        
        if total_raw_samples >= total_samples:
            total_packed_samples = batch_idx * batch_size + sample_idx
            break
        
    print(f"The number of samples have been changed from {total_samples} to {total_packed_samples}.")
    print(f"[build_pack_dataset] Finished. Batches saved under {temp_dir}")
    mark_rank_done(rank, args)
    print(f"NOTE: Please run sft with  `--is-tokenized-data`")

def parse_extra_args():
    """parse extra args"""
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-path", type=str, default=None,
                        help="Custom output path for preprocessed dataset")
    extra_args, remaining = parser.parse_known_args()
    sys.argv = [sys.argv[0]] + remaining  
    return extra_args

def main():
    """main function"""
    extra_args = parse_extra_args()
    if extra_args.output_path is None:
        print("[ERROR] --output-path must be specified.")
        sys.exit(1)

    args = parse_train_args()
    args.output_path = extra_args.output_path

    # 清理 node*_slots 目录
    for slot_dir in glob.glob(os.path.join(args.output_path, "node*_slots")):
        if os.path.isdir(slot_dir):
            try:
                shutil.rmtree(slot_dir)
                print(f"[INFO] Removed old slot directory: {slot_dir}")
            except Exception as e:
                print(f"[WARN] Failed to remove {slot_dir}: {e}")
    
    # 清理 temp_rank_* 目录
    for temp_dir in glob.glob(os.path.join(args.output_path, "temp_rank_*")):
        if os.path.isdir(temp_dir):
            try:
                shutil.rmtree(temp_dir)
                print(f"[INFO] Removed old temp directory: {temp_dir}")
            except Exception as e:
                print(f"[WARN] Failed to remove {temp_dir}: {e}")    

    build_pack_dataset(args)


if __name__ == '__main__':
    main()
