import concurrent.futures
import json
import os
import re
import time
from argparse import ArgumentParser
from collections import defaultdict
from typing import Dict, List, Tuple

import aiak_fp8_quantizer
import torch
from safetensors.torch import load_file, save_file
from transformers.modeling_utils import SAFE_WEIGHTS_INDEX_NAME


# Now this QUANT_PATTERNS is only validated on DeepSeek V3.
QUANT_PATTERNS = [
    # Self-attention
    r".*\.self_attn\.q_a_proj\.weight$",
    r".*\.self_attn\.q_b_proj\.weight$", 
    r".*\.self_attn\.kv_a_proj_with_mqa\.weight$",
    r".*\.self_attn\.kv_b_proj\.weight$",
    r".*\.self_attn\.o_proj\.weight$",
    
    # MLP
    r".*\.mlp\.gate_proj\.weight$",
    r".*\.mlp\.up_proj\.weight$",
    r".*\.mlp\.down_proj\.weight$",
    
    # MoE
    r".*\.mlp\.shared_experts\.gate_proj\.weight$",
    r".*\.mlp\.shared_experts\.up_proj\.weight$",
    r".*\.mlp\.shared_experts\.down_proj\.weight$",
    
    r".*\.mlp\.experts\.\d+\.gate_proj\.weight$",
    r".*\.mlp\.experts\.\d+\.up_proj\.weight$", 
    r".*\.mlp\.experts\.\d+\.down_proj\.weight$",

    # r".*\.(gate|up|down)_proj\.weight$"
]

def should_quantize(weight_name):
    """
    Check if a weight tensor should be quantized based on its name pattern.
    
    Args:
        weight_name (str): The name of the weight tensor
        
    Returns:
        bool: True if the weight should be quantized, False otherwise
    """
    return any(re.match(pattern, weight_name) for pattern in QUANT_PATTERNS)

def quantize_shard(shard: Dict[str, torch.Tensor], gpu_id: int) -> Dict[str, torch.Tensor]:
    """
    Quantize a single shard of model weights to FP8 format.
    
    Args:
        shard (Dict[str, torch.Tensor]): Dictionary containing model weights for the shard
        gpu_id (int): ID of the GPU to use for quantization
        
    Returns:
        Dict[str, torch.Tensor]: Dictionary containing quantized weights and their scales
    """
    start_time = time.time()
    device = torch.device(f"cuda:{gpu_id}")
    torch.cuda.set_device(gpu_id)
    
    total_params = 0
    quantized_params = 0
    processed_keys = 0
    
    result = {}
    
    # To GPU
    for key, tensor in shard.items():
        shard[key] = tensor.to(device, non_blocking=True)
    
    # call aiak_fp8_quantizer
    for weight_name, weight in shard.items():
        total_params += weight.numel()
        
        if should_quantize(weight_name):
            fp8_weight, fp8_scale, _, _ = \
                aiak_fp8_quantizer.per_block_cast_to_fp8_fprop_vector(weight)
            
            result[weight_name] = fp8_weight
            result[f"{weight_name}_scale_inv"] = fp8_scale
            
            quantized_params += fp8_weight.numel() + fp8_scale.numel()
            processed_keys += 1
        else:
            # remain non-quantize weight
            result[weight_name] = weight
    
    # To CPU
    for key in result:
        result[key] = result[key].cpu()
    
    del shard
    torch.cuda.empty_cache()
    
    elapsed = time.time() - start_time
    size_mb = sum(tensor.numel() * tensor.element_size() for tensor in result.values()) / (1024 ** 2)
    quant_percent = (quantized_params / total_params) * 100 if total_params > 0 else 0
    
    print(f"[GPU {gpu_id}] Quantized {processed_keys} keys, "
          f"{quant_percent:.1f}% of params, {size_mb:.2f} MB in {elapsed:.2f}s "
          f"({size_mb/elapsed:.2f} MB/s)")
    
    return result

def process_shard_file(shard_file: str, input_dir: str, output_dir: str, gpu_id: int):
    """
    Process a single shard file: load, quantize, and save.
    
    Args:
        shard_file (str): Name of the shard file to process
        input_dir (str): Input directory containing the shard file
        output_dir (str): Output directory to save the processed shard
        gpu_id (int): ID of the GPU to use for processing
    """
    load_start = time.time()
    file_path = os.path.join(input_dir, shard_file)
    shard = load_file(file_path)
    load_time = time.time() - load_start
    size_mb = sum(tensor.numel() * tensor.element_size() for tensor in shard.values()) / (1024 ** 2)
    print(f"[LOAD] Loaded {shard_file} ({size_mb:.2f} MB) in {load_time:.2f}s")
    
    # quantize
    quant_start = time.time()
    quantized_shard = quantize_shard(shard, gpu_id)
    quant_time = time.time() - quant_start
    
    # save
    save_start = time.time()
    save_path = os.path.join(output_dir, shard_file)
    save_file(quantized_shard, save_path, metadata={"format": "pt"})
    save_time = time.time() - save_start
    size_mb = sum(tensor.numel() * tensor.element_size() for tensor in quantized_shard.values()) / (1024 ** 2)
    
    print(f"[SAVE] Saved {shard_file} ({size_mb:.2f} MB) in {save_time:.2f}s "
          f"Quant time: {quant_time:.2f}s")
    
    return {
        "file": shard_file,
        "load_time": load_time,
        "quant_time": quant_time,
        "save_time": save_time,
        "size_mb": size_mb,
        "gpu": gpu_id
    }

def convert_safetensors_dirs(input_dir: str, output_dir: str):
    """
    Convert model weights from BF16 to FP8 format in safetensors format.
    
    This function handles the entire conversion process including:
    1. Loading the input safetensors files
    2. Quantizing the weights to FP8 format
    3. Saving the quantized weights in safetensors format
    4. Creating/updating the index file
    
    Args:
        input_dir (str): Path to input directory containing BF16 safetensors files
        output_dir (str): Path to output directory for FP8 safetensors files
    """
    os.makedirs(output_dir, exist_ok=True)
    print(f"\n[START] Converting from {input_dir} to {output_dir}")
    
    safetensor_files = [f for f in os.listdir(input_dir) if f.endswith(".safetensors")]
    
    if not safetensor_files:
        raise FileNotFoundError(f"[ERROR] No safetensors files found in {input_dir}")
    
    index_file = os.path.join(input_dir, SAFE_WEIGHTS_INDEX_NAME)
    index_data = None
    if os.path.exists(index_file):
        with open(index_file, "r", encoding="utf-8") as f:
            index_data = json.load(f)
    
    print(f"[INFO] Found {len(safetensor_files)} safetensors files")
    if index_data:
        print(f"[INFO] Using index file: {SAFE_WEIGHTS_INDEX_NAME}")
    
    num_gpus = torch.cuda.device_count()
    gpu_rotation = [i % num_gpus for i in range(len(safetensor_files))]
    
    stats = []
    total_size = 0
    
    with concurrent.futures.ThreadPoolExecutor(max_workers=num_gpus) as executor:
        futures = []
        for i, shard_file in enumerate(safetensor_files):
            gpu_id = gpu_rotation[i]
            futures.append(executor.submit(
                process_shard_file, 
                shard_file, 
                input_dir, 
                output_dir, 
                gpu_id
            ))
        
        for future in concurrent.futures.as_completed(futures):
            try:
                result = future.result()
                stats.append(result)
                total_size += result["size_mb"]
            except Exception as e:
                print(f"[ERROR] Processing failed: {e}")
                raise
    
    # New index file
    if index_data:
        new_weight_map = {}
        
        # original index
        for tensor_name, shard_file in index_data["weight_map"].items():
            new_weight_map[tensor_name] = shard_file
            
            # Add quantize weight info
            if should_quantize(tensor_name):
                scale_inv_key = f"{tensor_name}_scale_inv"
                new_weight_map[scale_inv_key] = shard_file
        
        new_index = {
            "metadata": index_data.get("metadata", {}),
            "weight_map": new_weight_map
        }
        
        # save
        save_index_file = os.path.join(output_dir, SAFE_WEIGHTS_INDEX_NAME)
        with open(save_index_file, "w", encoding="utf-8") as f:
            json.dump(new_index, f, indent=2, sort_keys=True)
        print(f"[INDEX] Created new index file: {SAFE_WEIGHTS_INDEX_NAME}")
        
        # validate
        indexed_keys = set(new_weight_map.keys())
        actual_keys = set()
        for shard_file in safetensor_files:
            shard_path = os.path.join(output_dir, shard_file)
            shard = load_file(shard_path)
            actual_keys.update(shard.keys())
        
        missing_keys = indexed_keys - actual_keys
        extra_keys = actual_keys - indexed_keys
        
        if missing_keys:
            print(f"[WARNING] Index contains {len(missing_keys)} keys not in shards")
        if extra_keys:
            print(f"[WARNING] Shards contain {len(extra_keys)} keys not in index")
    
    print("\n[STATS] Conversion Summary:")
    print(f"  Total files:    {len(stats)}")
    print(f"  Total size:     {total_size:.2f} MB")

if __name__ == "__main__":
    print(f"[INFO] CUDA version: {torch.version.cuda}")
    print(f"[INFO] PyTorch version: {torch.__version__}")
    print(f"[INFO] Available GPUs: {torch.cuda.device_count()}")
    
    parser = ArgumentParser()
    parser.add_argument("--bf16-dir", type=str, required=True, help="Path to BF16 safetensors release")
    parser.add_argument("--fp8-dir", type=str, required=True, help="Path to output FP8 safetensors release")
    args = parser.parse_args()

    total_start = time.time()
    convert_safetensors_dirs(args.bf16_dir, args.fp8_dir)
    total_elapsed = time.time() - total_start
    print(f"\n[COMPLETE] Total conversion time: {total_elapsed:.2f} seconds")