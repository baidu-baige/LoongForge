#!/usr/bin/env python3
# Copyright 2026 The LoongForge Authors.
# SPDX-License-Identifier: Apache-2.0

import argparse
import hashlib
import json
import math
import os
import sys
from pathlib import Path

import torch
import torch.distributed as dist


def digest_json(value) -> str:
    return hashlib.sha256(
        json.dumps(value, ensure_ascii=True, separators=(",", ":")).encode()
    ).hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--report", type=Path, required=True)
    parser.add_argument("--steps", type=int, default=10)
    args = parser.parse_args()

    root = Path(__file__).resolve().parents[3]
    module_dir = root / "loongforge/models/foundation/glm5_next"
    sys.path.insert(0, str(module_dir))
    from glm5_next_config import Glm5NextConfig
    from glm5_next_model import Glm5NextModel

    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    torch.cuda.set_device(local_rank)
    dist.init_process_group("nccl")
    torch.manual_seed(20260901 + dist.get_rank())
    torch.cuda.manual_seed_all(20260901 + dist.get_rank())

    from megatron.core import parallel_state

    parallel_state.initialize_model_parallel(
        tensor_model_parallel_size=world_size,
        pipeline_model_parallel_size=1,
    )

    config = Glm5NextConfig(
        vocab_size=256,
        hidden_size=48,
        intermediate_size=64,
        moe_intermediate_size=32,
        num_hidden_layers=4,
        num_attention_heads=2,
        num_key_value_heads=2,
        n_shared_experts=1,
        n_routed_experts=8,
        num_experts_per_tok=4,
        routed_scaling_factor=1.0,
        q_lora_rank=32,
        kv_lora_rank=16,
        qk_rope_head_dim=0,
        qk_nope_head_dim=16,
        v_head_dim=16,
        index_head_dim=16,
        index_n_heads=2,
        index_topk=6,
        index_kpool=3,
        linear_num_heads=2,
        linear_head_dim=16,
        linear_conv_kernel_dim=2,
        layer_types=[
            "linear_attention",
            "linear_attention",
            "linear_attention",
            "deepseek_sparse_attention",
        ],
        mlp_layer_types=["dense", "dense", "dense", "sparse"],
        indexer_types=["full", "full", "full", "full"],
        max_position_embeddings=32,
        pad_token_id=0,
    )
    model = Glm5NextModel(config).cuda().to(torch.bfloat16).train()
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)
    generator = torch.Generator(device="cpu").manual_seed(20260902)
    input_ids = torch.randint(9, config.vocab_size, (1, 8), generator=generator).cuda()
    attention_mask = torch.ones_like(input_ids)
    labels = input_ids.clone()
    losses = []
    for _ in range(args.steps):
        optimizer.zero_grad(set_to_none=True)
        output = model(input_ids, attention_mask=attention_mask, labels=labels)
        output.loss.backward()
        optimizer.step()
        losses.append(float(output.loss.detach()))

    if not all(math.isfinite(loss) for loss in losses):
        raise RuntimeError(f"non-finite smoke losses: {losses}")
    dist.barrier()
    if dist.get_rank() == 0:
        production_command = ["python", str(root / "loongforge/train.py")]
        report = {
            "schema_version": 1,
            "optimizer_steps": args.steps,
            "losses": losses,
            "nan_detected": False,
            "initialization": "random",
            "checkpoint_loaded": False,
            "world_size": world_size,
            "execution_paths": ["text"],
            "example_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
            "production_command_sha256": digest_json(production_command),
        }
        args.report.parent.mkdir(parents=True, exist_ok=True)
        args.report.write_text(json.dumps(report, indent=2) + "\n")
        print(json.dumps(report, indent=2))
    parallel_state.destroy_model_parallel()
    dist.destroy_process_group()


if __name__ == "__main__":
    main()
