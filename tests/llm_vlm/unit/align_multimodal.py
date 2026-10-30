#!/usr/bin/env python3
# Copyright 2026 The LoongForge Authors.
# SPDX-License-Identifier: Apache-2.0

import argparse
import json
import sys
from pathlib import Path

import torch
import torch.distributed as dist


def build_batch(config, device: torch.device) -> dict[str, torch.Tensor]:
    generator = torch.Generator(device="cpu").manual_seed(20260903)
    patch_width = (
        config.vision_config.in_channels
        * config.vision_config.temporal_patch_size
        * config.vision_config.patch_size
        * config.vision_config.patch_size
    )
    pixel_values = torch.randn(4, patch_width, generator=generator, dtype=torch.float32)
    input_ids = torch.tensor(
        [[9, config.image_token_id, config.image_token_id, config.image_token_id, config.image_token_id, 10, 11, 12]],
        dtype=torch.long,
    )
    return {
        "input_ids": input_ids.to(device),
        "attention_mask": torch.ones_like(input_ids).to(device),
        "labels": input_ids.to(device),
        "pixel_values": pixel_values.to(device),
        "image_grid_thw": torch.tensor([[1, 2, 2]], dtype=torch.long, device=device),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--transformers-source", type=Path, required=True)
    parser.add_argument("--target-root", type=Path, required=True)
    parser.add_argument("--report", type=Path, required=True)
    parser.add_argument("--atol", type=float, default=0.01)
    parser.add_argument(
        "--logits-atol",
        type=float,
        default=0.02,
        help="BF16 logits tolerance; reductions remain checked with --atol.",
    )
    args = parser.parse_args()

    initialized_here = False
    if not dist.is_initialized():
        dist.init_process_group("nccl")
        initialized_here = True
    from megatron.core import parallel_state

    if not parallel_state.is_initialized():
        parallel_state.initialize_model_parallel(
            tensor_model_parallel_size=1,
            pipeline_model_parallel_size=1,
        )

    try:
        sys.path.insert(0, str(args.transformers_source))
        from transformers.models.glm5_next.modeling_glm5_next import (
            Glm5NextForConditionalGeneration,
            Glm5NextTextIndexer,
        )

        indexer_forward = Glm5NextTextIndexer.forward

        def indexer_forward_with_long_indices(self, *forward_args, **forward_kwargs):
            return indexer_forward(self, *forward_args, **forward_kwargs).long()

        Glm5NextTextIndexer.forward = indexer_forward_with_long_indices

        target_module = args.target_root / "loongforge/models/foundation/glm5_next"
        sys.path.insert(0, str(target_module))
        from glm5_next_model import Glm5NextModel

        device = torch.device("cuda")
        reference = Glm5NextForConditionalGeneration.from_pretrained(
            args.checkpoint,
            torch_dtype=torch.bfloat16,
            attn_implementation="eager",
        ).to(device).eval()
        target = Glm5NextModel.from_checkpoint(args.checkpoint, device=device).eval()
        batch = build_batch(target.config, device)

        with torch.no_grad():
            reference_vision = reference.get_image_features(
                batch["pixel_values"], batch["image_grid_thw"]
            ).pooler_output[0]
            target_vision = target.get_image_features(
                batch["pixel_values"], batch["image_grid_thw"]
            ).pooler_output[0]
            reference_output = reference(**batch, use_cache=False)
            target_output = target(**batch)

        metrics = {
            "vision_max_abs_diff": float((reference_vision.float() - target_vision.float()).abs().max()),
            "logits_max_abs_diff": float(
                (reference_output.logits.float() - target_output.logits.float()).abs().max()
            ),
            "reference_loss": float(reference_output.loss),
            "target_loss": float(target_output.loss),
            "loss_abs_diff": abs(float(reference_output.loss) - float(target_output.loss)),
        }
        metrics["passed"] = (
            metrics["vision_max_abs_diff"] <= args.atol
            and metrics["logits_max_abs_diff"] <= args.logits_atol
            and metrics["loss_abs_diff"] <= args.atol
        )
        args.report.parent.mkdir(parents=True, exist_ok=True)
        args.report.write_text(json.dumps(metrics, indent=2) + "\n")
        print(json.dumps(metrics, indent=2))
        if not metrics["passed"]:
            raise SystemExit(1)
    finally:
        if parallel_state.is_initialized():
            parallel_state.destroy_model_parallel()
        if initialized_here and dist.is_initialized():
            dist.destroy_process_group()


if __name__ == "__main__":
    main()
