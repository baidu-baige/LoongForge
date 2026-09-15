#!/usr/bin/env python3
# Copyright 2026 The LoongForge Authors.
# SPDX-License-Identifier: Apache-2.0
"""Validate GLM-5.3-Flash TP/PP/CP with Loong-Megatron's native schedule."""

import argparse
import json
import os
import sys
from pathlib import Path

import torch
import torch.distributed as dist


def _batch(config, device: torch.device, multimodal: bool):
    if not multimodal:
        input_ids = torch.tensor([[3, 5, 7, 9, 11, 13, 15, 17]], dtype=torch.long, device=device)
        return {"input_ids": input_ids, "attention_mask": torch.ones_like(input_ids)}
    input_ids = torch.tensor(
        [[9, config.image_token_id, config.image_token_id, config.image_token_id,
          config.image_token_id, 10, 11, 12]],
        dtype=torch.long,
        device=device,
    )
    patch_width = (
        config.vision_config.in_channels
        * config.vision_config.temporal_patch_size
        * config.vision_config.patch_size
        * config.vision_config.patch_size
    )
    pixel_values = torch.linspace(
        -1.0, 1.0, steps=4 * patch_width, dtype=torch.float32, device=device
    ).reshape(4, patch_width)
    return {
        "input_ids": input_ids,
        "attention_mask": torch.ones_like(input_ids),
        "pixel_values": pixel_values,
        "image_grid_thw": torch.tensor([[1, 2, 2]], dtype=torch.long, device=device),
    }


def _reconstruct_cp(value: torch.Tensor, cp_group) -> torch.Tensor:
    if cp_group.size() == 1:
        return value
    gathered = [torch.empty_like(value) for _ in range(cp_group.size())]
    dist.all_gather(gathered, value, group=cp_group)
    halves = [tensor.chunk(2, dim=1) for tensor in gathered]
    return torch.cat([pair[0] for pair in halves] + [pair[1] for pair in reversed(halves)], dim=1)


def _reconstruct_sp(value: torch.Tensor, tp_group) -> torch.Tensor:
    if tp_group.size() == 1:
        return value
    from megatron.core.tensor_parallel import gather_from_sequence_parallel_region

    sequence_first = value.transpose(0, 1).contiguous()
    sequence_first = gather_from_sequence_parallel_region(
        sequence_first, tensor_parallel_output_grad=False, group=tp_group
    )
    return sequence_first.transpose(0, 1).contiguous()


def _forward_step(data_iterator, model):
    batch = next(data_iterator)
    output = model(**batch)
    logits = output.logits if hasattr(output, "logits") else output

    def loss_or_data(output_tensor, non_loss_data=False):
        if non_loss_data:
            return output_tensor.detach()
        loss = output_tensor.float().square().mean()
        return loss, {"loss": loss.detach()}

    return logits, loss_or_data


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--target-root", type=Path, required=True)
    parser.add_argument("--baseline", type=Path, required=True)
    parser.add_argument("--report", type=Path, required=True)
    parser.add_argument("--tp-size", type=int, default=1)
    parser.add_argument("--pp-size", type=int, default=1)
    parser.add_argument("--cp-size", type=int, default=1)
    parser.add_argument("--write-baseline", action="store_true")
    parser.add_argument("--multimodal", action="store_true")
    parser.add_argument("--backward", action="store_true")
    parser.add_argument("--atol", type=float, default=0.03)
    args = parser.parse_args()

    world_size = int(os.environ["WORLD_SIZE"])
    if args.tp_size * args.pp_size * args.cp_size != world_size:
        raise ValueError("WORLD_SIZE must equal TP * PP * CP")

    local_rank = int(os.environ["LOCAL_RANK"])
    torch.cuda.set_device(local_rank)
    dist.init_process_group("nccl")

    target_module = args.target_root / "loongforge/models/foundation/glm5_next"
    sys.path.insert(0, str(target_module))
    sys.path.insert(0, str(args.target_root / "third_party/Loong-Megatron"))

    from megatron.core import parallel_state
    from megatron.core.enums import ModelType
    from megatron.core.pipeline_parallel import get_forward_backward_func
    from megatron.core.process_groups_config import ProcessGroupCollection
    from megatron.core.tensor_parallel.random import model_parallel_cuda_manual_seed
    from megatron.core.utils import get_batch_on_this_cp_rank

    from glm5_next_model import Glm5NextModel

    try:
        parallel_state.initialize_model_parallel(
            tensor_model_parallel_size=args.tp_size,
            pipeline_model_parallel_size=args.pp_size,
            context_parallel_size=args.cp_size,
        )
        model_parallel_cuda_manual_seed(20260904)
        pg_collection = ProcessGroupCollection.use_mpu_process_groups()
        config_overrides = {
            "tensor_model_parallel_size": args.tp_size,
            "pipeline_model_parallel_size": args.pp_size,
            "context_parallel_size": args.cp_size,
            "sequence_parallel": args.tp_size > 1,
            "pipeline_dtype": torch.bfloat16,
        }
        model = Glm5NextModel.from_checkpoint(
            args.checkpoint,
            device="cuda",
            config_overrides=config_overrides,
            pre_process=parallel_state.is_pipeline_first_stage(),
            post_process=parallel_state.is_pipeline_last_stage(),
            pg_collection=pg_collection,
        ).eval()
        model.model_type = ModelType.encoder_or_decoder

        batch = _batch(model.config, torch.device("cuda"), args.multimodal)
        if args.cp_size > 1:
            local_ids = get_batch_on_this_cp_rank({"input_ids": batch["input_ids"]})
            batch["input_ids"] = local_ids["input_ids"]
            batch["attention_mask"] = torch.ones_like(batch["input_ids"])

        schedule = get_forward_backward_func()
        context = torch.enable_grad() if args.backward else torch.no_grad()
        with context:
            outputs = schedule(
                forward_step_func=_forward_step,
                data_iterator=iter([batch]),
                model=[model],
                num_microbatches=1,
                seq_length=8,
                micro_batch_size=1,
                forward_only=not args.backward,
                collect_non_loss_data=not args.backward,
            )

        local_layers = [layer.layer_number - 1 for layer in model.model.language_model.layers]
        logits = None
        if not args.backward and parallel_state.is_pipeline_last_stage():
            logits = _reconstruct_sp(outputs[0], pg_collection.tp)
            logits = _reconstruct_cp(logits, pg_collection.cp).float().cpu()

        if args.write_baseline:
            if world_size != 1 or logits is None:
                raise ValueError("--write-baseline requires TP=PP=CP=1")
            args.baseline.parent.mkdir(parents=True, exist_ok=True)
            torch.save(logits, args.baseline)

        diff = torch.zeros((), dtype=torch.float32, device="cuda")
        if not args.backward:
            if logits is not None:
                baseline = torch.load(args.baseline, map_location="cpu", weights_only=True)
                diff = (logits - baseline).abs().max().cuda()
            dist.all_reduce(diff, op=dist.ReduceOp.MAX)

        grad_finite = torch.ones((), dtype=torch.int32, device="cuda")
        has_grad = torch.zeros((), dtype=torch.int32, device="cuda")
        if args.backward:
            for parameter in model.parameters():
                if parameter.grad is not None:
                    has_grad.fill_(1)
                    grad_finite.mul_(torch.isfinite(parameter.grad).all().to(torch.int32))
            dist.all_reduce(grad_finite, op=dist.ReduceOp.MIN)
            dist.all_reduce(has_grad, op=dist.ReduceOp.MIN)

        result = {
            "tp": args.tp_size,
            "pp": args.pp_size,
            "cp": args.cp_size,
            "world_size": world_size,
            "max_abs_diff": None if args.backward else float(diff),
            "atol": args.atol,
            "passed": bool(grad_finite and has_grad) if args.backward else bool(diff <= args.atol),
            "multimodal": args.multimodal,
            "backward": args.backward,
            "grad_finite": bool(grad_finite) if args.backward else None,
            "all_ranks_have_grad": bool(has_grad) if args.backward else None,
        }
        layer_receipts = [None for _ in range(world_size)]
        dist.all_gather_object(layer_receipts, local_layers)
        result["rank_layer_ownership"] = layer_receipts
        if dist.get_rank() == 0:
            args.report.parent.mkdir(parents=True, exist_ok=True)
            args.report.write_text(json.dumps(result, indent=2) + "\n")
            print(json.dumps(result, indent=2))
        if not result["passed"]:
            raise SystemExit(1)
    finally:
        if parallel_state.is_initialized():
            parallel_state.destroy_model_parallel()
        if dist.is_initialized():
            dist.destroy_process_group()


if __name__ == "__main__":
    main()
