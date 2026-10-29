# Copyright 2026 The LoongForge Authors.
# SPDX-License-Identifier: Apache-2.0
#
# Modified from Megatron-LM under the BSD 3-Clause License.
# Copyright (c) 2025, NVIDIA CORPORATION. All rights reserved.

"""default sft trainer for generative models like GPTS"""

import torch

from functools import partial


from megatron.core import mpu, tensor_parallel
from megatron.core.enums import ModelType
from megatron.core.packed_seq_params import PackedSeqParams
from megatron.core.utils import StragglerDetector
from megatron.core.datasets.utils import get_blend_from_list
from megatron.core.rerun_state_machine import get_rerun_state_machine
from megatron.core.num_microbatches_calculator import get_num_microbatches

from megatron.training import get_timers

from loongforge.engines.megatron import (
    get_args,
    get_model_config,
    get_tokenizer,
    get_chat_template,
)
from megatron.training.utils import print_rank_0

from loongforge.data.llm.sft_dataset import SFTDataset, SFTDatasetConfig
from loongforge.data.llm.blended_hf_dataset_builder import BlendedHuggingFaceDatasetBuilder
from loongforge.data.llm.sft_data_collator import DataCollatorForSupervisedDataset

from loongforge.data import IGNORE_INDEX

from loongforge.engines.megatron.training_utils import pretrain

from loongforge.engines.megatron.utils import get_batch_on_this_cp_rank
from loongforge.engines.megatron.data_padding import llm_pad_to_multiple_of
from loongforge.data.llm.blended_hf_dataset_config import get_dataset_blend_from_list
from loongforge.engines.megatron.checkpointing import load_dataloader_state
from loongforge.data.llm.sft_dataloader import (
    SFTLoaderConfig,
    build_sft_cyclic_iterators,
    build_sft_data_collator,
)
from loongforge.training.llm.pretrain import get_embedding_ranks, model_provider


stimer = StragglerDetector()


def _get_position_ids(data: torch.Tensor):
    """Create position ids."""
    current_device = data.device
    _, seq_length = data.shape
    position_ids = torch.arange(seq_length, dtype=torch.long, device=current_device)
    position_ids = position_ids.unsqueeze(0).expand_as(data)
    return position_ids


def _get_attention_mask(attention_mask: torch.Tensor, args) -> torch.Tensor:
    """Create attention mask for TE."""
    current_device = attention_mask.device
    batch_size, seq_length = attention_mask.shape

    # Only used attn_mask when attn_mask_type in [padding, padding_causal, arbitrary] in TE
    if args.context_parallel_size > 1:
        # Context parallel only supports causal mask in TE now.
        # When context-parallel is enabled, the input data is of a relatively long length,
        # and micro-batch-size does not need to be increased, nor padding occurs.
        # Create causal mask here, shape [B, 1, S, S].
        attention_mask = torch.tril(
            torch.ones(
                (batch_size, seq_length, seq_length),
                dtype=torch.long,
                device=current_device,
            )
        )
        attention_mask.unsqueeze_(1)
        attention_mask = (attention_mask < 0.5).bool()
    else:
        # Create mask for TE, shape [B, 1, 1, S]. attn_mask_type is padding_causal or causal.
        attention_mask.unsqueeze_(1).unsqueeze_(1)
        attention_mask = (attention_mask < 0.5).bool()

    return attention_mask


def _get_packed_sequence_params(attention_mask: torch.Tensor) -> PackedSeqParams:
    """Create packed sequence params."""
    # assume micro_batch_size == 1
    assert attention_mask.shape[0] == 1, "attention_mask should be of shape [1, S]"

    packed_seq_params = PackedSeqParams()
    packed_seq_params.qkv_format = "thd"

    # calculate cu_seqlens_q
    # example: mask = [[1, 1, 2, 2, 2, 3, 3, 4, 5, 5, 5, 0, 0]]
    # expected cu_seqlens_q = [0, 2, 5, 7, 8, 11, 13]
    max_num = attention_mask.max().item()
    reduced_mask = torch.bincount(attention_mask.view(-1), minlength=max_num + 1)
    reduced_mask = reduced_mask[1:].to(dtype=torch.int32, device=attention_mask.device)

    cu_seqlens = reduced_mask.cumsum(dim=0).to(torch.int32)
    zero = torch.zeros(1, dtype=torch.int32, device=attention_mask.device)
    # The lengths of padding tokens must also be taken into account in cu_seqlens;
    # otherwise, the attention calculation will be incorrect.
    cu_seqlens[-1] = attention_mask.shape[1]
    cu_seqlens = torch.cat((zero, cu_seqlens))

    packed_seq_params.cu_seqlens_q = cu_seqlens
    packed_seq_params.cu_seqlens_kv = cu_seqlens  # just for self-attention
    packed_seq_params.max_seqlen_q = (cu_seqlens[1:] - cu_seqlens[:-1]).max().item()
    packed_seq_params.max_seqlen_kv = packed_seq_params.max_seqlen_q

    return packed_seq_params


def _required_tp_keys(args):
    """Keys read from the data iterator on the tensor-parallel source rank."""
    keys = ["attention_mask"]
    if args.enable_chunkpipe:
        keys += ["chunk_group_size", "group_total_tokens"]
    need_labels = args.pipeline_model_parallel_size == 1 or mpu.is_pipeline_last_stage()
    if need_labels:
        keys += ["input_ids", "labels"]
        if not args.eod_mask_loss:
            keys.append("loss_mask")
    elif mpu.is_pipeline_first_stage():
        keys.append("input_ids")
    return keys


def _broadcast_chunkpipe_metadata(step_num_groups, component_sizes):
    """Broadcast the per-micro-batch (source-group count, composite descriptor) from TP rank 0."""
    meta = torch.tensor(
        [step_num_groups, len(component_sizes)],
        dtype=torch.long,
        device=torch.cuda.current_device(),
    )
    torch.distributed.broadcast(
        meta,
        mpu.get_tensor_model_parallel_src_rank(),
        group=mpu.get_tensor_model_parallel_group(),
    )
    n_comp = int(meta[1].item())
    if n_comp > 0:
        if mpu.get_tensor_model_parallel_rank() == 0:
            comp_tensor = torch.tensor(
                component_sizes, dtype=torch.long, device=torch.cuda.current_device()
            )
        else:
            comp_tensor = torch.zeros(
                n_comp, dtype=torch.long, device=torch.cuda.current_device()
            )
        torch.distributed.broadcast(
            comp_tensor,
            mpu.get_tensor_model_parallel_src_rank(),
            group=mpu.get_tensor_model_parallel_group(),
        )
        component_sizes = comp_tensor.tolist()
    else:
        component_sizes = []
    return meta[:1], component_sizes


def _build_sft_batch(args, tokenizer, data_b):
    """Turn the broadcast data dict into model inputs, slicing MTP tokens and shifting SFT labels."""
    sft_chunkpipe_mtp = (
        args.enable_chunkpipe
        and getattr(args, "sft_chunkpipe_mode", False)
        and getattr(args, "mtp_num_layers", 0)
        and args.mtp_num_layers > 0
    )
    base_length = args.chunksize if sft_chunkpipe_mtp else None

    tokens_full = data_b["input_ids"].long() if "input_ids" in data_b else None
    tokens = tokens_full
    mtp_tokens = None
    mtp_position_ids = None
    if tokens_full is not None:
        if sft_chunkpipe_mtp:
            expected_length = base_length + args.mtp_num_layers
            assert tokens_full.dim() == 2, (
                f"SFT chunkpipe MTP expects 2D tokens, got shape {tuple(tokens_full.shape)}."
            )
            assert tokens_full.size(1) == expected_length, (
                f"SFT chunkpipe MTP expects physical sequence length "
                f"{expected_length}, got {tokens_full.size(1)}."
            )
            mtp_tokens = tokens_full
            tokens = tokens_full[:, :base_length]
            assert tokens.size(1) == base_length, (
                f"SFT chunkpipe main tokens must have base length "
                f"{base_length}, got {tokens.size(1)}."
            )
            mtp_position_ids = _get_position_ids(mtp_tokens)
        position_ids = _get_position_ids(tokens)
    else:
        position_ids = None

    labels_full = data_b["labels"].long() if "labels" in data_b else None
    labels = labels_full
    mtp_labels = None
    if labels_full is not None:
        if sft_chunkpipe_mtp:
            mtp_labels = labels_full[:, :base_length + args.mtp_num_layers]
            labels = labels_full[:, :base_length]
        elif not args.enable_chunkpipe:
            # Shift labels for next-token prediction; chunkpipe data is already pre-shifted
            labels = torch.roll(labels, shifts=-1, dims=1)
            labels[:, -1] = IGNORE_INDEX

    loss_mask_full = data_b["loss_mask"].long() if "loss_mask" in data_b else None
    loss_mask = loss_mask_full
    mtp_loss_mask = None
    if loss_mask_full is not None:
        if sft_chunkpipe_mtp:
            mtp_loss_mask = loss_mask_full[:, :base_length + args.mtp_num_layers]
            loss_mask = loss_mask_full[:, :base_length]
        elif not args.enable_chunkpipe:
            # pp last && not eod_mask_loss; chunkpipe data is already pre-shifted
            loss_mask = torch.roll(loss_mask, shifts=-1, dims=1)
            loss_mask[:, -1] = 0
    elif labels is not None:
        # pp last && eod_mask_loss
        assert args.eod_mask_loss, "eod_mask_loss should be true here!"
        loss_mask = torch.ones(labels.size(), dtype=torch.float, device=labels.device)
        loss_mask[labels == IGNORE_INDEX] = 0.0
        loss_mask[labels == tokenizer.pad] = 0.0
        loss_mask[labels == tokenizer.eos] = 0.0
        if sft_chunkpipe_mtp and mtp_labels is not None:
            mtp_loss_mask = torch.ones(mtp_labels.size(), dtype=torch.float, device=mtp_labels.device)
            mtp_loss_mask[mtp_labels == IGNORE_INDEX] = 0.0
            mtp_loss_mask[mtp_labels == tokenizer.pad] = 0.0
            mtp_loss_mask[mtp_labels == tokenizer.eos] = 0.0

    attention_mask = None
    packed_seq_params = None
    attention_mask_data = data_b["attention_mask"].long()
    if sft_chunkpipe_mtp:
        attention_mask_data = attention_mask_data[:, :base_length]

    if not args.packing_sft_data:
        attention_mask = _get_attention_mask(attention_mask_data, args)
    else:
        # attention_mask will be ignored in te
        packed_seq_params = _get_packed_sequence_params(attention_mask_data)

    batch = {
        "tokens": tokens,
        "labels": labels,
        "loss_mask": loss_mask,
        "position_ids": position_ids,
        "attention_mask": attention_mask,
        "packed_seq_params": packed_seq_params,
    }
    if sft_chunkpipe_mtp:
        if mtp_tokens is not None:
            batch["mtp_tokens"] = mtp_tokens
            batch["mtp_position_ids"] = mtp_position_ids
        if mtp_labels is not None:
            batch["mtp_labels"] = mtp_labels
        if mtp_loss_mask is not None:
            batch["mtp_loss_mask"] = mtp_loss_mask
    if args.enable_chunkpipe and "chunk_group_size" in data_b:
        batch["chunk_group_size"] = data_b["chunk_group_size"]
        batch["group_total_tokens"] = data_b["group_total_tokens"]
    return batch


def get_batch(data_iterator):
    """Generate a batch"""
    args = get_args()
    tokenizer = get_tokenizer()

    data = next(data_iterator) if data_iterator is not None else None

    # The iterator wrapper attaches the sampler's per-micro-batch metadata to each batch.
    # Only TP rank 0 of the VP last stage reads it (loss_func runs there); every other
    # rank starts from 0 / [] and TP ranks receive rank 0's values by broadcast.
    step_num_groups = 0
    component_sizes = []
    vp_size = mpu.get_virtual_pipeline_model_parallel_world_size()
    is_vpp = vp_size is not None and vp_size > 1
    reads_metadata = mpu.get_tensor_model_parallel_rank() == 0 and (
        not is_vpp or mpu.get_virtual_pipeline_model_parallel_rank() == vp_size - 1
    )
    if args.enable_chunkpipe and reads_metadata and data is not None and "chunkpipe_step_num_groups" in data:
        step_num_groups = data["chunkpipe_step_num_groups"]
        component_sizes = data["chunkpipe_component_sizes"]

    data_b = tensor_parallel.broadcast_data(_required_tp_keys(args), data, torch.int64)
    batch = _build_sft_batch(args, tokenizer, data_b)
    if args.enable_chunkpipe and "chunk_group_size" in data_b:
        batch["step_num_groups"], batch["composite_component_sizes"] = (
            _broadcast_chunkpipe_metadata(step_num_groups, component_sizes)
        )

    # slice batch along sequence dimension for context parallelism
    batch = get_batch_on_this_cp_rank(batch)

    # Extract chunk_group_size into the model config for model-side code.
    if args.enable_chunkpipe and "chunk_group_size" in batch:
        group_size = batch["chunk_group_size"][0].item()
        config = get_model_config()
        if config is not None and getattr(config, 'sft_chunkpipe_mode', False):
            # Always write chunkpipe_current_group_size from the batch data, regardless of
            # chunk_idx_in_group. This breaks the circular dependency where:
            #   1. group_size_cache missing group N → fallback gives wrong chunk_idx
            #   2. wrong chunk_idx ≠ 0 → get_batch() would not write group_size
            #   3. scheduler reads stale _discovered_gs → group_size_cache[N] = wrong value
            # By always writing, the scheduler post-step always gets the correct group_size
            # to update group_size_cache, even when chunk_idx was computed via fallback.
            config.chunkpipe_current_group_size = group_size

            # Loong-Megatron scheduler: the scheduler must set
            # config.chunkpipe_chunk_idx_in_group = 0 **before** invoking forward_step, so
            # that get_batch (called inside forward_step) observes the correct index here.
            #
            # Timing chain:
            #   scheduler sets chunk_idx = 0
            #     → forward_step()
            #       → get_batch() reads chunk_group_size from batch
            #         → writes config.chunkpipe_current_group_size   ← here
            #           → model code reads config.chunkpipe_current_group_size
            #
            # The ordering is currently correct but depends on this implicit call sequence.
            # If the scheduler logic is refactored, ensure this contract is preserved.
            if config.chunkpipe_chunk_idx_in_group == 0:
                # Composite scheduling info: applied at every real-group start
                # within the composite. The composite descriptor is the same
                # for every chunk inside one composite (sampler repeats it
                # per micro-batch), so re-writing on each real-group's chunk 0
                # is idempotent and preserves the value across the composite
                # for downstream readers (schedule.py + MLA chunk_keys).
                if "composite_component_sizes" in batch:
                    components = batch["composite_component_sizes"]
                    if components:
                        config.chunkpipe_component_sizes = list(components)
                    else:
                        # No composite info → fall back to trivial (single
                        # real group). Keeps behavior sane if the queues are
                        # somehow not populated (e.g. ad-hoc test paths).
                        config.chunkpipe_component_sizes = [group_size]

        # N_g and G for the per-sample loss path. N_g is per-chunk (same value
        # for all chunks in one source sequence), G_total is per-step (same
        # for all chunks in one step) and is the cross-rank sum of source
        # group counts. Both are read by loss_func from args. Because
        # micro_batch_size >= 1 chunk all share the same step, picking index 0
        # is correct.
        if "group_total_tokens" in batch:
            args.chunkpipe_group_total_tokens = batch["group_total_tokens"][0].item()
        if "step_num_groups" in batch:
            step_num_groups = batch["step_num_groups"][0].item()
            args.chunkpipe_step_num_groups = step_num_groups
            if config is not None and getattr(config, 'sft_chunkpipe_mode', False):
                config.chunkpipe_step_num_groups = step_num_groups

    mtp_batch = None
    if args.enable_chunkpipe and getattr(args, "sft_chunkpipe_mode", False):
        if "mtp_tokens" in batch:
            expected_length = args.chunksize + (args.mtp_num_layers or 0)
            assert batch["tokens"].size(1) == args.chunksize, (
                f"SFT chunkpipe main tokens must have base length "
                f"{args.chunksize}, got {batch['tokens'].size(1)}."
            )
            assert batch["mtp_tokens"].size(1) == expected_length, (
                f"SFT chunkpipe MTP tokens must have physical length "
                f"{expected_length}, got {batch['mtp_tokens'].size(1)}."
            )
            step_num_groups = batch.get("step_num_groups")
            mtp_batch = {
                "tokens": batch["mtp_tokens"],
                "position_ids": batch["mtp_position_ids"],
                "labels": batch.get("mtp_labels"),
                "loss_mask": batch.get("mtp_loss_mask"),
                "group_total_tokens": batch.get("group_total_tokens"),
                "step_num_groups": step_num_groups,
                # MTP loss logging scale dp * M / G, computed in fp32 like the MTP loss
                # it multiplies. The model applies it without reading parallel state.
                "log_scale": (
                    mpu.get_data_parallel_world_size()
                    * get_num_microbatches()
                    / (step_num_groups.to(torch.float32) if step_num_groups is not None else 1.0)
                ),
            }

    output = (
        batch["tokens"],
        batch["labels"],
        batch["loss_mask"],
        batch["position_ids"],
        batch["attention_mask"],
        batch["packed_seq_params"],
        mtp_batch,
    )

    return output


# define spiky loss as a loss that's 10x the max loss observed
SPIKY_LOSS_FACTOR = 10


def loss_func(loss_mask: torch.Tensor, output_tensor: torch.Tensor):
    """Loss function.

    Args:
        loss_mask (torch.Tensor): Used to mask out some portions of the loss
        output_tensor (torch.Tensor): The tensor with the losses
        num_input_tokens (int): The number of tokens in the batch

    Returns:
        the loss scalar for this micro-batch
        the number of non-padded tokens in this microbatch
        a dict containing reporting metrics on the loss and number of tokens across the data parallel ranks
    """
    args = get_args()

    losses = output_tensor.view(-1).float()
    loss_mask = loss_mask.view(-1).float()

    # Explicitly zero masked-out positions before summing. ``sum(losses * mask)``
    # turns a non-finite masked loss into NaN via ``inf * 0``; masked tokens must
    # contribute exactly 0. This keeps the loss (and its backward gradients)
    # finite on CP ranks whose shard contains only/mostly masked tokens.
    loss = torch.sum(torch.where(loss_mask > 0, losses, torch.zeros_like(losses)) * loss_mask)

    # Check individual rank losses are not NaN prior to DP all-reduce.
    rerun_state_machine = get_rerun_state_machine()
    if args.check_for_nan_in_loss_and_grad:
        rerun_state_machine.validate_result(
            result=loss,
            rejection_func=torch.isnan,
            message="found NaN in local forward loss calculation",
            tolerance=0.0,        # forward pass calculations are determinisic
            fatal=True,
        )
        rerun_state_machine.validate_result(
            result=loss,
            rejection_func=torch.isinf,
            message="found Inf in local forward loss calculation",
            tolerance=0.0,        # forward pass calculations are determinisic
            fatal=True,
        )

    # Check for spiky loss
    if args.check_for_spiky_loss:
        rerun_state_machine.validate_result(
            result=loss,
            rejection_func=partial(
                rerun_state_machine.is_unexpectedly_large,
                threshold=SPIKY_LOSS_FACTOR,
                context="loss",
            ),
            message="Spiky loss",
            tolerance=0.0,        # forward pass calculations are determinisic
            fatal=False,
        )

    num_tokens = loss_mask.sum().clone().detach().to(torch.int)

    # SFT chunkpipe per-sample loss path (default when sft_chunkpipe_mode=True
    # and --calculate-per-token-loss is NOT set). Goal: final gradient equals
    # (1/G_total) * sum_g (1/N_g) * sum_k d S_{g,k}/d theta, i.e. per-source-
    # sequence equal weighting independent of chunk count per sequence and
    # robust to per-rank G_local fluctuation.
    #
    # Derivation (legacy=True 2-tuple path, no CP for brevity; cp=1):
    #   loss_func returns scaled = D * S_{g,k} * M_raw / (N_g * G_total * cp)
    #   schedules legacy path multiplies by cp / M_raw
    #   → backward sees D * S_{g,k} / (N_g * G_total)
    #   DDP grad all-reduce averages by 1/D
    #   → grad sees S_{g,k} / (N_g * G_total) accumulated over all chunks on
    #     all ranks: (1/G_total) sum_g (1/N_g) sum_k S_{g,k}  ✓
    #
    # The D pre-compensation symmetrically applies to the reporting tensor so
    # that the post-aggregation report (sum-across-mbs → all-reduce-sum →
    # divide-by-DPxCP-world-size in engines/megatron/training_utils.py) recovers the same
    # (1/G_total) sum_g (1/N_g) sum_k S_{g,k} value. Using G_total instead of
    # G_local also fixes a latent precision bug in the equal-DP case where
    # LPT did not strictly guarantee G_local = G_total / D across ranks.
    config = get_model_config()
    sft_cp_mode = config is not None and getattr(config, "sft_chunkpipe_mode", False)
    if sft_cp_mode and not args.calculate_per_token_loss:
        N_g = args.chunkpipe_group_total_tokens
        G_total = args.chunkpipe_step_num_groups
        M_raw = get_num_microbatches()
        cp = mpu.get_context_parallel_world_size()
        D = mpu.get_data_parallel_world_size()

        denom = N_g * G_total * cp
        scaled = loss * (D * M_raw / denom)
        # Pre-compensate D*cp for the all_reduce + divide by (D*cp) in engines/megatron/training_utils.py.
        # Without cp factor, the logged loss would be 1/cp times smaller than correct value.
        reporting = (D * cp * loss / (N_g * G_total)).detach()
        loss_reduced_dict = {'lm loss': reporting}
        return scaled, loss_reduced_dict

    reporting_loss = torch.cat([loss.clone().detach().view(1), num_tokens.view(1)])

    # With context parallelism each CP rank only sees a shard of every sample,
    # and a contiguous-CP shard can contain zero valid tokens (all supervised
    # tokens live on the other rank). Reporting the local [S, n] pair would
    # then make train_step.py compute S/n = 0/0 = NaN and silently drop
    # 'lm loss' from the log line. Aggregate the reporting pair across the CP
    # group first (same as Megatron-LM's loss_func) so downstream reporting
    # always divides whole-sample sums. The backward `loss` and the returned
    # `num_tokens` stay local and are NOT touched.
    if args.context_parallel_size > 1:
        torch.distributed.all_reduce(
            reporting_loss, group=mpu.get_context_parallel_group()
        )

    loss_reduced_dict = {'lm loss': reporting_loss if not args.legacy_reporting_loss_reduction
                         else reporting_loss[0] / reporting_loss[1].clamp(min=1)}

    # calculate the number of tokens for this micro-batch
    if args.variable_seq_lengths:
        # for variable seq length, we need to calculate the number of tokens on fly
        # model output tensor shape is [B, S, H]
        num_input_tokens = output_tensor.shape[0] * output_tensor.shape[1]
        input_tokens = torch.tensor(num_input_tokens, dtype=torch.int, device=output_tensor.device)
        # sum across all dp ranks
        torch.distributed.all_reduce(input_tokens, group=mpu.get_data_parallel_group())
        loss_reduced_dict["total_inputs"] = input_tokens * args.context_parallel_size

    return loss, num_tokens, loss_reduced_dict


def forward_step(data_iterator, model, return_schedule_plan: bool = False):
    """Forward training step.

    Args:
        data_iterator : Input data iterator
        model: Megatron Model

    Returns:
        output_tensor: Output tensor
        loss_func: Loss function
        num_tokens: Number of tokens
    """
    args = get_args()
    timers = get_timers()

    # Get the batch.
    timers("batch-generator", log_level=2).start()

    with stimer(bdata=True):
        (
            tokens,
            labels,
            loss_mask,
            position_ids,
            attention_mask,
            packed_seq_params,
            mtp_batch,
        ) = get_batch(data_iterator)

    timers("batch-generator").stop()

    # SFT chunkpipe + MTP: MTP bridge tokens are appended during preprocessing.
    # The main model consumes the base chunksize slice; the MTP branch consumes
    # the full chunksize + mtp_num_layers tensors via extra_block_kwargs.
    extra_block_kwargs = None
    if mtp_batch is not None and mpu.is_pipeline_last_stage():
        extra_block_kwargs = {"mtp_batch": mtp_batch}

    with stimer:
        if return_schedule_plan:
            assert args.overlap_moe_expert_parallel_comm, \
                "overlap_moe_expert_parallel_comm must be enabled to return the schedule plan"
            schedule_plan = model.build_schedule_plan(
                input_ids=tokens,
                position_ids=position_ids,
                attention_mask=attention_mask, 
                labels=labels,
                packed_seq_params=packed_seq_params,
                loss_mask=loss_mask,
            )
            return schedule_plan, partial(loss_func, loss_mask)
        else:
            output_tensor = model(
                input_ids=tokens,
                position_ids=position_ids,
                attention_mask=attention_mask,
                labels=labels,
                packed_seq_params=packed_seq_params,
                loss_mask=loss_mask,
                extra_block_kwargs=extra_block_kwargs,
            )
            if hasattr(output_tensor, "loss") and output_tensor.loss is not None:
                output_tensor = output_tensor.loss

    return output_tensor, partial(loss_func, loss_mask)


def train_valid_test_datasets_provider(train_val_test_num_samples, vp_stage=None):
    """
    Build the train test and validation datasets.

    Args:
        train_val_test_num_samples: List[int]

    Returns:
        train_iter: Iterator
        valid_iter: Iterator
        test_iter: Iterator
    """
    args = get_args()

    config = SFTDatasetConfig(
        random_seed=args.seed,
        sequence_length=args.seq_length,  # max sequence length
        enable_discard_sample=args.enable_discard_sample,
        blend=get_blend_from_list(args.data_path),
        blend_per_split=[
            get_blend_from_list(args.train_data_path),
            get_blend_from_list(args.valid_data_path),
            get_blend_from_list(args.test_data_path),
        ],
        split=args.split,
        path_to_cache=args.data_cache_path,
        tokenizer=get_tokenizer(),
        dataset=get_dataset_blend_from_list(args.sft_dataset),
        dataset_per_split=[
            get_dataset_blend_from_list(args.sft_train_dataset),
            get_dataset_blend_from_list(args.sft_valid_dataset),
            get_dataset_blend_from_list(args.sft_test_dataset),
        ],
        dataset_config_file=args.sft_dataset_config,
        streaming=args.sft_data_streaming,
        streaming_buffer_size=args.streaming_buffer_size,
        mix_strategy=args.sft_data_mix_strategy,
        chat_template=get_chat_template(),
        num_preprocess_workers=args.sft_num_preprocess_workers,
        train_on_prompt=args.train_on_prompt,
        ignore_index=IGNORE_INDEX,
        eod_mask_loss=args.eod_mask_loss,
        history_mask_loss=args.history_mask_loss,
        is_tokenized=args.is_tokenized_data,
        packing=args.packing_sft_data,
        sort_batch=args.sft_sort_batch,
        packing_buffer_size=args.packing_buffer_size,
        context_parallel_size=args.context_parallel_size,
        enable_chunkpipe=args.enable_chunkpipe,
        chunksize=args.chunksize,
        mtp_num_layers=args.mtp_num_layers or 0,
    )

    print_rank_0(
        f"> building sft train, validation, and test datasets for {args.model_name} ..."
    )

    train_ds, valid_ds, test_ds = BlendedHuggingFaceDatasetBuilder(
        cls=SFTDataset,
        sizes=train_val_test_num_samples,  # NOTE: not use now!
        is_built_on_rank=lambda: mpu.get_tensor_model_parallel_rank() == 0,
        config=config,
        vp_rank=mpu.get_virtual_pipeline_model_parallel_rank(),
        distributed=mpu.get_world_size() > 1,
        is_global_rank_0=mpu.get_rank() == 0,
        global_rank=mpu.get_rank(),
    ).build()

    # will use external dataloader type for sft
    loader_config = SFTLoaderConfig.from_args(args)
    data_collator = build_sft_data_collator(
        DataCollatorForSupervisedDataset,
        loader_config,
        get_tokenizer(),
        pad_to_multiple_of=llm_pad_to_multiple_of(args),
    )
    train_iter, valid_iter, test_iter = build_sft_cyclic_iterators(
        train_ds,
        valid_ds,
        test_ds,
        data_collator,
        loader_config,
        mpu.get_data_parallel_rank(),
        mpu.get_data_parallel_world_size(),
        args.consumed_train_samples,
        args.consumed_valid_samples,
        args.skip_train,
        state=load_dataloader_state(args),
    )
    print_rank_0(f"> finished creating {args.model_name} sft datasets ...")

    return train_iter, valid_iter, test_iter

def train(args):
    """Run LLM SFT."""
    train_valid_test_datasets_provider.is_distributed = True
    pretrain(
        args,
        train_valid_test_datasets_provider,
        model_provider,
        ModelType.encoder_or_decoder,
        forward_step,
        get_embedding_ranks=get_embedding_ranks,
    )
