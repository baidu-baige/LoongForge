# Copyright 2026 The LoongForge Authors.
# SPDX-License-Identifier: Apache-2.0

import torch

from megatron.core import mpu, tensor_parallel
from megatron.core.packed_seq_params import PackedSeqParams

from loongforge import constants
from loongforge.engines.mcore import get_args, get_tokenizer


def _get_position_ids(data: torch.Tensor):
    """Create position ids."""
    current_device = data.device
    _, seq_length = data.shape
    position_ids = torch.arange(seq_length, dtype=torch.long, device=current_device)
    position_ids = position_ids.unsqueeze(0).expand_as(data)
    return position_ids


def _get_attention_mask(attention_mask: torch.Tensor, args=None) -> torch.Tensor:
    """Create attention mask for TE."""
    if args is None:
        args = get_args()
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


def gather_variable_shape_embeddings(
    local_embedding: torch.Tensor,
    dst_rank: int = 0,
    group: torch.distributed.ProcessGroup = None,
) -> list[torch.Tensor] | None:
    """
    Gather N-D tensors with different dim-0 sizes from all ranks to dst_rank.

    Args:
        local_embedding: shape = [batch_i, ...], batch_i may differ across ranks, other dims must match
        dst_rank:        global rank of the receiver
        group:           process group, None means default group

    Returns:
        dst_rank: list[Tensor], the i-th element has shape = [batch_i, ...]
        other ranks: None
    """
    world_size = torch.distributed.get_world_size(group)
    local_rank = torch.distributed.get_rank(group)
    device = local_embedding.device
    dtype = local_embedding.dtype
    other_dims = local_embedding.shape[1:]

    # Step 1: all_gather batch_size from all ranks
    local_batch = torch.tensor([local_embedding.shape[0]], dtype=torch.long, device=device)
    all_batches = [torch.zeros(1, dtype=torch.long, device=device) for _ in range(world_size)]
    torch.distributed.all_gather(all_batches, local_batch, group=group)
    batch_sizes = [b[0].item() for b in all_batches]
    max_batch = max(batch_sizes)

    # Step 2: pad local_embedding to max_batch
    pad_len = max_batch - local_embedding.shape[0]
    if pad_len > 0:
        pad = torch.zeros(pad_len, *other_dims, dtype=dtype, device=device)
        padded = torch.cat([local_embedding, pad], dim=0)
    else:
        padded = local_embedding

    # Step 3: gather to dst_rank
    gather_list = (
        [torch.zeros(max_batch, *other_dims, dtype=dtype, device=device)
         for _ in range(world_size)]
        if local_rank == dst_rank else None
    )

    dst_global_rank = torch.distributed.get_global_rank(group, dst_rank)
    torch.distributed.gather(padded, gather_list=gather_list, dst=dst_global_rank, group=group)

    # Step 4: unpad
    if local_rank == dst_rank:
        return [t[:batch_sizes[i]] for i, t in enumerate(gather_list)]
    return None

def scatter_variable_shape_embeddings(
    embeddings: list[torch.Tensor] | None,
    local_embedding_ref: torch.Tensor,
    src_rank: int = 0,
    group: torch.distributed.ProcessGroup = None,
) -> torch.Tensor:
    """
    Scatter a list of tensors from src_rank back to each rank.
    This is the inverse operation of gather_variable_shape_embeddings.

    Args:
        embeddings:          list[Tensor] on src_rank, the i-th element has shape = [batch_i, ...]
                             pass None on non-src ranks
        local_embedding_ref: local tensor used to retrieve shape/dtype/device info
        src_rank:            global rank of the sender
        group:               process group, None means default group

    Returns:
        The local Tensor for this rank, shape = [batch_i, ...]
    """
    world_size = torch.distributed.get_world_size(group)
    local_rank = torch.distributed.get_rank(group)
    device = local_embedding_ref.device
    dtype = local_embedding_ref.dtype
    other_dims = local_embedding_ref.shape[1:]

    # Step 1: all_gather batch_size from all ranks (symmetric with the gather side)
    local_batch = torch.tensor([local_embedding_ref.shape[0]], dtype=torch.long, device=device)
    all_batches = [torch.zeros(1, dtype=torch.long, device=device) for _ in range(world_size)]
    torch.distributed.all_gather(all_batches, local_batch, group=group)
    batch_sizes = [b[0].item() for b in all_batches]
    max_batch = max(batch_sizes)

    # Step 2: pad each tensor to max_batch on src_rank
    if local_rank == src_rank:
        scatter_list = []
        for i, emb in enumerate(embeddings):
            pad_len = max_batch - emb.shape[0]
            if pad_len > 0:
                pad = torch.zeros(pad_len, *other_dims, dtype=dtype, device=device)
                scatter_list.append(torch.cat([emb, pad], dim=0))
            else:
                scatter_list.append(emb)
    else:
        scatter_list = None

    # Step 3: scatter to each rank
    recv = torch.zeros(max_batch, *other_dims, dtype=dtype, device=device)
    src_global_rank = torch.distributed.get_global_rank(group, src_rank)
    torch.distributed.scatter(recv, scatter_list=scatter_list, src=src_global_rank, group=group)

    # Step 4: unpad to restore the actual batch_size of this rank
    return recv[:batch_sizes[local_rank]]


def get_batch_on_this_tp_rank(data_iterator):
    """get batch on this tp rank"""
    args = get_args()
    tokenizer = get_tokenizer()

    if data_iterator is not None:
        data = next(data_iterator)
    else:
        data = None

    # broadcast required keys across tp
    required_keys = ["attention_mask"]
    if args.enable_chunkpipe:
        required_keys.append("chunk_group_size")
        required_keys.append("group_total_tokens")

    if args.pipeline_model_parallel_size == 1:
        required_keys += ["input_ids", "labels"] + (
            ["loss_mask"] if not args.eod_mask_loss else []
        )

    elif mpu.is_pipeline_first_stage():
        required_keys.append("input_ids")

    elif mpu.is_pipeline_last_stage():
        required_keys += ["input_ids", "labels"] + (
            ["loss_mask"] if not args.eod_mask_loss else []
        )

    data_b = tensor_parallel.broadcast_data(required_keys, data, torch.int64)

    sft_chunkpipe_mtp = (
        args.enable_chunkpipe
        and getattr(args, "sft_chunkpipe_mode", False)
        and getattr(args, "mtp_num_layers", 0)
        and args.mtp_num_layers > 0
    )
    base_length = args.chunksize if sft_chunkpipe_mtp else None

    # tokens & position ids
    tokens_full = data_b["input_ids"].long() if "input_ids" in data_b else None
    tokens = tokens_full
    mtp_tokens = None
    mtp_position_ids = None
    if tokens_full is not None:
        if sft_chunkpipe_mtp:
            expected_length = base_length + args.mtp_num_layers
            assert tokens_full.dim() == 2, (
                f"SFT chunkpipe MTP expects 2D tokens, got shape "
                f"{tuple(tokens_full.shape)}."
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

    # labels & loss mask
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
            labels[:, -1] = constants.IGNORE_INDEX
        # labels[labels == tokenizer.pad] == constants.IGNORE_INDEX
        # labels[labels == tokenizer.eos] == constants.IGNORE_INDEX

    # create loss mask
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
        loss_mask[labels == constants.IGNORE_INDEX] = 0.0
        loss_mask[labels == tokenizer.pad] = 0.0
        loss_mask[labels == tokenizer.eos] = 0.0
        if sft_chunkpipe_mtp and mtp_labels is not None:
            mtp_loss_mask = torch.ones(mtp_labels.size(), dtype=torch.float, device=mtp_labels.device)
            mtp_loss_mask[mtp_labels == constants.IGNORE_INDEX] = 0.0
            mtp_loss_mask[mtp_labels == tokenizer.pad] = 0.0
            mtp_loss_mask[mtp_labels == tokenizer.eos] = 0.0

    # attention mask
    attention_mask = None
    packed_seq_params = None
    attention_mask_data = data_b["attention_mask"].long()
    if sft_chunkpipe_mtp:
        attention_mask_data = attention_mask_data[:, :base_length]

    if not args.packing_sft_data:
        attention_mask = _get_attention_mask(attention_mask_data)
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

        # Per-step G (source sequence count). Unlike per-chunk fields, G is not
        # carried through the dataset/collator path; it's produced by the
        # sampler yield-time and delivered via a FIFO deque on args. TP rank 0
        # pops the queue, other TP ranks receive the value via broadcast.
        #
        # VPP mode: get_batch is called multiple times per step (once per VP stage).
        # To avoid popping the queue multiple times, only pop on the first VP stage.
        vp_size = mpu.get_virtual_pipeline_model_parallel_world_size()
        vp_stage = mpu.get_virtual_pipeline_model_parallel_rank()
        tp_rank = mpu.get_tensor_model_parallel_rank()
        is_vpp_enabled = vp_size is not None and vp_size > 1

        # Determine if this rank should pop the queue:
        # - Non-VPP: only TP rank 0 pops
        # - VPP: only TP rank 0 AND VP last stage pops，because args.chunkpipe_step_g_queue is last vp_stage,
        #           and only last vp_stage will have loss_func calculations
        should_pop = (tp_rank == 0) and ((not is_vpp_enabled) or (vp_stage == (vp_size - 1)))

        if should_pop:
            # Per-microbatch G_total (cross-rank source-group sum) and composite
            # descriptor. Both are produced by the sampler yield-time and
            # delivered via FIFO deques on args. TP rank 0 pops the queues; other
            # TP ranks receive values via broadcast. Composite descriptor is
            # variable-length, so we broadcast (G_total, num_components) first and
            # then the components themselves.
            step_num_groups = args.chunkpipe_step_g_queue.popleft()
            component_sizes = list(args.chunkpipe_composite_queue.popleft())
        else:
            step_num_groups = 0
            component_sizes = []
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
                    component_sizes,
                    dtype=torch.long,
                    device=torch.cuda.current_device(),
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
        batch["step_num_groups"] = meta[:1]
        batch["composite_component_sizes"] = component_sizes

    return batch

