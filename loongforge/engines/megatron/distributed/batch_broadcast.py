# Copyright 2026 The LoongForge Authors.
# SPDX-License-Identifier: Apache-2.0

import torch

from megatron.core import mpu

# Longest shape broadcast_tensor can describe.
_MAX_BROADCAST_DIMS = 8

# Dtypes broadcast_tensor can carry, identified by index in the header tensor.
_DTYPE_CODES = [
    torch.float64,
    torch.float32,
    torch.float16,
    torch.bfloat16,
    torch.int64,
    torch.int32,
    torch.int16,
    torch.int8,
    torch.uint8,
    torch.bool,
]


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


def broadcast_from_tp_src(tensor: torch.Tensor) -> torch.Tensor:
    """Broadcast ``tensor`` in place from the tensor-parallel source rank."""
    torch.distributed.broadcast(
        tensor,
        mpu.get_tensor_model_parallel_src_rank(),
        group=mpu.get_tensor_model_parallel_group(),
    )
    return tensor


def broadcast_tensor(tensor: torch.Tensor, src_rank: int, group) -> torch.Tensor:
    """Broadcast a CUDA tensor of any shape and dtype.

    Non-source ranks pass ``tensor=None`` and receive a new tensor on the source device.
    """
    device = torch.cuda.current_device()
    # Fixed-size header: every rank must pass the same element count to the collective.
    header = torch.zeros(3 + _MAX_BROADCAST_DIMS, dtype=torch.long, device=device)
    if tensor is not None:
        if tensor.dim() > _MAX_BROADCAST_DIMS:
            raise ValueError(
                f"broadcast_tensor supports at most {_MAX_BROADCAST_DIMS} dimensions, got {tensor.dim()}"
            )
        header[0] = _DTYPE_CODES.index(tensor.dtype) + 1
        header[1] = tensor.dim()
        header[2] = tensor.numel()
        if tensor.dim() > 0:
            header[3:3 + tensor.dim()] = torch.tensor(tensor.shape, dtype=torch.long, device=device)
    torch.distributed.broadcast(header, src_rank, group=group)
    code, ndim = int(header[0]), int(header[1])
    if code == 0:
        return None
    shape = tuple(int(v) for v in header[3:3 + ndim])
    dtype = _DTYPE_CODES[code - 1]
    if tensor is None:
        tensor = torch.empty(shape, dtype=dtype, device=device)
    torch.distributed.broadcast(tensor, src_rank, group=group)
    return tensor


def broadcast_optional_tensor(tensor, src_rank: int, group):
    """Broadcast ``tensor``, propagating ``None``: every rank gets ``None`` when the source has none."""
    device = torch.cuda.current_device()
    present = torch.tensor([0 if tensor is None else 1], dtype=torch.long, device=device)
    torch.distributed.broadcast(present, src_rank, group=group)
    if not int(present.item()):
        return None
    return broadcast_tensor(tensor, src_rank, group)
