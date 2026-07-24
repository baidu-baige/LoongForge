# Copyright 2026 The LoongForge Authors.
# SPDX-License-Identifier: Apache-2.0
#
# Modified from Cosmos (NVIDIA cosmos-framework) under the OpenMDW-1.1 License.
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: OpenMDW-1.1

"""Attention dispatch and sequence packing utilities for joint attention."""

import torch
from torch import Tensor
from torch.nn.attention.flex_attention import BlockMask, create_block_mask, flex_attention as _flex_attention
from flash_attn.flash_attn_interface import flash_attn_func, flash_attn_varlen_func
from enum import Enum

flex_attention = torch.compile(_flex_attention)


class CausalType(str, Enum):
    """Attention causal mask type."""

    DontCare = "DontCare"
    TopLeft = "TopLeft"
    BottomRight = "BottomRight"


def merge_attentions(
    outputs: list[Tensor] | None = None,
    lse_tensors: list[Tensor] | None = None,
    torch_compile: bool = False,
    attn_out_1: Tensor | None = None,
    lse_1: Tensor | None = None,
    attn_out_2: Tensor | None = None,
    lse_2: Tensor | None = None,
) -> tuple[Tensor, Tensor]:
    """Merge attention outputs using log-sum-exp for numerical stability."""
    if outputs is not None:
        assert lse_tensors is not None and len(outputs) == len(lse_tensors)
        result = outputs[0]
        result_lse = lse_tensors[0]
        for i in range(1, len(outputs)):
            max_lse = torch.maximum(result_lse, lse_tensors[i])
            exp_1 = torch.exp(result_lse - max_lse)
            exp_2 = torch.exp(lse_tensors[i] - max_lse)
            denom = exp_1 + exp_2
            result = (result * exp_1.unsqueeze(-1) + outputs[i] * exp_2.unsqueeze(-1)) / denom.unsqueeze(-1)
            result_lse = max_lse + torch.log(denom)
        return result, result_lse
    # Legacy positional interface
    max_lse = torch.maximum(lse_1, lse_2)
    exp_1 = torch.exp(lse_1 - max_lse)
    exp_2 = torch.exp(lse_2 - max_lse)
    denom = exp_1 + exp_2
    out = (attn_out_1 * exp_1.unsqueeze(-1) + attn_out_2 * exp_2.unsqueeze(-1)) / denom.unsqueeze(-1)
    new_lse = max_lse + torch.log(denom)
    return out, new_lse


class SplitInfo:
    """Metadata for split-based attention routing."""

    def __init__(
        self,
        split_lens: list[int],
        attn_modes: list[str],
        sample_lens: list[int],
        actual_len: int,
        is_three_way: bool = False,
        vision_token_shapes: list[tuple[int, int, int]] | None = None,
        action_token_shapes: list[tuple[int, ...]] | None = None,
        num_action_tokens_per_supertoken: int = 0,
        null_action_supertokens: bool = False,
    ):
        """
        Actual len is the actual non-padded length of the packed sequence.
        It's used to trim split_lens, attn_modes and sample_lens, which were
        originally padded to max sequence length (likely for flex attention).
        """
        assert sum(sample_lens) == sum(split_lens), (
            f"Sum of new sample lens {sum(sample_lens)} is not equal to sum of new split lens {sum(split_lens)}"
        )

        max_causal_len = 0
        max_full_len = 0
        for split_len, attn_mode in zip(split_lens, attn_modes):
            if attn_mode == "causal":
                max_causal_len = max(max_causal_len, split_len)
            elif attn_mode == "full":
                max_full_len = max(max_full_len, split_len)

        self.max_causal_len = max_causal_len
        self.max_full_len = max_full_len
        self.max_sample_len = max(sample_lens)

        self.split_lens = split_lens
        self.attn_modes = attn_modes
        self.sample_lens = sample_lens

        self.is_three_way = is_three_way
        self.vision_token_shapes = vision_token_shapes
        self.action_token_shapes = action_token_shapes
        self.num_action_tokens_per_supertoken = num_action_tokens_per_supertoken
        self.null_action_supertokens = null_action_supertokens


AttentionMaskType = BlockMask | SplitInfo


_dotproduct_attention_cache = {}


from .sequence_packing import (
    FactoredSequencePack,
    JointSequencePack,
    create_sparse_mask,
    factored_from_joint_sequence,
    from_joint,
    from_mode_splits,
    generate_natten_metadata,
    generate_temporal_causal_natten_metadata,
    get_all_seq,
    get_causal_seq,
    get_full_only_seq,
    joint_from_joint_sequence,
)


def two_way_attention(
    packed_query_states: FactoredSequencePack | JointSequencePack,
    packed_key_states: FactoredSequencePack | JointSequencePack,
    packed_value_states: FactoredSequencePack | JointSequencePack,
):
    """
    Performs two-way attention with causal and full attention.
    """

    causal_q, causal_q_offsets = get_causal_seq(packed_query_states)
    causal_k, causal_k_offsets = get_causal_seq(packed_key_states)
    causal_v, _ = get_causal_seq(packed_value_states)
    full_q, full_q_offsets = get_full_only_seq(packed_query_states)

    sample_offsets = packed_query_states["sample_offsets"]

    use_dont_care_mask = causal_q_offsets is causal_k_offsets


    causal_res = attention(
        causal_q.unsqueeze(0),  # [1,N_und,heads,head_dim]
        causal_k.unsqueeze(0),  # [1,N_und,heads,head_dim]
        causal_v.unsqueeze(0),  # [1,N_und,heads,head_dim]
        cumulative_seqlen_Q=causal_q_offsets,
        cumulative_seqlen_KV=causal_k_offsets,
        max_seqlen_Q=packed_query_states["max_causal_len"],
        max_seqlen_KV=packed_query_states["max_causal_len"],
        is_causal=True,
        causal_type=CausalType.DontCare if use_dont_care_mask else CausalType.TopLeft,
        deterministic=torch.are_deterministic_algorithms_enabled()
    )  # [1,N_und,heads,head_dim]

    # [1,N_und,heads,head_dim] -> [N_und,heads,head_dim] -> [N_und,heads*head_dim]
    causal_out = causal_res.squeeze(0).flatten(-2, -1)  # type: ignore  # [N_und,heads*head_dim]

    full_res = attention(
        full_q.unsqueeze(0),  # [1,N_full,heads,head_dim]
        get_all_seq(packed_key_states).unsqueeze(0),  # [1,N_all,heads,head_dim]
        get_all_seq(packed_value_states).unsqueeze(0),  # [1,N_all,heads,head_dim]
        cumulative_seqlen_Q=full_q_offsets,
        cumulative_seqlen_KV=sample_offsets,
        max_seqlen_Q=packed_query_states["max_full_len"],
        max_seqlen_KV=packed_query_states["max_sample_len"],
        deterministic=torch.are_deterministic_algorithms_enabled()
    )  # [1,N_full,heads,head_dim]

    # [1,N_full,heads,head_dim] -> [N_full,heads,head_dim] -> [N_full,heads*head_dim]
    full_out = full_res.squeeze(0).flatten(-2, -1)  # type: ignore  # [N_full,heads*head_dim]

    out_all = from_mode_splits(causal_out, full_out, packed_query_states)
    return out_all


def three_way_attention(
    packed_query_states: FactoredSequencePack | JointSequencePack,
    packed_key_states: FactoredSequencePack | JointSequencePack,
    packed_value_states: FactoredSequencePack | JointSequencePack,
    natten_metadata: dict | None,
    attention_meta: SplitInfo | None = None,
):
    """
    Performs three-way attention, with understanding and generations attentions fully decomposed,
    and allows sparsity / multi-dimensional masking in the generation tower.

    When attention_meta is provided with null_action_supertokens=True, zeros V for the first
    num_action_tokens_per_supertoken tokens of each sample's GEN sequence (null action
    supertokens for temporal causal training). The metadata encodes is_causal=(True, False):
    causal across T supertokens, full within each supertoken S.

    NOTE: the three-way decomposition is only done so we can handle sparsity in the gen tower,
    but a KEY assumption is that the "full" tokens all correspond to the same modality!
    We should be careful when extending this to beyond t2i and t2v.
    """

    causal_q, causal_q_offsets = get_causal_seq(packed_query_states)
    causal_k, causal_k_offsets = get_causal_seq(packed_key_states)
    causal_v, _ = get_causal_seq(packed_value_states)
    full_q, full_q_offsets = get_full_only_seq(packed_query_states)
    full_k, full_k_offsets = get_full_only_seq(packed_key_states)
    full_v, _ = get_full_only_seq(packed_value_states)

    sample_offsets = packed_query_states["sample_offsets"]

    if attention_meta is not None and attention_meta.null_action_supertokens:
        # Zero V for the first num_action_tokens_per_supertoken tokens of each
        # sample's GEN sequence (null action supertokens at t=0).
        # out_i = Σ_j softmax(QKᵀ/√d)_j · V_j — terms with V_j=0 contribute exactly 0 to the output,
        # regardless of attention weights. Softmax mass is still allocated to these positions (not
        # redistributed), so this differs from hard key masking, but the output contribution is 0.
        full_v = full_v.clone()
        starts = full_q_offsets[:-1].long()  # [B]
        null_positions = (
            starts.unsqueeze(1) + torch.arange(attention_meta.num_action_tokens_per_supertoken, device=starts.device)
        ).reshape(-1)
        full_v[null_positions] = 0

    use_dont_care_mask = causal_q_offsets is causal_k_offsets


    causal_res = attention(
        causal_q.unsqueeze(0),  # [1,N_und,heads,head_dim]
        causal_k.unsqueeze(0),  # [1,N_und,heads,head_dim]
        causal_v.unsqueeze(0),  # [1,N_und,heads,head_dim]
        cumulative_seqlen_Q=causal_q_offsets,
        cumulative_seqlen_KV=causal_k_offsets,
        max_seqlen_Q=packed_query_states["max_causal_len"],
        max_seqlen_KV=packed_query_states["max_causal_len"],
        is_causal=True,
        causal_type=CausalType.DontCare if use_dont_care_mask else CausalType.TopLeft,
        deterministic=torch.are_deterministic_algorithms_enabled()
    )  # [1,N_und,heads,head_dim]
    # [1,N_und,heads,head_dim] -> [N_und,heads,head_dim] -> [N_und,heads*head_dim]
    causal_out = causal_res.squeeze(0).flatten(-2, -1)  # type: ignore  # [N_und,heads*head_dim]

    # If there's no metadata, it's a dense layer
    if natten_metadata is None:
        full_sa, full_sa_lse = attention(
            full_q.unsqueeze(0),  # [1,N_full,heads,head_dim]
            full_k.unsqueeze(0),  # [1,N_full,heads,head_dim]
            full_v.unsqueeze(0),  # [1,N_full,heads,head_dim]
            cumulative_seqlen_Q=full_q_offsets,
            cumulative_seqlen_KV=full_k_offsets,
            max_seqlen_Q=packed_query_states["max_full_len"],
            max_seqlen_KV=packed_query_states["max_full_len"],
            return_lse=True,
            deterministic=torch.are_deterministic_algorithms_enabled()
        )  # full_sa: [1,N_full,heads,head_dim], full_sa_lse: [1,N_full,heads]
    else:
        raise NotImplementedError("NATTEN multi_dimensional_attention_varlen is not supported")

    full_ca, full_ca_lse = attention(
        full_q.unsqueeze(0),  # [1,N_full,heads,head_dim]
        causal_k.unsqueeze(0),  # [1,N_und,heads,head_dim]
        causal_v.unsqueeze(0),  # [1,N_und,heads,head_dim]
        cumulative_seqlen_Q=full_q_offsets,
        cumulative_seqlen_KV=causal_k_offsets,
        max_seqlen_Q=packed_query_states["max_full_len"],
        max_seqlen_KV=packed_query_states["max_causal_len"],
        return_lse=True,
        deterministic=torch.are_deterministic_algorithms_enabled()
    )  # full_ca: [1,N_full,heads,head_dim], full_ca_lse: [1,N_full,heads]

    assert full_sa.shape == full_ca.shape
    full_res, _ = merge_attentions(
        outputs=[full_sa, full_ca], lse_tensors=[full_sa_lse, full_ca_lse], torch_compile=False
    )  # [1,N_full,heads,head_dim]

    # [1,N_full,heads,head_dim] -> [N_full,heads,head_dim] -> [N_full,heads*head_dim]
    full_out = full_res.squeeze(0).flatten(-2, -1)  # type: ignore  # [N_full,heads*head_dim]

    out_all = from_mode_splits(causal_out, full_out, packed_query_states)
    return out_all


def pad_sequence(tensor, pad_size):
    """
    Pad a tensor along the second-to-last dimension.

    Args:
        tensor: Input tensor to pad
        pad_size: Number of padding elements to add

    Returns:
        Padded tensor with zeros added along dim=-2
    """
    if pad_size <= 0:
        return tensor
    pad_shape = list(tensor.shape)
    pad_shape[-2] = pad_size
    padding = torch.zeros(pad_shape, dtype=tensor.dtype, device=tensor.device)
    return torch.cat([tensor, padding], dim=-2)  # [...,S+pad_size,...]


def block_flex_attention(
    packed_query_states: FactoredSequencePack | JointSequencePack,
    packed_key_states: FactoredSequencePack | JointSequencePack,
    packed_value_states: FactoredSequencePack | JointSequencePack,
    attention_mask: BlockMask,
    block_size: int | None = None,
):
    """Perform attention using flex_attention with a block mask."""
    packed_queries = get_all_seq(packed_query_states)  # [N,heads,head_dim]
    packed_keys = get_all_seq(packed_key_states)  # [N,heads,head_dim]
    packed_values = get_all_seq(packed_value_states)  # [N,heads,head_dim]
    max_num_tokens = packed_query_states["max_num_tokens"]

    num_attention_heads = packed_queries.shape[1]
    head_dim = packed_queries.shape[2]

    # Handle block mask attention with flex_attention
    pad_size = max_num_tokens - packed_queries.shape[0]
    packed_queries_padded = pad_sequence(packed_queries.permute(1, 0, 2), pad_size)  # [heads,max_num_tokens,head_dim]
    packed_keys_padded = pad_sequence(packed_keys.permute(1, 0, 2), pad_size)  # [heads,max_num_tokens,head_dim]
    packed_values_padded = pad_sequence(packed_values.permute(1, 0, 2), pad_size)  # [heads,max_num_tokens,head_dim]

    packed_attn_output = flex_attention(
        packed_queries_padded.unsqueeze(0),  # [1,heads,max_num_tokens,head_dim]
        packed_keys_padded.unsqueeze(0),  # [1,heads,max_num_tokens,head_dim]
        packed_values_padded.unsqueeze(0),  # [1,heads,max_num_tokens,head_dim]
        enable_gqa=True,
        block_mask=attention_mask,
    )  # [1,heads,max_num_tokens,head_dim]
    assert isinstance(packed_attn_output, torch.Tensor)

    end_index = packed_attn_output.shape[2] - pad_size
    packed_attn_output = packed_attn_output[0, :, :end_index, :]  # [heads,N,head_dim]
    packed_attn_output = packed_attn_output.transpose(0, 1).reshape(
        -1, num_attention_heads * head_dim
    )  # [N,heads*head_dim]

    return from_joint(packed_attn_output, packed_query_states)


def dispatch_attention(
    packed_query_states: FactoredSequencePack | JointSequencePack,
    packed_key_states: FactoredSequencePack | JointSequencePack,
    packed_value_states: FactoredSequencePack | JointSequencePack,
    attention_mask: BlockMask | SplitInfo,
    natten_metadata: dict | None = None,
) -> tuple[FactoredSequencePack | JointSequencePack | None]:
    """Dispatch to two-way, three-way, or flex attention based on mask type."""
    if isinstance(attention_mask, SplitInfo) and attention_mask.is_three_way:
        return three_way_attention(
            packed_query_states,
            packed_key_states,
            packed_value_states,
            natten_metadata=natten_metadata,
            attention_meta=attention_mask,
        )
    elif isinstance(attention_mask, SplitInfo):
        return two_way_attention(
            packed_query_states,
            packed_key_states,
            packed_value_states
        )
    else:
        return block_flex_attention(
            packed_query_states,
            packed_key_states,
            packed_value_states,
            attention_mask
        )


def build_packed_sequence(
    joint_attn_implementation: str,
    *,
    packed_sequence: torch.Tensor,
    attn_modes: list[str],
    split_lens: list[int],
    sample_lens: list[int],
    packed_und_token_indexes: torch.LongTensor,
    packed_gen_token_indexes: torch.LongTensor,
    num_heads: int,
    head_dim: int,
    num_layers: int,
    token_shapes: list[tuple[int, int, int]] | None = None,
    natten_parameter_list: list | None = None,
    block_size: int = 128,
    is_image_batch: bool = False,
    cp_world_size: int = 1,
    video_temporal_causal: bool = False,
    use_rolling_kv_cache: bool = False,
    vision_token_shapes: list[tuple[int, int, int]] | None = None,
    action_token_shapes: list[tuple[int, ...]] | None = None,
    num_action_tokens_per_supertoken: int = 0,
    null_action_supertokens: bool = False,
    pad_for_cuda_graphs: bool = False,
) -> tuple[FactoredSequencePack | JointSequencePack, AttentionMaskType, list | None]:
    """
    Build the model input pack and attention meta for joint attention.
    Returns a tuple: (input_pack, attention_meta).
    """
    device = packed_sequence.device
    natten_metadata_list = None
    if joint_attn_implementation == "flex":
        sparse_mask = create_sparse_mask(sample_lens, split_lens, attn_modes, device)
        seqlen = sum(sample_lens)
        attention_meta = create_block_mask(
            sparse_mask,
            B=1,
            H=num_heads,
            Q_LEN=seqlen,
            KV_LEN=seqlen,
            device=device,
            BLOCK_SIZE=block_size,
            _compile=True,
        )
        make_pack = joint_from_joint_sequence
    elif joint_attn_implementation == "two_way":
        attention_meta = SplitInfo(
            split_lens=split_lens,
            attn_modes=attn_modes,
            sample_lens=sample_lens,
            actual_len=int(packed_sequence.shape[0]),
        )
        make_pack = factored_from_joint_sequence
    elif joint_attn_implementation == "three_way":
        attention_meta = SplitInfo(
            split_lens=split_lens,
            attn_modes=attn_modes,
            sample_lens=sample_lens,
            actual_len=int(packed_sequence.shape[0]),
            is_three_way=True,
            vision_token_shapes=vision_token_shapes,
            action_token_shapes=action_token_shapes,
            num_action_tokens_per_supertoken=num_action_tokens_per_supertoken,
            null_action_supertokens=null_action_supertokens,
        )
        make_pack = factored_from_joint_sequence
        # The rolling KV-cache path implements temporal causality in
        # three_way_attention_with_kv_cache; skip NATTEN metadata.
        if not use_rolling_kv_cache:
            # Temporal causal: encode (T, S) supertoken layout; spatial NATTEN: encode (H, W) layout.
            if video_temporal_causal:
                natten_metadata_list = generate_temporal_causal_natten_metadata(
                    vision_token_shapes=vision_token_shapes,
                    num_action_tokens_per_supertoken=num_action_tokens_per_supertoken,
                    num_layers=num_layers,
                    head_dim=head_dim,
                    device=device,
                    dtype=packed_sequence.dtype,
                    requires_grad=packed_sequence.requires_grad,
                )
            else:
                natten_metadata_list = generate_natten_metadata(
                    token_shapes=token_shapes,
                    head_dim=head_dim,
                    num_layers=num_layers,
                    device=device,
                    dtype=packed_sequence.dtype,
                    requires_grad=packed_sequence.requires_grad,
                    natten_parameter_list=natten_parameter_list,
                )
    else:
        raise ValueError(
            f"Invalid joint_attn_implementation: {joint_attn_implementation}. "
            "Must be 'two_way', 'three_way', or 'flex'."
        )

    input_pack = make_pack(
        packed_sequence=packed_sequence,
        attn_modes=attn_modes,
        split_lens=split_lens,
        sample_lens=sample_lens,
        packed_und_token_indexes=packed_und_token_indexes.to(device),
        packed_gen_token_indexes=packed_gen_token_indexes.to(device),
        is_image_batch=is_image_batch,
        cp_world_size=cp_world_size,
        pad_for_cuda_graphs=pad_for_cuda_graphs,
    )
    # Not needed anymore, can cause recompilations.
    input_pack.pop("split_lens", None)
    input_pack.pop("attn_modes", None)
    return input_pack, attention_meta, natten_metadata_list




def flash2_attention(
    query: Tensor,
    key: Tensor,
    value: Tensor,
    is_causal: bool = False,
    causal_type: CausalType | None = None,
    scale: float | None = None,
    cumulative_seqlen_Q: Tensor | None = None,
    cumulative_seqlen_KV: Tensor | None = None,
    max_seqlen_Q: int | None = None,
    max_seqlen_KV: int | None = None,
    return_lse: bool = False,
    backend_kwargs: dict | None = None,
    deterministic: bool = False,
) -> Tensor | tuple[Tensor, Tensor]:
    """
    Runs Flash Attention v2 on given operands (Q, K, V) with the heads-last contiguous layout
        (`[batch, seqlen, heads, head_dim]`).

    Parameters:
        query (Tensor): 4-D query tensor, with the heads-last contiguous layout
            (`[batch, seqlen, heads, head_dim]`)

        key (Tensor): 4-D key tensor, with the heads-last contiguous layout
            (`[batch, seqlen_kv, heads_kv, head_dim]`)

        value (Tensor): 4-D value tensor, with heads-last contiguous layout
            (`[batch, seqlen_kv, heads_kv, head_dim_v]`)

        is_causal (bool): whether or not causal masking is enabled. Default is False.

        causal_type (CausalType): causal masking mode. Choices: `CausalType.TopLeft`,
            `CausalType.BottomRight`. Required when `is_causal = True`.

        scale (float | None): Dot product scale (attention scale). Defaults to head_dim ** -0.5.

        cumulative_seqlen_Q (Tensor | None): (varlen) Optional 1-D tensor with size `batch + 1`
            indicating the cumulative sum of number of query tokens in each batch, with an
            additional 0 element in the beginning. Must be passed together with
            `cumulative_seqlen_KV` and `max_seqlen_{Q,KV}`.

        cumulative_seqlen_KV (Tensor | None): (varlen) Optional 1-D tensor with size `batch + 1`
            indicating the cumulative sum of number of key/value tokens in each batch, with an
            additional 0 element in the beginning. Must be passed together with
            `cumulative_seqlen_Q` and `max_seqlen_{Q,KV}`.

        max_seqlen_Q (int | None): (varlen) Optional integer indicating the maximum query
            sequence length in all batches. Must be passed together with `cumulative_seqlen_{Q,KV}`
            and `max_seqlen_KV`.

        max_seqlen_KV (int | None): (varlen) Optional integer indicating the maximum key/value
            sequence length in all batches. Must be passed together with `cumulative_seqlen_{Q,KV}`
            and `max_seqlen_Q`.

    Other Parameters:
        return_lse (bool): Whether to return the logsumexp values. Default is False.

        backend_kwargs (dict | None): Key-value pair for passing arguments specific to Flash's
            attention operator, if any.

        deterministic (bool): Deterministic backward pass required.

    Returns:
        output (Tensor): 4-D output tensor, with the heads-last contiguous layout
            (`[batch, seqlen, heads, head_dim_v]`).

        logsumexp (Tensor): logsumexp tensor, with the heads-last contiguous layout
            (`[batch, seqlen, heads, 1]`). Only returned when return_lse is True.
            NOTE: this tensor is not contiguous in this backend (Flash2) and it should not be made
            contiguous unless we can guarantee its results aren't merged via `merge_attentions`.
    """

    is_varlen = cumulative_seqlen_Q is not None

    backend_kwargs = backend_kwargs.copy() if backend_kwargs is not None else {}
    # Determinism in backend_kwargs supersedes primary flag, if set to True
    if "deterministic" in backend_kwargs:
        deterministic = deterministic or backend_kwargs["deterministic"]
        del backend_kwargs["deterministic"]

    # This check introduces recompiles
    if not is_torch_compiling():
        if is_varlen and max_seqlen_Q == max_seqlen_KV == 0:
            raise NotImplementedError(
                "You're trying to use varlen attention with the flash2 backend and "
                "an empty batch, which is not yet supported by flash2."
            )

    scale = scale if scale is not None else query.shape[-1] ** -0.5

    if is_varlen:
        assert query.shape[0] == key.shape[0] == value.shape[0] == 1
        q = query.squeeze(0)  # [total_tokens,H,D]
        k = key.squeeze(0)  # [total_tokens,Hkv,D]
        v = value.squeeze(0)  # [total_tokens,Hkv,Dv]
        assert q.dim() == k.dim() == v.dim() == 3
        out, lse_, _ = flash_attn_varlen_func(
            q=query.squeeze(0),
            k=key.squeeze(0),
            v=value.squeeze(0),
            cu_seqlens_q=cumulative_seqlen_Q,
            cu_seqlens_k=cumulative_seqlen_KV,
            max_seqlen_q=max_seqlen_Q,
            max_seqlen_k=max_seqlen_KV,
            softmax_scale=scale,
            causal=is_causal,
            return_attn_probs=True,
            deterministic=deterministic,
            **backend_kwargs,
            # window_size=(-1, -1),
            # dropout_p=0.0,
            # softcap=0.0, # 0.0 means deactivated
            # alibi_slopes=None,
            # block_table=None,
        )
        assert out.dim() == 3  # [total_tokens,H,Dv]
        assert lse_.dim() == 2  # [H,total_tokens]

        output = out.unsqueeze(0)  # [1,total_tokens,H,Dv]
        lse = lse_.unsqueeze(0)  # [1,H,total_tokens]

    else:
        output, lse, _ = flash_attn_func(  # output: [B,N,H,Dv], lse: [B,H,N]
            q=query,
            k=key,
            v=value,
            softmax_scale=scale,
            causal=is_causal,
            return_attn_probs=True,
            deterministic=deterministic,
            **backend_kwargs,
            # window_size=(-1, -1),
            # dropout_p=0.0,
            # softcap=0.0, # 0.0 means deactivated
            # alibi_slopes=None,
        )

    assert isinstance(output, Tensor)
    assert isinstance(lse, Tensor)
    assert output.dim() == 4  # [B,N,H,Dv] or [1,total_tokens,H,Dv]
    assert lse.dim() == 3  # [B,H,N] or [1,H,total_tokens]

    # NOTE: Do NOT call .contiguous on LSE, otherwise Attention Merging backward pass will be
    # incorrect. All output and lse tensors passed into `merge_attentions` must have the same data
    # pointer as their corresponding attention autograd ops!
    lse = lse.permute(0, 2, 1)  # [B,N,H] or [1,total_tokens,H]

    if return_lse:
        return output, lse

    return output


def is_torch_compiling() -> bool:
    """is_torch_compiling."""
    try:
        return torch.compiler.is_compiling()
    except Exception as e:
        return False


# Use the local attention function
attention = flash2_attention

