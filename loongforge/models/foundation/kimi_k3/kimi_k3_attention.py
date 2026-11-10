# Copyright 2026 The LoongForge Authors.
# SPDX-License-Identifier: Apache-2.0
# Adapted from Megatron-Bridge under the Apache-2.0 License:
# https://github.com/NVIDIA-NeMo/Megatron-Bridge/blob/main/src/megatron/bridge/models/kimi/kimi_k3_layers.py

"""Kimi K3 KDA and no-RoPE MLA attention implementations."""

import copy

import torch
import torch.nn.functional as F
from einops import rearrange
from fla.modules import FusedRMSNormGated, ShortConvolution
from megatron.core.dist_checkpointing.mapping import ShardedStateDict
from megatron.core.extensions.transformer_engine import (
    TEColumnParallelLinear,
    TEDotProductAttention,
    TELinear,
    TERowParallelLinear,
)
from megatron.core.inference.contexts import BaseInferenceContext
from megatron.core.packed_seq_params import PackedSeqParams
from megatron.core.process_groups_config import ProcessGroupCollection
from megatron.core.tensor_parallel.layers import set_tensor_model_parallel_attributes
from megatron.core.tensor_parallel.mappings import (
    copy_to_tensor_model_parallel_region,
    gather_from_sequence_parallel_region,
    scatter_to_sequence_parallel_region,
)
from megatron.core.transformer.enums import AttnMaskType
from megatron.core.transformer.module import MegatronModule
from megatron.core.transformer.utils import (
    ensure_metadata_has_dp_cp_group,
    make_sharded_tensors_for_checkpoint,
)
from torch import nn

from loongforge.models.common.chunkpipe_state_cache import ChunkPipeStateCacheMixin

from .kimi_k3_ops import RMSNorm, kda, sum_grads_across_tp


def _linear(module: nn.Module, inputs: torch.Tensor) -> torch.Tensor:
    output, bias = module(inputs)
    if bias is not None:
        raise ValueError(
            "Kimi K3 requires bias-free projections, "
            f"got bias from {type(module).__name__}"
        )
    return output


class KimiK3ShortConvolution(ShortConvolution):
    """KDA short convolution, TP-sharded along the channel dimension.

    The weight starts in FP32 for checkpoint compatibility and follows the model
    dtype when ``Float16Module`` wraps it.
    """

    def __init__(self, *args, tp_group=None, **kwargs) -> None:
        self.tp_group = tp_group
        super().__init__(*args, **kwargs)
        self.weight.data = self.weight.data.float()
        set_tensor_model_parallel_attributes(self.weight, True, 0, 1)

    def sharded_state_dict(
        self,
        prefix: str = "",
        sharded_offsets: tuple = (),
        metadata: dict | None = None,
    ) -> ShardedStateDict:
        """Return the TP-sharded convolution state."""
        metadata = ensure_metadata_has_dp_cp_group(metadata)
        return make_sharded_tensors_for_checkpoint(
            self.state_dict(prefix="", keep_vars=True),
            prefix,
            {"weight": 0},
            sharded_offsets,
            tp_group=self.tp_group,
            dp_cp_group=metadata["dp_cp_group"],
        )


class KimiK3Attention(ChunkPipeStateCacheMixin, MegatronModule):
    """Select KDA or Kimi's no-RoPE MLA by global layer number."""

    def __init__(
        self,
        config,
        layer_number: int,
        cp_comm_type: str | None = None,
        pg_collection=None,
        pp_layer_offset: int | None = None,
        name: str | None = None,
    ) -> None:
        super().__init__(config=config)
        del pp_layer_offset, name
        self.cp_comm_type = cp_comm_type

        if pg_collection is None:
            pg_collection = ProcessGroupCollection.use_mpu_process_groups(
                required_pgs=["tp", "cp"]
            )
        elif not hasattr(pg_collection, "tp") or not hasattr(pg_collection, "cp"):
            raise ValueError("KimiK3Attention requires TP and CP process groups")
        self.pg_collection = pg_collection
        self.tp_group = pg_collection.tp
        self.tp_size = self.tp_group.size()
        self.cp_group = pg_collection.cp
        self.cp_size = self.cp_group.size()
        self.sequence_parallel = config.sequence_parallel
        self.linear_config = copy.copy(config)
        self.linear_config.sequence_parallel = False

        self.layer_idx = layer_number - 1
        self.is_kda = layer_number in config.kimi_kda_layers
        if self.is_kda and self.cp_size > 1:
            raise ValueError("Kimi K3 KDA context parallelism is not supported yet")
        if config.enable_chunkpipe:
            self.setup_chunkpipe_cache_config()
        if self.is_kda:
            self._init_kda(config)
        else:
            self._init_mla(config)

    def _duplicated_linear(self, input_size: int, output_size: int) -> TELinear:
        return TELinear(
            input_size,
            output_size,
            config=self.linear_config,
            init_method=self.config.init_method,
            bias=False,
            skip_bias_add=False,
            skip_weight_param_allocation=False,
            parallel_mode="duplicated",
        )

    def _column_linear(self, input_size: int, output_size: int) -> TEColumnParallelLinear:
        return TEColumnParallelLinear(
            input_size,
            output_size,
            config=self.linear_config,
            init_method=self.config.init_method,
            bias=False,
            gather_output=False,
            skip_bias_add=False,
            is_expert=False,
            tp_group=self.tp_group,
        )

    def _row_linear(self, input_size: int, output_size: int) -> TERowParallelLinear:
        return TERowParallelLinear(
            input_size,
            output_size,
            config=self.linear_config,
            init_method=self.config.init_method,
            bias=False,
            input_is_parallel=True,
            skip_bias_add=False,
            is_expert=False,
            tp_group=self.tp_group,
        )

    def _init_kda(self, config) -> None:
        hidden_size = config.hidden_size
        device = torch.cuda.current_device()
        dtype = config.params_dtype
        self.num_heads = config.kimi_linear_num_heads
        if self.num_heads % self.tp_size:
            raise ValueError(
                f"KDA heads {self.num_heads} must be divisible by TP size {self.tp_size}"
            )
        self.local_num_heads = self.num_heads // self.tp_size
        self.head_dim = config.kimi_linear_head_dim
        self.projection_size = self.num_heads * self.head_dim
        self.local_projection_size = self.local_num_heads * self.head_dim

        self.q_proj = self._column_linear(hidden_size, self.projection_size)
        self.k_proj = self._column_linear(hidden_size, self.projection_size)
        self.v_proj = self._column_linear(hidden_size, self.projection_size)
        conv_kwargs = {
            "hidden_size": self.local_projection_size,
            "kernel_size": config.kimi_linear_conv_kernel_size,
            "activation": "silu",
            "device": device,
            "dtype": dtype,
            "tp_group": self.tp_group,
        }
        self.q_conv1d = KimiK3ShortConvolution(**conv_kwargs)
        self.k_conv1d = KimiK3ShortConvolution(**conv_kwargs)
        self.v_conv1d = KimiK3ShortConvolution(**conv_kwargs)

        self.f_a_proj = self._duplicated_linear(hidden_size, self.head_dim)
        self.f_b_proj = self._column_linear(self.head_dim, self.projection_size)
        self.b_proj = self._column_linear(hidden_size, self.num_heads)
        self.g_proj = self._column_linear(hidden_size, self.projection_size)

        # Stored in FP32 for checkpoint compatibility, then cast with the model.
        self.A_log = nn.Parameter(
            torch.empty(self.local_num_heads, dtype=torch.float32, device=device)
        )
        self.dt_bias = nn.Parameter(
            torch.empty(self.local_projection_size, dtype=torch.float32, device=device)
        )
        self.A_log._keep_in_float32 = True
        self.dt_bias._keep_in_float32 = True
        set_tensor_model_parallel_attributes(self.A_log, True, 0, 1)
        set_tensor_model_parallel_attributes(self.dt_bias, True, 0, 1)

        self.o_norm = FusedRMSNormGated(
            self.head_dim,
            eps=config.layernorm_epsilon,
            activation="sigmoid",
            device=device,
            dtype=dtype,
        )
        sum_grads_across_tp(self.o_norm)
        self.o_proj = self._row_linear(self.projection_size, hidden_size)
        self.gate_lower_bound = config.kimi_kda_gate_lower_bound
        # KDA has no TE dot-product attention; the chunkpipe scheduler's
        # cache helpers do `getattr(self_attention.core_attention, 'indexer', None)`,
        # so expose the attribute as None to keep that no-op path working.
        self.core_attention = None

        if config.enable_chunkpipe:
            # Recurrent boundary state is fp32 (chunk_kda asserts it): [b, HV, K, V].
            self.register_state_cache(
                "kda_state",
                (config.micro_batch_size, self.local_num_heads, self.head_dim, self.head_dim),
                torch.float32,
            )
            # Short-convolution boundary states: [b, local_projection_size, kernel_size].
            conv_state_shape = (
                config.micro_batch_size,
                self.local_projection_size,
                config.kimi_linear_conv_kernel_size,
            )
            for name in ("q_conv_state", "k_conv_state", "v_conv_state"):
                self.register_state_cache(name, conv_state_shape, dtype)

    def _init_mla(self, config) -> None:
        hidden_size = config.hidden_size
        device = torch.cuda.current_device()
        dtype = config.params_dtype
        self.num_heads = config.num_attention_heads
        if self.num_heads % self.tp_size:
            raise ValueError(
                f"MLA heads {self.num_heads} must be divisible by TP size {self.tp_size}"
            )
        self.local_num_heads = self.num_heads // self.tp_size
        self.q_lora_rank = config.q_lora_rank
        self.kv_lora_rank = config.kv_lora_rank
        self.qk_nope_head_dim = config.qk_head_dim
        self.qk_extra_head_dim = config.qk_pos_emb_head_dim
        self.v_head_dim = config.v_head_dim
        self.q_head_dim = self.qk_nope_head_dim + self.qk_extra_head_dim
        # FlashAttention-2 needs equal QK and V head dims; pad V up to the QK dim
        # (same rule as MCore MultiLatentAttention) and slice the output back.
        self.v_channels = self.v_head_dim
        self.padding_v_head_dim = False
        if (
            (config.padding_v_to_qk_dim or config.enable_chunkpipe)
            and self.q_head_dim > self.v_head_dim
        ):
            self.v_channels = self.q_head_dim
            self.padding_v_head_dim = True

        self.q_a_proj = self._duplicated_linear(hidden_size, self.q_lora_rank)
        self.q_a_layernorm = RMSNorm(self.q_lora_rank, config.layernorm_epsilon).to(
            device=device, dtype=dtype
        )
        self.q_b_proj = self._column_linear(
            self.q_lora_rank, self.num_heads * self.q_head_dim
        )
        self.kv_a_proj_with_mqa = self._duplicated_linear(
            hidden_size, self.kv_lora_rank + self.qk_extra_head_dim
        )
        self.kv_a_layernorm = RMSNorm(self.kv_lora_rank, config.layernorm_epsilon).to(
            device=device, dtype=dtype
        )
        self.kv_b_proj = self._column_linear(
            self.kv_lora_rank,
            self.num_heads * (self.qk_nope_head_dim + self.v_head_dim),
        )
        self.g_proj = self._column_linear(
            hidden_size, self.num_heads * self.v_head_dim
        )
        self.o_proj = self._row_linear(
            self.num_heads * self.v_head_dim, hidden_size
        )

        self.core_attention = TEDotProductAttention(
            config=self.config,
            layer_number=self.layer_idx + 1,
            attn_mask_type=AttnMaskType.causal,
            attention_type="self",
            softmax_scale=self.q_head_dim**-0.5,
            k_channels=self.q_head_dim,
            v_channels=self.v_channels,
            cp_comm_type=self.cp_comm_type,
            pg_collection=self.pg_collection,
        )

        if config.enable_chunkpipe:
            if config.sft_chunkpipe_mode:
                raise NotImplementedError(
                    "K3 MLA chunk KV cache currently supports pretrain only"
                )
            self.init_chunk_key_value_cache_mla()

    # ------------------------------------------------------------------
    # Chunk KV cache (MLA, pretrain): mirrors multi_latent_attention.py,
    # except K3's kv_a_proj_with_mqa is a duplicated linear, so the compressed
    # KV and key-extra caches keep the full sequence on every rank (no // tp_size).
    # ------------------------------------------------------------------
    def init_chunk_key_value_cache_mla(self) -> None:
        device = next(self.parameters()).device
        dtype = next(self.parameters()).dtype
        slot_size = self.config.max_chunk_size or self.config.chunksize
        total_cache_tokens = self.kv_cache_chunk_size * slot_size
        self.kv_compressed_cache = torch.zeros(
            (total_cache_tokens, self.config.micro_batch_size, self.kv_lora_rank),
            device=device,
            dtype=dtype,
        )
        self.key_extra_cache = torch.zeros(
            (total_cache_tokens, self.config.micro_batch_size, self.qk_extra_head_dim),
            device=device,
            dtype=dtype,
        )
        self.kv_compressed_cache_grad = {}
        self.key_extra_cache_grad = {}

    def append_chunk_key_value_cache_mla(self, normed_kv_compressed, key_extra) -> None:
        """Cache a chunk of compressed KV and key-extra for MLA."""
        if not self.config.enable_chunkpipe:
            raise RuntimeError("Chunk key-value cache operations require chunkpipe to be enabled.")
        # Backward recompute reuses forward-stored values; last chunk has no successor.
        if not self.config.chunkpipe_forward:
            return
        if (self.config.chunkpipe_forward_microbatch + 1) % self.num_chunks_per_seq == 0:
            return
        if not self.empty_chunk_indices:
            raise RuntimeError("No available cache chunks. Consider increasing cache size.")

        chunk_id = self.empty_chunk_indices.pop(0)
        self.micro_batch_to_cache_chunk_map[self.config.chunkpipe_forward_microbatch] = chunk_id

        slot_size = self.config.max_chunk_size or self.config.chunksize
        kv_len = normed_kv_compressed.shape[0]
        indices = torch.arange(kv_len, device=normed_kv_compressed.device) + (chunk_id * slot_size)
        if self.is_enable_grad_chunkpipe():
            self.kv_compressed_cache[indices] = normed_kv_compressed.clone().detach()
            self.key_extra_cache[indices] = key_extra.clone().detach()
        else:
            self.kv_compressed_cache[indices] = normed_kv_compressed
            self.key_extra_cache[indices] = key_extra

    def recover_key_value_up_proj_tensors(self, normed_kv_compressed, key_extra):
        """Recover full key/value from the compressed KV latent + key-extra."""
        key_value = _linear(self.kv_b_proj, normed_kv_compressed)
        key_value = key_value.view(
            *key_value.shape[:-1],
            self.local_num_heads,
            self.qk_nope_head_dim + self.v_head_dim,
        )
        key_nope, value = torch.split(
            key_value, [self.qk_nope_head_dim, self.v_head_dim], dim=-1
        )
        if self.padding_v_head_dim:
            value = F.pad(value, [0, self.v_channels - self.v_head_dim])
        value = value.contiguous()
        key_extra = copy_to_tensor_model_parallel_region(key_extra, group=self.tp_group)
        key_extra = key_extra.unsqueeze(-2).expand(*key_nope.shape[:-1], -1)
        key = torch.cat((key_nope, key_extra), dim=-1).contiguous()
        return key, value

    def concat_cached_chunk_key_value_mla(self, curr_key, curr_value):
        """Concatenate cached previous chunks with the current chunk's key/value."""
        if not self.config.enable_chunkpipe:
            raise RuntimeError("Chunk concatenation requires chunkpipe to be enabled.")

        is_forward = self.config.chunkpipe_forward
        microbatch_idx = (
            self.config.chunkpipe_forward_microbatch
            if is_forward
            else self.config.chunkpipe_backward_microbatch
        )
        current_chunk_idx = microbatch_idx % self.num_chunks_per_seq
        start_microbatch_idx = microbatch_idx - current_chunk_idx

        cu = self.config.cu_chunks
        chunk_sizes = self.config.chunk_sizes
        slot_size = self.config.max_chunk_size or self.config.chunksize
        total_tokens = cu[current_chunk_idx + 1]

        concatenated_key = torch.zeros(
            (total_tokens, self.config.micro_batch_size, self.local_num_heads, self.q_head_dim),
            device=self.kv_compressed_cache.device,
            dtype=self.kv_compressed_cache.dtype,
        )
        concatenated_value = torch.zeros(
            (total_tokens, self.config.micro_batch_size, self.local_num_heads, self.v_channels),
            device=self.kv_compressed_cache.device,
            dtype=self.kv_compressed_cache.dtype,
        )

        def kv_hook_fn(chunk_index):
            def hook_fn(grad):
                grads = self.kv_compressed_cache_grad
                if chunk_index not in grads:
                    grads[chunk_index] = grad
                else:
                    grads[chunk_index] = grads[chunk_index] + grad
                return grad
            return hook_fn

        def key_extra_hook_fn(chunk_index):
            def hook_fn(grad):
                grads = self.key_extra_cache_grad
                if chunk_index not in grads:
                    grads[chunk_index] = grad
                else:
                    grads[chunk_index] = grads[chunk_index] + grad
                return grad
            return hook_fn

        current_pos = 0
        for prev_chunk_idx in range(current_chunk_idx):
            cache_chunk_idx = self.micro_batch_to_cache_chunk_map[
                start_microbatch_idx + prev_chunk_idx
            ]
            prev_len = chunk_sizes[prev_chunk_idx]
            indices = torch.arange(prev_len, device=self.kv_compressed_cache.device) + (
                cache_chunk_idx * slot_size
            )
            cached_kv_compressed = self.kv_compressed_cache[indices]
            cached_key_extra = self.key_extra_cache[indices]

            if self.is_enable_grad_chunkpipe():
                cached_kv_compressed.requires_grad = True
                cached_key_extra.requires_grad = True
                cached_kv_compressed.register_hook(kv_hook_fn(prev_chunk_idx))
                cached_key_extra.register_hook(key_extra_hook_fn(prev_chunk_idx))

            cached_key, cached_value = self.recover_key_value_up_proj_tensors(
                cached_kv_compressed, cached_key_extra
            )
            concatenated_key[current_pos : current_pos + prev_len] = cached_key
            concatenated_value[current_pos : current_pos + prev_len] = cached_value
            current_pos += prev_len

        curr_len = curr_key.shape[0]
        concatenated_key[current_pos : current_pos + curr_len] = curr_key
        concatenated_value[current_pos : current_pos + curr_len] = curr_value
        return concatenated_key, concatenated_value

    def _make_mla_kv_consume_hook(self):
        def hook_fn(grad):
            chunks_in_sequence = self.config.chunkpipe_backward_microbatch % self.num_chunks_per_seq
            if chunks_in_sequence == self.num_chunks_per_seq - 1:
                return grad
            return grad + self.kv_compressed_cache_grad.pop(chunks_in_sequence)
        return hook_fn

    def _make_mla_key_extra_consume_hook(self):
        def hook_fn(grad):
            chunks_in_sequence = self.config.chunkpipe_backward_microbatch % self.num_chunks_per_seq
            if chunks_in_sequence == self.num_chunks_per_seq - 1:
                return grad
            return grad + self.key_extra_cache_grad.pop(chunks_in_sequence)
        return hook_fn

    def sharded_state_dict(
        self,
        prefix: str = "",
        sharded_offsets: tuple = (),
        metadata: dict | None = None,
    ) -> ShardedStateDict:
        """Return attention state with explicit KDA TP sharding."""
        metadata = ensure_metadata_has_dp_cp_group(metadata)
        sharded_state_dict = super().sharded_state_dict(
            prefix, sharded_offsets, metadata
        )
        if not self.is_kda:
            return sharded_state_dict

        sharded_state_dict.update(
            make_sharded_tensors_for_checkpoint(
                {"A_log": self.A_log, "dt_bias": self.dt_bias},
                prefix,
                {"A_log": 0, "dt_bias": 0},
                sharded_offsets,
                tp_group=self.tp_group,
                dp_cp_group=metadata["dp_cp_group"],
            )
        )
        return sharded_state_dict

    def _run_kda_conv(self, name, module, x, cu_seqlens, ties) -> torch.Tensor:
        """Run a KDA short convolution, carrying its boundary state across chunks."""
        use_chunkpipe = self.config.enable_chunkpipe
        kwargs = {"output_final_state": use_chunkpipe, "cu_seqlens": cu_seqlens}
        if use_chunkpipe:
            kwargs["cache"] = self.load_prev_state(name)
        y, final = module(x=x, **kwargs)
        if use_chunkpipe:
            self.store_current_state(name, final)
            tie = self.inject_state_grad(name, final)
            if tie is not None:
                ties.append(tie)
        return y

    def _forward_kda(
        self,
        hidden_states: torch.Tensor,
        packed_seq_params: PackedSeqParams | None,
    ) -> torch.Tensor:
        x = hidden_states.transpose(0, 1)
        cu_seqlens = (
            packed_seq_params.cu_seqlens_q if packed_seq_params is not None else None
        )
        if packed_seq_params is not None and cu_seqlens is None:
            raise ValueError("Packed KDA input requires cu_seqlens_q")

        ties: list[torch.Tensor] = []
        q = self._run_kda_conv(
            "q_conv_state", self.q_conv1d, _linear(self.q_proj, x), cu_seqlens, ties
        )
        k = self._run_kda_conv(
            "k_conv_state", self.k_conv1d, _linear(self.k_proj, x), cu_seqlens, ties
        )
        v = self._run_kda_conv(
            "v_conv_state", self.v_conv1d, _linear(self.v_proj, x), cu_seqlens, ties
        )
        q = rearrange(q, "b s (h d) -> b s h d", h=self.local_num_heads)
        k = rearrange(k, "b s (h d) -> b s h d", h=self.local_num_heads)
        v = rearrange(v, "b s (h d) -> b s h d", h=self.local_num_heads)
        forget_gate = rearrange(
            _linear(self.f_b_proj, _linear(self.f_a_proj, x)),
            "b s (h d) -> b s h d",
            h=self.local_num_heads,
        )
        beta = _linear(self.b_proj, x).float().sigmoid()

        kda_kwargs = {
            "output_final_state": self.config.enable_chunkpipe,
            "cu_seqlens": cu_seqlens,
        }
        if self.config.enable_chunkpipe:
            kda_kwargs["initial_state"] = self.load_prev_state("kda_state")
        kda_result = kda(
            q,
            k,
            v,
            forget_gate,
            beta,
            self.A_log,
            self.dt_bias,
            self.gate_lower_bound,
            **kda_kwargs,
        )
        if self.config.enable_chunkpipe:
            output, kda_final = kda_result
            self.store_current_state("kda_state", kda_final)
            tie = self.inject_state_grad("kda_state", kda_final)
            if tie is not None:
                ties.append(tie)
        else:
            output = kda_result

        gate = rearrange(
            _linear(self.g_proj, x),
            "b s (h d) -> b s h d",
            h=self.local_num_heads,
        )
        output = self.o_norm(
            output.reshape(-1, self.head_dim), gate.reshape(-1, self.head_dim)
        )
        output = output.view(*gate.shape).flatten(-2)
        output = _linear(self.o_proj, output.to(hidden_states.dtype)).transpose(0, 1)
        for tie in ties:
            output = output + tie.to(output.dtype)
        return output

    def _forward_mla(
        self,
        hidden_states: torch.Tensor,
        packed_seq_params: PackedSeqParams | None,
    ) -> torch.Tensor:
        x = hidden_states
        query = _linear(self.q_b_proj, self.q_a_layernorm(_linear(self.q_a_proj, x)))
        query = query.view(*query.shape[:-1], self.local_num_heads, self.q_head_dim)

        compressed_kv = _linear(self.kv_a_proj_with_mqa, x)
        kv_latent, key_extra = torch.split(
            compressed_kv,
            [self.kv_lora_rank, self.qk_extra_head_dim],
            dim=-1,
        )
        normed_kv_compressed = self.kv_a_layernorm(kv_latent)

        if self.config.enable_chunkpipe:
            # The current chunk's compressed KV / key-extra feed its own key/value
            # (so they are in the loss graph).  Register consume hooks that absorb
            # the successor chunk's cached gradient, then store this chunk's values.
            if self.is_enable_grad_chunkpipe():
                normed_kv_compressed.register_hook(self._make_mla_kv_consume_hook())
                key_extra.register_hook(self._make_mla_key_extra_consume_hook())
            self.append_chunk_key_value_cache_mla(normed_kv_compressed, key_extra)

        key, value = self.recover_key_value_up_proj_tensors(
            normed_kv_compressed, key_extra
        )

        attn_mask_type = AttnMaskType.causal
        if self.config.enable_chunkpipe:
            key, value = self.concat_cached_chunk_key_value_mla(key, value)
            # Decoder-with-cache semantics: the kernel applies the causal diagonal
            # at the current chunk's offset (inferred from s_kv - s_q).
            attn_mask_type = AttnMaskType.causal_bottom_right

        is_thd = packed_seq_params is not None and packed_seq_params.qkv_format == "thd"
        if is_thd:
            query = query.squeeze(1)
            key = key.squeeze(1)
            value = value.squeeze(1)
        output = self.core_attention(
            query,
            key,
            value,
            None,
            packed_seq_params=packed_seq_params,
            attn_mask_type=attn_mask_type,
        )
        if is_thd:
            output = output.reshape(output.size(0), 1, -1)
        if self.padding_v_head_dim:
            prefix = output.shape[:-1]
            output = output.reshape(*prefix, self.local_num_heads, self.v_channels)
            output = output[..., : self.v_head_dim].reshape(*prefix, -1)
        output = output * torch.sigmoid(_linear(self.g_proj, x))
        return _linear(self.o_proj, output)

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: torch.Tensor | None = None,
        key_value_states: torch.Tensor | None = None,
        inference_context: BaseInferenceContext | None = None,
        rotary_pos_emb: torch.Tensor | None = None,
        rotary_pos_cos: torch.Tensor | None = None,
        rotary_pos_sin: torch.Tensor | None = None,
        rotary_pos_cos_sin: torch.Tensor | None = None,
        attention_bias: torch.Tensor | None = None,
        packed_seq_params: PackedSeqParams | None = None,
        sequence_len_offset: int | None = None,
        **kwargs,
    ) -> tuple[torch.Tensor, None]:
        """Run the attention implementation selected for this layer."""
        del (
            attention_mask,
            key_value_states,
            inference_context,
            rotary_pos_emb,
            rotary_pos_cos,
            rotary_pos_sin,
            rotary_pos_cos_sin,
            attention_bias,
            sequence_len_offset,
            kwargs,
        )
        if self.sequence_parallel:
            hidden_states = gather_from_sequence_parallel_region(
                hidden_states,
                tensor_parallel_output_grad=False,
                group=self.tp_group,
            )
        output = (
            self._forward_kda(hidden_states, packed_seq_params)
            if self.is_kda
            else self._forward_mla(hidden_states, packed_seq_params)
        )
        if self.sequence_parallel:
            output = scatter_to_sequence_parallel_region(output, group=self.tp_group)
        return output, None


__all__ = ["KimiK3Attention", "KimiK3ShortConvolution"]
