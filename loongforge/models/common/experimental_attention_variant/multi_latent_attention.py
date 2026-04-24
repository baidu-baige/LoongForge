# Copyright 2026 The LoongForge Authors.
# SPDX-License-Identifier: Apache-2.0
#
# Modified from Megatron-LM under the BSD 3-Clause License.
# Copyright (c) 2025, NVIDIA CORPORATION. All rights reserved.

"""Omni MLA wrapper carrying DSA absorb logic with CLI-gated fused routing."""

import copy
import math
from typing import Optional

import torch

try:
    from einops import rearrange

    HAVE_EINOPS = True
except ImportError:
    HAVE_EINOPS = False

from megatron.core import parallel_state, tensor_parallel
from megatron.core.models.common.embeddings import apply_rotary_pos_emb
from megatron.core.pipeline_parallel.fine_grained_activation_offload import (
    fine_grained_offloading_group_commit,
    fine_grained_offloading_group_start,
    get_fine_grained_offloading_context,
)
from megatron.core.process_groups_config import ProcessGroupCollection
from megatron.core.tensor_parallel.mappings import (
    gather_from_sequence_parallel_region,
    gather_from_tensor_model_parallel_region,
    scatter_to_sequence_parallel_region,
)
from megatron.core.transformer.enums import AttnMaskType
from megatron.core.transformer.multi_latent_attention import (
    MLASelfAttention,
    MLASelfAttentionSubmodules,
)
from megatron.core.transformer.spec_utils import build_module
from megatron.core.transformer.transformer_config import MLATransformerConfig
from megatron.core.utils import deprecate_inference_params
from megatron.core.fp8_utils import is_float8tensor

from loongforge.utils import get_args
from loongforge.models.common.experimental_attention_variant.dsa_fused_utils import (
    shard_packed_cu_seqlens_for_sp_rank,
)

try:
    from megatron.core.extensions.transformer_engine import TEGroupedLinear
except ImportError:
    TEGroupedLinear = None

try:
    from megatron.core.extensions.transformer_engine import TELinear as _TELinear
except ImportError:
    _TELinear = None

from .dsa_fused_kernels import (
    fused_apply_mla_rope,
    fused_apply_mla_rope_for_absorb_kv,
    fused_rope_permute_cat,
)

class MLASelfAttentionFused(MLASelfAttention):
    """Omni-side MLA class that preserves the rolled-back DSA absorb path."""

    def __init__(
        self,
        config: MLATransformerConfig,
        submodules: MLASelfAttentionSubmodules,
        layer_number: int,
        attn_mask_type: AttnMaskType = AttnMaskType.padding,
        cp_comm_type: Optional[str] = None,
        pg_collection: ProcessGroupCollection = None,
    ) -> None:
        if config.enable_chunkpipe:
            patched_config = config
        else:
            patched_config = copy.copy(config)

        super().__init__(
            config=patched_config,
            submodules=submodules,
            layer_number=layer_number,
            attn_mask_type=attn_mask_type,
            cp_comm_type=cp_comm_type,
            pg_collection=pg_collection,
        )
        # Note: chunkpipe cache is initialized in parent class MLASelfAttention.__init__
        # via init_chunk_key_value_cache_mla() when config.enable_chunkpipe is True

        if self.config.enable_chunkpipe:
            self.v_channels = self.config.v_head_dim
            self.padding_v_head_dim = False

        self.absorb_backend = getattr(self.config, 'absorb_backend', None)
        if self.absorb_backend is None:
            from megatron.training import get_args
            self.absorb_backend = getattr(get_args(), 'absorb_backend', 'te')

        if self.absorb_backend == "torch":
            # torch backend: use einsum with sliced weights from linear_kv_up_proj
            # No extra modules needed; linear_kv_up_proj stays trainable
            pass
        else:
            # TE backend: use TEGroupedLinear absorb modules
            if TEGroupedLinear is None:
                raise ImportError(
                    "--use-dsa-fused requires TEGroupedLinear from transformer_engine."
                )

            self.linear_kv_up_proj_absorb_q = build_module(
                TEGroupedLinear,
                self.num_attention_heads_per_partition,
                self.config.qk_head_dim,
                self.config.kv_lora_rank,
                parallel_mode=None,
                config=self.config,
                init_method=self.config.init_method,
                bias=False,
                skip_bias_add=False,
                is_expert=False,
                tp_comm_buffer_name="kv_up_proj_absorb_q",
            )
            self.linear_kv_up_proj_absorb_output = build_module(
                TEGroupedLinear,
                self.num_attention_heads_per_partition,
                self.config.kv_lora_rank,
                self.config.v_head_dim,
                parallel_mode=None,
                config=self.config,
                init_method=self.config.init_method,
                bias=False,
                skip_bias_add=False,
                is_expert=False,
                tp_comm_buffer_name="kv_up_proj_absorb_output",
            )

            self.linear_kv_up_proj.weight.requires_grad = False

        # Checkpoint load hooks
        if self.absorb_backend == "te":
            # TE: decompose kv_up_proj into per-head absorb weights
            self.register_load_state_dict_pre_hook(self._pre_load_hook)
            self.register_load_state_dict_post_hook(self._post_load_hook)
        else:
            # Torch: reconstruct kv_up_proj from absorb keys if checkpoint is TE format
            self.register_load_state_dict_pre_hook(self._pre_load_hook_torch)
            self.register_load_state_dict_post_hook(self._post_load_hook)

        # Checkpoint save hook (TE absorb only — reconstructs kv_up_proj from
        # per-head absorb weights; torch backend keeps kv_up_proj directly).
        if self.absorb_backend == "te":
            self.register_state_dict_post_hook(
                lambda mod, sd, pfx, meta: self._state_dict_post_hook(mod, sd, pfx, meta)
            )

        # SP-First: convert all 4 linear modules from TP-sharded to duplicated.
        args = get_args()
        self.use_dsa_sp_first = getattr(args, "use_dsa_sp_first", False) if args is not None else False
        if self.use_dsa_sp_first:
            self._convert_to_sp_first()
            if self.absorb_backend == "te":
                self._sync_kv_up_to_absorb_weights(all_gather_for_sp_first=True)

        if self.absorb_backend == "te":
            # Eagerly materialize absorb weights from linear_kv_up_proj during
            # module construction so the optimizer sees the real derived values
            # instead of placeholder initialization.
            self._sync_kv_up_to_absorb_weights()

        if self.absorb_backend == "te":
            # kv_up_proj is fully decomposed into absorb modules; free it.
            del self.linear_kv_up_proj

    @staticmethod
    def _gather_tp_weight(
        weight: torch.Tensor,
        tp_group,
        tp_world_size: int,
        cat_dim: int,
    ):
        """All-gather a TP-sharded weight tensor across the tensor-parallel group.

        Handles FP8 tensors by dequantizing before the collective and also
        gathers the high-precision init value when present.

        Args:
            weight: The (possibly FP8) weight parameter to gather.
            tp_group: The tensor-parallel process group.
            tp_world_size: Size of the tensor-parallel group.
            cat_dim: Dimension along which shards are concatenated.
                - 0 for ColumnParallel (output-dim sharded)
                - 1 for RowParallel   (input-dim sharded)

        Returns:
            Tuple of ``(full_weight, full_hp_weight)`` where ``full_hp_weight``
            is ``None`` when no high-precision init value is available.
        """
        shard = weight.detach()
        shard_original_device = shard.device
        is_fp8 = is_float8tensor(shard)
        if is_fp8:
            shard = shard.dequantize()
        shard = shard.to(torch.cuda.current_device())
        buf = [torch.empty_like(shard) for _ in range(tp_world_size)]
        torch.distributed.all_gather(buf, shard, group=tp_group)
        full = torch.cat(buf, dim=cat_dim)
        full = full.to(shard_original_device)

        hp_full = None
        if is_fp8 and hasattr(weight, 'get_high_precision_init_val'):
            hp_shard = weight.get_high_precision_init_val().to(torch.cuda.current_device())
            hp_buf = [torch.empty_like(hp_shard) for _ in range(tp_world_size)]
            torch.distributed.all_gather(hp_buf, hp_shard, group=tp_group)
            hp_full = torch.cat(hp_buf, dim=cat_dim)
            hp_full = hp_full.cpu()  # high prec val always offload to cpu

        return full, hp_full

    @staticmethod
    def _shard_tp_weight(
        weight: torch.Tensor,
        tp_rank: int,
        tp_world_size: int,
        shard_dim: int,
    ) -> torch.Tensor:
        """Extract the local TP shard from a full (duplicated) weight tensor.

        Handles FP8 tensors (including blockwise) by dequantizing before the
        slice and re-quantizing afterwards.  For blockwise FP8 the resulting
        tensor is stripped of its columnwise data via ``update_usage`` so that
        only rowwise data is serialised to the checkpoint.

        Args:
            weight: The (possibly FP8) weight tensor – full / non-TP.
            tp_rank: Rank within the tensor-parallel group.
            tp_world_size: Size of the tensor-parallel group.
            shard_dim: Dimension along which to shard.
                - 0 for ColumnParallel (output-dim sharded)
                - 1 for RowParallel   (input-dim sharded)

        Returns:
            The local TP shard, in the same dtype / quantisation format as *weight*.
        """
        weight_is_fp8 = is_float8tensor(weight)
        w = weight.dequantize() if weight_is_fp8 else weight
        chunk_size = w.shape[shard_dim] // tp_world_size
        shard = w.narrow(shard_dim, tp_rank * chunk_size, chunk_size).contiguous()
        if weight_is_fp8:
            q = weight._get_quantizer()
            q.set_usage(rowwise=True, columnwise=False)
            shard = q.quantize(shard)
        return shard

    def _convert_to_sp_first(self):
        """Convert all 4 linear modules from TP-sharded to duplicated for SP-First.

        Deletes each TP-sharded module and re-initialises it as a duplicated
        (non-TP) variant.  Weight synchronisation is deferred to the
        ``_post_load_absorb_weights`` hook which runs after checkpoint loading.

        The four modules handled are:
        - ``linear_q_up_proj``  — ColumnParallel → duplicated TELinear
        - ``linear_proj``       — RowParallel    → duplicated TELinear
        - ``linear_kv_up_proj_absorb_q``     — TEGroupedLinear (partial heads) → full heads
        - ``linear_kv_up_proj_absorb_output`` — TEGroupedLinear (partial heads) → full heads
        """
        if _TELinear is None:
            raise ImportError(
                "--use-dsa-sp-first requires TELinear from transformer_engine."
            )

        # ---- Re-init all 4 modules as duplicated (non-TP) ----
        del self.linear_q_up_proj
        self.linear_q_up_proj = build_module(
            _TELinear,
            self.config.q_lora_rank,
            self.config.num_attention_heads * self.q_head_dim,
            parallel_mode="duplicated",
            config=self.config,
            init_method=self.config.init_method,
            bias=False,
            skip_bias_add=False,
            skip_weight_param_allocation=False,
            tp_comm_buffer_name="q_up_proj",
        )

        del self.linear_proj
        self.linear_proj = build_module(
            _TELinear,
            self.config.num_attention_heads * self.config.v_head_dim,
            self.config.hidden_size,
            parallel_mode="duplicated",
            config=self.config,
            init_method=self.config.output_layer_init_method,
            bias=self.config.add_bias_linear,
            skip_bias_add=True,
            skip_weight_param_allocation=False,
            tp_comm_buffer_name="proj",
        )

        if self.absorb_backend == "te":
            # TE absorb modules: re-init with full heads
            del self.linear_kv_up_proj_absorb_q
            self.linear_kv_up_proj_absorb_q = build_module(
                TEGroupedLinear,
                self.config.num_attention_heads,
                self.config.qk_head_dim,
                self.config.kv_lora_rank,
                parallel_mode=None,
                config=self.config,
                init_method=self.config.init_method,
                bias=False,
                skip_bias_add=False,
                is_expert=False,
                tp_comm_buffer_name="kv_up_proj_absorb_q",
            )

            del self.linear_kv_up_proj_absorb_output
            self.linear_kv_up_proj_absorb_output = build_module(
                TEGroupedLinear,
                self.config.num_attention_heads,
                self.config.kv_lora_rank,
                self.config.v_head_dim,
                parallel_mode=None,
                config=self.config,
                init_method=self.config.init_method,
                bias=False,
                skip_bias_add=False,
                is_expert=False,
                tp_comm_buffer_name="kv_up_proj_absorb_output",
            )
        else:
            # Torch backend: re-init linear_kv_up_proj as duplicated (full heads)
            del self.linear_kv_up_proj
            self.linear_kv_up_proj = build_module(
                _TELinear,
                self.config.kv_lora_rank,
                self.config.num_attention_heads * (self.config.qk_head_dim + self.config.v_head_dim),
                parallel_mode="duplicated",
                config=self.config,
                init_method=self.config.init_method,
                bias=False,
                skip_bias_add=False,
                skip_weight_param_allocation=False,
                tp_comm_buffer_name="kv_up_proj",
            )

        # Update heads-per-partition to full heads count for SP-First
        self.num_attention_heads_per_partition = self.config.num_attention_heads

        # ---- Set SP flags ----
        # sequence_parallel=True: triggers grad AllReduce in finalize_model_grads.
        # tensor_model_parallel=False: avoids grad norm overcounting across TP ranks.
        sp_modules = [self.linear_q_up_proj, self.linear_proj]
        if self.absorb_backend == "te":
            sp_modules.extend([self.linear_kv_up_proj_absorb_q, self.linear_kv_up_proj_absorb_output])
        else:
            sp_modules.append(self.linear_kv_up_proj)
        for module in sp_modules:
            for param in module.parameters():
                setattr(param, "sequence_parallel", True)
                setattr(param, "tensor_model_parallel", False)

    def _sync_kv_up_to_absorb_weights(self, *, all_gather_for_sp_first=False):
        """Synchronize absorb weights from the current linear_kv_up_proj parameter.

        Splits the combined kv_up_proj weight into per-head K-up and V-up
        components and copies them into ``linear_kv_up_proj_absorb_q`` and
        ``linear_kv_up_proj_absorb_output`` respectively.

        By default this is a purely local operation using
        ``num_attention_heads_per_partition`` heads.  When
        ``all_gather_for_sp_first=True`` the kv_up_proj weight is first
        all-gathered across the TP group and split into
        ``num_attention_heads`` (full) heads.  This should only be set after
        ``_convert_to_sp_first()`` has re-initialised the absorb modules with
        full heads (e.g. during ``_post_load_absorb_weights``).
        """
        assert self.linear_kv_up_proj.parallel_mode == "column", (
            "DSA currently only supports linear_kv_up_proj with column parallel mode. "
            "Row parallel mode support is under development and will be available soon."
        )
        assert self.linear_kv_up_proj.use_bias is False, (
            "DSA currently only supports linear_kv_up_proj with no bias. "
            "Bias support is under development and will be avaliable soon."
        )

        with torch.no_grad():
            kv_up_weight = self.linear_kv_up_proj.weight.clone().detach()
            kv_up_weight_hp = None
            if is_float8tensor(kv_up_weight):
                if hasattr(self.linear_kv_up_proj.weight, 'get_high_precision_init_val'):
                    kv_up_weight_hp = self.linear_kv_up_proj.weight.get_high_precision_init_val()
                kv_up_weight = kv_up_weight.dequantize()

            if all_gather_for_sp_first:
                tp_group = parallel_state.get_tensor_model_parallel_group()
                world_size = parallel_state.get_tensor_model_parallel_world_size()
                gathered_list = [torch.empty_like(kv_up_weight) for _ in range(world_size)]
                torch.distributed.all_gather(gathered_list, kv_up_weight, group=tp_group)
                kv_up_weight = torch.cat(gathered_list, dim=0)
                if kv_up_weight_hp is not None:
                    kv_up_weight_hp = kv_up_weight_hp.to(kv_up_weight.device)
                    hp_gathered = [torch.empty_like(kv_up_weight_hp) for _ in range(world_size)]
                    torch.distributed.all_gather(hp_gathered, kv_up_weight_hp, group=tp_group)
                    kv_up_weight_hp = torch.cat(hp_gathered, dim=0)
                num_heads = self.config.num_attention_heads
            else:
                num_heads = self.num_attention_heads_per_partition

            kv_up_weight = kv_up_weight.view(
                num_heads, -1, self.config.kv_lora_rank
            )
            k_up_proj, v_up_proj = torch.split(
                kv_up_weight,
                [self.config.qk_head_dim, self.config.v_head_dim],
                dim=-2,
            )
            k_up_proj = k_up_proj.transpose(1, 2).contiguous()
            v_up_proj = v_up_proj.contiguous()

            k_up_proj_hp = v_up_proj_hp = None
            if kv_up_weight_hp is not None:
                kv_up_weight_hp = kv_up_weight_hp.view(
                    num_heads, -1, self.config.kv_lora_rank
                )
                k_up_proj_hp, v_up_proj_hp = torch.split(
                    kv_up_weight_hp,
                    [self.config.qk_head_dim, self.config.v_head_dim],
                    dim=-2,
                )
                k_up_proj_hp = k_up_proj_hp.transpose(1, 2).contiguous()
                v_up_proj_hp = v_up_proj_hp.contiguous()

            for head_idx in range(num_heads):
                q_absorb_weight = getattr(
                    self.linear_kv_up_proj_absorb_q, f"weight{head_idx}"
                )
                if is_float8tensor(q_absorb_weight):
                    q_absorb_weight.quantize_(k_up_proj[head_idx])
                    if k_up_proj_hp is not None and hasattr(
                        q_absorb_weight, 'set_high_precision_init_val'
                    ):
                        q_absorb_weight.set_high_precision_init_val(k_up_proj_hp[head_idx])
                else:
                    q_absorb_weight.copy_(k_up_proj[head_idx])

                output_absorb_weight = getattr(
                    self.linear_kv_up_proj_absorb_output, f"weight{head_idx}"
                )
                if is_float8tensor(output_absorb_weight):
                    output_absorb_weight.quantize_(v_up_proj[head_idx])
                    if v_up_proj_hp is not None and hasattr(
                        output_absorb_weight, 'set_high_precision_init_val'
                    ):
                        output_absorb_weight.set_high_precision_init_val(
                            v_up_proj_hp[head_idx]
                        )
                else:
                    output_absorb_weight.copy_(v_up_proj[head_idx])

    def concat_cached_chunk_key_value_dsa(self, curr_key):
        """Concatenate all cached key chunks for DSA path in chunkpipe.
        
        Unlike standard MLA, DSA stores key as [kv_compressed, k_pos_emb] directly
        without up-projection, and value is always None.
        
        Args:
            attention_mask (Tensor): Attention mask tensor for the current chunk
            curr_key (Tensor): Current chunk's key tensor with shape 
                [chunksize, batch_size, kv_lora_rank + qk_pos_emb_head_dim]
        
        Returns:
            tuple: (concatenated_key, None)
        """
        if not self.config.enable_chunkpipe:
            raise RuntimeError("Chunk concatenation requires chunkpipe to be enabled.")
        
        is_forward = self.config.chunkpipe_forward
        microbatch_idx = (self.config.chunkpipe_forward_microbatch if is_forward 
                         else self.config.chunkpipe_backward_microbatch)
        current_chunk_idx = microbatch_idx % self.num_chunks_per_seq
        start_microbatch_idx = microbatch_idx - current_chunk_idx

        # Calculate total sequence length after concatenation
        total_concatenated_tokens = (current_chunk_idx + 1) * self.config.chunksize
        # DSA key shape: [seq, batch, kv_lora_rank + qk_pos_emb_head_dim]
        key_hidden_size = self.config.kv_lora_rank + self.config.qk_pos_emb_head_dim
        concatenated_key_shape = (total_concatenated_tokens, self.config.micro_batch_size, key_hidden_size)
        
        concatenated_key = torch.zeros(concatenated_key_shape,
            device=self.kv_compressed_cache.device, dtype=self.kv_compressed_cache.dtype)

        def kv_compressed_hook_fn(chunk_index):
            """Hook function to accumulate gradient for cached compressed KV."""
            def hook_fn(grad):
                if chunk_index not in self.kv_compressed_cache_grad:
                    self.kv_compressed_cache_grad[chunk_index] = grad
                else:
                    self.kv_compressed_cache_grad[chunk_index] += grad
                return grad           
            return hook_fn

        def key_pos_emb_hook_fn(chunk_index):
            """Hook function to accumulate gradient for cached key position embeddings."""
            def hook_fn(grad):
                if chunk_index not in self.key_pos_emb_cache_grad:
                    self.key_pos_emb_cache_grad[chunk_index] = grad
                else:
                    self.key_pos_emb_cache_grad[chunk_index] += grad
                return grad           
            return hook_fn
        
        # Retrieve all previous chunks from cache
        current_pos = 0
        for prev_chunk_idx in range(current_chunk_idx):
            cache_chunk_idx = self.micro_batch_to_cache_chunk_map[start_microbatch_idx + prev_chunk_idx]
            
            tp_size = parallel_state.get_tensor_model_parallel_world_size()
            kv_indices = torch.arange(self.config.chunksize // tp_size) + \
                (cache_chunk_idx * self.config.chunksize // tp_size)
            pos_indices = torch.arange(self.config.chunksize) + (cache_chunk_idx * self.config.chunksize)

            cached_kv_compressed = self.kv_compressed_cache[kv_indices, :, :]
            cached_key_pos_emb = self.key_pos_emb_cache[pos_indices, :, 0:1, :]

            # Set up gradient hooks for backward pass
            if self.is_enable_grad_chunkpipe():
                cached_kv_compressed.requires_grad = True
                cached_key_pos_emb.requires_grad = True
                cached_kv_compressed.register_hook(kv_compressed_hook_fn(prev_chunk_idx))
                cached_key_pos_emb.register_hook(key_pos_emb_hook_fn(prev_chunk_idx))
            
            # For DSA: need to gather kv_compressed if sequence parallel, then cat
            if self.config.sequence_parallel:
                cached_kv_compressed_all = gather_from_sequence_parallel_region(cached_kv_compressed)
            else:
                cached_kv_compressed_all = cached_kv_compressed
            
            # Reconstruct key as [kv_compressed, k_pos_emb]
            cached_key = torch.cat([cached_kv_compressed_all, cached_key_pos_emb.squeeze(1)], dim=-1)
            
            concatenated_key[current_pos : current_pos + self.config.chunksize, :, :] = cached_key
            current_pos += self.config.chunksize

        # Add current chunk's key
        concatenated_key[current_pos : current_pos + self.config.chunksize, :, :] = curr_key
            
        return concatenated_key, None

    def _pre_load_hook(
        self, module, state_dict, prefix, local_metadata, strict,
        missing_keys, unexpected_keys, error_msgs,
    ):
        """Pre-hook: decompose kv_up_proj and inject per-head absorb weights into state_dict.

        For SP-First, also all-gathers q_up_proj, proj, and kv_up_proj TP shards,
        then removes kv_up_key since the module is deleted.
        """
        tp_group = parallel_state.get_tensor_model_parallel_group()
        tp_world_size = parallel_state.get_tensor_model_parallel_world_size()

        # All-gather TP-sharded `q_up_proj` and `linear_proj` to match duplicated module shapes.
        if self.use_dsa_sp_first:
            # Ensure we use CUDA device for all_gather (NCCL only supports CUDA)
            cuda_device = torch.cuda.current_device()
            with torch.no_grad():
                q_up_key = prefix + "linear_q_up_proj.weight"
                ckpt_w = state_dict[q_up_key]
                if ckpt_w.shape[0] != self.linear_q_up_proj.weight.shape[0]:
                    full, _ = self._gather_tp_weight(
                        ckpt_w.to(cuda_device),
                        tp_group, tp_world_size, cat_dim=0,
                    )
                    state_dict[q_up_key] = full.cpu()

                proj_key = prefix + "linear_proj.weight"
                ckpt_w = state_dict[proj_key]
                if ckpt_w.shape[1] != self.linear_proj.weight.shape[1]:
                    full, _ = self._gather_tp_weight(
                        ckpt_w.to(cuda_device),
                        tp_group, tp_world_size, cat_dim=1,
                    )
                    state_dict[proj_key] = full.cpu()

        # Decompose kv_up_proj into per-head absorb weights (all-gather first if SP-First).
        kv_up_key = prefix + "linear_kv_up_proj.weight"
        assert kv_up_key in state_dict, f"Missing required key: {kv_up_key}"
        with torch.no_grad():
            if self.use_dsa_sp_first and tp_world_size > 1:
                cuda_device = torch.cuda.current_device()
                kv_up_weight, _ = self._gather_tp_weight(
                    state_dict[kv_up_key].to(cuda_device),
                    tp_group, tp_world_size, cat_dim=0,
                )
                num_heads = self.config.num_attention_heads
            else:
                kv_up_weight = state_dict[kv_up_key].detach()
                if is_float8tensor(kv_up_weight):
                    kv_up_weight = kv_up_weight.dequantize()
                num_heads = self.num_attention_heads_per_partition

            kv_up_weight = kv_up_weight.view(num_heads, -1, self.config.kv_lora_rank)
            k_up_proj, v_up_proj = torch.split(
                kv_up_weight,
                [self.config.qk_head_dim, self.config.v_head_dim],
                dim=-2,
            )
            k_up_proj = k_up_proj.transpose(1, 2).contiguous()
            v_up_proj = v_up_proj.contiguous()

            absorb_q_pfx = prefix + "linear_kv_up_proj_absorb_q."
            absorb_out_pfx = prefix + "linear_kv_up_proj_absorb_output."
            for head_idx in range(num_heads):
                state_dict[absorb_q_pfx + f"weight{head_idx}"] = k_up_proj[head_idx].cpu()
                state_dict[absorb_out_pfx + f"weight{head_idx}"] = v_up_proj[head_idx].cpu()

        # linear_kv_up_proj module is deleted; remove from state_dict.
        del state_dict[kv_up_key]

    def _pre_load_hook_torch(
        self, module, state_dict, prefix, local_metadata, strict,
        missing_keys, unexpected_keys, error_msgs,
    ):
        """Pre-hook for torch absorb backend: reconstruct kv_up_proj from absorb keys if needed.

        When loading a checkpoint saved with TE absorb backend, the state_dict
        contains per-head absorb_q/absorb_output keys instead of kv_up_proj.
        This hook reconstructs kv_up_proj.weight from those keys and removes
        the absorb keys so that load_state_dict succeeds.

        For SP-First, also all-gathers TP-sharded q_up_proj and linear_proj.
        """
        tp_group = parallel_state.get_tensor_model_parallel_group()
        tp_world_size = parallel_state.get_tensor_model_parallel_world_size()

        # SP-First: all-gather TP-sharded q_up_proj and linear_proj
        if self.use_dsa_sp_first:
            cuda_device = torch.cuda.current_device()
            with torch.no_grad():
                q_up_key = prefix + "linear_q_up_proj.weight"
                ckpt_w = state_dict[q_up_key]
                if ckpt_w.shape[0] != self.linear_q_up_proj.weight.shape[0]:
                    full, _ = self._gather_tp_weight(
                        ckpt_w.to(cuda_device),
                        tp_group, tp_world_size, cat_dim=0,
                    )
                    state_dict[q_up_key] = full.cpu()

                proj_key = prefix + "linear_proj.weight"
                ckpt_w = state_dict[proj_key]
                if ckpt_w.shape[1] != self.linear_proj.weight.shape[1]:
                    full, _ = self._gather_tp_weight(
                        ckpt_w.to(cuda_device),
                        tp_group, tp_world_size, cat_dim=1,
                    )
                    state_dict[proj_key] = full.cpu()

        kv_up_key = prefix + "linear_kv_up_proj.weight"
        absorb_q_pfx = prefix + "linear_kv_up_proj_absorb_q."
        absorb_out_pfx = prefix + "linear_kv_up_proj_absorb_output."

        # Check if checkpoint has absorb keys (TE absorb format)
        absorb_keys = [k for k in state_dict if k.startswith(absorb_q_pfx) or k.startswith(absorb_out_pfx)]
        if not absorb_keys:
            # Standard checkpoint with kv_up_proj.weight — may still need all-gather for SP-First
            if self.use_dsa_sp_first and kv_up_key in state_dict:
                ckpt_w = state_dict[kv_up_key]
                if ckpt_w.shape[0] != self.linear_kv_up_proj.weight.shape[0]:
                    cuda_device = torch.cuda.current_device()
                    with torch.no_grad():
                        full, _ = self._gather_tp_weight(
                            ckpt_w.to(cuda_device),
                            tp_group, tp_world_size, cat_dim=0,
                        )
                        state_dict[kv_up_key] = full.cpu()
            return

        with torch.no_grad():
            # Count heads by finding weightN keys
            num_heads = 0
            while (absorb_q_pfx + f"weight{num_heads}") in state_dict:
                num_heads += 1

            if num_heads == 0:
                return  # No per-head weights found

            q_absorb_list = []
            output_absorb_list = []
            for head_idx in range(num_heads):
                q_w = state_dict[absorb_q_pfx + f"weight{head_idx}"]
                if is_float8tensor(q_w):
                    q_w = q_w.dequantize()
                q_absorb_list.append(q_w.detach())

                out_w = state_dict[absorb_out_pfx + f"weight{head_idx}"]
                if is_float8tensor(out_w):
                    out_w = out_w.dequantize()
                output_absorb_list.append(out_w.detach())

            # Reconstruct kv_up_proj: absorb_q is [kv_lora_rank, qk_head_dim] per head
            # Need to transpose back: k_up_proj was transposed(1,2) when decomposed
            q_absorb = torch.stack(q_absorb_list, dim=0)   # [num_heads, kv_lora_rank, qk_head_dim]
            q_absorb = q_absorb.transpose(1, 2).contiguous()  # [num_heads, qk_head_dim, kv_lora_rank]
            output_absorb = torch.stack(output_absorb_list, dim=0)  # [num_heads, v_head_dim, kv_lora_rank]

            # Concatenate back: [num_heads, (qk_head_dim + v_head_dim), kv_lora_rank]
            kv_up_weight = torch.cat([q_absorb, output_absorb], dim=-2)
            kv_up_weight = kv_up_weight.contiguous().view(-1, self.config.kv_lora_rank)

            # SP-First: all-gather TP-sharded kv_up_proj to full heads
            if self.use_dsa_sp_first:
                cuda_device = torch.cuda.current_device()
                kv_up_weight, _ = self._gather_tp_weight(
                    kv_up_weight.to(cuda_device),
                    tp_group, tp_world_size, cat_dim=0,
                )
                kv_up_weight = kv_up_weight.cpu()

            # Inject reconstructed kv_up_proj into state_dict
            state_dict[kv_up_key] = kv_up_weight

        # Remove all absorb keys from state_dict
        for k in absorb_keys:
            del state_dict[k]

    def _post_load_hook(self, module, incompatible_keys):
        """Post-hook: suppress missing-key warnings for derived/deleted absorb weights."""
        if incompatible_keys is not None:
            suppress_markers = (
                ".linear_kv_up_proj.",
                ".linear_kv_up_proj_absorb_q.",
                ".linear_kv_up_proj_absorb_output.",
            )
            incompatible_keys.missing_keys[:] = [
                key
                for key in incompatible_keys.missing_keys
                if not any(marker in key for marker in suppress_markers)
            ]

    def _reconstruct_kv_up_weight(self, num_heads):
        """Reconstruct kv_up_proj weight from absorb_q and absorb_output modules.

        If the absorb weights are FP8, the output is re-quantized back to the
        same FP8 type so that the caller receives the original dtype.
        """
        first_absorb = getattr(self.linear_kv_up_proj_absorb_q, "weight0", None)
        input_is_fp8 = first_absorb is not None and is_float8tensor(first_absorb)

        q_absorb_list = []
        output_absorb_list = []
        for head_idx in range(num_heads):
            head_q_absorb = getattr(
                self.linear_kv_up_proj_absorb_q, f"weight{head_idx}"
            ).clone().detach()
            if is_float8tensor(head_q_absorb):
                q_absorb_list.append(head_q_absorb.dequantize())
            else:
                q_absorb_list.append(head_q_absorb)

            head_output_absorb = getattr(
                self.linear_kv_up_proj_absorb_output, f"weight{head_idx}"
            ).clone().detach()
            if is_float8tensor(head_output_absorb):
                output_absorb_list.append(head_output_absorb.dequantize())
            else:
                output_absorb_list.append(head_output_absorb)

        q_absorb = torch.stack(q_absorb_list, dim=0)
        output_absorb = torch.stack(output_absorb_list, dim=0)
        q_absorb = q_absorb.transpose(1, 2).contiguous()

        kv_up_weight = torch.cat([q_absorb, output_absorb], dim=-2)
        kv_up_weight = kv_up_weight.contiguous().view(-1, self.config.kv_lora_rank)

        if input_is_fp8:
            q = first_absorb._get_quantizer()
            q.set_usage(rowwise=True, columnwise=False)
            kv_up_weight = q.quantize(kv_up_weight)

        return kv_up_weight

    def _state_dict_post_hook(self, module, state_dict, prefix, local_metadata):
        """Post-hook: reconstruct kv_up_proj from absorb weights and inject into state_dict.

        Also removes the redundant per-head absorb weight keys, since they can be
        fully reconstructed from linear_kv_up_proj.weight during loading (_pre_load_hook).

        For SP-First, also slices the duplicated (non-TP) q_up_proj and linear_proj
        weights back to per-rank TP shards so that checkpoints remain TP-compatible.

        Finally, all FP8 blockwise tensors whose internal data sits in an
        oversized shared storage are compacted (cloned to right-sized
        independent storage) to prevent checkpoint size inflation.
        """
        with torch.no_grad():
            if self.use_dsa_sp_first:
                tp_rank = parallel_state.get_tensor_model_parallel_rank()
                world_size = parallel_state.get_tensor_model_parallel_world_size()

                # --- kv_up_proj: reconstruct full weight, then extract TP shard ---
                kv_up_weight = self._shard_tp_weight(
                    self._reconstruct_kv_up_weight(self.config.num_attention_heads),
                    tp_rank, world_size, shard_dim=0,
                )

                # --- q_up_proj: ColumnParallel, shard along dim-0 (output dim) ---
                q_up_key = prefix + "linear_q_up_proj.weight"
                state_dict[q_up_key] = self._shard_tp_weight(
                    state_dict[q_up_key], tp_rank, world_size, shard_dim=0,
                )

                # --- linear_proj: RowParallel, shard along dim-1 (input dim) ---
                proj_key = prefix + "linear_proj.weight"
                state_dict[proj_key] = self._shard_tp_weight(
                    state_dict[proj_key], tp_rank, world_size, shard_dim=1,
                )
            else:
                kv_up_weight = self._reconstruct_kv_up_weight(self.num_attention_heads_per_partition)

        state_dict[prefix + "linear_kv_up_proj.weight"] = kv_up_weight

        # Remove redundant per-head absorb weights: they are derived from kv_up_proj.weight
        # and will be re-decomposed by _pre_load_hook at load time.
        absorb_prefixes = (
            prefix + "linear_kv_up_proj_absorb_q.",
            prefix + "linear_kv_up_proj_absorb_output.",
        )
        keys_to_delete = [k for k in state_dict if k.startswith(absorb_prefixes)]
        for k in keys_to_delete:
            del state_dict[k]

    def _get_kv_up_slices(self):
        """Return (k_up, v_up) weight slices from linear_kv_up_proj for torch einsum path.

        Results are cached and auto-invalidated when the underlying weight changes
        (after optimizer step), so they stay valid across micro-batches within a step.

        Returns:
            k_up: [num_heads_per_partition, qk_head_dim, kv_lora_rank]
            v_up: [num_heads_per_partition, v_head_dim, kv_lora_rank]
        """
        w = self.linear_kv_up_proj.weight
        cache_key = (w.data_ptr(), w._version)
        if getattr(self, "_cached_kv_up_key", None) == cache_key:
            return self._cached_kv_up_slices
        if is_float8tensor(w):
            w = w.dequantize()
        num_heads = self.num_attention_heads_per_partition
        w = w.view(num_heads, self.config.qk_head_dim + self.config.v_head_dim, self.config.kv_lora_rank)
        k_up, v_up = torch.split(
            w, [self.config.qk_head_dim, self.config.v_head_dim], dim=1
        )
        self._cached_kv_up_slices = (k_up, v_up)
        self._cached_kv_up_key = cache_key
        return k_up, v_up

    def forward(
        self,
        hidden_states,
        attention_mask,
        key_value_states=None,
        inference_context=None,
        rotary_pos_emb=None,
        rotary_pos_cos=None,
        rotary_pos_sin=None,
        rotary_pos_cos_sin=None,
        attention_bias=None,
        packed_seq_params=None,
        position_ids=None,
        sequence_len_offset=None,
        *,
        inference_params=None,
    ):
        """Forward pass with Omni absorb-output projection before linear_proj."""
        assert rotary_pos_emb is None, "Rotary position embeddings should not be passed into MLA."
        assert attention_bias is None, "Attention bias should not be passed into MLA."
        assert (
            rotary_pos_cos is None and rotary_pos_sin is None
        ), "MLA does not support Flash Decoding"
        assert not rotary_pos_cos_sin, "Flash-infer rope has not been tested with MLA."
        assert not (
            self.training and self.cache_mla_latents
        ), "cache_mla_latents conflicts with training."

        inference_context = deprecate_inference_params(inference_context, inference_params)
        if inference_context and not inference_context.is_static_batching():
            assert (
                self.config.cache_mla_latents
            ), "currently to use dynamic backend for MLA cache mla latents must be true"

        if self.config.cache_mla_latents:
            self.prepare_for_absorption()

        query, key, value, q_compressed, kv_compressed = self.get_query_key_value_tensors(
            hidden_states,
            key_value_states,
            position_ids,
            packed_seq_params,
            inference_context=inference_context,
        )

        query, key, value, _, attn_mask_type, block_table = self._adjust_key_value_for_inference(
            inference_context, query, key, value, rotary_pos_emb=None
        )

        # Chunkpipe: concatenate cached KV chunks with current chunk
        if self.config.enable_chunkpipe:
            key, value = self.concat_cached_chunk_key_value_dsa(key)

        query = query.contiguous()
        key = key.contiguous()
        if value is not None:
            value = value.contiguous()

        if self.checkpoint_core_attention and self.training:
            core_attn_out = self._checkpointed_attention_forward(
                query, key, value, attention_mask, packed_seq_params=packed_seq_params
            )
        else:
            if self.offload_core_attention and self.training:
                query = fine_grained_offloading_group_start(query, name="core_attn")

            if inference_context is None or inference_context.is_static_batching():
                extra_kwargs = {
                    "x": hidden_states,
                    "qr": q_compressed,
                }
                with get_fine_grained_offloading_context(self.offload_core_attention):
                    core_attn_out = self.core_attention(
                        query,
                        key,
                        value,
                        attention_mask,
                        packed_seq_params=packed_seq_params,
                        attn_mask_type=attn_mask_type,
                        **extra_kwargs,
                    )

                if self.absorb_backend == "torch":
                    # core_attn_out: [..., num_heads, kv_lora_rank]
                    # v_up: [num_heads, v_head_dim, kv_lora_rank]
                    _, v_up = self._get_kv_up_slices()
                    core_attn_out = torch.einsum(
                        "...hk,hdk->...hd", core_attn_out, v_up
                    )
                else:
                    num_heads_out = (
                        self.config.num_attention_heads
                        if self.use_dsa_sp_first
                        else self.num_attention_heads_per_partition
                    )
                    m_splits_v = [math.prod(core_attn_out.size()[:-2])] * num_heads_out
                    core_attn_out_permute = core_attn_out.movedim(-2, 0).contiguous()
                    core_attn_out, _ = self.linear_kv_up_proj_absorb_output(
                        core_attn_out_permute, m_splits_v
                    )
                    core_attn_out = core_attn_out.transpose(0, core_attn_out.ndim - 2)
                core_attn_out = core_attn_out.flatten(-2, -1).contiguous()

            elif self.cache_mla_latents:
                q, k, v = (query, key, value)
                cu_query_lengths, max_seqlen_q = inference_context.cu_query_lengths()
                cu_kv_lengths, kv_lengths, max_seqlen_k = inference_context.cu_kv_lengths()

                core_attn_out = self.flash_decode_and_prefill(
                    q,
                    k,
                    v,
                    max_seqlen_q,
                    max_seqlen_k,
                    cu_query_lengths,
                    cu_kv_lengths,
                    kv_lengths,
                    block_table,
                )
                if not inference_context.is_decode_only():
                    core_attn_out = rearrange(core_attn_out, "s b h d -> s b (h d)")
            if self.offload_core_attention and self.training:
                (core_attn_out,) = fine_grained_offloading_group_commit(
                    core_attn_out, name="core_attn", forced_released_tensors=[query, key, value]
                )

        if self.cache_mla_latents and inference_context.is_decode_only():
            core_attn_out = torch.einsum("sbhc,hdc->sbhd", core_attn_out, self.up_v_weight)
            core_attn_out = core_attn_out.contiguous()
            core_attn_out = core_attn_out.view(core_attn_out.size(0), core_attn_out.size(1), -1)

        if self.padding_v_head_dim:
            _prefix = core_attn_out.shape[:-1]
            core_attn_out = core_attn_out.reshape(*_prefix, -1, self.v_channels)
            core_attn_out = core_attn_out[..., : self.config.v_head_dim].reshape(*_prefix, -1)

        if packed_seq_params is not None and packed_seq_params.qkv_format == "thd":
            core_attn_out = core_attn_out.reshape(core_attn_out.size(0), 1, -1)

        if self.recompute_up_proj:
            assert self.qkv_up_checkpoint is not None
            self.qkv_up_checkpoint.discard_output_and_register_recompute(core_attn_out)
            self.qkv_up_checkpoint = None

        if self.offload_attn_proj:
            core_attn_out = fine_grained_offloading_group_start(core_attn_out, name="attn_proj")
        with get_fine_grained_offloading_context(self.offload_attn_proj):
            output, bias = self.linear_proj(core_attn_out)
        if self.offload_attn_proj:
            output, bias = fine_grained_offloading_group_commit(
                output, bias, name="attn_proj", forced_released_tensors=[core_attn_out]
            )

        return output, bias

    def get_query_key_value_tensors(
        self,
        hidden_states,
        key_value_states=None,
        position_ids=None,
        packed_seq_params=None,
        inference_context=None,
        *,
        inference_params=None,
    ):
        """Derive query/key/value tensors using Omni fused DSA absorb-q path."""
        if self.cache_mla_latents:
            return super().get_query_key_value_tensors(
                hidden_states=hidden_states,
                key_value_states=key_value_states,
                position_ids=position_ids,
                packed_seq_params=packed_seq_params,
                inference_context=inference_context,
                inference_params=inference_params,
            )

        assert (
            hidden_states.ndim == 3
        ), f"hidden_states should be 3D, [s, b, n*h], got {hidden_states.ndim}D"

        inference_context = deprecate_inference_params(inference_context, inference_params)
        rotary_seq_len = self.rotary_pos_emb.get_rotary_seq_len(
            inference_context, None, hidden_states, self.config, packed_seq_params
        )

        # Calculate position embedding offset for chunkpipe
        pos_emb_offset = 0
        if self.config.enable_chunkpipe:
            ck_fwd_mic = self.config.chunkpipe_forward_microbatch % self.num_chunks_per_seq
            if not self.config.chunkpipe_forward:
                ck_fwd_mic = self.config.chunkpipe_backward_microbatch % self.num_chunks_per_seq
            pos_emb_offset = ck_fwd_mic * self.config.chunksize

        mscale = 1.0
        packed_seq = packed_seq_params is not None and packed_seq_params.qkv_format == "thd"
        if self.config.rope_type == "rope":
            rotary_pos_emb = self.rotary_pos_emb(rotary_seq_len, offset=pos_emb_offset, packed_seq=packed_seq)
        else:
            if self.config.apply_rope_fusion:
                rotary_pos_cos, rotary_pos_sin = self.rotary_pos_emb.get_cached_cos_sin(
                    rotary_seq_len, dtype=hidden_states.dtype, packed_seq=packed_seq
                )
                rotary_pos_emb = None
                assert inference_context is None, "Inference with MLA RoPE fusion is not supported"
            else:
                rotary_pos_emb, mscale = self.rotary_pos_emb(
                    rotary_seq_len, offset=pos_emb_offset, packed_seq=packed_seq)

        if packed_seq_params is not None:
            if packed_seq_params.cu_seqlens_q_padded is not None:
                cu_seqlens_q = packed_seq_params.cu_seqlens_q_padded
            else:
                cu_seqlens_q = packed_seq_params.cu_seqlens_q
            if packed_seq_params.cu_seqlens_kv_padded is not None:
                cu_seqlens_kv = packed_seq_params.cu_seqlens_kv_padded
            else:
                cu_seqlens_kv = packed_seq_params.cu_seqlens_kv
        else:
            cu_seqlens_q = cu_seqlens_kv = None

        if self.config.q_lora_rank is not None:
            q_compressed, _ = self.linear_q_down_proj(hidden_states)
            if q_compressed.size(-1) != self.config.q_lora_rank:
                q_compressed = gather_from_tensor_model_parallel_region(q_compressed)
                if self.config.sequence_parallel:
                    q_compressed = scatter_to_sequence_parallel_region(q_compressed)
        else:
            q_compressed = hidden_states

        kv_combined, _ = self.linear_kv_down_proj(hidden_states)
        if kv_combined.size(-1) != self.config.kv_lora_rank + self.config.qk_pos_emb_head_dim:
            kv_combined = gather_from_tensor_model_parallel_region(kv_combined)
            kv_compressed, k_pos_emb = torch.split(
                kv_combined, [self.config.kv_lora_rank, self.config.qk_pos_emb_head_dim], dim=-1
            )
            if self.config.sequence_parallel:
                kv_compressed = scatter_to_sequence_parallel_region(kv_compressed)
        else:
            kv_compressed, k_pos_emb = torch.split(
                kv_combined, [self.config.kv_lora_rank, self.config.qk_pos_emb_head_dim], dim=-1
            )
            if (
                parallel_state.get_tensor_model_parallel_world_size() > 1
                and self.config.sequence_parallel
            ):
                k_pos_emb = gather_from_sequence_parallel_region(k_pos_emb)

        if packed_seq_params is not None:
            q_compressed = q_compressed.squeeze(1)
            kv_compressed = kv_compressed.squeeze(1)
            k_pos_emb = k_pos_emb.squeeze(1)

        if self.config.q_lora_rank is not None:
            q_compressed = self.q_layernorm(q_compressed)
        kv_compressed = self.kv_layernorm(kv_compressed)

        def qkv_up_proj_and_rope_apply_for_dsa(
            q_compressed, kv_compressed, k_pos_emb, rotary_pos_emb
        ):
            assert self.absorb_backend == "torch" or self.linear_kv_up_proj_absorb_q is not None, (
                "get_query_kv_tensor() can only be called when absorb_backend is 'torch' or "
                "linear_kv_up_proj_absorb_q is not None."
            )

            if self.config.q_lora_rank is not None:
                q, _ = self.linear_q_up_proj(q_compressed)
            else:
                q, _ = self.linear_q_proj(q_compressed)

            if self.use_dsa_sp_first:
                num_heads_q = self.config.num_attention_heads
            else:
                num_heads_q = self.num_attention_heads_per_partition
            q = q.view(*q.size()[:-1], num_heads_q, self.q_head_dim)

            k_pos_emb_local = torch.unsqueeze(k_pos_emb, -2)

            if self.config.sequence_parallel and not self.config.enable_chunkpipe:
                kv_compressed = gather_from_sequence_parallel_region(kv_compressed)

            q_len = q.size()[0]
            k_len = k_pos_emb_local.size(0)

            if self.config.apply_rope_fusion:
                cp_rank = self.pg_collection.cp.rank()
                cp_size = self.pg_collection.cp.size()

                if self.use_dsa_sp_first:
                    rope_offset = parallel_state.get_tensor_model_parallel_rank() * q_len
                    q_cos = rotary_pos_cos[rope_offset:rope_offset + q_len]
                    q_sin = rotary_pos_sin[rope_offset:rope_offset + q_len]
                else:
                    q_cos = rotary_pos_cos[0:q_len]
                    q_sin = rotary_pos_sin[0:q_len]

                q_no_pe, q_pos_emb_raw = torch.split(
                    q, [self.config.qk_head_dim, self.config.qk_pos_emb_head_dim], dim=-1
                )

                # Absorb key up projection weight into query
                if self.absorb_backend == "torch":
                    k_up, _ = self._get_kv_up_slices()
                    q_content = torch.einsum("...hd,hdk->...hk", q_no_pe, k_up)
                    # Convert SBHD/THD -> HSD to match fused_rope_permute_cat's expected layout
                    if q_content.ndim == 4:
                        # SBHD [s, b, h, kv_lora_rank] -> HSD [h, s*b, kv_lora_rank]
                        q_content = q_content.permute(2, 0, 1, 3).reshape(
                            num_heads_q, -1, q_content.shape[-1]
                        ).contiguous()
                    else:
                        # THD [t, h, kv_lora_rank] -> HSD [h, t, kv_lora_rank]
                        q_content = q_content.permute(1, 0, 2).contiguous()
                else:
                    q_no_pe_4d = q_no_pe.unsqueeze(1) if q_no_pe.ndim == 3 else q_no_pe
                    q_content, _ = self.linear_kv_up_proj_absorb_q(
                        q_no_pe_4d.permute(2, 0, 1, 3).contiguous(),
                        [q_len * q_no_pe_4d.size(1)] * num_heads_q,
                    )

                query = fused_rope_permute_cat(
                    q_content,
                    q_pos_emb_raw,
                    q_cos,
                    q_sin,
                    cu_seqlens_q,
                    cp_rank,
                    cp_size,
                )

                kv_cached = fused_apply_mla_rope_for_absorb_kv(
                    kv_compressed.unsqueeze(1),
                    k_pos_emb_local,
                    rotary_pos_cos,
                    rotary_pos_sin,
                    self.config.qk_pos_emb_head_dim,
                    self.config.kv_lora_rank,
                    cu_seqlens_kv,
                    cp_rank,
                    cp_size,
                )
                kv_cached = kv_cached.squeeze(1)
            else:
                if inference_context is not None:
                    sequence_start = inference_context.sequence_len_offset
                    sequence_end = sequence_start + q_len
                    rotary_pos_emb_k = rotary_pos_emb_q = rotary_pos_emb[sequence_start:sequence_end]
                elif packed_seq_params is None or self.config.context_parallel_size == 1:
                    if self.use_dsa_sp_first:
                        if packed_seq_params is not None:
                            # Packed-seq + SP-first: pass the full rotary_pos_emb as freqs,
                            # because per-fragment offsets (from shard_packed_cu_seqlens_for_sp_rank)
                            # will index into it to get the correct within-sequence positions.
                            rotary_pos_emb_q = rotary_pos_emb
                        else:
                            rope_offset = parallel_state.get_tensor_model_parallel_rank() * q_len
                            rotary_pos_emb_q = rotary_pos_emb[rope_offset:rope_offset + q_len]
                        rotary_pos_emb_k = rotary_pos_emb[0:k_len]
                    else:
                        rotary_pos_emb_k = rotary_pos_emb_q = rotary_pos_emb[0:q_len]
                else:
                    rotary_pos_emb_k = rotary_pos_emb_q = rotary_pos_emb

                q_no_pe, q_pos_emb = torch.split(
                    q, [self.config.qk_head_dim, self.config.qk_pos_emb_head_dim], dim=-1
                )

                # In SP-first mode, q_pos_emb is sharded along the sequence dim by
                # TP size, but cu_seqlens_q still describes the full packed sequence.
                # Shard cu_seqlens to match the local token range on this TP rank,
                # and pass per-fragment offsets so RoPE uses the correct global positions.
                cu_seqlens_q_rope = cu_seqlens_q
                rope_extra_kwargs_q = {}
                if self.use_dsa_sp_first and cu_seqlens_q is not None:
                    tp_rank = parallel_state.get_tensor_model_parallel_rank()
                    tp_size = parallel_state.get_tensor_model_parallel_world_size()
                    if tp_size > 1:
                        cu_seqlens_q_rope, offsets_q = shard_packed_cu_seqlens_for_sp_rank(
                            cu_seqlens_q,
                            sp_rank=tp_rank,
                            sp_world_size=tp_size,
                        )
                        rope_extra_kwargs_q["offsets"] = offsets_q

                q_pos_emb = apply_rotary_pos_emb(
                    q_pos_emb,
                    rotary_pos_emb_q,
                    config=self.config,
                    cu_seqlens=cu_seqlens_q_rope,
                    mscale=mscale,
                    cp_group=self.pg_collection.cp,
                    **rope_extra_kwargs_q,
                )
                k_pos_emb_local = apply_rotary_pos_emb(
                    k_pos_emb_local,
                    rotary_pos_emb_k,
                    config=self.config,
                    cu_seqlens=cu_seqlens_kv,
                    mscale=mscale,
                    cp_group=self.pg_collection.cp,
                )

                # Chunkpipe: register gradient hooks and cache KV
                if self.config.enable_chunkpipe:
                    def kv_compressed_hook_fn(grad):
                        """Hook function to combine compressed KV gradients from subsequent chunk."""
                        chunks_in_current_sequence = self.config.chunkpipe_backward_microbatch % self.num_chunks_per_seq
                        if chunks_in_current_sequence == self.num_chunks_per_seq - 1:
                            return grad
                        else:
                            grad_from_prev_chunk = self.kv_compressed_cache_grad.pop(chunks_in_current_sequence)
                            return grad + grad_from_prev_chunk
                    
                    def key_pos_emb_hook_fn(grad):
                        """Hook function to combine key position embedding gradients from subsequent chunk."""
                        chunks_in_current_sequence = self.config.chunkpipe_backward_microbatch % self.num_chunks_per_seq
                        if chunks_in_current_sequence == self.num_chunks_per_seq - 1:
                            return grad
                        else:
                            grad_from_prev_chunk = self.key_pos_emb_cache_grad.pop(chunks_in_current_sequence)
                            return grad + grad_from_prev_chunk

                    if self.is_enable_grad_chunkpipe():
                        kv_compressed.register_hook(kv_compressed_hook_fn)
                        k_pos_emb_local.register_hook(key_pos_emb_hook_fn)
                    self.append_chunk_key_value_cache_mla(kv_compressed, k_pos_emb_local)

                    if self.config.sequence_parallel:
                        kv_compressed = gather_from_sequence_parallel_region(kv_compressed)

                kv_cached = torch.cat([kv_compressed, k_pos_emb_local.squeeze(1)], dim=-1)

                # Absorb key up projection weight into query
                if self.absorb_backend == "torch":
                    k_up, _ = self._get_kv_up_slices()
                    q_content = torch.einsum("...hd,hdk->...hk", q_no_pe, k_up)
                    q_content = q_content.squeeze(1) if packed_seq_params is not None else q_content
                else:
                    q_no_pe_4d = q_no_pe.unsqueeze(1) if q_no_pe.ndim == 3 else q_no_pe
                    q_content, _ = self.linear_kv_up_proj_absorb_q(
                        q_no_pe_4d.permute(2, 0, 1, 3).contiguous(),
                        [q_len * q_no_pe_4d.size(1)] * num_heads_q,
                    )
                    q_content = q_content.permute(1, 2, 0, 3).contiguous()
                    q_content = q_content.squeeze(1) if packed_seq_params is not None else q_content

                query = torch.cat([q_content, q_pos_emb], dim=-1)

            key = kv_cached
            value = None
            return query.contiguous(), key.contiguous(), value

        if self.recompute_up_proj:
            self.qkv_up_checkpoint = tensor_parallel.CheckpointWithoutOutput(fp8=self.config.fp8)
            query, key, value = self.qkv_up_checkpoint.checkpoint(
                qkv_up_proj_and_rope_apply_for_dsa,
                q_compressed,
                kv_compressed,
                k_pos_emb,
                rotary_pos_emb,
            )
        else:
            query, key, value = qkv_up_proj_and_rope_apply_for_dsa(
                q_compressed, kv_compressed, k_pos_emb, rotary_pos_emb
            )

        return query, key, value, q_compressed, kv_compressed
