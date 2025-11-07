"""attention module for aiak"""

import torch

from megatron.core.transformer.attention import SelfAttention
from megatron.core.transformer.transformer_config import TransformerConfig

try:
    import transformer_engine

    HAVE_TE = True
except ImportError:
    HAVE_TE = False

if HAVE_TE:
    from megatron.core.extensions.transformer_engine import SplitAlongDim
else:
    SplitAlongDim = None


class CogvlmSelfAttention(SelfAttention):
    """
    SelfAttention for cogvlm
    """

    def __init__(self, config: TransformerConfig, **kwargs):
        super().__init__(config, **kwargs)

    def forward(
        self,
        hidden_states,
        attention_mask,
        attn_mask_type=None,
        key_value_states=None,
        inference_params=None,
        rotary_pos_emb=None,
        rotary_pos_cos=None,
        rotary_pos_sin=None,
        attention_bias=None,
        packed_seq_params=None,
        sequence_len_offset=None,
        **kwargs,
    ):
        """
        forward pass for self attention
        """
        # hidden_states: [sq, b, h]

        # For self attention we just duplicate the rotary_pos_emb if it isn't already
        if rotary_pos_emb is not None and not isinstance(rotary_pos_emb, tuple):
            rotary_pos_emb = (rotary_pos_emb,) * 2

        # =====================
        # Query, Key, and Value
        # =====================
        # Get the query, key and value tensors based on the type of attention -
        # self or cross attn.

        query, key, value = self.get_query_key_value_tensors(
            hidden_states,
            key_value_states,
            **kwargs,
        )

        # ===================================================
        # Adjust key, value, and rotary_pos_emb for inference
        # ===================================================
        query, key, value, rotary_pos_emb, attn_mask_type = (
            self._adjust_key_value_for_inference(
                inference_params,
                query,
                key,
                value,
                rotary_pos_emb,
                rotary_pos_cos,
                rotary_pos_sin,
                attn_mask_type,
                sequence_len_offset,
            )
        )

        if packed_seq_params is not None:
            query = query.squeeze(1)
            key = key.squeeze(1)
            value = value.squeeze(1)

        # ================================================
        # relative positional embedding (rotary embedding)
        # ================================================
        if rotary_pos_emb is not None:
            q_pos_emb, k_pos_emb = rotary_pos_emb

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

            query = self.apply_rotary_fn(
                query,
                q_pos_emb,
                config=self.config,
                cu_seqlens=cu_seqlens_q,
                **kwargs,
            )
            key = self.apply_rotary_fn(
                key,
                k_pos_emb,
                config=self.config,
                cu_seqlens=cu_seqlens_kv,
                **kwargs,
            )

            # TODO, can apply positional embedding to value_layer so it has
            # absolute positional embedding.
            # otherwise, only relative positional embedding takes effect
            # value_layer = self.apply_rotary_fn(value_layer, k_pos_emb)

        # ==================================
        # core attention computation
        # ==================================
        if self.checkpoint_core_attention and self.training:
            core_attn_out = self._checkpointed_attention_forward(
                query,
                key,
                value,
                attention_mask,
                attn_mask_type=attn_mask_type,
                packed_seq_params=packed_seq_params,
            )
        else:
            core_attn_out = self.core_attention(
                query,
                key,
                value,
                attention_mask,
                attn_mask_type=attn_mask_type,
                packed_seq_params=packed_seq_params,
            )

        if packed_seq_params is not None:
            # reshape to same output shape as unpacked case
            # (t, np, hn) -> (t, b=1, h=np*hn)
            # t is the pack size = sum (sq_i)
            # note that batch is a dummy dimension in the packed case
            core_attn_out = core_attn_out.reshape(core_attn_out.size(0), 1, -1)

        # =================
        # Output. [sq, b, h]
        # =================
        output, bias = self.linear_proj(core_attn_out, **kwargs)

        return output, bias

    def get_query_key_value_tensors(
        self, hidden_states, key_value_states=None, **kwargs
    ):
        """
        Derives `query`, `key` and `value` tensors from `hidden_states`.
        """
        # Attention heads [sq, b, h] --> [sq, b, ng * (np/ng + 2) * hn)]
        mixed_qkv, _ = self.linear_qkv(hidden_states, **kwargs)

        # [sq, b, hp] --> [sq, b, ng, (np/ng + 2) * hn]
        new_tensor_shape = mixed_qkv.size()[:-1] + (
            self.num_query_groups_per_partition,
            (
                (
                    self.num_attention_heads_per_partition
                    // self.num_query_groups_per_partition
                    + 2
                )
                * self.hidden_size_per_attention_head
            ),
        )
        mixed_qkv = mixed_qkv.view(*new_tensor_shape)

        split_arg_list = [
            (
                self.num_attention_heads_per_partition
                // self.num_query_groups_per_partition
                * self.hidden_size_per_attention_head
            ),
            self.hidden_size_per_attention_head,
            self.hidden_size_per_attention_head,
        ]

        if SplitAlongDim is not None:

            # [sq, b, ng, (np/ng + 2) * hn] --> [sq, b, ng, np/ng * hn], [sq, b, ng, hn], [sq, b, ng, hn]
            (query, key, value) = SplitAlongDim(
                mixed_qkv,
                3,
                split_arg_list,
            )
        else:

            # [sq, b, ng, (np/ng + 2) * hn] --> [sq, b, ng, np/ng * hn], [sq, b, ng, hn], [sq, b, ng, hn]
            (query, key, value) = torch.split(
                mixed_qkv,
                split_arg_list,
                dim=3,
            )

        # [sq, b, ng, np/ng * hn] -> [sq, b, np, hn]
        query = query.reshape(
            query.size(0), query.size(1), -1, self.hidden_size_per_attention_head
        )

        if self.q_layernorm is not None:
            query = self.q_layernorm(query)

        if self.k_layernorm is not None:
            key = self.k_layernorm(key)

        if self.config.test_mode:
            self.run_realtime_tests()

        return query, key, value
