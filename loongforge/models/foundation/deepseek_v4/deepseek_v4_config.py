# Copyright 2026 The LoongForge Authors.
# SPDX-License-Identifier: Apache-2.0

"""DeepSeek-V4 model config."""

from typing import Iterable, List, Optional, Sequence, Union
from dataclasses import dataclass

from omegaconf import ListConfig

from loongforge.models.common.base_model_config import BaseModelMLAConfig
from loongforge.utils.constants import LanguageModelFamilies


_COMPRESS_RATIO_SYMBOLS = {0: "W", 4: "C", 128: "H"}


def _decoder_layer_is_moe(
    layer_id: int,
    num_layers: int,
    moe_layer_freq: Union[int, Sequence[int], None],
) -> bool:
    """Return whether a decoder layer owns a MoE FFN."""
    if moe_layer_freq is None:
        moe_layer_freq = 1
    if isinstance(moe_layer_freq, (list, tuple, ListConfig)):
        if len(moe_layer_freq) != num_layers:
            raise ValueError(
                f"moe_layer_freq length ({len(moe_layer_freq)}) must equal "
                f"num_layers ({num_layers})."
            )
        value = moe_layer_freq[layer_id]
        if value not in (0, 1, False, True):
            raise ValueError(
                "DeepSeek-V4 hybrid patterns require binary moe_layer_freq entries."
            )
        return bool(value)
    if not isinstance(moe_layer_freq, int) or moe_layer_freq <= 0:
        raise ValueError("moe_layer_freq must be a positive integer or a binary layer list.")
    return layer_id % moe_layer_freq == 0


def _ratio_symbol(ratio: int) -> str:
    try:
        return _COMPRESS_RATIO_SYMBOLS[ratio]
    except KeyError as exc:
        raise ValueError(
            f"Unsupported DeepSeek-V4 compress ratio {ratio}; "
            f"expected one of {sorted(_COMPRESS_RATIO_SYMBOLS)}."
        ) from exc


def build_deepseek_v4_hybrid_layer_pattern(
    csa_compress_ratios: Iterable[int],
    num_layers: int,
    mtp_num_layers: int = 0,
    moe_layer_freq: Union[int, Sequence[int], None] = 1,
) -> str:
    """Build the unified HybridModel pattern for DeepSeek-V4."""
    ratios: List[int] = list(csa_compress_ratios)
    expected_length = num_layers + mtp_num_layers
    if len(ratios) != expected_length:
        raise ValueError(
            f"csa_compress_ratios length ({len(ratios)}) must equal "
            f"num_layers + mtp_num_layers ({expected_length})."
        )

    decoder_parts: List[str] = []
    for layer_id in range(num_layers):
        decoder_parts.append(_ratio_symbol(ratios[layer_id]))
        decoder_parts.append(
            "E" if _decoder_layer_is_moe(layer_id, num_layers, moe_layer_freq) else "-"
        )

    pattern = "".join(decoder_parts)
    if mtp_num_layers <= 0:
        return pattern

    mtp_ratios = ratios[num_layers:]
    if len(mtp_ratios) != mtp_num_layers:
        raise ValueError("MTP compress-ratio count does not match mtp_num_layers.")

    mtp_symbol = _ratio_symbol(mtp_ratios[0])
    if any(_ratio_symbol(ratio) != mtp_symbol for ratio in mtp_ratios[1:]):
        raise ValueError(
            "MCore Hybrid MTP requires the same attention pattern at every MTP depth."
        )
    mtp_pattern = f"{mtp_symbol}E"
    return pattern + "/" + "/".join([mtp_pattern] * mtp_num_layers)


def expand_deepseek_v4_hybrid_compress_ratios(
    pattern: str,
    csa_compress_ratios: Iterable[int],
) -> List[int]:
    """Expand logical V4 attention ratios to one entry per physical Hybrid layer."""
    ratios = iter(csa_compress_ratios)
    expanded: List[int] = []
    for symbol in pattern:
        if symbol == "/":
            continue
        if symbol in _COMPRESS_RATIO_SYMBOLS.values():
            try:
                expanded.append(next(ratios))
            except StopIteration as exc:
                raise ValueError(
                    "csa_compress_ratios does not cover every attention symbol "
                    f"in hybrid pattern {pattern!r}."
                ) from exc
        else:
            expanded.append(0)
    try:
        next(ratios)
    except StopIteration:
        return expanded
    raise ValueError(
        f"csa_compress_ratios contains unused entries for hybrid pattern {pattern!r}."
    )


@dataclass
class DeepseekV4Config(BaseModelMLAConfig):
    """Configuration for DeepSeek-V4 model (MLA + MoE + Hyper-Connections + CSA/HCA).

    Fields beyond BaseModelMLAConfig defaults:
    hc_mult, hc_sinkhorn_iters, hc_eps, swiglu_limit,
    o_groups, o_lora_rank, head_dim, qk_rope_head_dim,
    csa_compress_ratios, csa_compress_rotary_base, csa_window_size,
    index_head_dim, index_n_heads, index_topk,
    moe_n_hash_layers, moe_router_topk_scaling_factor,
    moe_router_score_function, moe_router_enable_expert_bias,
    norm_topk_prob, actual_vocab_size
    """

    # ── Required fields (NO default values, filled by YAML) ──────────────────
    num_layers: int
    hidden_size: int
    ffn_hidden_size: int
    num_attention_heads: int

    # ── MQA / MLA ──────────────────────────────────────────────────────────────
    group_query_attention: bool = True
    num_query_groups: int = 1
    q_lora_rank: int = None
    head_dim: int = None
    qk_rope_head_dim: int = None
    multi_latent_attention: bool = True

    # ── MoE ────────────────────────────────────────────────────────────────────
    num_experts: int = None
    moe_ffn_hidden_size: int = None
    moe_shared_expert_intermediate_size: int = None
    moe_layer_freq: Optional[Union[int, List[int]]] = 1
    moe_n_hash_layers: int = 0
    actual_vocab_size: Optional[int] = None
    moe_router_topk_scaling_factor: Optional[float] = None
    moe_router_score_function: str = "sqrtsoftplus"
    moe_router_enable_expert_bias: bool = True
    norm_topk_prob: bool = True

    # ── Hyper-Connections ──────────────────────────────────────────────────────
    hc_mult: int = 4
    hc_sinkhorn_iters: int = 20
    hc_eps: float = 1e-6

    def __post_init__(self, **kwargs):
        if self.num_moe_experts is None:
            self.num_moe_experts = self.num_experts
        if self.moe_ffn_hidden_size is None:
            self.moe_ffn_hidden_size = self.ffn_hidden_size
        if self.moe_layer_freq is None:
            self.moe_layer_freq = 1

        # Native MCore hash routing uses different public field names than the
        # DeepSeek-V4 HF/config contract.
        self.moe_num_hash_layers = self.moe_n_hash_layers
        if self.actual_vocab_size is not None:
            self.hash_moe_vocab_size = self.actual_vocab_size

        # Map V4-specific HC fields to the MCore 0.20 mHC configuration.
        self.enable_mhc_connections = self.hc_mult > 1
        self.mhc_num_residual_streams = self.hc_mult
        self.mhc_sinkhorn_iterations = self.hc_sinkhorn_iters
        self.mhc_init_gating_factor = self.hc_eps
        self.return_mhc_multistream_for_mtp = True

        # W layers use full qk_pos_emb_dim RoPE. Yarn C/H layers encode their
        # percent through YarnRotaryEmbedding and do not consume this value.
        self.rotary_percent = 1.0

        # Map V4-specific DSA indexer fields to Megatron TransformerConfig fields
        self.dsa_indexer_head_dim = self.index_head_dim
        self.dsa_indexer_n_heads = self.index_n_heads
        self.dsa_indexer_topk = self.index_topk

        # DeepSeek-V4 always uses qk_layernorm
        self.qk_layernorm = True
        # MCore encodes MLA as its own attention mode and rejects the legacy
        # group-query flag even though the YAML keeps it for HF compatibility.
        self.group_query_attention = False

        # DSv4 hybrid derives MLA latent dimensions from the HF-style head dimensions.
        self.qk_head_dim = self.v_head_dim - self.qk_pos_emb_head_dim
        self.kv_lora_rank = self.qk_head_dim
        self.hetereogenous_dist_checkpoint = True

        # Native grouped low-rank output projection fields.
        self.output_projection_groups = self.o_groups
        self.output_projection_lora_rank = self.o_lora_rank

        # Map swiglu_limit to activation_func_clamp_value for shared expert MLP
        self.activation_func_clamp_value = self.swiglu_limit

        # LoongForge's historical DSA backend names are not all valid for
        # MCore's native DSV4 hybrid path. Preserve the reference/safe choices
        # and let MCore choose its default for "auto".
        legacy_dsa_backend = self.dsa_kernel_backend
        dsa_backend_map = {
            "auto": None,
            "sm100": "cudnn",
            "sm80": "none",
            "torch_ref": "none",
            "tilelang": "none",
        }
        if legacy_dsa_backend in dsa_backend_map:
            self.dsa_kernel_backend = dsa_backend_map[legacy_dsa_backend]
        elif legacy_dsa_backend not in (None, "none", "cudnn"):
            raise ValueError(
                f"Unsupported dsa_kernel_backend for DeepSeek-V4: {legacy_dsa_backend!r}"
            )

        super().__post_init__(**kwargs)
        self.rotary_percent = 1.0

        assert self.num_moe_experts and self.num_moe_experts > 0, "DeepSeek-V4 requires MoE."
        assert self.multi_latent_attention, "DeepSeek-V4 requires multi_latent_attention."
        assert self.csa_compress_ratios is not None, "csa_compress_ratios must be set."
        mtp_layers = self.mtp_num_layers or 0
        expected_len = self.num_layers + mtp_layers
        assert len(self.csa_compress_ratios) == expected_len, (
            f"csa_compress_ratios length ({len(self.csa_compress_ratios)}) must equal "
            f"num_layers + mtp_num_layers ({self.num_layers} + {mtp_layers} = {expected_len})."
        )
        assert all(ratio in [0, 4, 128] for ratio in self.csa_compress_ratios), (
            "csa_compress_ratios must contain only 0, 4, or 128."
        )
        assert not getattr(self, "qk_clip", False), (
            "QK clipping is not supported with DeepSeek-V4 hybrid attention."
        )
        assert not getattr(self, "mla_down_proj_fusion", False), (
            "MLA down projection fusion must be disabled for DeepSeek-V4 hybrid attention."
        )

        if self.moe_n_hash_layers > 0:
            assert self.actual_vocab_size is not None, (
                "actual_vocab_size must be set when moe_n_hash_layers > 0."
            )
            assert self.hash_moe_vocab_size == self.actual_vocab_size, (
                "hash_moe_vocab_size must match actual_vocab_size for DeepSeek-V4."
            )
            assert self.moe_num_hash_layers == self.moe_n_hash_layers, (
                "moe_num_hash_layers must match moe_n_hash_layers for DeepSeek-V4."
            )

        # HybridModel uses num_layers for physical main-pattern execution and
        # metric slot sizing. Preserve the HF/checkpoint logical layer count for
        # ratio derivation and checkpoint conversion.
        logical_num_layers = self.num_layers
        hybrid_layer_pattern = build_deepseek_v4_hybrid_layer_pattern(
            csa_compress_ratios=self.csa_compress_ratios,
            num_layers=logical_num_layers,
            mtp_num_layers=mtp_layers,
            moe_layer_freq=self.moe_layer_freq,
        )
        if (
            self.hybrid_layer_pattern is not None
            and self.hybrid_layer_pattern != hybrid_layer_pattern
        ):
            raise ValueError(
                "hybrid_layer_pattern does not match csa_compress_ratios and "
                f"moe_layer_freq: configured {self.hybrid_layer_pattern!r}, "
                f"derived {hybrid_layer_pattern!r}."
            )
        self.hybrid_layer_pattern = hybrid_layer_pattern
        self.csa_compress_ratios = expand_deepseek_v4_hybrid_compress_ratios(
            hybrid_layer_pattern,
            self.csa_compress_ratios,
        )
        self.num_attention_layers = logical_num_layers
        self.num_layers = len(hybrid_layer_pattern.split("/")[0])

    # ── CSA / HCA (Compressed Sparse Attention) ──────────────────────────────
    csa_window_size: int = 128
    csa_compress_ratios: Optional[List[int]] = None
    csa_compress_rotary_base: float = 160000.0
    csa_dense_mode: bool = False

    # ── Lightning Indexer ──────────────────────────────────────────────────────
    index_head_dim: int = 128
    index_n_heads: int = 64
    index_topk: int = 512

    # ── Grouped Output Projection ──────────────────────────────────────────────
    o_groups: int = 8
    o_lora_rank: int = 1024

    # ── SwiGLU Clamping ──────────────────────────────────────────────────────
    swiglu_limit: float = 10.0

    # ── RoPE ──────────────────────────────────────────────────────────────────
    position_embedding_type: str = "rope"
    add_position_embedding: bool = False
    rotary_interleaved: bool = False
    apply_rope_fusion: bool = True

    # ── Normalization ─────────────────────────────────────────────────────────
    normalization: str = "RMSNorm"

    # ── FFN ────────────────────────────────────────────────────────────────────
    swiglu: bool = True

    # ── Dropout ────────────────────────────────────────────────────────────────
    attention_dropout: float = 0
    hidden_dropout: float = 0

    # ── Linear bias ────────────────────────────────────────────────────────────
    add_bias_linear: bool = False
    add_qkv_bias: bool = False
    qk_layernorm: bool = True

    # ── Vocab / Embedding ──────────────────────────────────────────────────────
    untie_embeddings_and_output_weights: bool = True
    vocab_size_in_config_file: int = None
    make_vocab_size_divisible_by: int = 128

    # ── MTP ────────────────────────────────────────────────────────────────────
    mtp_num_layers: int = 0
    mtp_loss_coef: float = 0.1
    hybrid_layer_pattern: Optional[str] = None
    num_attention_layers: Optional[int] = None

    model_spec = [
        "loongforge.models.foundation.deepseek_v4.deepseek_v4_model",
        "get_deepseek_v4_hybrid_stack_spec",
    ]
    model_type = LanguageModelFamilies.DEEPSEEK_V4
    model_backend = "hybrid"
