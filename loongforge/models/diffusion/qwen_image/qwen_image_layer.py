# Copyright 2026 The LoongForge Authors.
# SPDX-License-Identifier: Apache-2.0

"""Qwen-Image transformer layer.

Follows the Megatron TransformerLayer contract so it plugs into
TransformerBlock and honors --recompute-* / TP configuration:

  * forward(hidden_states, attention_mask, context, context_mask,
            rotary_pos_emb, timestep_mod, modulate_index, ...)
    where ``hidden_states`` carries the image stream and ``context``
    carries the text stream. Returns ``(hidden_states, context)`` so the
    parent block iterates naturally.

  * Layout is [s, b, h] to match TE fused attention / TransformerBlock.

  * QKV / output / FFN projections use TE Column/Row parallel linears.
    ``img_mod`` / ``txt_mod`` are TE ColumnParallelLinear with
    ``gather_output=True`` so their 6*hidden output is available on every
    TP rank.
"""

from dataclasses import dataclass
from typing import Optional, Tuple, Union

import torch
import torch.nn as nn
from einops import rearrange

from megatron.core.process_groups_config import ProcessGroupCollection
from megatron.core.transformer.enums import AttnMaskType
from megatron.core.transformer.identity_op import IdentityOp
from megatron.core.transformer.spec_utils import ModuleSpec, build_module
from megatron.core.transformer.transformer_layer import (
    TransformerLayer,
    TransformerLayerSubmodules,
)
from megatron.core.utils import divide, make_viewless_tensor

from .qwen_image_modules import QwenFeedForward, RMSNorm, linear_out
from .qwen_image_rope import apply_rotary_emb_qwen, apply_rotary_emb_qwen_fused_sbnd


@dataclass
class QwenImageLayerSubmodules(TransformerLayerSubmodules):
    """Submodules used by the Qwen-Image TransformerLayer wrapper."""

    linear: Union[ModuleSpec, type] = IdentityOp
    column_linear: Union[ModuleSpec, type] = IdentityOp
    row_linear: Union[ModuleSpec, type] = IdentityOp
    core_attention: Union[ModuleSpec, type] = IdentityOp


class QwenDoubleStreamAttention(nn.Module):
    """Joint attention over image + text streams, TP-aware.

    Q/K/V projections are TE ColumnParallelLinear, output projections are
    TE RowParallelLinear, and the fused kernel is TEDotProductAttention.

    Inputs/outputs use the Megatron [s, b, h] convention.
    """

    def __init__(
        self,
        config,
        submodules: QwenImageLayerSubmodules,
        dim_a: int,
        dim_b: int,
        num_heads: int,
        head_dim: int,
        layer_number: int,
        attn_mask_type: AttnMaskType = AttnMaskType.no_mask,
        pg_collection: Optional[ProcessGroupCollection] = None,
    ):
        super().__init__()
        self.config = config
        self.tp_size = config.tensor_model_parallel_size
        self.num_heads = divide(num_heads, self.tp_size)
        self.head_dim = head_dim
        self.attn_mask_type = attn_mask_type
        self.layer_number = layer_number

        def col(in_dim, out_dim, name):
            return build_module(
                submodules.column_linear,
                in_dim,
                out_dim,
                config=config,
                init_method=config.init_method,
                gather_output=False,
                bias=True,
                skip_bias_add=False,
                is_expert=False,
                skip_weight_param_allocation=False,
                tp_comm_buffer_name=name,
            )

        def row(in_dim, out_dim, name):
            return build_module(
                submodules.row_linear,
                in_dim,
                out_dim,
                config=config,
                init_method=config.init_method,
                bias=True,
                input_is_parallel=True,
                skip_bias_add=False,
                is_expert=False,
                tp_comm_buffer_name=name,
            )

        # Image stream Q/K/V (column-parallel; head dim is shared across ranks).
        self.to_q = col(dim_a, dim_a, "img_q")
        self.to_k = col(dim_a, dim_a, "img_k")
        self.to_v = col(dim_a, dim_a, "img_v")
        self.norm_q = RMSNorm(head_dim, eps=1e-6)
        self.norm_k = RMSNorm(head_dim, eps=1e-6)

        # Text stream Q/K/V.
        self.add_q_proj = col(dim_b, dim_b, "txt_q")
        self.add_k_proj = col(dim_b, dim_b, "txt_k")
        self.add_v_proj = col(dim_b, dim_b, "txt_v")
        self.norm_added_q = RMSNorm(head_dim, eps=1e-6)
        self.norm_added_k = RMSNorm(head_dim, eps=1e-6)

        # Output projections (row-parallel).
        self.to_out = nn.Sequential(row(dim_a, dim_a, "img_o"))
        self.to_add_out = row(dim_b, dim_b, "txt_o")

        # Fused TE attention kernel. Inputs must be [s, b, np, hn].
        self.core_attention = build_module(
            submodules.core_attention,
            config=config,
            layer_number=layer_number,
            attn_mask_type=attn_mask_type,
            attention_type="self",
            pg_collection=pg_collection,
        )

    def _project(self, module, x):
        """Project [s, b, h] -> [s, b, np, hn] with TE (Linear, bias) tuple handling."""
        out = linear_out(module, x)
        sq, b, _ = out.shape
        return out.view(sq, b, self.num_heads, self.head_dim)

    def _apply_rope(self, img_q, img_k, txt_q, txt_k, image_rotary_emb):
        """Apply Qwen RoPE.

        If ``config.use_fused_qwen_image_rope`` (or env ``QWEN_IMAGE_USE_FUSED_ROPE=1``)
        is set, dispatches to the fused Triton kernel that operates on the
        SBND layout directly. Otherwise runs the reference PyTorch path
        (permutes to [b, np, s, hn], complex-multiply, permute back).
        """
        if image_rotary_emb is None:
            return img_q, img_k, txt_q, txt_k
        img_freqs, txt_freqs = image_rotary_emb

        import os
        use_fused = getattr(self.config, "use_fused_qwen_image_rope", False) or (
            os.environ.get("QWEN_IMAGE_USE_FUSED_ROPE", "0") == "1"
        )

        if use_fused:
            return (
                apply_rotary_emb_qwen_fused_sbnd(img_q, img_freqs),
                apply_rotary_emb_qwen_fused_sbnd(img_k, img_freqs),
                apply_rotary_emb_qwen_fused_sbnd(txt_q, txt_freqs),
                apply_rotary_emb_qwen_fused_sbnd(txt_k, txt_freqs),
            )

        def apply_reference_rope(x, freqs):
            # [s, b, np, hn] -> [b, np, s, hn] for freqs broadcast, then back.
            y = x.permute(1, 2, 0, 3).contiguous()
            y = apply_rotary_emb_qwen(y, freqs)
            return y.permute(2, 0, 1, 3).contiguous()

        return (
            apply_reference_rope(img_q, img_freqs),
            apply_reference_rope(img_k, img_freqs),
            apply_reference_rope(txt_q, txt_freqs),
            apply_reference_rope(txt_k, txt_freqs),
        )

    def forward(
        self,
        image: torch.Tensor,
        text: torch.Tensor,
        image_rotary_emb: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
        attention_mask: Optional[torch.Tensor] = None,
    ):
        """Run joint image/text attention and return the two updated streams."""
        # image, text: [s, b, h] (already TP-partitioned by ColumnParallelLinear callers)
        img_q = self._project(self.to_q, image)
        img_k = self._project(self.to_k, image)
        img_v = self._project(self.to_v, image)
        txt_q = self._project(self.add_q_proj, text)
        txt_k = self._project(self.add_k_proj, text)
        txt_v = self._project(self.add_v_proj, text)

        # Per-head RMSNorm.
        img_q = self.norm_q(img_q)
        img_k = self.norm_k(img_k)
        txt_q = self.norm_added_q(txt_q)
        txt_k = self.norm_added_k(txt_k)

        # RoPE (applies to Q/K only).
        img_q, img_k, txt_q, txt_k = self._apply_rope(
            img_q, img_k, txt_q, txt_k, image_rotary_emb
        )

        # Joint [s_txt + s_img, b, np, hn].
        seq_txt = txt_q.shape[0]
        joint_q = torch.cat([txt_q, img_q], dim=0)
        joint_k = torch.cat([txt_k, img_k], dim=0)
        joint_v = torch.cat([txt_v, img_v], dim=0)

        core_out = self.core_attention(
            joint_q,
            joint_k,
            joint_v,
            attention_mask,
            attn_mask_type=self.attn_mask_type,
        )
        # TE returns [s, b, np*hn].
        txt_attn = core_out[:seq_txt]
        img_attn = core_out[seq_txt:]

        img_out = linear_out(self.to_out[0], img_attn)
        txt_out = linear_out(self.to_add_out, txt_attn)
        return img_out, txt_out


class QwenImageLayer(TransformerLayer):
    """A single Qwen-Image transformer layer.

    Subclasses Megatron ``TransformerLayer`` so ``TransformerBlock`` can
    manage recompute / TP / FSDP through the standard code paths.
    """

    def __init__(
        self,
        config,
        submodules: QwenImageLayerSubmodules,
        layer_number: int = 1,
        hidden_dropout: float = None,
        pg_collection: Optional[ProcessGroupCollection] = None,
        vp_stage: Optional[int] = None,
        attn_mask_type: AttnMaskType = AttnMaskType.no_mask,
    ):
        super().__init__(
            config=config,
            submodules=submodules,
            layer_number=layer_number,
            hidden_dropout=hidden_dropout,
            pg_collection=pg_collection,
            vp_stage=vp_stage,
        )

        dim = config.hidden_size
        self.dim = dim
        self.num_attention_heads = config.num_attention_heads
        self.attention_head_dim = config.kv_channels

        # 6*hidden modulation projection.
        # TE ColumnParallelLinear does not support gather_output=True, and we
        # need the full 6*dim tensor on every TP rank to slice per-token
        # shift/scale/gate. Keep this as plain nn.Linear (matches Wan's
        # ``time_projection``); it is a single per-layer projection so the
        # extra memory is negligible compared to a full FFN.
        self.img_mod = nn.Sequential(nn.SiLU(), nn.Linear(dim, 6 * dim, bias=True))
        self.txt_mod = nn.Sequential(nn.SiLU(), nn.Linear(dim, 6 * dim, bias=True))

        self.img_norm1 = nn.LayerNorm(dim, elementwise_affine=False, eps=config.norm_epsilon)
        self.img_norm2 = nn.LayerNorm(dim, elementwise_affine=False, eps=config.norm_epsilon)
        self.txt_norm1 = nn.LayerNorm(dim, elementwise_affine=False, eps=config.norm_epsilon)
        self.txt_norm2 = nn.LayerNorm(dim, elementwise_affine=False, eps=config.norm_epsilon)

        self.attn = QwenDoubleStreamAttention(
            config=config,
            submodules=submodules,
            dim_a=dim,
            dim_b=dim,
            num_heads=config.num_attention_heads,
            head_dim=config.kv_channels,
            layer_number=layer_number,
            attn_mask_type=attn_mask_type,
            pg_collection=pg_collection,
        )
        self.img_mlp = QwenFeedForward(
            dim=dim,
            dim_out=dim,
            linear_cls=submodules.linear,
            column_linear_cls=submodules.column_linear,
            row_linear_cls=submodules.row_linear,
            config=config,
        )
        self.txt_mlp = QwenFeedForward(
            dim=dim,
            dim_out=dim,
            linear_cls=submodules.linear,
            column_linear_cls=submodules.column_linear,
            row_linear_cls=submodules.row_linear,
            config=config,
        )

    @staticmethod
    def _modulate(x, mod_params, modulate_index=None):
        """Apply (shift, scale, gate) modulation.

        x: [s, b, h].
        mod_params: [b, 3*h] (regular) or [2*b, 3*h] when zero_cond_t is used.
        modulate_index: None, or [s] selecting between the two modulations
        along the token dim.
        """
        shift, scale, gate = mod_params.chunk(3, dim=-1)
        if modulate_index is not None:
            actual_batch = shift.size(0) // 2
            # Split into the two modulation banks.
            shift_0, shift_1 = shift[:actual_batch], shift[actual_batch:]
            scale_0, scale_1 = scale[:actual_batch], scale[actual_batch:]
            gate_0, gate_1 = gate[:actual_batch], gate[actual_batch:]
            # modulate_index shape [s] -> [s, 1, 1] to broadcast over batch/hidden.
            idx = modulate_index.view(-1, 1, 1)
            # shift_i shape [b, h] -> [1, b, h].
            shift_res = torch.where(idx == 0, shift_0.unsqueeze(0), shift_1.unsqueeze(0))
            scale_res = torch.where(idx == 0, scale_0.unsqueeze(0), scale_1.unsqueeze(0))
            gate_res = torch.where(idx == 0, gate_0.unsqueeze(0), gate_1.unsqueeze(0))
        else:
            # [b, h] -> [1, b, h] broadcast over seq.
            shift_res = shift.unsqueeze(0)
            scale_res = scale.unsqueeze(0)
            gate_res = gate.unsqueeze(0)
        return x * (1 + scale_res) + shift_res, gate_res

    def forward(
        self,
        hidden_states: torch.Tensor = None,
        attention_mask: Optional[torch.Tensor] = None,
        context: Optional[torch.Tensor] = None,
        context_mask: Optional[torch.Tensor] = None,
        rotary_pos_emb: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
        rotary_pos_cos: Optional[torch.Tensor] = None,
        rotary_pos_sin: Optional[torch.Tensor] = None,
        attention_bias: Optional[torch.Tensor] = None,
        inference_context=None,
        packed_seq_params=None,
        sequence_len_offset=None,
        timestep_mod: Optional[torch.Tensor] = None,
        modulate_index: Optional[torch.Tensor] = None,
        **kwargs,
    ):
        """Forward pass matching TransformerBlock's contract.

        ``hidden_states`` is the image stream, ``context`` is the text stream;
        both are in [s, b, h] layout. ``rotary_pos_emb`` is a tuple
        ``(image_freqs, text_freqs)``. ``timestep_mod`` is the shared
        conditioning tensor ``[b, h]`` produced by the parent model
        (may be ``[2*b, h]`` when ``zero_cond_t`` is active).
        """
        if timestep_mod is None:
            raise RuntimeError(
                "QwenImageLayer.forward() requires timestep_mod. "
                "Ensure QwenImageModel passes it to the decoder."
            )

        image = hidden_states
        text = context
        temb = timestep_mod

        img_mod_attn, img_mod_mlp = linear_out(
            self.img_mod[1], self.img_mod[0](temb)
        ).chunk(2, dim=-1)
        # Text stream does not use zero_cond_t modulation split; take the first bank.
        if modulate_index is not None:
            temb_txt = torch.chunk(temb, 2, dim=0)[0]
        else:
            temb_txt = temb
        txt_mod_attn, txt_mod_mlp = linear_out(
            self.txt_mod[1], self.txt_mod[0](temb_txt)
        ).chunk(2, dim=-1)

        # --- Attention block ---
        img_modulated, img_gate = self._modulate(
            self.img_norm1(image), img_mod_attn, modulate_index=modulate_index
        )
        txt_modulated, txt_gate = self._modulate(self.txt_norm1(text), txt_mod_attn)

        img_attn_out, txt_attn_out = self.attn(
            image=img_modulated,
            text=txt_modulated,
            image_rotary_emb=rotary_pos_emb,
            attention_mask=None,
        )
        image = image + img_gate * img_attn_out
        text = text + txt_gate * txt_attn_out

        # --- FFN block ---
        img_modulated_2, img_gate_2 = self._modulate(
            self.img_norm2(image), img_mod_mlp, modulate_index=modulate_index
        )
        txt_modulated_2, txt_gate_2 = self._modulate(self.txt_norm2(text), txt_mod_mlp)
        image = image + img_gate_2 * self.img_mlp(img_modulated_2)
        text = text + txt_gate_2 * self.txt_mlp(txt_modulated_2)

        image = make_viewless_tensor(
            inp=image, requires_grad=image.requires_grad, keep_graph=True
        )
        text = make_viewless_tensor(
            inp=text, requires_grad=text.requires_grad, keep_graph=True
        )
        # Return (hidden_states, context) so TransformerBlock forwards both.
        return image, text
