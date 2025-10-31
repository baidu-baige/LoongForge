"""wan2.1 model"""

import torch
import torch.nn as nn
from torch import Tensor

import torch.nn.functional as F
import math
from typing import Tuple, Dict, Literal, Optional
from einops import rearrange
from megatron.core.models.common.vision_module.vision_module import VisionModule
from aiak_training_omni.models.custom.transformer.vision.stdit_transformer_config import (
    StditTransformerConfig,
)
from megatron.core.transformer.spec_utils import ModuleSpec
from megatron.core.transformer.transformer_block import TransformerBlock
from megatron.core.models.common.embeddings.rotary_pos_embedding import RotaryEmbedding
from aiak_training_omni.models.custom.transformer.vision.stdit_model_embedding import (
    TimestepEmbedder,
)
from aiak_training_omni.models.stdit.communications import (
    split_forward_gather_backward,
    gather_forward_split_backward,
)
from megatron.core.parallel_state import (
    get_context_parallel_group,
)
from torch.cuda.amp import autocast as autocast
from megatron.training import print_rank_0
import re


def modulate(x: torch.Tensor, shift: torch.Tensor, scale: torch.Tensor):
    """
    对输入张量x进行调制。

    Args:
        x (torch.Tensor): 输入张量。
        shift (torch.Tensor): 平移参数。
        scale (torch.Tensor): 缩放参数。

    Returns:
        torch.Tensor: 调制后的张量。

    """
    return x * (1 + scale) + shift


def sinusoidal_embedding_1d(dim, position):
    """
    计算一维正弦嵌入表示。

    Args:
        dim (int): 嵌入的维度。
        position (torch.Tensor): 位置索引，大小为 (N,)。

    Returns:
        torch.Tensor: 一维正弦嵌入表示，大小为 (N, dim)。

    """
    sinusoid = torch.outer(
        position.type(torch.float64),
        torch.pow(
            10000,
            -torch.arange(dim // 2, dtype=torch.float64, device=position.device).div(
                dim // 2
            ),
        ),
    )
    x = torch.cat([torch.cos(sinusoid), torch.sin(sinusoid)], dim=1)
    return x.to(position.dtype)


def precompute_freqs_cis_3d(dim: int, end: int = 1024, theta: float = 10000.0):
    """
    3d rope计算
    """
    # 3d rope precompute
    f_freqs_cis = precompute_freqs_cis(dim - 2 * (dim // 3), end, theta)
    h_freqs_cis = precompute_freqs_cis(dim // 3, end, theta)
    w_freqs_cis = precompute_freqs_cis(dim // 3, end, theta)
    return f_freqs_cis, h_freqs_cis, w_freqs_cis


def precompute_freqs_cis(dim: int, end: int = 1024, theta: float = 10000.0):
    """
    1d rope precompute

    """

    freqs = 1.0 / (theta ** (torch.arange(0, dim, 2)[: (dim // 2)].double() / dim))
    freqs = torch.outer(torch.arange(end, device=freqs.device), freqs)
    freqs_cis = torch.polar(torch.ones_like(freqs), freqs)  # complex64
    return freqs_cis


def rope_apply(x, freqs, num_heads):
    """
    计算 rope

    """
    x = rearrange(x, "b s (n d) -> b s n d", n=num_heads)
    x_out = torch.view_as_complex(
        x.to(torch.float64).reshape(x.shape[0], x.shape[1], x.shape[2], -1, 2)
    )
    x_out = torch.view_as_real(x_out * freqs).flatten(2)
    return x_out.to(x.dtype)


class MLP(torch.nn.Module):
    """
    MLP class 定义
    """

    def __init__(self, in_dim, out_dim, has_pos_emb=False):
        """
        初始化方法
        """
        super().__init__()
        self.proj = torch.nn.Sequential(
            nn.LayerNorm(in_dim),
            nn.Linear(in_dim, in_dim),
            nn.GELU(),
            nn.Linear(in_dim, out_dim),
            nn.LayerNorm(out_dim),
        )
        self.has_pos_emb = has_pos_emb
        if has_pos_emb:
            self.emb_pos = torch.nn.Parameter(torch.zeros((1, 514, 1280)))

    def forward(self, x):
        """
        前向函数
        """
        if self.has_pos_emb:
            x = x + self.emb_pos.to(dtype=x.dtype, device=x.device)
        return self.proj(x)


class Head(nn.Module):
    """
    Head class
    """

    def __init__(
        self, dim: int, out_dim: int, patch_size: Tuple[int, int, int], eps: float
    ):
        """
        初始化函数
        """
        super().__init__()
        self.dim = dim
        self.patch_size = patch_size
        self.norm = nn.LayerNorm(dim, eps=eps, elementwise_affine=False)
        self.head = nn.Linear(dim, out_dim * math.prod(patch_size))
        self.modulation = nn.Parameter(torch.randn(1, 2, dim) / dim**0.5)

    def forward(self, x, t_mod):
        """
        前向函数
        """
        shift, scale = (
            self.modulation.to(dtype=t_mod.dtype, device=t_mod.device) + t_mod
        ).chunk(2, dim=1)
        x = self.head(self.norm(x) * (1 + scale) + shift)
        return x


class WanModel(VisionModule):
    """
    Wan Transformer language model
    """

    def __init__(
        self,
        config: StditTransformerConfig,
        transformer_layer_spec: ModuleSpec,
        vocab_size: int,
        max_sequence_length: int,
        ##
        dim: int,
        in_dim: int,
        ffn_dim: int,
        out_dim: int,
        text_dim: int,
        freq_dim: int,
        eps: float,
        patch_size: Tuple[int, int, int],
        num_heads: int,
        num_layers: int,
        has_image_input: bool,
        has_image_pos_emb: bool = False,
        ##
        pre_process: bool = True,
        post_process: bool = True,
        fp16_lm_cross_entropy: bool = False,
        parallel_output: bool = True,
        share_embeddings_and_output_weights: bool = True,
        position_embedding_type: Literal["learned_absolute", "rope"] = "rope",
        rotary_percent: float = 1.0,
        rotary_base: int = 10000,
        seq_len_interpolation_factor: Optional[float] = None,
    ):
        super().__init__(config=config)
        self.pre_process = pre_process
        self.post_process = post_process

        self.dim = dim
        self.freq_dim = freq_dim
        self.has_image_input = has_image_input
        self.patch_size = patch_size
        self.share_embeddings_and_output_weights = share_embeddings_and_output_weights
        self.patch_embedding = nn.Conv3d(
            in_dim, dim, kernel_size=patch_size, stride=patch_size
        )
        self.text_embedding = nn.Sequential(
            nn.Linear(text_dim, dim), nn.GELU(approximate="tanh"), nn.Linear(dim, dim)
        )
        self.time_embedding = nn.Sequential(
            nn.Linear(freq_dim, dim), nn.SiLU(), nn.Linear(dim, dim)
        )

        self.head = Head(dim, out_dim, patch_size, eps)
        head_dim = dim // num_heads
        self.freqs = precompute_freqs_cis_3d(head_dim)

        if has_image_input:
            self.img_emb = MLP(
                1280, dim, has_pos_emb=has_image_pos_emb
            )  # clip_feature_dim = 1280
        self.has_image_pos_emb = has_image_pos_emb
        # timestep embedding
        self.time_projection = nn.Sequential(
            nn.SiLU(), nn.Linear(config.hidden_size, 6 * config.hidden_size, bias=True)
        )

        # Transformer.
        self.decoder = TransformerBlock(
            config=self.config,
            spec=transformer_layer_spec,
            pre_process=self.pre_process,
            post_process=self.post_process,
            post_layer_norm=False,
        )

        # if self.post_process:
        #     self.head = Head(dim, out_dim, patch_size, eps)

    def patchify(self, x: torch.Tensor):
        """
        将输入张量进行patchify操作。
        """
        x = self.patch_embedding(x)
        grid_size = x.shape[2:]
        x = rearrange(x, "b c f h w -> b (f h w) c").contiguous()
        return x, grid_size  # x, grid_size: (f, h, w)

    def unpatchify(self, x: torch.Tensor, grid_size: torch.Tensor):
        """
        将分块张量重新组合成完整的张量。
        """

        return rearrange(
            x,
            "b (f h w) (x y z c) -> b c (f x) (h y) (w z)",
            f=grid_size[0],
            h=grid_size[1],
            w=grid_size[2],
            x=self.patch_size[0],
            y=self.patch_size[1],
            z=self.patch_size[2],
        )

    def pad_image(self, clip):
        """image padding"""
        seq = clip.shape[1]
        pad_num = (
            self.config.context_parallel_size - seq % self.config.context_parallel_size
        )
        pad_num = pad_num % self.config.context_parallel_size
        if pad_num != 0:
            pad = torch.zeros(clip.shape[0], pad_num, clip.shape[2]).to(clip.device)
            clip = torch.cat([clip, pad], dim=1)
        return pad_num, clip

    def reorganize_x(self, hidden_state, context, timestep, t_s, clip_emb=None):
        """reorganize hidden_state, context, timestep, t_s for cp parallel"""
        cp = self.config.context_parallel_size
        assert context.shape[0] % cp == 0
        hidden_states = torch.chunk(hidden_state, cp, dim=0)
        contexts = torch.chunk(context, cp, dim=0)
        cated_x = []
        if clip_emb is not None:
            clip_embs = torch.chunk(clip_emb, cp, dim=0)
            for i in range(len(contexts)):
                cated_x.append(
                    torch.cat(
                        [hidden_states[i], clip_embs[i], contexts[i], timestep, t_s],
                        dim=0,
                    )
                )
        else:
            for i in range(len(contexts)):
                cated_x.append(
                    torch.cat([hidden_states[i], contexts[i], timestep, t_s], dim=0)
                )
        x = torch.cat(cated_x, dim=0)
        return x

    def forward(
        self,
        x: torch.Tensor,
        timestep: torch.Tensor,
        context: torch.Tensor,
        clip_feature: Optional[torch.Tensor] = None,
        y: Optional[torch.Tensor] = None,
        use_gradient_checkpointing: bool = False,
        use_gradient_checkpointing_offload: bool = False,
        **kwargs,
    ):
        """
        wan 的前向
        """
        f, h, w = (21, 30, 52)
        freqs = torch.cat(
            [
                self.freqs[0][:f].view(f, 1, 1, -1).expand(f, h, w, -1),
                self.freqs[1][:h].view(1, h, 1, -1).expand(f, h, w, -1),
                self.freqs[2][:w].view(1, 1, w, -1).expand(f, h, w, -1),
            ],
            dim=-1,
        ).reshape(f * h * w, 1, -1)

        if self.has_image_input:
            pad_num, clip_feature = self.pad_image(clip_feature)

        if self.pre_process:
            t = self.time_embedding(sinusoidal_embedding_1d(self.freq_dim, timestep))
            t_s = t.unsqueeze(0)

            timestep = self.time_projection(t).unflatten(1, (6, self.dim))
            context = self.text_embedding(context)
            if self.has_image_input:
                x = torch.cat([x, y], dim=1)  # (b, c_x + c_y, f, h, w)
                with autocast(dtype=torch.bfloat16):
                    clip_embdding = self.img_emb(clip_feature)
                    clip_embdding = rearrange(
                        clip_embdding, f"B S C ->S B C"
                    ).contiguous()

            x, (f, h, w) = self.patchify(x)
            assert (f, h, w) == (21, 30, 52)

            x = rearrange(x, f"B S C ->S B C").contiguous()
            timestep = rearrange(timestep, f"B S C ->S B C").contiguous()
            context = rearrange(context, f"B S C ->S B C").contiguous()
            # 拼接context
            if self.has_image_input:
                x = self.reorganize_x(x, context, timestep, t_s, clip_embdding)
            else:
                x = self.reorganize_x(x, context, timestep, t_s)
            ## context 并行
            if self.config.context_parallel_size > 1:
                x = split_forward_gather_backward(
                    x, get_context_parallel_group(), dim=0, grad_scale="down"
                )
        else:
            x = None
        extra_block_kwargs = {}
        x = self.decoder(
            hidden_states=x,
            attention_mask=None,
            context=None,
            context_mask=None,
            inference_params=None,
            packed_seq_params=None,
            rotary_pos_emb=freqs,
            **(extra_block_kwargs or {}),
        )

        if not self.post_process:
            return x
        if self.config.context_parallel_size > 1:
            x = gather_forward_split_backward(
                x, get_context_parallel_group(), dim=0, grad_scale="up"
            )
        t = x[-1:, :, :].squeeze(1).to(torch.bfloat16)
        vido_length = self.config.max_video_length

        ##cat latent
        chunk_size = x.shape[0] // self.config.context_parallel_size
        sp_latent_len = vido_length // self.config.context_parallel_size
        chunks = []
        for i in range(self.config.context_parallel_size):
            chunk = x[i * chunk_size : (i + 1) * chunk_size]
            front_x = chunk[:sp_latent_len]
            chunks.append(front_x)
        x = torch.cat(chunks, dim=0).to(torch.bfloat16)

        x = rearrange(x, f"S B C ->B S C").contiguous()
        x = self.head(x, t)
        x = self.unpatchify(x, (f, h, w))
        return x

    def set_input_tensor(self, input_tensor: Tensor) -> None:
        """Sets input tensor to the model.

        See megatron.model.transformer.set_input_tensor()

        Args:
            input_tensor (Tensor): Sets the input tensor for the model.
        """
        # This is usually handled in schedules.py but some inference code still
        # gives us non-lists or None
        if not isinstance(input_tensor, list):
            input_tensor = [input_tensor]
        assert len(input_tensor) == 1, "input_tensor should only be length 1"
        self.decoder.set_input_tensor(input_tensor[0])


def convert_state_dict_from_hg_t2v_1p3b(state_dict):
    """
    HuggingFace state_dict转为megatron 格式的 state_dict。

    """
    num_layers = 30
    inside_blk_replace_dict = {
        "blocks.0.modulation": "decoder.layers.0.modulation",
        "blocks.0.ffn.0.weight": "decoder.layers.0.ffn.0.weight",
        "blocks.0.ffn.0.bias": "decoder.layers.0.ffn.0.bias",
        "blocks.0.ffn.2.weight": "decoder.layers.0.ffn.2.weight",
        "blocks.0.ffn.2.bias": "decoder.layers.0.ffn.2.bias",
        "blocks.0.norm3.weight": "decoder.layers.0.norm3.weight",
        "blocks.0.norm3.bias": "decoder.layers.0.norm3.bias",
    }
    outside_blk_replace_dict = [
        "patch_embedding.weight",
        "patch_embedding.bias",
        "text_embedding.0.weight",
        "text_embedding.0.bias",
        "text_embedding.2.weight",
        "text_embedding.2.bias",
        "time_embedding.0.weight",
        "time_embedding.0.bias",
        "time_embedding.2.weight",
        "time_embedding.2.bias",
        "time_projection.1.weight",
        "time_projection.1.bias",
        "head.modulation",
        "head.head.weight",
        "head.head.bias",
    ]

    new_state_dict = {}
    for key in outside_blk_replace_dict:
        new_state_dict[key] = state_dict[key]

    for i in range(num_layers):
        ## self_attention qkv合并
        q_weight = state_dict["blocks." + str(i) + ".self_attn.q.weight"]
        q_bias = state_dict["blocks." + str(i) + ".self_attn.q.bias"]
        k_weight = state_dict["blocks." + str(i) + ".self_attn.k.weight"]
        k_bias = state_dict["blocks." + str(i) + ".self_attn.k.bias"]
        v_weight = state_dict["blocks." + str(i) + ".self_attn.v.weight"]
        v_bias = state_dict["blocks." + str(i) + ".self_attn.v.bias"]
        # convert to huggingface linear_qkv
        concat_qkv_weight = torch.concat([q_weight, k_weight, v_weight], dim=0)
        concat_qkv_weight = rearrange(
            concat_qkv_weight, "(R N D) H -> (N R D) H", R=3, N=12, D=128, H=1536
        )
        concat_qkv_bias = torch.concat([q_bias, k_bias, v_bias], dim=0)
        concat_qkv_bias = rearrange(
            concat_qkv_bias, "(R N D H) -> (N R D H)", R=3, N=12, D=128, H=1
        )

        new_state_dict["decoder.layers." + str(i) + ".self_attn.linear_qkv.weight"] = (
            concat_qkv_weight
        )
        new_state_dict["decoder.layers." + str(i) + ".self_attn.linear_qkv.bias"] = (
            concat_qkv_bias
        )
        # 转 o
        o_weight = state_dict["blocks." + str(i) + ".self_attn.o.weight"]
        o_weight = rearrange(
            o_weight, "(R N D) H -> (N R D) H", R=1, N=12, D=128, H=1536
        )
        o_bias = state_dict["blocks." + str(i) + ".self_attn.o.bias"]
        o_bias = rearrange(o_bias, "(R N D H) -> (N R D H)", R=1, N=12, D=128, H=1)

        new_state_dict["decoder.layers." + str(i) + ".self_attn.linear_proj.bias"] = (
            o_bias
        )
        new_state_dict["decoder.layers." + str(i) + ".self_attn.linear_proj.weight"] = (
            o_weight
        )

        ## cross_attention kv 合并
        cross_q_w = state_dict["blocks." + str(i) + ".cross_attn.q.weight"]
        cross_q_w = rearrange(
            cross_q_w, "(R N D) H -> (N R D) H", R=1, N=12, D=128, H=1536
        )
        cross_q_b = state_dict["blocks." + str(i) + ".cross_attn.q.bias"]
        cross_q_b = rearrange(
            cross_q_b, "(R N D H) -> (N R D H)", R=1, N=12, D=128, H=1
        )
        new_state_dict["decoder.layers." + str(i) + ".cross_attn.linear_q.weight"] = (
            cross_q_w
        )
        new_state_dict["decoder.layers." + str(i) + ".cross_attn.linear_q.bias"] = (
            cross_q_b
        )

        cross_attn_k_weight = state_dict["blocks." + str(i) + ".cross_attn.k.weight"]
        cross_attn_k_bias = state_dict["blocks." + str(i) + ".cross_attn.k.bias"]
        cross_attn_v_weight = state_dict["blocks." + str(i) + ".cross_attn.v.weight"]
        cross_attn_v_bias = state_dict["blocks." + str(i) + ".cross_attn.v.bias"]
        concat_kv_weight = torch.concat(
            [cross_attn_k_weight, cross_attn_v_weight], dim=0
        )
        concat_kv_weight = rearrange(
            concat_kv_weight, "(R N D) H -> (N R D) H", R=2, N=12, D=128, H=1536
        )
        concat_kv_bias = torch.concat([cross_attn_k_bias, cross_attn_v_bias], dim=0)
        concat_kv_bias = rearrange(
            concat_kv_bias, "(R N D H) -> (N R D H)", R=2, N=12, D=128, H=1
        )
        new_state_dict["decoder.layers." + str(i) + ".cross_attn.linear_kv.weight"] = (
            concat_kv_weight
        )
        new_state_dict["decoder.layers." + str(i) + ".cross_attn.linear_kv.bias"] = (
            concat_kv_bias
        )

        cross_o_weight = state_dict["blocks." + str(i) + ".cross_attn.o.weight"]
        cross_o_weight = rearrange(
            cross_o_weight, "(R N D) H -> (N R D) H", R=1, N=12, D=128, H=1536
        )
        cross_o_bias = state_dict["blocks." + str(i) + ".cross_attn.o.bias"]
        cross_o_bias = rearrange(
            cross_o_bias, "(R N D H) -> (N R D H)", R=1, N=12, D=128, H=1
        )
        new_state_dict[
            "decoder.layers." + str(i) + ".cross_attn.linear_proj.weight"
        ] = cross_o_weight
        new_state_dict["decoder.layers." + str(i) + ".cross_attn.linear_proj.bias"] = (
            cross_o_bias
        )

        ## 一般性替换
        for key, value in inside_blk_replace_dict.items():
            key = key.replace("blocks.0", "blocks." + str(i))
            value = value.replace("decoder.layers.0", "decoder.layers." + str(i))
            new_state_dict[value] = state_dict[key]

    return new_state_dict


def convert_state_dict_from_hg_i2v_14b(state_dict):
    """HuggingFace state_dict转为megatron格式的state_dict。"""
    num_layers = 40
    inside_blk_replace_dict = {
        # "blocks.0.modulation": "decoder.layers.0.modulation",
        "blocks.0.ffn.0.weight": "decoder.layers.0.ffn.0.weight",
        "blocks.0.ffn.0.bias": "decoder.layers.0.ffn.0.bias",
        "blocks.0.ffn.2.weight": "decoder.layers.0.ffn.2.weight",
        "blocks.0.ffn.2.bias": "decoder.layers.0.ffn.2.bias",
        "blocks.0.norm3.weight": "decoder.layers.0.norm3.weight",
        "blocks.0.norm3.bias": "decoder.layers.0.norm3.bias",
        "blocks.0.self_attn.norm_q.weight": "decoder.layers.0.self_attn.q_layernorm.weight",
        "blocks.0.self_attn.norm_k.weight": "decoder.layers.0.self_attn.k_layernorm.weight",
        "blocks.0.cross_attn.norm_q.weight": "decoder.layers.0.cross_attn.q_layernorm.weight",
        "blocks.0.cross_attn.norm_k.weight": "decoder.layers.0.cross_attn.k_layernorm.weight",
        "blocks.0.cross_attn.norm_k_img.weight": "decoder.layers.0.cross_attn.k_img_layernorm.weight",
    }
    outside_blk_replace_dict = [
        "patch_embedding.weight",
        "patch_embedding.bias",
        "text_embedding.0.weight",
        "text_embedding.0.bias",
        "text_embedding.2.weight",
        "text_embedding.2.bias",
        "time_embedding.0.weight",
        "time_embedding.0.bias",
        "time_embedding.2.weight",
        "time_embedding.2.bias",
        "time_projection.1.weight",
        "time_projection.1.bias",
        "head.modulation",
        "head.head.weight",
        "head.head.bias",
        "img_emb.proj.0.weight",
        "img_emb.proj.0.bias",
        "img_emb.proj.1.weight",
        "img_emb.proj.1.bias",
        "img_emb.proj.3.weight",
        "img_emb.proj.3.bias",
        "img_emb.proj.4.weight",
        "img_emb.proj.4.bias",
    ]

    new_state_dict = {}
    for key in outside_blk_replace_dict:
        new_state_dict[key] = state_dict[key]

    for i in range(num_layers):
        ## self_attention qkv合并
        q_weight = state_dict["blocks." + str(i) + ".self_attn.q.weight"]
        q_bias = state_dict["blocks." + str(i) + ".self_attn.q.bias"]
        k_weight = state_dict["blocks." + str(i) + ".self_attn.k.weight"]
        k_bias = state_dict["blocks." + str(i) + ".self_attn.k.bias"]
        v_weight = state_dict["blocks." + str(i) + ".self_attn.v.weight"]
        v_bias = state_dict["blocks." + str(i) + ".self_attn.v.bias"]
        # convert to huggingface linear_qkv
        concat_qkv_weight = torch.concat([q_weight, k_weight, v_weight], dim=0)
        concat_qkv_weight = rearrange(
            concat_qkv_weight, "(R N D) H -> (N R D) H", R=3, N=40, D=128, H=5120
        )
        concat_qkv_bias = torch.concat([q_bias, k_bias, v_bias], dim=0)
        concat_qkv_bias = rearrange(
            concat_qkv_bias, "(R N D H) -> (N R D H)", R=3, N=40, D=128, H=1
        )

        new_state_dict["decoder.layers." + str(i) + ".self_attn.linear_qkv.weight"] = (
            concat_qkv_weight
        )
        new_state_dict["decoder.layers." + str(i) + ".self_attn.linear_qkv.bias"] = (
            concat_qkv_bias
        )
        # 转 o
        o_weight = state_dict["blocks." + str(i) + ".self_attn.o.weight"]
        o_weight = rearrange(
            o_weight, "(R N D) H -> (N R D) H", R=1, N=40, D=128, H=5120
        )
        o_bias = state_dict["blocks." + str(i) + ".self_attn.o.bias"]
        o_bias = rearrange(o_bias, "(R N D H) -> (N R D H)", R=1, N=40, D=128, H=1)

        new_state_dict["decoder.layers." + str(i) + ".self_attn.linear_proj.bias"] = (
            o_bias
        )
        new_state_dict["decoder.layers." + str(i) + ".self_attn.linear_proj.weight"] = (
            o_weight
        )

        ## cross_attention kv 合并
        cross_q_w = state_dict["blocks." + str(i) + ".cross_attn.q.weight"]
        cross_q_w = rearrange(
            cross_q_w, "(R N D) H -> (N R D) H", R=1, N=40, D=128, H=5120
        )
        cross_q_b = state_dict["blocks." + str(i) + ".cross_attn.q.bias"]
        cross_q_b = rearrange(
            cross_q_b, "(R N D H) -> (N R D H)", R=1, N=40, D=128, H=1
        )
        new_state_dict["decoder.layers." + str(i) + ".cross_attn.linear_q.weight"] = (
            cross_q_w
        )
        new_state_dict["decoder.layers." + str(i) + ".cross_attn.linear_q.bias"] = (
            cross_q_b
        )

        cross_attn_k_weight = state_dict["blocks." + str(i) + ".cross_attn.k.weight"]
        cross_attn_k_bias = state_dict["blocks." + str(i) + ".cross_attn.k.bias"]
        cross_attn_v_weight = state_dict["blocks." + str(i) + ".cross_attn.v.weight"]
        cross_attn_v_bias = state_dict["blocks." + str(i) + ".cross_attn.v.bias"]
        concat_kv_weight = torch.concat(
            [cross_attn_k_weight, cross_attn_v_weight], dim=0
        )
        concat_kv_weight = rearrange(
            concat_kv_weight, "(R N D) H -> (N R D) H", R=2, N=40, D=128, H=5120
        )
        concat_kv_bias = torch.concat([cross_attn_k_bias, cross_attn_v_bias], dim=0)
        concat_kv_bias = rearrange(
            concat_kv_bias, "(R N D H) -> (N R D H)", R=2, N=40, D=128, H=1
        )
        new_state_dict["decoder.layers." + str(i) + ".cross_attn.linear_kv.weight"] = (
            concat_kv_weight
        )
        new_state_dict["decoder.layers." + str(i) + ".cross_attn.linear_kv.bias"] = (
            concat_kv_bias
        )

        # cross_attention imge_kv 合并
        cross_attn_k_img_weight = state_dict[
            "blocks." + str(i) + ".cross_attn.k_img.weight"
        ]
        cross_attn_k_img_bias = state_dict[
            "blocks." + str(i) + ".cross_attn.k_img.bias"
        ]
        cross_attn_v_img_weight = state_dict[
            "blocks." + str(i) + ".cross_attn.v_img.weight"
        ]
        cross_attn_v_img_bias = state_dict[
            "blocks." + str(i) + ".cross_attn.v_img.bias"
        ]
        concat_kv_img_weight = torch.concat(
            [cross_attn_k_img_weight, cross_attn_v_img_weight], dim=0
        )
        concat_kv_img_weight = rearrange(
            concat_kv_img_weight, "(R N D) H -> (N R D) H", R=2, N=40, D=128, H=5120
        )
        concat_kv_img_bias = torch.concat(
            [cross_attn_k_img_bias, cross_attn_v_img_bias], dim=0
        )
        concat_kv_img_bias = rearrange(
            concat_kv_img_bias, "(R N D H) -> (N R D H)", R=2, N=40, D=128, H=1
        )
        new_state_dict[
            "decoder.layers." + str(i) + ".cross_attn.linear_kv_img.weight"
        ] = concat_kv_img_weight
        new_state_dict[
            "decoder.layers." + str(i) + ".cross_attn.linear_kv_img.bias"
        ] = concat_kv_img_bias

        cross_o_weight = state_dict["blocks." + str(i) + ".cross_attn.o.weight"]
        cross_o_weight = rearrange(
            cross_o_weight, "(R N D) H -> (N R D) H", R=1, N=40, D=128, H=5120
        )
        cross_o_bias = state_dict["blocks." + str(i) + ".cross_attn.o.bias"]
        cross_o_bias = rearrange(
            cross_o_bias, "(R N D H) -> (N R D H)", R=1, N=40, D=128, H=1
        )
        new_state_dict[
            "decoder.layers." + str(i) + ".cross_attn.linear_proj.weight"
        ] = cross_o_weight
        new_state_dict["decoder.layers." + str(i) + ".cross_attn.linear_proj.bias"] = (
            cross_o_bias
        )
        # 1, 6, 5120 -> 6, 1, 5120
        modulation = state_dict["blocks." + str(i) + ".modulation"]
        new_state_dict["decoder.layers." + str(i) + ".modulation"] = rearrange(
            modulation, "D M L -> M D L"
        )
        ## 一般性替换
        for key, value in inside_blk_replace_dict.items():
            key = key.replace("blocks.0", "blocks." + str(i))
            value = value.replace("decoder.layers.0", "decoder.layers." + str(i))
            new_state_dict[value] = state_dict[key]

    return new_state_dict
