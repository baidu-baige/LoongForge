import math
from typing import Optional

import torch
import torch.distributed as dist
import torch.nn.functional as F
from diffusers.models.embeddings import apply_rotary_emb
from einops import rearrange, repeat
from einops.layers.torch import Rearrange
from torch import nn

from ....acceleration import all_to_all, get_sequence_parallel_group, split_forward_gather_backward


class PatchEmbed(nn.Module):
    """Embeds video patches (spatial and temporal) into a higher-dimensional
    space."""

    def __init__(
        self,
        spatial_patch_size: int,
        temporal_patch_size: int,
        in_channels: int = 3,
        out_channels: int = 768,
        bias: bool = True,
    ):
        super().__init__()
        self.proj = nn.Sequential(
            Rearrange(
                'b c (t r) (h m) (w n) -> b t h w (c r m n)',
                r=temporal_patch_size,
                m=spatial_patch_size,
                n=spatial_patch_size,
            ),
            nn.Linear(in_channels * spatial_patch_size * spatial_patch_size * temporal_patch_size, out_channels, bias=bias),
        )

    def forward(self, x):
        return self.proj(x)


class VideoRopePosition3DEmb(nn.Module):
    """Generates 3D rotary positional embeddings for video data (temporal,
    height, width)."""

    def __init__(
        self,
        head_dim: int,
        len_h: int,
        len_w: int,
        len_t: int,
        base_fps: int = 24,
        h_extrapolation_ratio: float = 1.0,
        w_extrapolation_ratio: float = 1.0,
        t_extrapolation_ratio: float = 1.0,
    ):
        super().__init__()
        self.base_fps = base_fps
        self.max_h = len_h
        self.max_w = len_w
        self.max_t = len_t
        dim = head_dim
        dim_h = dim // 6 * 2
        dim_w = dim_h
        dim_t = dim - 2 * dim_h
        self.dim_h = dim_h
        self.dim_w = dim_w
        self.dim_t = dim_t
        self.h_ntk_factor = h_extrapolation_ratio ** (dim_h / (dim_h - 2))
        self.w_ntk_factor = w_extrapolation_ratio ** (dim_w / (dim_w - 2))
        self.t_ntk_factor = t_extrapolation_ratio ** (dim_t / (dim_t - 2))

    def forward(self, x_B_T_H_W_C: torch.Tensor, fps=Optional[torch.Tensor]) -> torch.Tensor:
        B_T_H_W_C = x_B_T_H_W_C.shape
        sp_group = get_sequence_parallel_group()
        if sp_group is not None:
            world_size = dist.get_world_size(sp_group)
            B, T, H, W, C = B_T_H_W_C
            B_T_H_W_C = (B, T * world_size, H, W, C)
        embeddings = self.generate_embedding_for_batch(
            B_T_H_W_C,
            device=x_B_T_H_W_C.device,
            fps=fps,
        )
        if sp_group is not None:
            embeddings = split_forward_gather_backward(embeddings, dim=0, group=sp_group)
        return embeddings

    def generate_embedding_for_batch(
        self,
        B_T_H_W_C: torch.Size,
        device: torch.device,
        fps: int,
    ):
        h_theta = 10000.0 * self.h_ntk_factor
        w_theta = 10000.0 * self.w_ntk_factor
        t_theta = 10000.0 * self.t_ntk_factor

        seq = torch.arange(max(self.max_h, self.max_w, self.max_t), device=device, dtype=torch.float32)
        dim_h_range = torch.arange(0, self.dim_h, 2, device=device, dtype=torch.float32)[: (self.dim_h // 2)] / self.dim_h
        dim_w_range = torch.arange(0, self.dim_w, 2, device=device, dtype=torch.float32)[: (self.dim_w // 2)] / self.dim_w
        dim_t_range = torch.arange(0, self.dim_t, 2, device=device, dtype=torch.float32)[: (self.dim_t // 2)] / self.dim_t

        h_spatial_freqs = 1.0 / (h_theta**dim_h_range)
        w_spatial_freqs = 1.0 / (w_theta**dim_w_range)
        temporal_freqs = 1.0 / (t_theta**dim_t_range)

        B, T, H, W, _ = B_T_H_W_C
        half_emb_h = torch.outer(seq[:H], h_spatial_freqs)
        half_emb_w = torch.outer(seq[:W], w_spatial_freqs)
        half_emb_t = torch.outer(seq[:T] / fps * self.base_fps, temporal_freqs)
        em_T_H_W_D = torch.cat(
            [
                repeat(half_emb_t, 't d -> t h w d', h=H, w=W),
                repeat(half_emb_h, 'h d -> t h w d', t=T, w=W),
                repeat(half_emb_w, 'w d -> t h w d', t=T, h=H),
            ]
            * 2,
            dim=-1,
        )
        em_T_H_W_D = rearrange(em_T_H_W_D, 't h w d -> (t h w) d').float()
        return em_T_H_W_D


class Timesteps(nn.Module):
    """Generates sinusoidal timestep embeddings for diffusion models."""

    def __init__(self, num_channels: int):
        super().__init__()
        self.num_channels = num_channels

    def forward(self, timesteps: torch.Tensor):
        in_dype = timesteps.dtype
        half_dim = self.num_channels // 2
        exponent = -math.log(10000) * torch.arange(half_dim, dtype=torch.float32, device=timesteps.device)
        exponent = exponent / (half_dim - 0.0)
        emb = torch.exp(exponent)
        emb = timesteps[:, None].float() * emb[None, :]
        sin_emb = torch.sin(emb)
        cos_emb = torch.cos(emb)
        emb = torch.cat([cos_emb, sin_emb], dim=-1)
        return emb.to(in_dype)


class TimestepEmbedding(nn.Module):
    """Projects timestep embeddings through two linear layers with
    activation."""

    def __init__(self, in_features: int, out_features: int):
        super().__init__()
        self.linear_1 = nn.Linear(in_features, out_features, bias=False)
        self.activation = nn.SiLU()
        self.linear_2 = nn.Linear(out_features, 3 * out_features, bias=False)

    def forward(self, sample: torch.Tensor) -> torch.Tensor:
        emb = self.linear_1(sample)
        emb = self.activation(emb)
        emb = self.linear_2(emb)
        return sample, emb


class FeedForward(nn.Module):
    """Standard feed-forward MLP block used in transformers."""

    def __init__(
        self,
        d_model: int,
        d_ff: int,
        activation=nn.GELU(),
        bias: bool = False,
    ) -> None:
        super().__init__()
        self.layer1 = nn.Linear(d_model, d_ff, bias=bias)
        self.layer2 = nn.Linear(d_ff, d_model, bias=bias)
        self.activation = activation

    def forward(self, x: torch.Tensor):
        x = self.layer1(x)
        x = self.activation(x)
        x = self.layer2(x)
        return x


class MoEGate(nn.Module):
    """Mixture-of-Experts gating mechanism for routing tokens to experts."""

    def __init__(
        self,
        embed_dim: int,
        num_routed_experts: int = 4,
        num_activated_experts: int = 2,
        aux_loss_alpha: float = 0.01,
    ):
        super().__init__()
        self.top_k = num_activated_experts
        self.n_routed_experts = num_routed_experts
        self.scoring_func = 'softmax'
        self.alpha = aux_loss_alpha
        self.seq_aux = False
        # topk selection algorithm
        self.norm_topk_prob = False
        self.gating_dim = embed_dim
        self.weight = nn.Parameter(torch.randn(self.n_routed_experts, self.gating_dim) / embed_dim**0.5)

    def forward(self, hidden_states: torch.Tensor):
        bsz, seq_len, h = hidden_states.shape
        # compute gating score
        hidden_states = hidden_states.reshape(-1, h)
        logits = F.linear(hidden_states, self.weight, None)
        if self.scoring_func == 'softmax':
            scores = logits.softmax(dim=-1)
        else:
            raise NotImplementedError(f'insupportable scoring function for MoE gating: {self.scoring_func}')
        # select top-k experts
        topk_weight, topk_idx = torch.topk(scores, k=self.top_k, dim=-1, sorted=False)
        # norm gate to sum 1
        if self.top_k > 1 and self.norm_topk_prob:
            denominator = topk_weight.sum(dim=-1, keepdim=True) + 1e-20
            topk_weight = topk_weight / denominator
        elif self.top_k == 1:
            denominator = topk_weight.sum(dim=-1, keepdim=True) + 1e-20
            topk_weight = topk_weight / denominator
        # expert-level computation auxiliary loss
        if self.training and self.alpha > 0.0:
            scores_for_aux = scores
            aux_topk = self.top_k
            # always compute aux loss based on the naive greedy topk method
            topk_idx_for_aux_loss = topk_idx.view(bsz, -1)
            if self.seq_aux:
                scores_for_seq_aux = scores_for_aux.view(bsz, seq_len, -1)
                ce = torch.zeros(bsz, self.n_routed_experts, device=hidden_states.device)
                ce.scatter_add_(1, topk_idx_for_aux_loss, torch.ones(bsz, seq_len * aux_topk, device=hidden_states.device)).div_(
                    seq_len * aux_topk / self.n_routed_experts
                )
                aux_loss = (ce * scores_for_seq_aux.mean(dim=1)).sum(dim=1).mean() * self.alpha
            else:
                mask_ce = F.one_hot(topk_idx_for_aux_loss.view(-1), num_classes=self.n_routed_experts)
                ce = mask_ce.float().mean(0)
                pi = scores_for_aux.mean(0)
                fi = ce * self.n_routed_experts
                aux_loss = (pi * fi).sum() * self.alpha
        else:
            aux_loss = None
        return topk_idx, topk_weight, aux_loss


class MOEFeedForward(nn.Module):
    """Mixture-of-Experts feed-forward block with expert selection and
    auxiliary loss."""

    def __init__(
        self,
        x_dim: int,
        mlp_ratio: float = 4.0,
        bias: bool = False,
        num_routed_experts: int = 4,
        num_activated_experts: int = 2,
        shared_experts: bool = False,
    ):
        super().__init__()
        self.shared_experts = shared_experts
        if shared_experts:
            self.shared_experts = FeedForward(x_dim, int(x_dim * mlp_ratio), bias=bias)
        self.experts = nn.ModuleList([FeedForward(x_dim, int(x_dim * mlp_ratio), bias=bias) for i in range(num_routed_experts)])
        self.gate = MoEGate(
            embed_dim=x_dim,
            num_routed_experts=num_routed_experts,
            num_activated_experts=num_activated_experts,
        )
        self.num_activated_experts = num_activated_experts

    def forward(self, x: torch.Tensor):
        identity = x
        b, t, h, w = x.shape[:4]
        x = rearrange(x, 'b t h w d -> b (t h w) d')
        topk_idx, topk_weight, aux_loss = self.gate(x)
        x = rearrange(x, 'b s d -> (b s) d')
        flat_topk_idx = topk_idx.view(-1)
        if self.training:
            x = x.repeat_interleave(self.num_activated_experts, dim=0)
            y = torch.empty_like(x, dtype=x.dtype)
            for i, expert in enumerate(self.experts):
                y[flat_topk_idx == i] = expert(x[flat_topk_idx == i]).to(dtype=x.dtype)
            y = (y.reshape(*topk_weight.shape, -1) * topk_weight.unsqueeze(-1)).sum(dim=1)
            y = AddAuxiliaryLoss.apply(y, aux_loss)
        else:
            y = self.moe_infernce(x, flat_topk_idx, topk_weight.view(-1, 1))
        if self.shared_experts:
            y = y + self.shared_experts(identity)
        y = rearrange(y, '(b t h w) d-> b t h w d', b=b, t=t, h=h, w=w)
        return y

    def moe_infernce(self, x: torch.Tensor, flat_expert_indices: torch.Tensor, flat_expert_weights: torch.Tensor):
        expert_cache = torch.zeros_like(x)
        idxs = flat_expert_indices.argsort()
        tokens_per_expert = flat_expert_indices.bincount().cpu().numpy().cumsum(0)
        token_idxs = idxs // self.num_activated_experts
        for i, end_idx in enumerate(tokens_per_expert):
            start_idx = 0 if i == 0 else tokens_per_expert[i - 1]
            if start_idx == end_idx:
                continue
            expert = self.experts[i]
            exp_token_idx = token_idxs[start_idx:end_idx]
            expert_tokens = x[exp_token_idx]
            expert_out = expert(expert_tokens)
            expert_out.mul_(flat_expert_weights[idxs[start_idx:end_idx]])
            expert_cache.scatter_reduce_(0, exp_token_idx.view(-1, 1).repeat(1, x.shape[-1]), expert_out, reduce='sum')
        return expert_cache


class AddAuxiliaryLoss(torch.autograd.Function):
    """Custom autograd function to add auxiliary loss for MoE training."""

    """The trick function of adding auxiliary (aux) loss, which includes the
    gradient of the aux loss during backpropagation."""

    @staticmethod
    def forward(ctx, x, loss):
        assert loss.numel() == 1
        ctx.dtype = loss.dtype
        ctx.required_aux_loss = loss.requires_grad
        return x

    @staticmethod
    def backward(ctx, grad_output):
        grad_loss = None
        if ctx.required_aux_loss:
            grad_loss = torch.ones(1, dtype=ctx.dtype, device=grad_output.device)
        return grad_output, grad_loss


class Attention(nn.Module):
    """Multi-head attention block supporting self/cross attention and different
    backends."""

    def __init__(
        self,
        query_dim: int,
        context_dim: int | None = None,
        heads: int = 8,
        dim_head: int = 64,
        qkv_bias: bool = False,
        out_bias: bool = False,
        natten_params: list = None,
    ) -> None:
        super().__init__()
        self.is_selfattn = context_dim is None  # self attention
        inner_dim = dim_head * heads
        context_dim = query_dim if context_dim is None else context_dim
        norm_dim = dim_head
        self.heads = heads
        self.dim_head = dim_head
        self.to_q = nn.Sequential(
            nn.Linear(query_dim, inner_dim, bias=qkv_bias),
            nn.RMSNorm(norm_dim, eps=1e-6),
        )
        self.to_k = nn.Sequential(
            nn.Linear(context_dim, inner_dim, bias=qkv_bias),
            nn.RMSNorm(norm_dim, eps=1e-6),
        )
        self.to_v = nn.Sequential(
            nn.Linear(context_dim, inner_dim, bias=qkv_bias),
        )
        self.to_out = nn.Sequential(
            nn.Linear(inner_dim, query_dim, bias=out_bias),
        )
        if natten_params is not None:
            assert self.is_selfattn
            self.set_attn_backend('natten', natten_params=natten_params)
        else:
            self.set_attn_backend('torch')
        self.context_parallel = False

    def set_attn_backend(self, backend, **kwargs):
        self.attn_backend = backend
        if backend == 'natten':
            from .neighborhood_attn import NeighborhoodAttention

            self.attn_op = NeighborhoodAttention(**kwargs)
        elif backend in ['torch', 'sage']:
            self.attn_op = None
        else:
            raise ValueError(f'Attn Backend {backend} not found')

    def apply_rotary_emb(self, x, rope_emb):
        x = rearrange(x, 'b s h d -> b h s d')
        rope_emb_cos = torch.cos(rope_emb)
        rope_emb_sin = torch.sin(rope_emb)
        x = apply_rotary_emb(x, [rope_emb_cos, rope_emb_sin], use_real=True, use_real_unbind_dim=-2)
        x = rearrange(x, 'b h s d -> b s h d')
        return x

    def cal_qkv(self, x, context=None, rope_emb=None):
        q = self.to_q[0](x)
        context = x if context is None else context
        k = self.to_k[0](context)
        v = self.to_v[0](context)
        q, k, v = map(
            lambda t: rearrange(t, 'b s (h d) -> b s h d', h=self.heads, d=self.dim_head),
            (q, k, v),
        )
        q = self.to_q[1](q)
        k = self.to_k[1](k)
        if self.is_selfattn and rope_emb is not None:
            q = self.apply_rotary_emb(q, rope_emb)
            k = self.apply_rotary_emb(k, rope_emb)
        return q, k, v

    def cal_attn(self, q, k, v, video_size=None):
        sp_group = get_sequence_parallel_group()
        if sp_group is not None:
            q = all_to_all(q, scatter_dim=2, gather_dim=1, group=sp_group)
            if self.is_selfattn:
                k = all_to_all(k, scatter_dim=2, gather_dim=1, group=sp_group)
                v = all_to_all(v, scatter_dim=2, gather_dim=1, group=sp_group)
            else:
                k = split_forward_gather_backward(k, dim=2, group=sp_group)
                v = split_forward_gather_backward(v, dim=2, group=sp_group)
        if self.attn_backend == 'natten':
            out = self.attn_op(q, k, v, video_size=video_size)
            out = rearrange(out, 'b s h d -> b s (h d)')
            if sp_group is not None:
                out = all_to_all(out, scatter_dim=1, gather_dim=2, group=sp_group)
            out = self.to_out(out)
        elif self.attn_backend in ['torch', 'sage']:
            q = rearrange(q, 'b s h d -> b h s d')
            k = rearrange(k, 'b s h d -> b h s d')
            v = rearrange(v, 'b s h d -> b h s d')
            if self.attn_backend == 'torch':
                out = F.scaled_dot_product_attention(q, k, v)
            else:
                from sageattention import sageattn

                out = sageattn(q, k, v)
            out = rearrange(out, 'b h s d -> b s (h d)')
            if sp_group is not None:
                out = all_to_all(out, scatter_dim=1, gather_dim=2, group=sp_group)
            out = self.to_out(out)
        else:
            raise ValueError(f'Attn Backend {self.attn_backend} not found')
        return out

    def forward(self, x, context=None, rope_emb=None):
        t, h, w = x.shape[1:4]
        sp_group = get_sequence_parallel_group()
        if sp_group is not None:
            world_size = dist.get_world_size(sp_group)
            video_size = (world_size * t, h, w)
        else:
            video_size = (t, h, w)
        x = rearrange(x, 'b t h w d -> b (t h w) d')
        q, k, v = self.cal_qkv(x, context, rope_emb=rope_emb)
        out = self.cal_attn(q, k, v, video_size=video_size)
        out = rearrange(out, 'b (t h w) d -> b t h w d', h=h, w=w)
        return out


class SingleBlock(nn.Module):
    """
    A single transformer block: can be attention, cross-attention, or MLP (optionally MoE).
    """

    def __init__(
        self,
        block_type: str,
        x_dim: int,
        context_dim: Optional[int],
        num_heads: int,
        mlp_ratio: float = 4.0,
        bias: bool = False,
        adaln_lora_dim: int = 256,
        natten_params: dict = None,
        moe_params: dict = None,
    ) -> None:
        super().__init__()
        block_type = block_type.lower()
        if block_type in ['full_attn', 'fa']:
            self.block = Attention(
                query_dim=x_dim,
                context_dim=None,
                heads=num_heads,
                dim_head=x_dim // num_heads,
                qkv_bias=bias,
                out_bias=bias,
                natten_params=natten_params,
            )
        elif block_type in ['cross_attn', 'ca']:
            self.block = Attention(
                query_dim=x_dim,
                context_dim=context_dim,
                heads=num_heads,
                dim_head=x_dim // num_heads,
                qkv_bias=bias,
                out_bias=bias,
            )
        elif block_type in ['mlp', 'ff']:
            if moe_params is None:
                self.block = FeedForward(x_dim, int(x_dim * mlp_ratio), bias=bias)
            else:
                self.block = MOEFeedForward(x_dim, mlp_ratio, bias=bias, **moe_params)
        else:
            raise ValueError(f'Unknown block type: {block_type}')
        self.block_type = block_type
        self.n_adaln_chunks = 3
        self.norm_state = nn.LayerNorm(x_dim, elementwise_affine=False, eps=1e-6)
        self.adaLN_modulation = nn.Sequential(
            nn.SiLU(),
            nn.Linear(x_dim, adaln_lora_dim, bias=False),
            nn.Linear(adaln_lora_dim, self.n_adaln_chunks * x_dim, bias=False),
        )

    def forward(
        self,
        x: torch.Tensor,
        emb: torch.Tensor,
        crossattn_emb: torch.Tensor,
        rope_emb: Optional[torch.Tensor],
        adaln_lora: Optional[torch.Tensor],
    ) -> torch.Tensor:
        shift, scale, gate = (self.adaLN_modulation(emb) + adaln_lora).chunk(self.n_adaln_chunks, dim=1)
        B = x.shape[0]
        shift, scale, gate = (
            rearrange(shift, '(b t) d -> b t 1 1 d', b=B),
            rearrange(scale, '(b t) d -> b t 1 1 d', b=B),
            rearrange(gate, '(b t) d -> b t 1 1 d', b=B),
        )
        norm_x = self.norm_state(x) * (1 + scale) + shift
        if self.block_type in ['full_attn', 'fa']:
            x = x + gate * self.block(
                norm_x,
                context=None,
                rope_emb=rope_emb,
            )
        elif self.block_type in ['cross_attn', 'ca']:
            x = x + gate * self.block(
                norm_x,
                context=crossattn_emb,
                rope_emb=rope_emb,
            )
        elif self.block_type in ['mlp', 'ff']:
            x = x + gate * self.block(norm_x)
        else:
            raise ValueError(f'Unknown block type: {self.block_type}')
        return x


class TransformerBlock(nn.Module):
    """Stacks multiple SingleBlock modules as a transformer block."""

    def __init__(
        self,
        block_config: str,
        x_dim: int,
        context_dim: int,
        num_heads: int,
        mlp_ratio: float = 4.0,
        adaln_lora_dim: int = 256,
        natten_params: dict = None,
        moe_params: dict = None,
    ):
        super().__init__()
        self.blocks = nn.ModuleList()
        for block_type in block_config.split('-'):
            self.blocks.append(
                SingleBlock(
                    block_type=block_type,
                    x_dim=x_dim,
                    context_dim=context_dim,
                    num_heads=num_heads,
                    mlp_ratio=mlp_ratio,
                    adaln_lora_dim=adaln_lora_dim,
                    natten_params=natten_params,
                    moe_params=moe_params,
                )
            )

    def forward(
        self,
        x: torch.Tensor,
        emb: torch.Tensor,
        crossattn_emb: torch.Tensor,
        rope_emb: Optional[torch.Tensor],
        adaln_lora: Optional[torch.Tensor],
    ) -> torch.Tensor:
        for block in self.blocks:
            x = block(
                x=x,
                emb=emb,
                crossattn_emb=crossattn_emb,
                rope_emb=rope_emb,
                adaln_lora=adaln_lora,
            )
        return x


class FinalLayer(nn.Module):
    """Final projection layer for the transformer output, with adaptive layer
    norm."""

    def __init__(
        self,
        hidden_size: int,
        spatial_patch_size: int,
        temporal_patch_size: int,
        out_channels: int,
        adaln_lora_dim: int = 256,
    ):
        super().__init__()
        self.norm_final = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
        self.linear = nn.Linear(hidden_size, spatial_patch_size * spatial_patch_size * temporal_patch_size * out_channels, bias=False)
        self.hidden_size = hidden_size
        self.adaLN_modulation = nn.Sequential(
            nn.SiLU(),
            nn.Linear(hidden_size, adaln_lora_dim, bias=False),
            nn.Linear(adaln_lora_dim, 2 * hidden_size, bias=False),
        )

    def forward(self, x, emb, adaln_lora):
        shift, scale = (self.adaLN_modulation(emb) + adaln_lora[:, : 2 * self.hidden_size]).chunk(2, dim=1)
        x = self.norm_final(x) * (1 + scale.unsqueeze(1)) + shift.unsqueeze(1)
        x = self.linear(x)
        return x
