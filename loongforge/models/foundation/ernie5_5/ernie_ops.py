# Copyright 2026 The LoongForge Authors.
# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Adapted from NVIDIA-NeMo/Megatron-Bridge at 8f580c64376877e9e2bc3140ce3476afc0f75d76.

"""ERNIE residual, activation and output operations."""

import math

import torch
import torch.nn.functional as F
from megatron.core.extensions.transformer_engine import TENorm
from megatron.core.jit import jit_fuser
from megatron.core.transformer.hyper_connection import HyperConnectionModule


class Ernie55Norm(TENorm):
    def __new__(cls, config, hidden_size, eps=1e-6, **kwargs):
        norm = TENorm(config, hidden_size, eps=eps, **kwargs)
        norm.register_forward_pre_hook(_downcast_norm_input)
        return norm


def _downcast_norm_input(module, args):
    return (args[0].to(module.weight.dtype), *args[1:])


class Ernie55SiTU(torch.nn.Module):
    def __init__(self, config):
        super().__init__()
        self.beta = config.activation_situ_beta
        self.linear_beta = config.activation_situ_linear_beta

    def forward(self, value):
        return situ_glu(value, self.beta, self.linear_beta)


@jit_fuser
def tanh_soft_clamp(value, scale: float):
    return (scale * torch.tanh(value.float() / scale)).to(value.dtype)


@jit_fuser
def situ(value, scale: float):
    return tanh_soft_clamp(value, scale) * torch.sigmoid(value)


@jit_fuser
def situ_glu(value, beta: float, linear_beta: float):
    gate, linear = value.chunk(2, dim=-1)
    linear = tanh_soft_clamp(linear, linear_beta)
    return situ(gate, beta) * (linear + 0.0)


class Ernie55HyperConnection(HyperConnectionModule):
    def __init__(self, config, layer_number):
        super().__init__(config, layer_number)
        self._h_post_bda_op = ernie55_h_post_bda
        for parameter in self.parameters():
            parameter.keep_in_fp32 = True

    def forward(self, value, mhc_recompute_manager=None):
        mapping_input, aggregate_input, residual = Ernie55ResidualSplit.apply(value)
        pre, post, matrix = self.compute_mappings(mapping_input)
        return self.aggregate(aggregate_input, pre), (matrix, residual), post

    def fused_h_res_h_post_bda(
        self,
        residual_state,
        original_residual,
        post,
        layer_output_with_bias,
        dropout_prob,
        training,
        fused,
        manager=None,
    ):
        # Carry the residual alias through the three-result TransformerLayer API.
        matrix, residual = residual_state
        shape = residual.shape
        residual = residual.view(*shape[:2], self.n, self.hidden_size)
        return ernie55_h_post_bda(matrix, residual, post, *layer_output_with_bias).view(shape)

    def compute_mappings(self, value):
        seq, batch, width = value.shape
        n = self.n
        value = value.float()
        with torch.autocast(device_type=value.device.type, enabled=False):
            projection = value @ self.mapping_proj.weight.float().T.contiguous()
            scale = 1 / (value.norm(dim=-1, keepdim=True) / math.sqrt(width) + self.norm_eps)
            alpha = torch.cat((self.alpha_pre.expand(n), self.alpha_post.expand(n), self.alpha_res.expand(n * n)))
            mapping = scale * projection * alpha + self.bias
            pre = mapping[..., :n].sigmoid() + self.compute_h_eps
            post = 2 * mapping[..., n : 2 * n].sigmoid()
            residual = mapping[..., 2 * n :].reshape(seq, batch, n, n).softmax(-1) + self.sinkhorn_eps
            residual = residual / (residual.sum(-2, keepdim=True) + self.sinkhorn_eps)
            for _ in range(self.sinkhorn_iterations - 1):
                residual = residual / (residual.sum(-1, keepdim=True) + self.sinkhorn_eps)
                residual = residual / (residual.sum(-2, keepdim=True) + self.sinkhorn_eps)
        return pre, post, residual

    def aggregate(self, value, pre):
        return (value.view(*value.shape[:-1], self.n, self.hidden_size) * pre.unsqueeze(-1)).sum(-2)


class Ernie55ResidualSplit(torch.autograd.Function):
    @staticmethod
    def forward(ctx, value):
        return value.view_as(value), value.view_as(value), value.view_as(value)

    @staticmethod
    def backward(ctx, mapping_grad, aggregate_grad, residual_grad):
        return add_residual_grads(mapping_grad, aggregate_grad, residual_grad)


@jit_fuser
def add_residual_grads(mapping_grad, aggregate_grad, residual_grad):
    return mapping_grad + aggregate_grad + residual_grad


def ernie55_h_post_bda(residual_map, residual, post, value, bias):
    seq, batch, streams, hidden = residual.shape
    mixed = torch.bmm(
        residual_map.reshape(seq * batch, streams, streams).transpose(1, 2),
        residual.float().reshape(seq * batch, streams, hidden),
    ).reshape(seq, batch, streams, hidden)
    output = post.unsqueeze(-1) * value.float().unsqueeze(2)
    if bias is not None:
        output = output + post.unsqueeze(-1) * bias.float().view(1, 1, 1, hidden)
    return (output + mixed).to(residual.dtype)


def vha_postmix(value, u, v):
    heads = value.shape[-2]
    mix = torch.addmm(torch.eye(heads, dtype=value.dtype, device=value.device), v.to(value.dtype), u.to(value.dtype).T)
    return torch.matmul(mix, value)


class Ernie55Multimax(torch.autograd.Function):
    @staticmethod
    def forward(ctx, logits, ranges, scales):
        ctx.save_for_backward(logits, ranges, scales)
        output = torch.empty_like(logits)
        rows = logits.reshape(-1, logits.shape[-1])
        output_rows = output.view_as(rows)
        for start in range(0, rows.shape[0], 512):
            output_rows[start : start + 512] = multimax(
                rows[start : start + 512].float(), ranges.float(), scales.float()
            )
        return output

    @staticmethod
    def backward(ctx, grad_output):
        logits, ranges, scales = ctx.saved_tensors
        rows = logits.reshape(-1, logits.shape[-1])
        grads = grad_output.reshape_as(rows)
        grad_logits = torch.empty_like(rows)
        grad_ranges = torch.zeros_like(ranges, dtype=torch.float32)
        grad_scales = torch.zeros_like(scales, dtype=torch.float32)
        with torch.enable_grad():
            r = ranges.detach().float().requires_grad_()
            t = scales.detach().float().requires_grad_()
            for start in range(0, rows.shape[0], 512):
                chunk = rows[start : start + 512].detach().requires_grad_()
                result = multimax(chunk.float(), r, t).to(logits.dtype)
                dx, dr, dt = torch.autograd.grad(result, (chunk, r, t), grads[start : start + 512])
                grad_logits[start : start + 512] = dx
                grad_ranges += dr
                grad_scales += dt
        return grad_logits.view_as(logits), grad_ranges.to(ranges.dtype), grad_scales.to(scales.dtype)


def multimax(value, ranges, scales):
    r0, r1, r2, r3 = ranges.unbind()
    t0, t1, t2, t3 = scales.unbind()
    return (
        value
        + t0 * F.relu(r0 - value)
        + t1 * F.relu(value - r1)
        + t2 * F.relu(r2 - value).square()
        + t3 * F.relu(value - r3).square()
    )
