# Copyright 2026 The LoongForge Authors.
# SPDX-License-Identifier: Apache-2.0
"""Isolated candidate kernel prototype; not integrated or used by baseline."""
import torch
import triton
import triton.language as tl

@triton.jit
def _rope_fp64_kernel(X, F, Y, PAIRS:tl.constexpr, SEQ:tl.constexpr, HEADS:tl.constexpr, HALF:tl.constexpr, BLOCK:tl.constexpr):
    idx=tl.program_id(0)*BLOCK+tl.arange(0,BLOCK)
    active=idx<PAIRS
    # x layout [B,S,H,half,2]; freqs [S,1,half] complex128.
    fi=((idx//(HEADS*HALF))%SEQ)*HALF+idx%HALF
    ar=tl.load(X+2*idx,active,0).to(tl.float64)
    ai=tl.load(X+2*idx+1,active,0).to(tl.float64)
    fr=tl.load(F+2*fi,active,0)
    fimag=tl.load(F+2*fi+1,active,0)
    real=ar*fr-ai*fimag
    imag=ar*fimag+ai*fr
    # PyTorch double->BF16 conversion goes through the float constructor.
    tl.store(Y+2*idx,real.to(tl.float32).to(Y.dtype.element_ty),active)
    tl.store(Y+2*idx+1,imag.to(tl.float32).to(Y.dtype.element_ty),active)

def rope_fp64(x,freqs,num_heads,out=None):
    assert x.is_cuda and x.is_contiguous() and x.dtype==torch.bfloat16
    assert freqs.dtype==torch.complex128 and freqs.is_contiguous()
    assert not torch.is_grad_enabled() or not x.requires_grad
    batch,seq,width=x.shape
    half=width//num_heads//2
    assert freqs.shape==(seq,1,half)
    y=torch.empty_like(x) if out is None else out
    if (y.shape != x.shape or y.dtype != x.dtype or y.device != x.device
            or y.stride() != x.stride() or not y.is_contiguous()):
        raise ValueError("RoPE output scratch has an incompatible layout")
    if y.data_ptr() == x.data_ptr() and (torch.is_grad_enabled() or x.requires_grad):
        raise ValueError("In-place RoPE requires frozen no_grad execution")
    # Each program loads both real/imaginary values before storing the same
    # pair, so exact elementwise in-place rotation is safe and avoids a full
    # [B,S,D] output allocation at large GBS256 batch sizes.
    pairs=x.numel()//2
    _rope_fp64_kernel[(triton.cdiv(pairs,256),)](x,torch.view_as_real(freqs),y,pairs,seq,num_heads,half,256,num_warps=4,enable_fp_fusion=False)
    return y
