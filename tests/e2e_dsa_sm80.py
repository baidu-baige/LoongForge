# Copyright 2026 The LoongForge Authors.
# SPDX-License-Identifier: Apache-2.0
"""Module-level check of the fused DSA indexer + sparse MLA on SM80.

Drives the real ``DSAIndexerKernel`` with GLM-5's indexer geometry
(``dsa_indexer_n_heads=32``, ``dsa_indexer_head_dim=128``) one chunk at a time, each
chunk seeing all keys emitted so far.  This exercises the sm80 kernels through the
production indexer module rather than through ``DSADotProductAttentionFunction``
directly.

SCOPE -- this is NOT a ChunkPipe test.  ``config.enable_chunkpipe`` is never set and
none of the ChunkPipe machinery in ``dsa_fused.py`` runs: no indexer key cache
(``append/delete/clear_chunk_indexer_key_cache``), no ``indexer_key_cache_grad``
hook, no PP-dependent cache sizing, no ``chunkpipe_forward_microbatch`` offset
arithmetic, and no pipeline parallelism.  The growing ``ik[:kv_len]`` slice
reproduces the *access pattern* ChunkPipe produces, which is what the kernels have
to get right (``q_start_index_s`` offsetting, ``s_kv > s_q`` causal masking, padded
top-k).  Verifying ChunkPipe itself needs a multi-rank run with
``--enable-chunkpipe``.

Run under torchrun with a single rank:
    PYTHONPATH=$MEGATRON:$LOONGFORGE torchrun --nproc_per_node 1 tests/e2e_dsa_sm80.py
"""

import os
import sys

import torch


def _init_distributed():
    os.environ.setdefault("MASTER_ADDR", "127.0.0.1")
    os.environ.setdefault("MASTER_PORT", "29531")
    os.environ.setdefault("RANK", "0")
    os.environ.setdefault("WORLD_SIZE", "1")
    os.environ.setdefault("LOCAL_RANK", "0")
    torch.cuda.set_device(int(os.environ["LOCAL_RANK"]))
    if not torch.distributed.is_initialized():
        torch.distributed.init_process_group(backend="nccl")
    from megatron.core import parallel_state

    if not parallel_state.model_parallel_is_initialized():
        parallel_state.initialize_model_parallel(
            tensor_model_parallel_size=1, pipeline_model_parallel_size=1)


def main():
    _init_distributed()
    # loongforge.models has a circular import with loongforge.train; importing the
    # latter first is what the rest of the tree relies on.
    import loongforge.train  # noqa: F401

    from loongforge.models.common.experimental_attention_variant.dsa_kernel_backend import (
        get_dsa_backend,
    )
    from loongforge.models.common.experimental_attention_variant.dsa_fused_kernels import (
        DSADotProductAttentionFunction,
        DSAIndexerKernel,
    )

    backend = get_dsa_backend()
    print(f"device : {torch.cuda.get_device_name(0)}")
    print(f"backend: {backend.name}")

    # GLM-5 geometry, scaled down in sequence length only.
    h_index, d_index, d_qk, d_v, h_q = 32, 128, 576, 512, 128
    chunksize, n_chunks, topk = 64, 4, 64
    seq = chunksize * n_chunks
    dev, dt = "cuda", torch.bfloat16
    g = torch.Generator(device=dev).manual_seed(0)

    kernel = DSAIndexerKernel()

    q = (torch.randn(seq, h_q, d_qk, device=dev, dtype=dt, generator=g) * 0.25).requires_grad_(True)
    kv = (torch.randn(seq, 1, d_qk, device=dev, dtype=dt, generator=g) * 0.25).requires_grad_(True)
    iq = (torch.randn(seq, h_index, d_index, device=dev, dtype=dt, generator=g) * 0.25
          ).requires_grad_(True)
    ik = (torch.randn(seq, d_index, device=dev, dtype=dt, generator=g) * 0.25).requires_grad_(True)
    w = (torch.randn(seq, h_index, device=dev, dtype=torch.float32, generator=g).abs() * 0.1
         ).requires_grad_(True)

    total_loss = 0.0
    for c in range(n_chunks):
        lo, hi = c * chunksize, (c + 1) * chunksize
        kv_len = hi  # ChunkPipe: keys accumulate across chunks

        # The real indexer module: BF16 MQA logits + causal mask + padded top-k.
        score_topk, sel = kernel(
            iq[lo:hi], ik[:kv_len], w[lo:hi], topk, lo, None)
        assert torch.isfinite(score_topk).all() or torch.isneginf(score_topk).any()
        assert sel.shape[0] == chunksize and sel.shape[-1] == topk, sel.shape
        k = min(topk, kv_len)
        # padded_flashinfer_topk returns int64; dsa_fused.py casts at the call site.
        sel = sel.unsqueeze(1).int()

        out, p_out = DSADotProductAttentionFunction.apply(
            q[lo:hi], kv[:kv_len], sel, lo, d_qk**-0.5, d_v, True, None, None, None, 0, False)
        assert out.shape == (1, chunksize, h_q, d_v), out.shape
        assert torch.isfinite(out).all(), f"chunk {c}: non-finite attention output"
        assert torch.isfinite(p_out).all() or torch.isneginf(p_out).any()

        loss = out.float().square().mean() + score_topk.float().nan_to_num(
            neginf=0.0).square().mean() * 1e-3
        loss.backward()
        total_loss += loss.item()
        print(f"chunk {c}: kv_len={kv_len:4d} valid_topk={k:4d} loss={loss.item():.6f}")

    for name, t in [("q", q), ("kv", kv), ("indexer_q", iq), ("indexer_k", ik), ("weights", w)]:
        assert t.grad is not None, f"{name} received no gradient"
        assert torch.isfinite(t.grad).all(), f"{name} gradient is not finite"
        assert t.grad.abs().max() > 0, f"{name} gradient is all zero"
        print(f"grad {name:9s}: max|g|={t.grad.abs().max().item():.4e}")

    print(f"\nOK: {n_chunks} chunks, total loss {total_loss:.6f}, all gradients finite")


if __name__ == "__main__":
    sys.exit(main())
