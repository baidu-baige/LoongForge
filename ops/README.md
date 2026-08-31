# Custom Fused Operators

## Introduction

This directory contains custom fused operators that power LoongForge's training acceleration, including Sparse MLA Attention, Lightning Indexer implementations, groot_n1_7_op, and wall_oss_05_op.

### TileLang Operators

The `tilelang_ops/` directory provides the following operators built on [TileLang](https://github.com/tile-ai/tilelang):

- **Sparse MLA Forward** (`sparse_mla_fwd.py`)
- **Sparse MLA Backward** (`sparse_mla_bwd.py`)
- **Lightning Indexer** (`lightning_indexer.py`)

### Model operator packages

Installable packages that expose their operators as a normal Python import.
Each one has its own README with the full operator list, tensor layout
requirements, and build options:

- [`groot_n1_7_op`](cuda_source/groot_n1_7_op/README.md) — GR00T-N1.7 fused operators:
  Qwen3-VL vision/text RoPE, RMSNorm and SiLU-multiply; precision-compatible
  fused AdamW (eager / capturable / capturable with fused gradient scale);
  `c10d::Reducer` bucket initialization and inspection.
- [`wall_oss_05_op`](cuda_source/wall_oss_05_op/README.md) — WALL-OSS-0.5 operators:
  RoPE, M-RoPE, RotPosEmb, RMSNorm, SwiGLU, MoE permute/unpermute,
  GetRopeIndex and GetWindowIndex. Every operator falls back to pure PyTorch
  when the CUDA extension is unavailable.

## Requirements

- DSA kernels: SM90 / SM100, CUDA 12.8 and above (CUDA 12.9+ is required for
  SM100 kernels), See `requirements.txt` for dependencies.
- Operator packages: SM80 / SM120 by default, overridable via
  `TORCH_CUDA_ARCH_LIST`
- PyTorch 2.0 and above
```
