# Custom Fused Operators

## Introduction

This directory contains custom fused operators that power LoongForge's training acceleration, including Sparse MLA Attention and Lightning Indexer implementations.

Currently, only the **TileLang-based operators** (`tilelang_ops/`) are open-sourced. The CUDA kernel implementations are not included in this release; they are available for free on Baidu Baige Platform.

## TileLang Operators

The `tilelang_ops/` directory provides the following operators built on [TileLang](https://github.com/tile-ai/tilelang):

- **Sparse MLA Forward** (`sparse_mla_fwd.py`)
- **Sparse MLA Backward** (`sparse_mla_bwd.py`)
- **Lightning Indexer** (`lightning_indexer.py`)

### Requirements

See `requirements.txt` for dependencies.
```
