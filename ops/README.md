# DeepTraining

## Introduction

DeepTraining is a library of optimized kernels, powering the models training.
This directory contains the following implementations:

### DSA kernels

*These kernels power DeepSeek Sparse Attention (DSA), as introduced in
[DeepSeek-V3.2-Exp](https://github.com/deepseek-ai/DeepSeek-V3.2-Exp).*

- Sparse Attention forward — `cuda_source/sparse_mla_fwd/`
- Sparse Attention backward — `cuda_source/sparse_mla_bwd/`
- Lightning Indexer backward — `cuda_source/lightning_indexer_bwd/`

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
  SM100 kernels)
- Operator packages: SM80 / SM120 by default, overridable via
  `TORCH_CUDA_ARCH_LIST`
- PyTorch 2.0 and above

## Installation

All commands below are run from the repository root (`DeepTraining/`).

### Install sparse mla fwd

Sparse MLA forward is dependent on FlashMLA. Please manually clone FlashMLA first:

```bash
cd cuda_source/sparse_mla_fwd
git clone https://github.com/deepseek-ai/FlashMLA.git FlashMLA
cd FlashMLA
git checkout 47c35a712362f11bc235854ead51819ad76f5a81
git submodule update --init --recursive
cd ../
pip install -v .
```

### Install sparse mla bwd

Sparse MLA backward is dependent on FlashMLA. Please manually clone FlashMLA first:

```bash
cd cuda_source/sparse_mla_bwd
git clone https://github.com/deepseek-ai/FlashMLA.git FlashMLA
cd FlashMLA
git checkout 47c35a712362f11bc235854ead51819ad76f5a81
git submodule update --init --recursive
cd ../
pip install -v .
```

### Install lightning indexer bwd

Lightning Indexer backward is dependent on DeepGEMM. Please manually clone
DeepGEMM first, then run install.sh which will automatically install deep_gemm:

```bash
cd cuda_source/lightning_indexer_bwd
mkdir -p vendor
git clone --recurse-submodules https://github.com/deepseek-ai/DeepGEMM.git vendor/DeepGEMM
sh install.sh  # deep_gemm will be installed automatically
```

### Install the operator packages

Both packages are editable installs with no external dependency to clone:

```bash
cd cuda_source/groot_n1_7_op   && pip install --no-build-isolation -e . && cd ../..
cd cuda_source/wall_oss_05_op  && pip install --no-build-isolation -e . && cd ../..
```

After installation the operators are importable from any working directory:

```python
from groot_n1_7_op import qwen3_vl_fused_text_rope_forward, eager_step
from wall_oss_05_op import rope, rmsnorm, swiglu
```

## Test & benchmark

#### MLA prefill (Sparse)

```bash
python cuda_source/sparse_mla_fwd/tests/test_flash_mla_sparse_fwd.py
python cuda_source/sparse_mla_bwd/tests/test_flash_mla_sparse_bwd.py
```

#### Indexer backward

```bash
python cuda_source/lightning_indexer_bwd/tests/test_fp8_mqa_logits_bwd.py
```

#### Operator packages

Both suites require an installed package and a CUDA device; CUDA tests are
skipped when no device is available.

```bash
pytest -q cuda_source/groot_n1_7_op/tests/     # 87 tests
pytest -q cuda_source/wall_oss_05_op/test/     # 30 tests
```
