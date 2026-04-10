# DeepTraining

## Introduction

DeepTraining is library of optimized kernels, powering the models training. This repository contains the following implementations:

**Kernels**

*These kernels power DeepSeek Sparse Attention (DSA), as introduced in [this paper](https://github.com/deepseek-ai/DeepSeek-V3.2-Exp).*

- Sparse Attention forward
- Sparse Attention backward
- Lihtning Indexer backward

#### Test & benchmark MLA prefill (Sparse):

```bash
python tests/test_flash_mla_sparse_prefill.py
```

#### Test & benchmark Indexer backward:

```bash
python lightning_indexer_bwd/tests/test_autograd.py
```

## Requirements

- SM90 / SM100 (See the support matrix below)
- CUDA 12.8 and above (CUDA 12.9+ is required for SM100 kernels)
- PyTorch 2.0 and above

## Installation
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
Lightning Indexer backward is dependent on DeepGEMM. Please manually clone DeepGEMM first, then run install.sh which will automatically install deep_gemm:

```bash
cd cuda_source/lightning_indexer_bwd
mkdir -p vendor
git clone --recurse-submodules https://github.com/deepseek-ai/DeepGEMM.git vendor/DeepGEMM
sh install.sh  # deep_gemm will be installed automatically
```
