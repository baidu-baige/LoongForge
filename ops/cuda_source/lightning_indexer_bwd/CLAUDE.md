# CLAUDE.md - lightning_indexer_bwd

<!-- Bilingual maintenance note (for contributors):
     This is the English version, committed to git.
     Chinese version lives in CLAUDE.zh.md (git-ignored, local only).
     ALWAYS keep both files in sync when making changes.
     Rule: whenever this file (CLAUDE.md) is modified, also update CLAUDE.zh.md. -->

> Context file for AI agents (Ducc / Claude Code), **auto-loaded at the start of every session**.
> After reading this file, the agent should be able to handle 90% of daily tasks without scanning the full codebase.

---

## Project Overview

`lightning_indexer_bwd` is a single-operator CUDA library targeting **NVIDIA Blackwell (SM100)**,
implementing the **backward pass of sparse FP8 MQA Attention logits** (`fp8_mqa_logits_bwd`).
It serves as a low-level kernel module for Baidu's AIACC DeepTraining framework.

- **Languages**: CUDA C++17 + Python (pybind11 binding)
- **Compilation**: NVRTC JIT (runtime compilation, not AOT)
- **Single public function**: `fp8_mqa_logits_bwd()`, returns `(grad_q, grad_kv, grad_weights)`

---

## Repository Layout

```
lightning_indexer_bwd/
+-- CLAUDE.md                         # this file (English, committed)
+-- CLAUDE.zh.md                      # Chinese version (git-ignored, local only)
+-- setup.py                          # build entry (setuptools + CUDAExtension)
+-- install.sh                        # clean build + pip install in one step
+-- csrc/
|   +-- python_api.cpp                # pybind11 binding (only .cpp compiled into .so)
|   +-- apis/attention_bwd.hpp        # host-side API: input validation + GPU arch dispatch
|   +-- jit_kernels/impls/
|       +-- smxx_fp8_mqa_logits_bwd.hpp  # JIT launcher: generate_impl() + launch_impl()
+-- lightning_indexer_bwd/
|   +-- __init__.py                   # Python public interface
|   +-- include/lightning_indexer_bwd/impls/
|       +-- sm100_fp8_mqa_logits_bwd_sparse.cuh  # core kernel (~750 lines, modify with care)
+-- tests/
|   +-- test_autograd.py              # numerical accuracy test (vs. PyTorch reference)
+-- vendor/                           # auto-generated at build time, not in git
|   +-- DeepGEMM/                     # deepseek-ai/DeepGEMM (JIT framework)
|   +-- deep_gemm_csrc/               # csrc copied from DeepGEMM
+-- third_party/                      # local dev reference sources, not in git (see third_party/README.md)
    +-- DeepGEMM/                     # full DeepGEMM source including LaunchRuntime base class
    +-- cutlass/                      # CuTe / CUTLASS, TMA and Swizzle reference implementations
```

---

## Quick Start

```bash
# Standard build and install
bash install.sh

# First-time build: auto-clone DeepGEMM (default expects manual clone into vendor/)
SKIP_DEEP_GEMM_CLONE=0 bash install.sh

# Skip CUDA compilation (Python-layer debugging only)
SKIP_CUDA_BUILD=1 pip install -e .

# Run accuracy tests
python tests/test_autograd.py
```

**Requirements:** CUDA >= 12.1, NVIDIA Blackwell GPU (SM100), PyTorch (matching CUDA version)

### Packaging for remote GPU test machines

Two scripts are provided for transferring to an offline/restricted GPU environment:

```bash
# Pack source code only (~20 KB)
bash pack.sh                          # output: ../lightning_indexer_bwd_<date>.tar.gz
bash pack.sh -o /tmp/src.tar.gz       # custom output path
bash pack.sh --dry-run                # list files without creating archive

# Pack third-party dependencies -- DeepGEMM + submodules (~34 MB, no .git history)
bash pack_deps.sh                     # output: ../deps_<date>.tar.gz
bash pack_deps.sh -o /tmp/deps.tar.gz
bash pack_deps.sh --dry-run           # show size breakdown
```

**On the target machine:**

```bash
# Unpack source
tar -xzf lightning_indexer_bwd_<date>.tar.gz -C /workspace/

# Unpack deps and place under vendor/ (skips git clone)
tar -xzf deps_<date>.tar.gz -C /workspace/lightning_indexer_bwd/
mv /workspace/lightning_indexer_bwd/deps/DeepGEMM  /workspace/lightning_indexer_bwd/vendor/
rmdir /workspace/lightning_indexer_bwd/deps

# Build (SKIP_DEEP_GEMM_CLONE=1 is the default -- reads from vendor/DeepGEMM directly)
cd /workspace/lightning_indexer_bwd && bash install.sh
```

---

## Core Constraints (Read Before Modifying)

### Hardcoded Parameters

All constraints are defined in `csrc/jit_kernels/impls/smxx_fp8_mqa_logits_bwd.hpp`.
**Changing any one of them requires syncing both the JIT launcher and the `static_assert`s in the `.cuh` kernel.**

| Parameter | Current Value | Notes |
|-----------|---------------|-------|
| `kNumHeads` | 64 | Template param, hardcoded in `generate_impl()` |
| `kHeadDim` | 128 | Same as above |
| `kBlockKV` | 128 | Affects TMA stride and swizzle layout |
| `kBlockQ` | 1 | |
| Target arch | SM100 (Blackwell) | Dispatch branch in `attention_bwd.hpp` |
| `seq_len_kv % 128` | Must be 0 | Asserted in `launch_impl()` |
| `max_seqlen_k` | Forced to 0 | Variable-length seqlen_k not yet implemented |

### Thread Block Structure (384 threads -- do not change)

```
Threads   0-127  (math warp-group)        epilogue: writes d_q / d_kv / d_weights
Threads 128-255  (specialized warp-group) TMA load + FP8 UMMA + BF16 UMMA scheduling
Threads 256-383  (sparse load warp-group) sparse KV token load + FP8->BF16 conversion
```

The three warp-groups are synchronized via `ClusterTransactionBarrier`, forming a double-buffered Q/KV pipeline. **384 is a hard pipeline constraint, not an arbitrary choice.**

### Shared Memory Swizzle (changes must be paired with TMA descriptor updates)

| Data type | Swizzle config | Affects |
|-----------|----------------|---------|
| FP8 | `Swizzle<3,4,3>` | TMA descriptor stride |
| BF16 | `Swizzle<2,4,2>` | UMMA matrix layout |

---

## Python Interface

```python
from lightning_indexer_bwd import fp8_mqa_logits_bwd

grad_q, grad_kv, grad_weights = fp8_mqa_logits_bwd(
    grad_logits,   # Tensor [bs, num_heads, seqlen_q, topk],     float32
    q,             # Tensor [bs, num_heads, seqlen_q, head_dim], fp8_e4m3
    kv,            # Tensor [bs, seqlen_k, head_dim],            fp8_e4m3
    kv_scales,     # Tensor, per-token KV dequantization scale,  float32
    weights,       # Tensor, attention weights,                  float32
    topk_indices,  # Tensor, sparse top-k indices,               int32
    cu_seqlens_q,  # Tensor, cumulative sequence lengths for q
    cu_seqlens_k,  # Tensor, cumulative sequence lengths for k
)
# Returns: grad_q (float32), grad_kv (float32), grad_weights (float32)
```

Input validation is performed via `DG_HOST_ASSERT` in `csrc/apis/attention_bwd.hpp`.

---

## Task Guides

### Task: Add support for new shapes (num_heads / head_dim)

1. `smxx_fp8_mqa_logits_bwd.hpp` -> `generate_impl()`: update the template parameter string
2. `smxx_fp8_mqa_logits_bwd.hpp` -> `launch_impl()`: update assertions and argument passing
3. `sm100_fp8_mqa_logits_bwd_sparse.cuh`: search for all `static_assert` and `constexpr`, verify each one

### Task: Add SM90 (Hopper) support

- `attention_bwd.hpp` already has an SM90 dispatch branch (currently throws) -- this is the only entry point
- Create a new `sm90_fp8_mqa_logits_bwd_sparse.cuh` (use SM100 version as reference)
- **Key difference**: SM90 has no TMEM; MMA accumulators must be stored in SMEM or registers; use `SM90_MMA_*` instructions

### Task: Debug JIT compilation errors

JIT errors originate from NVRTC and usually include a full compiler log:
1. Print the generated source string in `generate_impl()` to verify template instantiation
2. Check DeepGEMM's `LaunchRuntime` base class to locate the NVRTC call and error reporting
3. Common causes: missing header paths (verify `vendor/` was generated correctly), template parameter type mismatch

### Task: Modify or add tests

- Test file: `tests/test_autograd.py`
- Strategy: numerical comparison against a PyTorch reference with explicit `atol`/`rtol`
- **Confirm SM100 GPU is available before running** -- the kernel dispatch will throw on other architectures

---

## Development Sandbox Workflow

**All feature development, refactoring, and optimization work starts in `workspace/`.**
Never modify main library files directly during active development.

### Directory layout

```
workspace/
+-- <topic>/                # active working directory, kebab-case name
|   +-- MANIFEST            # declares how step files map to main library paths (required)
|   +-- step_01_<desc>.cuh  # sequential snapshots, each self-contained
|   +-- step_02_<desc>.cuh
|   +-- ...
|   +-- notes.md            # optional: decisions and observations per step
+-- archive.tar.gz          # single compressed backup of all graduated topics
```

### MANIFEST file (required in every workspace topic)

`pack.sh` reads MANIFEST to include active development files in the archive at the
correct main library paths. Without MANIFEST, the topic is silently skipped.

```
# workspace/<topic>/MANIFEST
# Format: <source> -> <destination relative to repo root>
# <source>:
#   step_latest        -- highest-numbered step_NN_* file in this directory
#   step_03_foo.cuh    -- an explicit step filename
#
# Example:
step_latest -> lightning_indexer_bwd/include/lightning_indexer_bwd/impls/sm100_fp8_mqa_logits_bwd_sparse.cuh
```

Multi-file topics: one rule per line.

### Iteration

Add a new `step_NN_*.cuh` for each meaningful change. Summarize what changed and why
in each response. Never delete earlier steps.

### Graduating to the main library

When the user confirms the final step is satisfactory, execute these steps **in order**:

**Step A -- Update the main library**

| Task type | Action |
|-----------|--------|
| Refactor / optimize | Overwrite the existing file(s) in the main library |
| New operator / new feature | Add new file(s) to the appropriate location in the main library |

**Step B -- Merge into single archive**

Append the topic directory into `workspace/archive.tar.gz` (creating it if absent),
then delete the topic directory:

```bash
# Run from repo root -- Ducc executes this block
python3 - workspace/<topic> workspace/archive.tar.gz <<'EOF'
import sys, os, tarfile, tempfile, shutil
topic_dir  = sys.argv[1]
archive    = sys.argv[2]
topic_name = os.path.basename(topic_dir)
os.makedirs(os.path.dirname(archive) or '.', exist_ok=True)
with tempfile.TemporaryDirectory() as tmp:
    if os.path.exists(archive):
        with tarfile.open(archive, 'r:gz') as t: t.extractall(tmp)
    shutil.copytree(topic_dir, os.path.join(tmp, topic_name))
    with tarfile.open(archive, 'w:gz') as t: t.add(tmp, arcname='')
print(f"Archived {topic_name} into {archive}")
EOF
rm -rf workspace/<topic>/
```

`workspace/archive.tar.gz` accumulates all graduated topics. It is git-ignored and
never travels with `pack.sh`. To inspect contents: `tar -tzf workspace/archive.tar.gz`

**Step C -- Verify**

```bash
python tests/test_autograd.py
```

### How pack.sh handles active workspace topics

`pack.sh` scans `workspace/` for active topic directories. For each topic with a MANIFEST:
- Resolves `step_latest` to the highest-numbered `step_NN_*` file
- Places that file in the archive **at the main library path** declared in MANIFEST
- If a main library file exists at that path, the workspace version wins

Result: the test machine unpacks and runs `bash install.sh` directly -- no manual
file placement needed. `workspace/` itself is never included in the archive.

---

## Hard Rules -- Never Violate

- **Never modify files under `vendor/`** -- they are overwritten at build time
- **Never modify Swizzle parameters without syncing the TMA descriptor stride** -- produces silent wrong results
- **Never change the total thread count (384) or the three-way warp-group split** -- breaks pipeline synchronization
- **Never run the full test suite without confirming SM100 GPU availability** -- will throw on non-Blackwell hardware
- **Never submit kernel changes without running the accuracy test**
- **Never modify original library files directly during active development** -- always use the workspace sandbox workflow above

---

## Code Style

- Header suffixes: `.hpp` for C++ host code, `.cuh` for CUDA device code
- Host-side assertions: use `DG_HOST_ASSERT` (from DeepGEMM), not standard `assert`
- Compile-time constant template parameters: `k` prefix -- `kBlockQ`, `kBlockKV`, `kNumHeads`, `kHeadDim`
- Thread roles are distinguished by `threadIdx.x` range; each range must have a comment naming its warp-group
- Build system: setuptools `CUDAExtension` only -- no CMake

### Git Commit Format

```
<type>(<scope>): <subject>    # subject <= 72 chars, imperative voice

[body]                        # explain why, not what
```

`type`: `feat`, `fix`, `perf`, `refactor`, `test`, `docs`, `build`
`scope`: `kernel`, `jit`, `api`, `test`, `build`

---

## Domain Routing (What to Look at for Task X)

| Task | Primary file | Secondary file |
|------|-------------|----------------|
| Modify Python interface signature | `lightning_indexer_bwd/__init__.py` | `csrc/python_api.cpp` |
| Input validation / arch dispatch | `csrc/apis/attention_bwd.hpp` | -- |
| Modify JIT template parameters | `csrc/jit_kernels/impls/smxx_fp8_mqa_logits_bwd.hpp` | `.cuh` kernel |
| Modify kernel compute logic | `sm100_fp8_mqa_logits_bwd_sparse.cuh` | JIT launcher |
| Modify SMEM layout / Swizzle | `.cuh` kernel | JIT launcher (TMA descriptor) |
| Build dependencies / vendor | `setup.py` | `install.sh` |
| Accuracy debugging | `tests/test_autograd.py` | `.cuh` kernel epilogue |
| JIT compile error | `generate_impl()` in JIT launcher | DeepGEMM `LaunchRuntime` base class |
| New architecture support (e.g. SM90) | `attention_bwd.hpp` (placeholder branch exists) | create new `sm90_*.cuh` |

---

## Core Concepts

**JIT compilation flow**: Python call -> `__init__.py` -> pybind11 -> `attention_bwd.hpp` (validate + dispatch) -> `smxx_fp8_mqa_logits_bwd.hpp` (`generate_impl` builds source string -> NVRTC compiles -> `launch_impl` executes). Compiled once on first call, cached thereafter.

**Why the three-way thread block is indivisible**: The three warp-groups collaborate via `ClusterTransactionBarrier` to form a double-buffered Q/KV pipeline. Removing any segment causes deadlock or data races, not merely a performance regression.

**Sparse indexing**: `topk_indices` specifies which KV token positions each query attends to. The sparse load warp-group loads them non-contiguously via `cp.async` PTX, rather than streaming all KV tokens sequentially.

**FP8 quantization**: `q` and `kv` are stored as `fp8_e4m3`; `kv_scales` provides per-token dequantization scales. All gradient outputs are `float32`.

---

## Known Limitations / TODO

- [ ] SM90 (Hopper) support (placeholder dispatch branch already exists in `attention_bwd.hpp`)
- [ ] Parameterize `kNumHeads` / `kHeadDim` (currently hardcoded to 64 / 128)
- [ ] Variable-length `seqlen_k` support (`max_seqlen_k` currently forced to 0)
- [ ] Missing user-facing README.md

---

*Last updated: 2026-03-06*
