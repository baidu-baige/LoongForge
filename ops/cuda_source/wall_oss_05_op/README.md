# wall_oss_05_op

Standalone CUDA operator library for Wall-OSS-0.5 (Qwen2.5-VL / GR00T-N1.7),
extracted from the LoongForge training framework.

Operators: RoPE, M-RoPE, RotPosEmb, RMSNorm, SwiGLU, MoE Permute/Unpermute,
GetRopeIndex, GetWindowIndex.

Each operator exposes a **two-level fallback**:
1. CUDA inline kernel (compiled extension, requires GPU)
2. Pure PyTorch (always available, no GPU required)

## Requirements

- Python 3.10+
- PyTorch 2.0+ with CUDA support
- CUDA 12.x, NVCC

## Build

```bash
cd cuda_source/wall_oss_05_op

# Editable install
pip install --no-build-isolation -e .

# Build options (prefix the install command to override)
#   TORCH_CUDA_ARCH_LIST="8.0 12.0"  target architectures (default: sm_80 + sm_120)
#   NVCC_THREADS=8                   parallel NVCC compile threads (default: 8)
```

## Usage

```python
from wall_oss_05_op import rope, m_rope, rot_pos_emb
from wall_oss_05_op import rmsnorm, swiglu
from wall_oss_05_op import permute, unpermute
from wall_oss_05_op import get_rope_index, get_window_index

# Check which backend is active (cuda_inline or pytorch)
from wall_oss_05_op import backend_inventory
print(backend_inventory())
```

## Test

```bash
pytest -q test/test_wall_oss_05_op.py
```

## Layout

```
wall_oss_05_op/
├── setup.py                       # build entry point
├── csrc/                          # CUDA kernel sources
│   ├── binding.cu                 # standard extension pybind
│   ├── binding_exact.cu           # bitwise-exact extension pybind
│   ├── rope/                      # RoPE kernels
│   ├── m_rope/                    # M-RoPE kernels
│   ├── rmsnorm_exact/             # bitwise-exact RMSNorm
│   ├── swiglu_exact/              # bitwise-exact SwiGLU
│   ├── permute_unpermute/         # MoE routing kernels
│   ├── rot_pos/                   # vision rotary position kernels
│   ├── get_rope_index/            # 3D RoPE index kernels
│   └── window_index/              # window attention index kernels
├── wall_oss_05_op/               # Python package (importable after install)
│   ├── __init__.py                # public API re-exports
│   ├── _cuda_ext.py               # extension loader (standalone + optional loongforge fallback)
│   ├── _cuda_wrappers.py          # autograd wrappers around raw kernels
│   ├── base.py                    # OpsProxy base class (lazy backend resolution)
│   ├── rope.py                    # RoPE / M-RoPE / RotPosEmb operators
│   ├── norm.py                    # RMSNorm operator
│   ├── activation.py              # SwiGLU operator
│   ├── moe.py                     # MoE permute/unpermute operators
│   └── index.py                   # GetRopeIndex / GetWindowIndex operators
└── test/
    └── test_wall_oss_05_op.py
```

## Notes

- The **exact** kernels (`rmsnorm_exact`, `swiglu_exact`) are compiled into a
  separate `_cuda_ext_exact_bin` module **without** `--use_fast_math`, preserving
  bitwise identity with eager PyTorch in forward and backward passes.
- The standard kernels use `--use_fast_math` for maximum throughput.
- `TORCH_CUDA_ARCH_LIST` defaults to `"8.0 12.0"` (sm_80 + sm_120).
  Override before building if other architectures are needed.
