# `groot_n1_7_op`

AOT packaging of the GR00T-N1.7 fused inference, optimizer, and DDP reducer
operators. The CUDA sources are migrated from the current LoongForge
GR00T-N1.7 implementation and keep its public operation names and rounding
behavior.

## Layout

- `groot_n1_7_op/qwen3_vl_fused_ops.py`: Python interface for the four Qwen3-VL
  inference entry points.
- `groot_n1_7_op/groot_fused_adamw.py`: eager and capturable fused AdamW paths.
- `groot_n1_7_op/groot_ddp_reducer_bucket_control.py`: bindings for internal
  `c10d::Reducer` bucket initialization and inspection.
- `src/qwen3_vl_fused_ops_bindings.cpp` and `qwen3_vl_fused_ops.cu`: vision/text RoPE, RMSNorm, and
  SiLU-multiply kernels.
- `src/groot_n1_7_fused_adamw_bindings.cpp` and `groot_n1_7_fused_adamw.cu`: precision-compatible AdamW
  kernels using PyTorch `MultiTensorApply`.
- `src/ddp_reducer_bucket_control.cpp`: pure C++ binding for reducer buckets.
- `tests/`: correctness and validation tests for all public entry points.

## Build and Test

```bash
cd cuda_source/groot_n1_7_op

# Editable install
pip install --no-build-isolation -e .

# Build options (prefix the install command to override)
#   TORCH_CUDA_ARCH_LIST="8.0 12.0"  target architectures (default: sm_80 + sm_120)
#   NVCC_THREADS=8                   parallel NVCC compile threads (default: 8)
```

## Python API

```python
from groot_n1_7_op import (
    capturable_grad_scaled_step,
    capturable_step,
    eager_step,
    get_buckets,
    initialize_buckets,
    qwen3_vl_fused_text_rope_forward,
    qwen3_vl_fused_text_rms_norm_forward,
    qwen3_vl_fused_text_silu_mul_forward,
    qwen3_vl_fused_vision_rope_forward,
)
```

The Qwen3-VL operators accept CUDA tensors in fp32, fp16, or bf16 where
documented by the individual validation checks. Vision RoPE uses `[S, H, D]`
q/k with fp32 `[S, D]` cos/sin. Text RoPE uses `[B, H, S, D]` q/k and
`[B, S, D]` cos/sin with a matching dtype. RMSNorm requires contiguous input,
fp32 weight, and returns fp32 output. SiLU-multiply requires contiguous fp16
or bf16 inputs with matching shape and dtype.

The AdamW operators mutate parameter and optimizer-state tensors in place.
They support contiguous CUDA fp32 tensors only. Capturable paths require CUDA
scalar tensors with the dtypes described by the wrapper signature.

Reducer bucket control is tied to PyTorch's internal `c10d::Reducer` ABI. It
must be built and used with a compatible PyTorch installation.

## Tests

```bash
cd cuda_source/groot_n1_7_op
pytest -q tests/
```

Expected: 87 passed (20 original + 67 extended tests covering all public entry points).
CUDA tests are skipped when CUDA is unavailable.
