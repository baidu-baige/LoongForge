# INT4 Fake Quantization for QAT

GPU-accelerated INT4 fake quantization kernels for Quantization-Aware Training (QAT) of MoE expert weights.

## Install

```bash
cd LoongForge/ops/cuda_source/int4_qat
pip install -e .
```

Requires: PyTorch with CUDA support, NVCC matching the PyTorch CUDA version.

To verify:

```bash
python -c "from int4_qat import fake_int4_quant; print('OK')"
```

## Usage

### In training (via LoongForge)

Pass CLI arguments to `train.py`:

```bash
--enable-int4-qat           # Enable INT4 QAT
--int4-qat-group-size 32    # Group size (default: 32)
```

This automatically applies fake INT4 quantization to `TEGroupedLinear` (MoE expert) weights during training forward, with STE gradient passthrough.

### Standalone API

```python
import torch
from int4_qat import fake_int4_quant, fake_int4_quantize_dequantize

w = torch.randn(256, 1024, device="cuda", dtype=torch.bfloat16)

# Low-level: quantize only → integer codes + scale + zero
q, scale, zero = fake_int4_quant(w, block_size=[1, 32], sym=True)

# Mid-level: quantize + dequantize → fake-quantized BF16 tensor
w_fq = fake_int4_quantize_dequantize(w, group_size=32, sym=True)
```

For training, use `apply_int4_qat()` which patches `TEGroupedLinear` modules with STE-enabled fake quantization:

```python
from int4_qat import apply_int4_qat

transform, count = apply_int4_qat(model, group_size=32, sym=True)
# Patches _get_weight_tensors on matching modules (MoE expert FC layers by default)
```

## CUDA Kernels

Three dispatch paths in `csrc/fake_int4_quant.cu`, selected by block size:

| Kernel | Block Size | Use Case |
|--------|-----------|----------|
| `int4_quant_1x32_kernel` | `[1, 32]` | Per-row grouping (most common for weight quant) |
| `int4_quant_32x1_kernel` | `[32, 1]` | Transpose / per-column grouping |
| `int4_quant_common_kernel` | arbitrary | Generic, requires `block_m * block_n % 32 == 0` |

Quantization modes:
- **Symmetric** (`sym=True`): range `[-7, 7]`, `scale = max(|x|) / 7`
- **Asymmetric** (`sym=False`): range `[0, 15]`, `scale = (max - min) / 15`, with zero-point

If the CUDA extension is not built, all APIs fall back to a pure PyTorch implementation automatically.

## Tests

```bash
cd LoongForge/ops/cuda_source/int4_qat
pytest tests/ -v
```