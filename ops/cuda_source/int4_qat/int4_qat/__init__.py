"""
INT4 fake quantization for Quantization-Aware Training (QAT).

Provides GPU-accelerated and pure-PyTorch fake INT4 quantization with
Straight-Through Estimator (STE) for training MoE expert weights.
"""
__version__ = "1.0.0"

from int4_qat.interface import (
    fake_int4_quant,
    fake_int4_quantize_dequantize,
)
from int4_qat.weight_transform import (
    FakeQuantWeightTransform,
    _FakeQuantSTE as FakeInt4QuantSTE,
    apply_int4_qat,
)
