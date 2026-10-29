# Copyright 2026 The LoongForge Authors.
# SPDX-License-Identifier: Apache-2.0

"""HF bridge arguments shared by training and checkpoint conversion."""

import argparse


def add_bridge_args(parser: argparse.ArgumentParser) -> argparse.ArgumentParser:
    """Register bridge arguments on *parser*."""
    group = parser.add_argument_group(title="extra-bridge")
    group.add_argument('--fp8-force-no-requant', '--fp8_force_no_requant', dest='fp8_force_no_requant',
                       action='store_true',
                       help=("If enabled, in converting FP8 checkpoint, skip the `dequantize + re-quantize`, "
                             "directly chunk/concate the quantized data.")
    )
    group.add_argument('--force-pow-2-scales', '--force_pow_2_scales', dest='force_pow_2_scales',
                       action='store_true',
                       help=("Define whether to force destination checkpoint's scale to be power-of-two.")
    )
    group.add_argument('--amax-epsilon', '--amax_epsilon', dest='amax_epsilon',
                       type=float, default=0.0,
                       help=("Epsilon value added to the amax calculation to avoid divised by zero "
                             "when converting to FP8. Only used in Transformer Engine FP8 conversion.")
    )
    group.add_argument('--hf-dequantize-int4', '--hf_dequantize_int4', dest='hf_dequantize_int4',
                       action='store_true',
                       help=("If enabled, on-the-fly dequantize compressed-tensors packed INT4 "
                             "weights to bf16/fp32 during HF checkpoint online loading.")
    )
    group.add_argument('--hf-dequantize-mxfp4', '--hf_dequantize_mxfp4', dest='hf_dequantize_mxfp4',
                       action='store_true',
                       help=("If enabled, on-the-fly dequantize MXFP4 packed FP4 weights "
                             "(E2M1 two-per-byte in uint8 + per-block E8M0 scale, block_size=32, "
                             "e.g. DeepSeek-V4 routed experts) to bf16/fp32 during HF checkpoint "
                             "online loading.")
    )
    group.add_argument('--hf-dequantize-dtype', '--hf_dequantize_dtype', dest='hf_dequantize_dtype',
                       default='bfloat16',
                       help=("Target dtype for INT4 dequantization (bfloat16 or float32). "
                             "Default: bfloat16")
    )
    group.add_argument('--hf-quant-config-file', '--hf_quant_config_file', dest='hf_quant_config_file',
                       default=None,
                       help=("Path to config.json carrying the compressed-tensors quantization_config; "
                             "used to locate the INT4-packed weight keys. Default: None")
    )
    group.add_argument('--quant-method', '--quant_method', dest='quant_method',
                       default='te', choices=['te', 'pt'],
                       help="fp8 quantization backend for bf16->fp8 conversion during bridge load. "
                            "Default: te (TransformerEngine)")
    group.add_argument('--pretrain-as-fp8', '--pretrain_as_fp8', dest='pretrain_as_fp8',
                       action='store_true',
                       help="Treat the loaded checkpoint as fp8 for pretrain (bf16 saved ckpt + pretrain fp8). "
                            "Default: False")
    group.add_argument(
        "--encoder-tensor-model-parallel-size",
        type=int,
        default=None,
        help="Encoder tensor-parallel size the HF bridge uses to shard encoder weights "
             "for heterogeneous TP. Default: None (use --tensor-model-parallel-size)",
    )
    return parser
