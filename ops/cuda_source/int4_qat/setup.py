"""
Setup for INT4 fake quantization CUDA kernels.

Provides GPU-accelerated fake INT4 quantization (quantize → integer codes)
and fused quantize-dequantize (3.4x faster) with symmetric and asymmetric modes,
used for Quantization-Aware Training (QAT) of MoE expert weights.
"""
import os
import subprocess
from setuptools import setup, find_packages
from torch.utils.cpp_extension import BuildExtension, CUDAExtension, CUDA_HOME


def get_arch_flags():
    """Detect GPU architectures and generate NVCC gencode flags."""
    import torch

    arch_set = set()
    if torch.cuda.is_available():
        for i in range(torch.cuda.device_count()):
            major, minor = torch.cuda.get_device_capability(i)
            arch_set.add(f"{major}{minor}")

    if not arch_set:
        env_arch = os.environ.get("TORCH_CUDA_ARCH_LIST", "")
        if env_arch:
            for a in env_arch.replace(";", " ").split():
                a = a.strip().replace("+PTX", "").replace(".", "")
                if a:
                    arch_set.add(a)
        else:
            arch_set = {"80", "86", "89", "90"}

    flags = []
    for arch in sorted(arch_set):
        flags += [f"-gencode=arch=compute_{arch},code=sm_{arch}"]
    # Always include sm_90a for Hopper/Blackwell
    flags += ["-gencode=arch=compute_90a,code=sm_90a"]
    return flags


this_dir = os.path.dirname(os.path.abspath(__file__))

nvcc_flags = [
    "-O3",
    "-std=c++17",
    "--expt-relaxed-constexpr",
    "-Xcompiler",
    "-fPIC",
    # NOTE: do NOT add --use_fast_math — it changes division rounding and
    # breaks bit-exact match between the fused kernel and the original.
] + get_arch_flags()

ext_modules = [
    CUDAExtension(
        name="int4_qat.cuda",
        sources=[os.path.join("csrc", "fake_int4_quant.cu")],
        extra_compile_args={
            "cxx": ["-O3", "-std=c++17"],
            "nvcc": nvcc_flags,
        },
    ),
    CUDAExtension(
        name="int4_qat.cuda_fused",
        sources=[os.path.join("csrc", "fake_int4_quant_dequant_fused.cu")],
        extra_compile_args={
            "cxx": ["-O3", "-std=c++17"],
            "nvcc": nvcc_flags,
        },
    ),
]

try:
    cmd = ["git", "rev-parse", "--short", "HEAD"]
    rev = "+" + subprocess.check_output(cmd, cwd=this_dir).decode("ascii").rstrip()
except Exception:
    rev = ""

setup(
    name="int4_qat",
    version="1.0.0" + rev,
    description="INT4 fake quantization CUDA kernels for QAT",
    packages=find_packages(include=["int4_qat"]),
    ext_modules=ext_modules,
    cmdclass={"build_ext": BuildExtension},
)
