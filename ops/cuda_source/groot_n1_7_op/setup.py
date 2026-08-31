"""Build the GR00T-N1.7 CUDA and DDP operator extensions."""

import os
from pathlib import Path

from setuptools import find_packages, setup
from torch.utils.cpp_extension import BuildExtension, CppExtension, CUDAExtension


ROOT = Path(__file__).resolve().parent
SRC = ROOT / "src"

CXX_FLAGS = ["-O3", "-std=c++17", "-DNDEBUG", "-Wno-deprecated-declarations"]
NVCC_FLAGS = [
    "-O3",
    "-std=c++17",
    "-DNDEBUG",
    "-Wno-deprecated-declarations",
    "--expt-relaxed-constexpr",
    "--expt-extended-lambda",
    # Undefine CUDA macros that suppress half / bf16 built-in operators so that
    # fp16 and bf16 arithmetic can be written without explicit function calls.
    "-U__CUDA_NO_HALF_OPERATORS__",
    "-U__CUDA_NO_HALF_CONVERSIONS__",
    "-U__CUDA_NO_HALF2_OPERATORS__",
    "-U__CUDA_NO_BFLOAT16_CONVERSIONS__",
    # Parallel intra-process NVCC threads (each .cu gets multiple threads).
    "--threads", os.getenv("NVCC_THREADS", "8"),
]

# Target only sm_80 and sm_120.
# Override with TORCH_CUDA_ARCH_LIST if a different set is needed.
if os.getenv("TORCH_CUDA_ARCH_LIST"):
    print(f"Using TORCH_CUDA_ARCH_LIST={os.environ['TORCH_CUDA_ARCH_LIST']}")
else:
    os.environ["TORCH_CUDA_ARCH_LIST"] = "8.0 12.0"
    print("Targeting sm_80 and sm_120 only.")


ext_modules = [
    CUDAExtension(
        name="groot_n1_7_op._qwen3_vl_fused_ops",
        sources=[
            str(SRC / "qwen3_vl_fused_ops_bindings.cpp"),
            str(SRC / "qwen3_vl_fused_ops.cu"),
        ],
        include_dirs=[str(SRC)],
        extra_compile_args={"cxx": CXX_FLAGS, "nvcc": NVCC_FLAGS},
    ),
    CUDAExtension(
        name="groot_n1_7_op._groot_n1_7_fused_adamw",
        sources=[
            str(SRC / "groot_n1_7_fused_adamw_bindings.cpp"),
            str(SRC / "groot_n1_7_fused_adamw.cu"),
        ],
        include_dirs=[str(SRC)],
        extra_compile_args={"cxx": CXX_FLAGS, "nvcc": NVCC_FLAGS},
    ),
    CppExtension(
        name="groot_n1_7_op._ddp_reducer_bucket_control",
        sources=[str(SRC / "ddp_reducer_bucket_control.cpp")],
        include_dirs=[str(SRC)],
        extra_compile_args={"cxx": CXX_FLAGS},
    ),
]


setup(
    name="groot_n1_7_op",
    version="1.0.0",
    description="GR00T-N1.7 fused CUDA and DDP operators",
    packages=find_packages(include=["groot_n1_7_op"]),
    ext_modules=ext_modules,
    cmdclass={"build_ext": BuildExtension},
    python_requires=">=3.9",
)
