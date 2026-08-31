# Copyright 2026 The LoongForge Authors.
# SPDX-License-Identifier: Apache-2.0
#
# Modified from Wall-X under the Apache-2.0 License.

"""Install-time compiled CUDA kernels for wall_oss_0_5 ops.

The extension modules are installed via ``pip install --no-build-isolation -e .``
and accessed as ``wall_oss_05_ops._cuda_ext_bin``.  Falls back to the loongforge
package path when the standalone package is not installed.
"""

_module = None


def load():
    """Load the installed CUDA extension module."""
    global _module
    if _module is not None:
        return _module

    # Primary: installed as part of the wall_oss_05_ops package.
    try:
        from wall_oss_05_ops import _cuda_ext_bin  # noqa: PLC0415
        _module = _cuda_ext_bin
        return _module
    except ImportError:
        pass

    # Fallback: installed inside the loongforge package.
    try:
        from loongforge.embodied.model.wall_oss_0_5.core.ops import _cuda_ext_bin  # noqa: PLC0415
        _module = _cuda_ext_bin
        return _module
    except ImportError as exc:
        raise ImportError(
            "wall_oss_0_5 CUDA operators were not found. Install the package:\n"
            "  pip install --no-build-isolation -e .\n"
            "from the cuda_source/wall_oss_05_ops directory."
        ) from exc


def is_available() -> bool:
    """Check whether the install-time CUDA extension can be imported."""
    try:
        load()
        return True
    except Exception:
        return False


_exact_module = None


def load_exact():
    """Load the bitwise-exact CUDA extension.

    The exact kernels (fused SwiGLU / RMSNorm) live in their own module because
    they must be compiled without ``--use_fast_math``.  Falls back gracefully
    so callers can choose the PyTorch path when unavailable.
    """
    global _exact_module
    if _exact_module is not None:
        return _exact_module

    # Primary: installed as part of the wall_oss_05_ops package.
    try:
        from wall_oss_05_ops import _cuda_ext_exact_bin  # noqa: PLC0415
        _exact_module = _cuda_ext_exact_bin
        return _exact_module
    except ImportError:
        pass

    # Fallback: installed inside the loongforge package.
    try:
        from loongforge.embodied.model.wall_oss_0_5.core.ops import _cuda_ext_exact_bin  # noqa: PLC0415
        _exact_module = _cuda_ext_exact_bin
        return _exact_module
    except ImportError as exc:
        raise ImportError(
            "wall_oss_0_5 bitwise-exact CUDA operators were not found. Install the package:\n"
            "  pip install --no-build-isolation -e .\n"
            "from the cuda_source/wall_oss_05_ops directory."
        ) from exc


def is_exact_available() -> bool:
    """Check whether the bitwise-exact CUDA extension can be imported."""
    try:
        load_exact()
        return True
    except Exception:
        return False
