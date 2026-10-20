import time
from typing import Any, Iterable, Optional, Tuple, Union

import torch
import torch.nn as nn
import torch.nn.functional as F
from calflops.calculate_pipline import CalFlopsPipline, module_flop_count, module_mac_count, old_functions
from calflops.pytorch_ops import wrapFunc
from calflops.utils import flops_to_string, macs_to_string, params_to_string
from torch import Tensor

try:
    from transformer_engine.pytorch.attention import DotProductAttention
except ImportError:
    DotProductAttention = None

try:
    import natten
except ImportError:
    natten = None


class FlopsAnalyzer:
    """Wrap a model to print flops/macs/params on first forward call."""

    def __init__(self, model: nn.Module):
        """Initialize FlopsAnalyzer with a model.

        Args:
            model: PyTorch model to analyze.
        """
        model._original_forward = model.forward
        model.forward = self.forward
        self.model = model
        self.first_time = True

    def reset(self) -> None:
        """Reset the analyzer to re-compute flops on next forward call."""
        self.first_time = True

    def clear(self) -> None:
        """Remove the analyzer wrapper and restore original forward method."""
        self.model.forward = self.model._original_forward
        self.reset()

    def forward(self, *args: Any, **kwargs: Any) -> Any:
        """Forward pass with flops counting on first call and timing on all calls.

        Args:
            *args: Positional arguments passed to model forward.
            **kwargs: Keyword arguments passed to model forward.

        Returns:
            Any: Model output.
        """
        if self.first_time:
            self.model.forward = self.model._original_forward
            flops, macs, params = flops_count(
                model=self.model,
                args=args,
                kwargs=kwargs,
                print_results=True,
                print_detailed=True,
            )
            print(f'{flops}, {macs}, {params}')
            self.model.forward = self.forward
            self.first_time = False
        cpu_start = time.time()
        gpu_start = torch.cuda.Event(enable_timing=True)
        gpu_end = torch.cuda.Event(enable_timing=True)
        torch.cuda.synchronize()
        gpu_start.record()
        output = self.model._original_forward(*args, **kwargs)
        gpu_end.record()
        torch.cuda.synchronize()
        gpu_time = gpu_start.elapsed_time(gpu_end) / 1000.0
        cpu_time = time.time() - cpu_start
        print(f'GPU time: {gpu_time}s, CPU time: {cpu_time}s')
        return output


def flops_count(
    model: nn.Module,
    args: Iterable = [],
    kwargs: dict = {},
    include_backPropagation: bool = False,
    compute_bp_factor: float = 2.0,
    print_results: bool = True,
    print_detailed: bool = True,
    output_as_string: bool = True,
    output_precision: int = 2,
    output_unit: Optional[str] = None,
    ignore_modules: Optional[Iterable] = None,
) -> Union[Tuple[str, str, str], Tuple[int, int, int]]:
    """Calculate FLOPs, MACs, and parameter count for a model.

    Args:
        model: PyTorch model to analyze.
        args: Positional arguments for model forward pass.
        kwargs: Keyword arguments for model forward pass.
        include_backPropagation: If True, include backward pass in FLOPs/MACs calculation.
        compute_bp_factor: Multiplier for backward pass (default 2.0).
        print_results: If True, print analysis results.
        print_detailed: If True, print detailed per-layer breakdown.
        output_as_string: If True, return formatted strings; otherwise return raw numbers.
        output_precision: Number of decimal places for formatted output.
        output_unit: Unit for output (e.g., "M", "G"); auto-selected if None.
        ignore_modules: Collection of module names or types to exclude from counting.

    Returns:
        Union[Tuple[str, str, str], Tuple[int, int, int]]: (flops, macs, params) as strings or ints.
    """
    calculate_flops_pipline = CalFlopsPipline(
        model=model,
        include_backPropagation=include_backPropagation,
        compute_bp_factor=compute_bp_factor,
    )
    calculate_flops_pipline.start_flops_calculate(ignore_list=ignore_modules)
    _patch_extra(old_functions, module_flop_count, module_mac_count)

    model(*args, **kwargs)

    flops = calculate_flops_pipline.get_total_flops()
    macs = calculate_flops_pipline.get_total_macs()
    params = calculate_flops_pipline.get_total_params()

    if print_results:
        calculate_flops_pipline.print_model_pipline(
            units=output_unit,
            precision=output_precision,
            print_detailed=print_detailed,
        )

    calculate_flops_pipline.end_flops_calculate()
    _reload_extra(old_functions)

    if include_backPropagation:
        flops = flops * (1 + compute_bp_factor)
        macs = macs * (1 + compute_bp_factor)

    if output_as_string:
        return (
            flops_to_string(flops, units=output_unit, precision=output_precision),
            macs_to_string(macs, units=output_unit, precision=output_precision),
            params_to_string(params, units=output_unit, precision=output_precision),
        )

    return flops, macs, params


def _prod(dims: Union[int, Iterable[int]]) -> int:
    """Calculate product of dimensions.

    Args:
        dims: Single dimension or iterable of dimensions.

    Returns:
        int: Product of all dimensions.
    """
    if isinstance(dims, int):
        return dims
    elif isinstance(dims, (tuple, list)):
        p = 1
        for v in dims:
            p *= v
        return p
    else:
        assert False


def _rms_norm_flops_compute(
    input: Tensor,
    normalized_shape: list[int],
    weight: Optional[Tensor] = None,
    eps: Optional[float] = None,
) -> Tuple[int, int]:
    """Compute FLOPs and MACs for RMSNorm operation.

    Args:
        input: Input tensor.
        normalized_shape: Shape over which to normalize.
        weight: Optional affine weight parameter.
        eps: Epsilon for numerical stability.

    Returns:
        Tuple[int, int]: (flops, macs) where macs is 0 for normalization.
    """
    has_affine = weight is not None
    # estimation
    return input.numel() * (5 if has_affine else 4), 0


def _attention_torch_flops_compute(query: Tensor, key: Tensor, value: Tensor, **kwargs: Any) -> Tuple[int, int]:
    """Compute FLOPs and MACs for PyTorch scaled dot-product attention.

    Args:
        query: Query tensor [batch, heads, seq_len, dim].
        key: Key tensor [batch, heads, seq_len_kv, dim].
        value: Value tensor [batch, heads, seq_len_kv, dim].
        **kwargs: Additional arguments (ignored).

    Returns:
        Tuple[int, int]: (flops, macs) for attention computation.
    """
    assert key.shape == value.shape
    b, h, s, d = query.shape
    bk, hk, sk, dk = key.shape
    assert b == bk and h == hk and d == dk
    count = query.numel() * sk
    return 4 * count, 2 * count


def _attention_te_flops_compute(self: Any, query: Tensor, key: Tensor, value: Tensor, **kwargs: Any) -> Tuple[int, int]:
    """Compute FLOPs and MACs for Transformer Engine DotProductAttention.

    Args:
        self: DotProductAttention instance.
        query: Query tensor [seq_len, batch, heads, dim].
        key: Key tensor [seq_len_kv, batch, heads, dim].
        value: Value tensor [seq_len_kv, batch, heads, dim].
        **kwargs: Additional arguments (ignored).

    Returns:
        Tuple[int, int]: (flops, macs) for attention computation.
    """
    assert key.shape == value.shape
    s, b, h, d = query.shape
    sk, bk, hk, dk = key.shape
    assert b == bk and h == hk and d == dk
    count = query.numel() * sk
    return 4 * count, 2 * count


def _natten_flops_compute(
    query: Tensor, key: Tensor, value: Tensor, kernel_size: Union[int, Iterable[int]], stride: Union[int, Iterable[int]], **kwargs: Any
) -> Tuple[int, int]:
    """Compute FLOPs and MACs for Neighborhood Attention (NATTEN).

    Args:
        query: Query tensor.
        key: Key tensor (same shape as query).
        value: Value tensor (same shape as query).
        kernel_size: Neighborhood window size(s).
        stride: Stride for neighborhood attention.
        **kwargs: Additional arguments (ignored).

    Returns:
        Tuple[int, int]: (flops, macs) for neighborhood attention.
    """
    assert query.shape == key.shape == value.shape
    count = query.numel() * _prod(kernel_size) / _prod(stride)
    return 4 * count, 2 * count


def _patch_extra(old_functions: dict, module_flop_count: dict, module_mac_count: dict) -> None:
    """Patch additional operations with custom FLOPs computation functions.

    Args:
        old_functions: Dictionary to store original function references.
        module_flop_count: Counter for FLOPs per module.
        module_mac_count: Counter for MACs per module.
    """
    F.rms_norm = wrapFunc(F.rms_norm, _rms_norm_flops_compute, old_functions, module_flop_count, module_mac_count)
    F.scaled_dot_product_attention = wrapFunc(
        F.scaled_dot_product_attention, _attention_torch_flops_compute, old_functions, module_flop_count, module_mac_count
    )
    if DotProductAttention is not None:
        DotProductAttention.forward = wrapFunc(
            DotProductAttention.forward, _attention_te_flops_compute, old_functions, module_flop_count, module_mac_count
        )
    if natten is not None:
        natten.functional.neighborhood_attention_generic = wrapFunc(
            natten.functional.neighborhood_attention_generic, _natten_flops_compute, old_functions, module_flop_count, module_mac_count
        )


def _reload_extra(old_functions: dict) -> None:
    """Restore original function implementations after FLOPs counting.

    Args:
        old_functions: Dictionary containing original function references.
    """
    F.rms_norm = old_functions[F.rms_norm.__str__]
    F.scaled_dot_product_attention = old_functions[F.scaled_dot_product_attention.__str__]
    if DotProductAttention is not None:
        DotProductAttention.forward = old_functions[DotProductAttention.forward.__str__]
    if natten is not None:
        natten.functional.neighborhood_attention_generic = old_functions[natten.functional.neighborhood_attention_generic.__str__]
