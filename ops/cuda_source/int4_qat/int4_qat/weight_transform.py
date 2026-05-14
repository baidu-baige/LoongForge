"""
Pluggable weight transforms for Quantization-Aware Training (QAT).

Provides:
  - FakeQuantWeightTransform: callable that fake-quantizes a weight tensor
    with STE (forward sees quantized values, backward passes gradient through).
  - apply_int4_qat(): monkey-patch ``_get_weight_tensors`` on matching modules.

The patching approach directly modifies ``_get_weight_tensors`` on the target
TE module without introducing wrapper ``nn.Module`` instances.  This avoids
increasing the module-tree depth which can trigger ``RecursionError`` during
``.train()`` traversal in deep models (DDP → Float16Module → VPP chunks → …).
"""
import logging
import re
from typing import Optional

import torch
import torch.nn as nn

from int4_qat.interface import fake_int4_quantize_dequantize

logger = logging.getLogger("int4_qat")

# Default regex matching MoE expert FC layers in DSv3.2.
_DEFAULT_QAT_REGEX = r'\.mlp\.experts\.linear_fc[12]$'

# TE-internal attributes that must be forwarded from the original weight
# to the STE-modified weight so that TE's backward kernels work correctly.
_TE_WEIGHT_ATTRS = (
    'main_grad',
    'grad_added_to_main_grad',
    'get_main_grad',
    'overwrite_main_grad',
    'zero_out_wgrad',
    'allreduce',
    'sequence_parallel',
)


def _copy_te_attrs(src: torch.Tensor, dst: torch.Tensor) -> None:
    """Copy TE-required bookkeeping attributes from *src* to *dst*.

    Also installs a writeback hook so that when TE's backward sets
    ``dst.grad_added_to_main_grad = True``, the flag propagates back to
    *src* (the real parameter).  Without this, Megatron's
    ``finalize_model_grads`` would not know that ``main_grad`` was already
    populated, resulting in NaN grad norms.
    """
    for attr in _TE_WEIGHT_ATTRS:
        if hasattr(src, attr):
            try:
                setattr(dst, attr, getattr(src, attr))
            except (AttributeError, RuntimeError) as e:
                logger.debug(f"[INT4 QAT] Failed to copy TE attr {attr!r}: {e}")

    # Stash reference to the real parameter for writeback.
    dst._qat_src_param = src

    # Install a lightweight __setattr__ hook via dynamic subclass.
    # When TE backward does  origin_weight.grad_added_to_main_grad = True,
    # origin_weight is our fq tensor, and this hook syncs the write to src.
    _make_writeback_tensor(dst)


# Cache the subclass so we only create it once per base class.
_writeback_cls_cache: dict[type, type] = {}


def _make_writeback_tensor(t: torch.Tensor) -> None:
    """Re-class *t* to a thin subclass with writeback __setattr__."""
    base = type(t)
    if base in _writeback_cls_cache:
        t.__class__ = _writeback_cls_cache[base]
        return
    # Skip if already a writeback subclass.
    if getattr(base, '_is_qat_writeback', False):
        return

    cls = type(
        f'{base.__name__}_QATwb',
        (base,),
        {
            '_is_qat_writeback': True,
            '__setattr__': _qat_setattr,
        },
    )
    _writeback_cls_cache[base] = cls
    t.__class__ = cls


def _qat_setattr(self, name: str, value) -> None:
    """Intercept attr writes; sync ``grad_added_to_main_grad`` back."""
    object.__setattr__(self, name, value)
    if name == 'grad_added_to_main_grad':
        src = getattr(self, '_qat_src_param', None)
        if src is not None:
            try:
                object.__setattr__(src, name, value)
            except (AttributeError, RuntimeError) as e:
                logger.debug(
                    f"[INT4 QAT] Writeback of {name!r} to src param failed: {e}"
                )


class _FakeQuantSTE(torch.autograd.Function):
    """Combined FP8-dequant + INT4 fake-quant with Straight-Through Estimator.

    Wrapping the entire dequant→fake-quant chain in a single autograd Function
    avoids Float8Tensor arithmetic in the Python dispatch layer.
    """

    @staticmethod
    def forward(ctx, weight, group_size, sym):
        """Apply INT4 fake-quantization with STE to the input weight.

        Args:
            ctx: PyTorch autograd context.
            weight: Input weight tensor, may be a Float8Tensor or plain Tensor.
            group_size: Group size for per-group quantization.
            sym: Whether to use symmetric quantization.

        Returns:
            torch.Tensor: Fake-quantized weight tensor in the same dtype as input.
        """
        w = weight
        if hasattr(weight, 'dequantize'):
            w = weight.dequantize()
        if not w.is_contiguous():
            w = w.contiguous()
        return fake_int4_quantize_dequantize(w, group_size, sym=sym)

    @staticmethod
    def backward(ctx, grad_output):
        """Straight-through estimator: pass gradients directly to weight.

        Args:
            ctx: PyTorch autograd context.
            grad_output: Gradient tensor from downstream.

        Returns:
            tuple: (grad_output, None, None) — gradient for weight, None for group_size and sym.
        """
        return grad_output, None, None


class FakeQuantWeightTransform:
    """Fake-quantize a weight tensor with STE for QAT."""

    def __init__(self, group_size: int = 32, sym: bool = True):
        self.group_size = group_size
        self.sym = sym
        self.enabled = True
        self._call_count = 0
        self._logged_first = False

    def __call__(self, weight: torch.Tensor) -> torch.Tensor:
        if not self.enabled:
            return weight

        try:
            fq = _FakeQuantSTE.apply(weight, self.group_size, self.sym)
        except RuntimeError as e:
            logger.warning(
                f"[INT4 QAT] fake_quant failed, disabling transform for this module "
                f"(weight shape={tuple(weight.shape)}, group_size={self.group_size}): {e}"
            )
            self.enabled = False
            return weight

        self._call_count += 1
        if not self._logged_first:
            is_quantized = hasattr(weight, 'dequantize')
            logger.info(
                f"[INT4 QAT] First fake_quant call — weight shape {tuple(weight.shape)}, "
                f"dtype={weight.dtype}, device={weight.device}, "
                f"fp8_param={is_quantized}"
            )
            self._logged_first = True

        _copy_te_attrs(weight, fq)
        return fq

    def __repr__(self) -> str:
        return (
            f"FakeQuantWeightTransform(group_size={self.group_size}, "
            f"sym={self.sym}, enabled={self.enabled})"
        )


def _patch_get_weight_tensors(
    module: nn.Module,
    transform: FakeQuantWeightTransform,
) -> bool:
    """Monkey-patch ``_get_weight_tensors`` on *module* for INT4 QAT."""
    if not hasattr(module, '_get_weight_tensors'):
        return False

    original_fn = module._get_weight_tensors

    def _patched_get_weights(
        _orig=original_fn, _mod=module, _xform=transform,
    ):
        weights = _orig()
        if _mod.training and _xform.enabled:
            return [
                _xform(w) if isinstance(w, torch.Tensor) else w
                for w in weights
            ]
        return weights

    module._get_weight_tensors = _patched_get_weights
    object.__setattr__(module, '_int4_qat_transform', transform)
    return True



def apply_int4_qat(
    model: nn.Module,
    transform: Optional[FakeQuantWeightTransform] = None,
    *,
    group_size: int = 32,
    sym: bool = True,
    filter_regex: Optional[str] = _DEFAULT_QAT_REGEX,
) -> tuple[FakeQuantWeightTransform, int]:
    """Patch ``_get_weight_tensors`` on matching modules for INT4 QAT."""
    if transform is None:
        transform = FakeQuantWeightTransform(group_size=group_size, sym=sym)

    compiled_re = re.compile(filter_regex) if filter_regex is not None else None

    logger.info(
        f"[INT4 QAT] apply_int4_qat — filter_regex={filter_regex!r}, "
        f"group_size={transform.group_size}, sym={transform.sym}"
    )

    to_patch = [
        (name, module)
        for name, module in model.named_modules()
        if compiled_re is None or compiled_re.search(name) is not None
    ]

    count = 0
    for name, module in to_patch:
        if _patch_get_weight_tensors(module, transform):
            count += 1

    logger.info(f"[INT4 QAT] Patched {count} modules (matched {len(to_patch)} candidates)")

    return transform, count
