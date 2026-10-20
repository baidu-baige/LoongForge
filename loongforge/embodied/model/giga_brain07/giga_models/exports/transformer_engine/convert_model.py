from types import MethodType
from typing import Iterable, Optional

import torch
import torch.nn as nn


@torch.no_grad()
def convert_model_torch_to_te(model: nn.Module, prefix: str = '', ignore_modules: Optional[Iterable] = None) -> nn.Module:
    """Recursively convert PyTorch layers to Transformer Engine counterparts.

    Currently supports ``nn.Linear``, ``nn.LayerNorm`` and ``nn.RMSNorm`` if
    available. Modules can be ignored via name substring or type match.

    Args:
        model: PyTorch model to convert.
        prefix: Module name prefix for tracking nested modules.
        ignore_modules: Collection of module names (str) or types to skip during conversion.

    Returns:
        nn.Module: The same model with converted layers in-place.
    """
    import transformer_engine.pytorch as te

    for name, module in model.named_children():
        submodule_prefix = prefix + ('.' if prefix else '') + name
        if _need_ignore(submodule_prefix, module, ignore_modules):
            continue
        if isinstance(module, nn.Linear):
            if any(p % 16 != 0 for p in module.weight.shape):
                continue
            has_bias = module.bias is not None
            te_module = te.Linear(module.in_features, module.out_features, bias=has_bias, params_dtype=module.weight.dtype)
            te_module.weight.copy_(module.weight)
            te_module.weight.requires_grad = module.weight.requires_grad
            if has_bias:
                te_module.bias.copy_(module.bias)
                te_module.bias.requires_grad = module.bias.requires_grad
            setattr(model, name, te_module)

        elif isinstance(module, nn.LayerNorm):
            if module.weight is None:
                continue
            te_module = te.LayerNorm(module.normalized_shape, eps=module.eps, dtype=module.weight.dtype)
            te_module.weight.copy_(module.weight)
            te_module.weight.requires_grad = module.weight.requires_grad
            if module.bias is not None:
                te_module.bias.copy_(module.bias)
                te_module.bias.requires_grad = module.bias.requires_grad
            setattr(model, name, te_module)

        elif isinstance(module, nn.RMSNorm):
            if module.weight is None:
                continue
            te_module = te.RMSNorm(module.normalized_shape, eps=module.eps, dtype=module.weight.dtype)
            te_module.weight.copy_(module.weight)
            te_module.weight.requires_grad = module.weight.requires_grad
            setattr(model, name, te_module)

        else:
            convert_model_torch_to_te(module, submodule_prefix, ignore_modules)
    return model


def apply_fp8_autowrap(model: nn.Module, **recipe_kwargs) -> nn.Module:
    """Apply FP8 autocast wrapper to model's forward method.

    Args:
        model: PyTorch model to wrap with FP8 autocast.
        **recipe_kwargs: Configuration for FP8 recipe, including:
            - fp8_format: FP8 format string (converted to te_recipe.Format enum)
            - use_autocast_during_eval: Whether to use autocast during eval mode
            - Other DelayedScaling recipe parameters

    Returns:
        nn.Module: Model with FP8-wrapped forward method.
    """
    from accelerate.utils import contextual_fp8_autocast
    from transformer_engine.common import recipe as te_recipe

    if 'fp8_format' in recipe_kwargs:
        recipe_kwargs['fp8_format'] = getattr(te_recipe.Format, recipe_kwargs['fp8_format'])
    use_during_eval = recipe_kwargs.pop('use_autocast_during_eval', False)
    fp8_recipe = te_recipe.DelayedScaling(**recipe_kwargs)
    new_forward = contextual_fp8_autocast(model.forward, fp8_recipe, use_during_eval)

    if hasattr(model.forward, '__func__'):
        model.forward = MethodType(new_forward, model)
    else:
        model.forward = new_forward

    return model


def _need_ignore(prefix: str, module: nn.Module, ignore_modules: Optional[Iterable]) -> bool:
    """Check if a module should be ignored during conversion.

    Args:
        prefix: Full module path prefix (e.g., "model.layer.0").
        module: Module instance to check.
        ignore_modules: Collection of module names (str) or types to ignore.

    Returns:
        bool: True if module should be ignored, False otherwise.
    """
    if ignore_modules is None:
        return False
    for ignore_module in ignore_modules:
        if isinstance(ignore_module, str):
            if ignore_module in prefix:
                return True
            if module.__class__.__name__ == ignore_module:
                return True
        elif isinstance(module, ignore_module):
            return True
    return False
