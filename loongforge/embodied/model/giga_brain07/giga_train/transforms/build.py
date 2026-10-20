from typing import Any

from ..registry import Registry, build_module

TRANSFORMS = Registry()


def build_transform(params_or_type: dict | str | None, *args: Any, **kwargs: Any):
    """Build a transform from registry using config dict or type string.

    Args:
        params_or_type: Dict with key ``'type'`` or a type name string.
        *args, **kwargs: Forwarded to the transform constructor.

    Returns:
        Any | None: Instantiated transform or ``None`` if type missing.
    """
    return build_module(TRANSFORMS, params_or_type, *args, **kwargs)
