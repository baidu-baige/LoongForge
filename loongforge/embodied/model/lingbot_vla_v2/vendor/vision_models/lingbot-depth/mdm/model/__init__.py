"""Vendor module implementation."""
import importlib
from typing import *

if TYPE_CHECKING:
    from .v2 import MDMModel as MDMModelV2


def import_model_class_by_version(version: str) -> Type[Union['MDMModelV2']]:
    """Vendor interface."""
    assert version in ['v2'], f'Unsupported model version: {version}'

    try:
        module = importlib.import_module(f'.{version}', __package__)
    except ModuleNotFoundError:
        raise ValueError(f'Model version "{version}" not found.')
    cls = getattr(module, 'MDMModel')
    return cls
