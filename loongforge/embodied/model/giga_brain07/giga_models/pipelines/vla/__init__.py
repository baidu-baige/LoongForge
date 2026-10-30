from importlib import import_module

from .giga_brain_0 import GigaBrain07Pipeline

__all__ = ['GigaBrain07Pipeline', 'Pi0Pipeline']


def __getattr__(name):
    if name != 'Pi0Pipeline':
        raise AttributeError(f'module {__name__!r} has no attribute {name!r}')
    value = import_module('.pi0', __name__).Pi0Pipeline
    globals()[name] = value
    return value
