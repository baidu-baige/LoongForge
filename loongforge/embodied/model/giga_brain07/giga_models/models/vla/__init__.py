from importlib import import_module

from .giga_brain_0 import GigaBrain07Policy

__all__ = ['GigaBrain07Policy', 'PI0Policy']


def __getattr__(name):
    if name != 'PI0Policy':
        raise AttributeError(f'module {__name__!r} has no attribute {name!r}')
    value = import_module('.pi0', __name__).PI0Policy
    globals()[name] = value
    return value
