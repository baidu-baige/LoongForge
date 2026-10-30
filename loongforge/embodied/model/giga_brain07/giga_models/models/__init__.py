from importlib import import_module

from .vla import GigaBrain07Policy

_LAZY_EXPORTS = {
    'Cosmos25ControlNet3DModel': '.diffusion',
    'Cosmos25MultiControlNet3DModel': '.diffusion',
    'Cosmos25Transformer3DModel': '.diffusion',
    'GigaWorld0Transformer3DModel': '.diffusion',
    'LoRAPeftWrapper': '.diffusion',
    'PI0Policy': '.vla',
}

__all__ = ['GigaBrain07Policy', *_LAZY_EXPORTS]


def __getattr__(name):
    module_name = _LAZY_EXPORTS.get(name)
    if module_name is None:
        raise AttributeError(f'module {__name__!r} has no attribute {name!r}')
    value = getattr(import_module(module_name, __name__), name)
    globals()[name] = value
    return value
