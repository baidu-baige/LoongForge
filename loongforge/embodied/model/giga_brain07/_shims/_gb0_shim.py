"""Permissive import stubs for optional deps of the giga_* libraries.

The giga_train / giga_datasets / giga_models packages import a handful of
third-party modules eagerly at module import time (evaluation metrics, video
utils, flop counters, ssh helpers). None of them is on the training code path
for GigaBrain-0 VLA finetuning, and installing them would drag in timm /
onnxruntime / opencv and risk the pinned torch + transformers + lerobot stack.

Each stub package simply calls install(globals()); any attribute access returns
a _Dummy that only raises if something actually tries to use it.
"""


class _Dummy:
    """Attribute- and call-tolerant placeholder for an unavailable symbol."""

    def __init__(self, name):
        self.__name__ = name
        self.__qualname__ = name

    def __call__(self, *args, **kwargs):
        raise RuntimeError(
            '{} belongs to a stubbed optional dependency and is not installed'.format(self.__name__)
        )

    def __getattr__(self, item):
        # Never fake dunders: torch/inspect probe them and expect real values.
        if item.startswith('__') and item.endswith('__'):
            raise AttributeError(item)
        return _Dummy('{}.{}'.format(self.__name__, item))

    def __getitem__(self, item):
        return _Dummy('{}[{!r}]'.format(self.__name__, item))

    def __repr__(self):
        return '<gb0 shim {}>'.format(self.__name__)


def install(module_globals):
    """Give a stub module a module-level __getattr__ handing out _Dummy objects."""
    name = module_globals.get('__name__', 'gb0_shim')

    def __getattr__(item):
        if item.startswith('__') and item.endswith('__'):
            raise AttributeError(item)
        return _Dummy('{}.{}'.format(name, item))

    module_globals['__getattr__'] = __getattr__
