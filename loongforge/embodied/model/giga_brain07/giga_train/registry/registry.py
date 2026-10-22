import copy
from typing import Any, Callable


class Registry(dict):
    r"""A helper class for managing registering modules, it extends a
    dictionary and provides a register function."""

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        super(Registry, self).__init__(*args, **kwargs)

    def register_module(self, module_class: Callable[..., Any], module_name: str | None = None):
        """Register a class/function to this registry.

        Args:
            module_class: Class or callable to register.
            module_name: Optional name override; defaults to class ``.__name__``.

        Returns:
            Any: The original class/function.
        """
        if module_name is None:
            module_name = module_class.__name__
        assert module_name not in self, module_name
        self[module_name] = module_class
        return module_class

    def register(self, module_class: str | Callable[..., Any]):
        """Decorator or direct registration helper.

        Args:
            module_class: Class or its target name as a string.

        Returns:
            Callable | Any: A decorator when ``module_class`` is str, otherwise the class itself.
        """
        if isinstance(module_class, str):

            def _register(_module_class):
                return self.register_module(_module_class, module_class)

            return _register
        else:
            return self.register_module(module_class)


def merge_params(params_or_type: dict | str | None, **kwargs: Any) -> dict:
    """Merge type/config dict into kwargs for builder functions.

    Args:
        params_or_type: Either a type name or a full config dict.
        **kwargs: Additional kwargs to forward to target constructor.

    Returns:
        dict: A single kwargs dict containing the merged configuration.
    """
    if isinstance(params_or_type, str):
        assert 'type' not in kwargs
        kwargs['type'] = params_or_type
    elif isinstance(params_or_type, dict):
        params = copy.deepcopy(params_or_type)
        for name, value in params.items():
            assert name not in kwargs
            kwargs[name] = value
    else:
        assert params_or_type is None
    return kwargs


def build_module(registry: Registry, params_or_type: dict | str | None, *args: Any, **kwargs: Any):
    """Instantiate an object from a registry using a config or type name.

    Args:
        registry: The target registry mapping names to constructors.
        params_or_type: Dict with key ``'type'`` or a plain type name string.
        *args, **kwargs: Positional/keyword args forwarded to constructor.

    Returns:
        Any | None: Constructed object or ``None`` if config/type not provided.
    """
    params = merge_params(params_or_type, **kwargs)
    if 'type' in params:
        obj_type = params.pop('type')
        assert obj_type in registry, '%s not in registry' % obj_type
        return registry[obj_type](*args, **params)
    else:
        return None
