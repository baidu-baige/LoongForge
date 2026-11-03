# Copyright 2026 The LoongForge Authors.
# SPDX-License-Identifier: Apache-2.0

"""Value schedulers + tiny registry for giga_brain07 (flattened, self-contained).

Merged from the former ``giga_train/{registry,schedulers}`` subpackage into a
single root module. Provides the ``WarmupCosineScheduler`` the route-B trainer
builds via :func:`build_scheduler` (giga WarmupCosine semantics, wrapped by
torch ``LambdaLR``). Logic is unchanged from the original giga_train code; only
the module location and the (previously cross-file) imports were inlined.
"""

import copy
import math
from importlib import import_module
from typing import Any, Callable

from torch.optim.lr_scheduler import LambdaLR


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


def import_function(function_name: str, sep: str = '.') -> Callable[..., Any]:
    """Import a callable given its fully-qualified dotted name."""
    parts = function_name.split(sep)
    module_name = '.'.join(parts[:-1])
    module = import_module(module_name)
    return getattr(module, parts[-1])


SCHEDULERS = Registry()


def build_scheduler(params_or_type: dict | str | None, optimizer, *args: Any, **kwargs: Any):
    """Build a custom value scheduler and wrap it by PyTorch LambdaLR."""
    scheduler = build_module(SCHEDULERS, params_or_type, *args, **kwargs)
    ld_scheduler = LambdaLR(optimizer=optimizer, lr_lambda=scheduler.get_value)
    return ld_scheduler



class BaseScheduler(object):
    """Base scheduler that produces a value by interpolating start/end.

    Args:
        epoch_size: Number of steps in one epoch.
        max_epochs: Total number of epochs.
        max_steps: Total number of steps.
        start_value: Starting value for interpolation.
        end_value: Ending value for interpolation.
    """

    def __init__(self, epoch_size: int, max_epochs: int, max_steps: int, start_value: float = 1.0, end_value: float = 0.0) -> None:
        self.epoch_size = epoch_size
        self.max_epochs = max_epochs
        self.max_steps = max_steps
        self.start_value = start_value
        self.end_value = end_value

    def get_value(self, num_update: int) -> float:
        assert 0 <= num_update <= self.max_steps
        start_factor, end_factor = self.get_factor(float(num_update))
        value = self.start_value * start_factor + self.end_value * end_factor
        return float(value)

    def get_factor(self, num_update: float) -> tuple[float, float]:
        raise NotImplementedError


@SCHEDULERS.register
class FuncScheduler(BaseScheduler):
    """Function Scheduler.

    Example:
        def _get_factor(self, num_update):
            end_factor = (num_update - 1) / (self.max_steps - 1)
            return 1 - end_factor, end_factor

        func_scheduler = FuncScheduler(_get_factor)
        # or
        _get_factor_func = os.path.relpath(__file__)[:-3].replace('/', '.') + '._get_factor'
        func_scheduler = FuncScheduler(_get_factor_func)
    """

    def __init__(self, func: Callable[['BaseScheduler', float], tuple[float, float]] | str, **kwargs) -> None:
        super(FuncScheduler, self).__init__(**kwargs)
        self.func = func
        if isinstance(func, str):
            self.func = import_function(func)
        assert callable(self.func)

    def get_factor(self, num_update: float) -> tuple[float, float]:
        """Delegate factor computation to the provided function.

        Args:
            num_update: Current global step.

        Returns:
            tuple[float, float]: (start_factor, end_factor) blending coefficients.
        """
        return self.func(self, num_update)


@SCHEDULERS.register
class ConstantScheduler(BaseScheduler):
    def __init__(self, **kwargs) -> None:
        super(ConstantScheduler, self).__init__(**kwargs)

    def get_factor(self, num_update: float) -> tuple[float, float]:
        """Always return constant start factor of 1 and end factor of 0."""
        return 1, 0


@SCHEDULERS.register
class LinearScheduler(BaseScheduler):
    def __init__(self, **kwargs) -> None:
        super(LinearScheduler, self).__init__(**kwargs)

    def get_factor(self, num_update: float) -> tuple[float, float]:
        if num_update == 0:
            return 1, 0
        start_factor = 1 - (num_update - 1) / (self.max_steps - 1)
        return start_factor, 1 - start_factor


@SCHEDULERS.register
class PolyScheduler(BaseScheduler):
    def __init__(self, power: int = 2, method: int = 1, **kwargs) -> None:
        super(PolyScheduler, self).__init__(**kwargs)
        self.power = power
        self.method = method

    def get_factor(self, num_update: float) -> tuple[float, float]:
        if num_update == 0:
            return 1, 0
        if self.method == 1:
            start_factor = pow(1 - (num_update - 1) / (self.max_steps - 1), self.power)
        elif self.method == 2:
            start_factor = 1 - pow((num_update - 1) / (self.max_steps - 1), self.power)
        else:
            assert False
        return start_factor, 1 - start_factor


@SCHEDULERS.register
class CosineScheduler(BaseScheduler):
    def __init__(self, weight: float = 1.0, **kwargs) -> None:
        super(CosineScheduler, self).__init__(**kwargs)
        self.weight = weight

    def get_factor(self, num_update: float) -> tuple[float, float]:
        if num_update == 0:
            return 1, 0
        start_factor = 0.5 * self.weight * (1 + math.cos(math.pi * (num_update - 1) / (self.max_steps - 1)))
        return start_factor, 1 - start_factor


@SCHEDULERS.register
class WarmupCosineScheduler(BaseScheduler):
    def __init__(self, warmup_steps: int, decay_steps: int, **kwargs) -> None:
        super(WarmupCosineScheduler, self).__init__(**kwargs)
        self.warmup_steps = warmup_steps
        self.decay_steps = decay_steps

    def get_factor(self, num_update: float) -> tuple[float, float]:
        if num_update == 0:
            factor = 1.0 / (self.warmup_steps + 1)
            return factor, 0
        elif num_update <= self.warmup_steps:
            factor = num_update / self.warmup_steps
            return factor, 0
        elif num_update <= self.decay_steps:
            alpha = (num_update - self.warmup_steps) / (self.decay_steps - self.warmup_steps)
            factor = 0.5 * (1 + math.cos(math.pi * alpha))
            return factor, 1 - factor
        else:
            factor = 0
            return factor, 1 - factor


@SCHEDULERS.register
class CycleScheduler(BaseScheduler):
    """Cycle Scheduler.

    refer to paper <Cyclical Learning Rates for Training Neural Networks>
    """

    def __init__(self, mode: str = 'triangular', gamma: float = 1, step_size: int = 0, step_epoch: int = 0, **kwargs) -> None:
        super(CycleScheduler, self).__init__(**kwargs)
        assert mode in ('triangular', 'triangular2', 'exp_range')
        if step_size > 0:
            assert step_epoch == 0
        else:
            assert step_epoch > 0
            step_size = int(step_epoch * self.epoch_size)
        self.mode = mode
        self.gamma = gamma
        self.step_size = step_size

    def get_factor(self, num_update: float) -> tuple[float, float]:
        if num_update == 0:
            return 1, 0
        cycle = math.floor(1 + (num_update - 1) / (2 * (self.step_size - 1)))
        x = abs((num_update - 1) / (self.step_size - 1) - 2 * cycle + 1)
        end_factor = max(0, (1 - x))
        if self.mode == 'triangular2':
            end_factor *= 1 / pow(2, cycle - 1)
        elif self.mode == 'exp_range':
            end_factor *= pow(self.gamma, num_update - 1)
        return 1 - end_factor, end_factor


@SCHEDULERS.register
class StepScheduler(BaseScheduler):
    def __init__(self, step_factor: float = 0.1, step_size_list: list[int] | None = None, step_epoch_list: list[int] | None = None, **kwargs) -> None:
        super(StepScheduler, self).__init__(**kwargs)
        if step_size_list is not None:
            assert step_epoch_list is None
        else:
            assert step_epoch_list is not None
            step_size_list = [int(s * self.epoch_size) for s in step_epoch_list]
        self.step_factor = step_factor
        self.step_size_list = step_size_list

    def get_factor(self, num_update: float) -> tuple[float, float]:
        if num_update == 0:
            return 1, 0
        count = sum([1 for s in self.step_size_list if s <= (num_update - 1)])
        start_factor = pow(self.step_factor, count)
        return start_factor, 0
