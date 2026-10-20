import math
from importlib import import_module
from itertools import accumulate
from typing import Any, Callable, Sequence

import numpy as np
import torch


def as_list(data: Any) -> list[Any]:
    """Ensure input is a list.

    Args:
        data (Any): Any object or list.

    Returns:
        list[Any]: A list containing ``data`` if it was not already a list; otherwise returns ``data`` unchanged.
    """
    if isinstance(data, list):
        return data
    else:
        return [data]


def repeat_n(data: Any | list[Any], n: int) -> list[Any]:
    """Repeat or validate a list to length ``n``.

    Args:
        data (Any | list[Any]): A value or list to be repeated/validated.
        n (int): Target length.

    Returns:
        list[Any]: A list of length ``n``.
    """
    if not isinstance(data, list):
        data = [data]
    if len(data) == 1:
        data = data * n
    else:
        assert len(data) == n
    return data


def stack_data(data_list: list[np.ndarray | torch.Tensor], pad_value: int | float = 0, is_equal: bool = False) -> np.ndarray | torch.Tensor:
    """Stack arrays/tensors with optional padding when shapes differ.

    Args:
        data_list (list[np.ndarray | torch.Tensor]): A non-empty list of arrays/tensors with the same dtype/ndim.
        pad_value (int | float): Value used to pad when shapes do not match.
        is_equal (bool): If True, requires all shapes to be exactly equal.

    Returns:
        np.ndarray | torch.Tensor: A stacked array/tensor with leading dimension equal to len(data_list).
    """
    # Fast path: if all shapes match, use torch/np stack directly
    equal_shape = True
    for data in data_list:
        if data.shape != data_list[0].shape:
            equal_shape = False
            break
    if is_equal:
        assert equal_shape
    if equal_shape:
        if isinstance(data_list[0], np.ndarray):
            new_data = np.stack(data_list)
        elif isinstance(data_list[0], torch.Tensor):
            new_data = torch.stack(data_list)
        else:
            assert False
    else:
        # Slow path: allocate a padded canvas and copy each sample in-place
        ndim = data_list[0].ndim
        dtype = data_list[0].dtype
        new_data_shape = [len(data_list)]
        for dim in range(ndim):
            new_data_shape.append(max(data.shape[dim] for data in data_list))
        if isinstance(data_list[0], np.ndarray):
            new_data = np.full(tuple(new_data_shape), pad_value, dtype=dtype)
        elif isinstance(data_list[0], torch.Tensor):
            new_data = torch.full(tuple(new_data_shape), pad_value, dtype=dtype)
        else:
            assert False
        for i, data in enumerate(data_list):
            if ndim == 1:
                new_data[i][: data.shape[0]] = data
            elif ndim == 2:
                new_data[i][: data.shape[0], : data.shape[1]] = data
            elif ndim == 3:
                new_data[i][: data.shape[0], : data.shape[1], : data.shape[2]] = data
            elif ndim == 4:
                new_data[i][: data.shape[0], : data.shape[1], : data.shape[2], : data.shape[3]] = data
            else:
                assert False
    return new_data


def split_data(data: Sequence[Any], world_size: int = 1, rank: int = 0) -> list[Any]:
    """Split a list-like dataset across ``world_size`` ranks.

    Args:
        data (Sequence[Any]): A list-like sequence to split.
        world_size (int): Total number of splits.
        rank (int): Current rank in [0, world_size).

    Returns:
        list[Any]: The split segment owned by ``rank``.
    """
    # Compute near-uniform splits with at-most-one element difference
    data_size = len(data)
    local_size = math.floor(data_size / world_size)
    local_size_list = [local_size for _ in range(world_size)]
    for i in range(data_size - local_size * world_size):
        local_size_list[i] += 1
    assert sum(local_size_list) == data_size
    local_size_list = [0] + list(accumulate(local_size_list))
    begin = local_size_list[rank]
    end = local_size_list[rank + 1]
    data = data[begin:end]
    return data


def wrap_call(func: Callable[..., Any]) -> Callable[..., Any]:
    """Retry wrapper that continues on exception and prints the error.

    Args:
        func (Callable[..., Any]): Function to wrap with retry-on-exception.

    Returns:
        Callable[..., Any]: A function that retries indefinitely upon exceptions, printing the error each time.
    """

    def f(*args: Any, **kwargs: Any) -> Any:
        while True:
            try:
                return func(*args, **kwargs)
            except Exception as e:
                # Log and continue retrying; user code must ensure eventual success/exit
                print(e)
                continue

    return f


def import_function(function_name: str, sep: str = '.') -> Any:
    """Import a function from a fully-qualified name.

    Args:
        function_name (str): Fully-qualified function path, e.g. ``pkg.module.func``.
        sep (str): Separator used to split module and attribute.

    Returns:
        Any: The resolved attribute object from the imported module.
    """
    parts = function_name.split(sep)
    module_name = '.'.join(parts[:-1])
    module = import_module(module_name)
    return getattr(module, parts[-1])
