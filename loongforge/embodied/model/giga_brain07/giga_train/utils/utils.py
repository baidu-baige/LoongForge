import os
import time
from importlib import import_module
from typing import Any, Callable

import psutil
import pynvml
import safetensors
import torch
from diffusers.utils import SAFETENSORS_WEIGHTS_NAME, WEIGHTS_NAME


def as_list(data: Any) -> list[Any]:
    """Return the input wrapped as a Python list.

    Args:
        data: Any Python object or sequence.

    Returns:
        list[Any]:
            - If ``data`` is a list, it is returned as-is.
            - If ``data`` is a tuple, it is converted to a list.
            - Otherwise, ``data`` is wrapped in a single-element list.
    """
    if isinstance(data, list):
        return data
    elif isinstance(data, tuple):
        return list(data)
    else:
        return [data]


def import_function(function_name: str, sep: str = '.') -> Callable[..., Any]:
    """Import a callable given its fully-qualified name.

    Args:
        function_name: Dotted path to a callable (e.g., ``"pkg.module.func"``).
        sep: Separator used to split module and attribute.

    Returns:
        Callable[..., Any]: The resolved callable object.
    """
    parts = function_name.split(sep)
    module_name = '.'.join(parts[:-1])
    module = import_module(module_name)
    return getattr(module, parts[-1])


def get_cur_time() -> str:
    """Return current time as a compact string.

    Returns:
        str: Current local time formatted as ``YYYY-MM-DD-HHMMSS``.
    """
    return time.strftime('%Y-%m-%d-%H%M%S', time.localtime(time.time()))


def wait_for_gpu_memory(gpu_ids: list[int], gpu_memory: float, unit: str = 'GB', seconds: int = 10, count_limit: int = -1) -> None:
    """Block until specified GPUs have at least ``gpu_memory`` free.

    Args:
        gpu_ids: GPU indices to monitor.
        gpu_memory: Required free memory threshold in ``unit``.
        unit: Display and threshold unit, one of {``'GB'``, ``'MB'``, ``'KB'``, ``'B'``}.
        seconds: Sleep interval between checks.
        count_limit: Maximum number of checks. ``-1`` waits indefinitely.

    Returns:
        None
    """
    factors = {
        'GB': 1024 * 1024 * 1024,
        'MB': 1024 * 1024,
        'KB': 1024,
        'B': 1,
    }
    factor = factors[unit]
    pynvml.nvmlInit()
    gpu_handles = [pynvml.nvmlDeviceGetHandleByIndex(gpu_id) for gpu_id in gpu_ids]
    count = 0
    while True:
        meet_need = True
        for gpu_id, gpu_handle in zip(gpu_ids, gpu_handles):
            mem_info = pynvml.nvmlDeviceGetMemoryInfo(gpu_handle)
            mem_total = mem_info.total / factor
            mem_used = mem_info.used / factor
            mem_free = mem_info.free / factor
            if mem_free < gpu_memory:
                meet_need = False
            msg = 'GPU {}:'.format(gpu_id)
            msg += ' Total {:.2f}{}'.format(mem_total, unit)
            msg += ', Used {:.2f}{}'.format(mem_used, unit)
            msg += ', Free {:.2f}{}'.format(mem_free, unit)
            print(msg)
        if meet_need:
            break
        else:
            count += 1
            if count > count_limit > 0:
                print('Timeout Exit')
                exit(0)
            else:
                print('Wait For GPU Memory: {:.2f}{} .....'.format(gpu_memory, unit))
                time.sleep(seconds)


def get_cpu_memory(unit: str = 'GB') -> str:
    """Return a concise memory usage string.

    Args:
        unit: Display unit, one of {``'GB'``, ``'MB'``, ``'KB'``, ``'B'``}.

    Returns:
        str: A ``"used/total UNIT"`` summary of system RAM.
    """
    factors = {
        'GB': 1024 * 1024 * 1024,
        'MB': 1024 * 1024,
        'KB': 1024,
        'B': 1,
    }
    factor = factors[unit]
    mem_info = psutil.virtual_memory()
    mem_total = mem_info.total / factor
    mem_used = mem_info.used / factor
    msg = f'{mem_used:.2f}/{mem_total:.2f} {unit}'
    return msg


def load_state_dict(weight_path: str, weights_only: bool = True):
    """Load a model state dictionary from a file or directory.

    Args:
        weight_path: File or directory path to the weights.
        weights_only: When using ``torch.load``, pass through to avoid loading optimizer states.

    Returns:
        dict[str, Any]: The loaded state dictionary.
    """
    if os.path.isdir(weight_path):
        if os.path.exists(os.path.join(weight_path, WEIGHTS_NAME)):
            return torch.load(os.path.join(weight_path, WEIGHTS_NAME), map_location='cpu', weights_only=weights_only)
        elif os.path.exists(os.path.join(weight_path, SAFETENSORS_WEIGHTS_NAME)):
            return safetensors.torch.load_file(os.path.join(weight_path, SAFETENSORS_WEIGHTS_NAME), device='cpu')
        else:
            assert False
    elif os.path.isfile(weight_path):
        if weight_path.endswith('.safetensors'):
            return safetensors.torch.load_file(weight_path, device='cpu')
        else:
            return torch.load(weight_path, map_location='cpu', weights_only=weights_only)
    else:
        assert False


def save_state_dict(state_dict: dict[str, Any], save_path: str) -> None:
    """Save a model state dictionary to disk.

    Args:
        state_dict: Mapping of parameter names to tensors.
        save_path: Output path; ``.safetensors`` uses safetensors, others use ``torch.save``.

    Returns:
        None
    """
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    if save_path.endswith('.safetensors'):
        safetensors.torch.save_file(state_dict, save_path)
    else:
        torch.save(state_dict, save_path)
