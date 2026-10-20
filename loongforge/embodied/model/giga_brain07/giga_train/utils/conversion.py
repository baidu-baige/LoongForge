from typing import Any

import numpy as np
import torch


def to_list(data: Any) -> list[Any]:
    """Convert common array/tensor containers to a Python list."""
    if isinstance(data, list):
        return data
    elif isinstance(data, (torch.Tensor, np.ndarray)):
        return data.tolist()
    elif isinstance(data, tuple):
        return list(data)
    elif hasattr(data, 'tensor'):
        return data.tensor.tolist()
    else:
        assert False


def to_tensor(data: Any):
    """Recursively convert nested containers to PyTorch tensors when
    possible."""
    if isinstance(data, dict):
        for key, val in data.items():
            data[key] = to_tensor(val)
    elif isinstance(data, list):
        data = [to_tensor(d) for d in data]
    elif isinstance(data, np.ndarray):
        data = torch.from_numpy(data)
    elif hasattr(data, 'tensor'):
        data = data.tensor
    elif not isinstance(data, torch.Tensor):
        data = torch.tensor(data)
    return data


def to_numpy(data: Any):
    """Recursively convert nested containers to NumPy arrays when possible."""
    if isinstance(data, dict):
        for key, val in data.items():
            data[key] = to_numpy(val)
    elif isinstance(data, list):
        data = [to_numpy(d) for d in data]
    elif isinstance(data, torch.Tensor):
        data = data.contiguous().cpu().numpy()
    elif hasattr(data, 'tensor'):
        data = data.numpy()
    elif not isinstance(data, np.ndarray):
        data = np.array(data)
    return data


def to_cuda(data: Any, device: int | torch.device | None = None):
    """Move tensors or nested containers to CUDA device."""
    if isinstance(data, dict):
        for key, val in data.items():
            data[key] = to_cuda(val)
    elif isinstance(data, list):
        data = [to_cuda(d) for d in data]
    elif isinstance(data, torch.Tensor) or hasattr(data, 'tensor'):
        data = data.cuda(device)
    return data


def to_cpu(data: Any):
    """Move tensors or nested containers to CPU."""
    if isinstance(data, dict):
        for key, val in data.items():
            data[key] = to_cpu(val)
    elif isinstance(data, list):
        data = [to_cpu(d) for d in data]
    elif isinstance(data, torch.Tensor) or hasattr(data, 'tensor'):
        data = data.cpu()
    return data


def to_dtype(data: Any, dtype: torch.dtype | np.dtype):
    """Cast tensors/arrays or nested containers to target dtype."""
    if isinstance(data, dict):
        for key, val in data.items():
            data[key] = to_dtype(val, dtype)
    elif isinstance(data, list):
        data = [to_dtype(d, dtype) for d in data]
    elif isinstance(data, torch.Tensor) or hasattr(data, 'tensor'):
        data = data.to(dtype)
    elif isinstance(data, np.ndarray):
        data = data.astype(dtype)
    return data
