import os
import socket

import safetensors
import torch
from diffusers.utils import SAFETENSORS_WEIGHTS_NAME, WEIGHTS_NAME


def wrap_call(func):
    """Wrap a callable and automatically retry on exceptions indefinitely."""

    def f(*args, **kwargs):
        while True:
            try:
                return func(*args, **kwargs)
            except Exception as e:
                print(e)
                continue

    return f


def load_state_dict(weight_path: str, weights_only: bool = True):
    """Load a model state dict from file or directory.

    Supports ``.pt`` and ``.safetensors`` files as well as Diffusers-style
    directory with ``pytorch_model.bin`` or ``model.safetensors``.
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


def save_state_dict(state_dict, save_path: str) -> None:
    """Save a state dict to ``save_path``.

    Chooses format by filename suffix.
    """
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    if save_path.endswith('.safetensors'):
        safetensors.torch.save_file(state_dict, save_path)
    else:
        torch.save(state_dict, save_path)


def find_free_port() -> int:
    """Ask OS for an available TCP port and return it.

    Note: there remains a race where the port can be taken before being used.
    """
    sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    # Binding to port 0 will cause the OS to find an available port for us
    sock.bind(('', 0))
    port = sock.getsockname()[1]
    sock.close()
    # NOTE: there is still a chance the port could be taken by other processes.
    return port
