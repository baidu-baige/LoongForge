import functools
import json
import operator
from collections import OrderedDict
from typing import Any, Union

import torch
import torch.distributed as dist
from tqdm import tqdm

from ..utils import load_state_dict, save_state_dict


def save_ema(ckpt_paths: list[str], save_path: str, decay: float = 0.9) -> None:
    """Compute EMA across multiple checkpoints and save to file."""
    state_dict_ema = None
    dtype = None
    for i in tqdm(range(len(ckpt_paths))):
        state_dict = load_state_dict(ckpt_paths[i])
        if i == 0:
            dtype = list(state_dict.values())[0].dtype
            state_dict_ema = {name: param.float() for name, param in state_dict.items()}
        else:
            for name, param in state_dict.items():
                state_dict_ema[name] = state_dict_ema[name] * decay + param.float() * (1 - decay)
    state_dict_ema = {name: param.to(dtype) for name, param in state_dict_ema.items()}
    save_state_dict(state_dict_ema, save_path)


class EMAModel:
    """Exponential Moving Average of model weights.

    Maintains sharded flat parameter buffers per-rank to support distributed accumulation and reconstruction.
    """

    def __init__(
        self,
        decay: float = 0.9999,
        min_decay: float = 0.0,
        optimization_step: int = 0,
        update_after_step: int = 0,
        use_ema_warmup: bool = False,
        inv_gamma: Union[float, int] = 1.0,
        power: Union[float, int] = 2 / 3,
        rank: int = 0,
        world_size: int = 1,
    ):
        """Initialize an EMA tracker for model parameters.

        Args:
            decay: Target EMA decay value (upper bound).
            min_decay: Minimum decay enforced after warmup.
            optimization_step: Starting global optimization step.
            update_after_step: Delay EMA updates until passing this step.
            use_ema_warmup: Use inverse-gamma warmup schedule for decay.
            inv_gamma: Inverse gamma parameter for warmup.
            power: Power schedule parameter for warmup.
            rank: Process rank in distributed setting.
            world_size: Total processes in distributed setting.
        """
        if world_size > 1:
            assert rank == dist.get_rank() and world_size == dist.get_world_size()
        self.decay = decay
        self.min_decay = min_decay
        self.optimization_step = optimization_step
        self.update_after_step = update_after_step
        self.use_ema_warmup = use_ema_warmup
        self.inv_gamma = inv_gamma
        self.power = power
        self.rank = rank
        self.world_size = world_size

        self._param_dict = OrderedDict()
        self._shape_dict = OrderedDict()
        self._slice_dict = OrderedDict()
        self._buffer_dict = OrderedDict()

    def get_decay(self, optimization_step: int) -> float:
        step = max(0, optimization_step - self.update_after_step - 1)

        if step <= 0:
            return 0.0

        if self.use_ema_warmup:
            cur_decay_value = 1 - (1 + step / self.inv_gamma) ** -self.power
        else:
            cur_decay_value = (1 + step) / (10 + step)

        cur_decay_value = min(cur_decay_value, self.decay)
        # make sure decay is not smaller than min_decay
        cur_decay_value = max(cur_decay_value, self.min_decay)
        return cur_decay_value

    @torch.no_grad()
    def step(self, state_dict: dict[str, Any]) -> None:
        self.optimization_step += 1
        decay = self.get_decay(self.optimization_step)
        for name, param in state_dict.items():
            if isinstance(param, torch.Tensor):
                if param.numel() == 0:
                    assert name in self._buffer_dict
                    continue
            else:
                assert name in self._buffer_dict
                self._buffer_dict[name] = param
                continue
            if name.endswith('_extra_state'):
                assert name in self._buffer_dict
                self._buffer_dict[name] = param
                continue
            s_param = self._param_dict[name]
            s_slice = self._slice_dict[name]
            param = param.view(-1)
            pad_size = (self.world_size - param.numel() % self.world_size) % self.world_size
            if pad_size > 0:
                param = torch.nn.functional.pad(param, [0, pad_size])
            param = param[s_slice[0] : s_slice[1]]
            s_param[:] = (s_param.float() * decay + param.float() * (1 - decay)).to(s_param.dtype)

    def load_state_dict(self, state_dict: dict[str, Any], device: torch.device | None = None, dtype: torch.dtype | None = None) -> None:
        self._param_dict = OrderedDict()
        self._shape_dict = OrderedDict()
        self._slice_dict = OrderedDict()
        for name, param in state_dict.items():
            if isinstance(param, torch.Tensor):
                if param.numel() == 0:
                    self._buffer_dict[name] = param
                    continue
            else:
                self._buffer_dict[name] = param
                continue
            if name.endswith('_extra_state'):
                self._buffer_dict[name] = param
                continue
            self._shape_dict[name] = param.shape
            param = param.view(-1)
            pad_size = (self.world_size - param.numel() % self.world_size) % self.world_size
            if pad_size > 0:
                param = torch.nn.functional.pad(param, [0, pad_size])
            local_size = param.numel() // self.world_size
            begin = self.rank * local_size
            end = (self.rank + 1) * local_size
            param = param[begin:end].clone()
            if device is not None:
                param = param.to(device)
            if dtype is not None:
                param = param.to(dtype)
            self._param_dict[name] = param
            self._slice_dict[name] = (begin, end)

    def state_dict(self, device: torch.device | None = None, dtype: torch.dtype | None = None) -> OrderedDict:
        param_dict = OrderedDict()
        for name, s_param in self._param_dict.items():
            param_shape = self._shape_dict[name]
            if self.world_size > 1:
                s_param_list = [torch.empty_like(s_param) for _ in range(self.world_size)]
                dist.all_gather(s_param_list, s_param)
                param_size = functools.reduce(operator.mul, param_shape)
                param = torch.cat(s_param_list)[:param_size].reshape(param_shape)
            else:
                param = s_param.reshape(param_shape)
            if device is not None:
                param = param.to(device)
            else:
                param = param.cpu()
            if dtype is not None:
                param = param.to(dtype)
            param_dict[name] = param
        param_dict.update(self._buffer_dict)
        return param_dict

    @property
    def config(self) -> dict[str, Any]:
        return {
            'decay': self.decay,
            'min_decay': self.min_decay,
            'optimization_step': self.optimization_step,
            'update_after_step': self.update_after_step,
            'use_ema_warmup': self.use_ema_warmup,
            'inv_gamma': self.inv_gamma,
            'power': self.power,
        }

    @classmethod
    def from_config(cls, config: dict[str, Any] | str):
        if not isinstance(config, dict):
            config = json.load(open(config, 'r'))
        return cls(**config)

    def save_config(self, save_path: str) -> None:
        with open(save_path, 'w', encoding='utf-8') as writer:
            json_string = json.dumps(self.config, indent=2, sort_keys=True) + '\n'
            writer.write(json_string)

    def load_config(self, config: dict[str, Any] | str) -> None:
        if not isinstance(config, dict):
            config = json.load(open(config, 'r'))
        for key, val in config.items():
            setattr(self, key, val)
