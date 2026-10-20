from collections import OrderedDict
from collections import abc as container_abcs
from collections.abc import Iterable, Iterator, Mapping
from typing import List, Optional

import torch.nn as nn
from torch._jit_internal import _copy_to_script_wrapper


class ModuleDict(nn.Module):
    r"""Holds submodules in a dictionary.

    :class:`~torch.nn.ModuleDict` can be indexed like a regular Python dictionary,
    but modules it contains are properly registered, and will be visible by all
    :class:`~torch.nn.Module` methods.

    :class:`~torch.nn.ModuleDict` is an **ordered** dictionary that respects

    * the order of insertion, and

    * in :meth:`~torch.nn.ModuleDict.update`, the order of the merged
      ``OrderedDict``, ``dict`` (started from Python 3.6) or another
      :class:`~torch.nn.ModuleDict` (the argument to
      :meth:`~torch.nn.ModuleDict.update`).

    Note that :meth:`~torch.nn.ModuleDict.update` with other unordered mapping
    types (e.g., Python's plain ``dict`` before Python version 3.6) does not
    preserve the order of the merged mapping.

    Args:
        modules (iterable, optional): a mapping (dictionary) of (string: module)
            or an iterable of key-value pairs of type (string, module)

    Example::

        class MyModule(nn.Module):
            def __init__(self) -> None:
                super().__init__()
                self.choices = nn.ModuleDict({
                        'conv': nn.Conv2d(10, 10, 3),
                        'pool': nn.MaxPool2d(3)
                })
                self.activations = nn.ModuleDict([
                        ['lrelu', nn.LeakyReLU()],
                        ['prelu', nn.PReLU()]
                ])

            def forward(self, x, choice, act):
                x = self.choices[choice](x)
                x = self.activations[act](x)
                return x
    """

    _modules: dict[str, nn.Module]  # type: ignore[assignment]

    def __init__(self, modules: Optional[Mapping[str, nn.Module]] = None) -> None:
        super().__init__()
        """Initialize an ordered dictionary of named submodules.

        Args:
            modules: Mapping from name to submodule to populate initially.
        """
        self.load_state_dict_mode = None
        if modules is not None:
            self.update(modules)

    @_copy_to_script_wrapper
    def __getitem__(self, key: str) -> nn.Module:
        return self._modules[key]

    def __setitem__(self, key: str, module: nn.Module) -> None:
        self.add_module(key, module)

    def __delitem__(self, key: str) -> None:
        del self._modules[key]

    @_copy_to_script_wrapper
    def __len__(self) -> int:
        return len(self._modules)

    @_copy_to_script_wrapper
    def __iter__(self) -> Iterator[str]:
        return iter(self._modules)

    @_copy_to_script_wrapper
    def __contains__(self, key: str) -> bool:
        return key in self._modules

    def clear(self) -> None:
        """Remove all items from the ModuleDict."""
        self._modules.clear()

    def pop(self, key: str) -> nn.Module:
        r"""Remove key from the ModuleDict and return its module.

        Args:
            key (str): key to pop from the ModuleDict
        """
        v = self[key]
        del self[key]
        return v

    @_copy_to_script_wrapper
    def keys(self) -> Iterable[str]:
        r"""Return an iterable of the ModuleDict keys."""
        return self._modules.keys()

    @_copy_to_script_wrapper
    def items(self) -> Iterable[tuple[str, nn.Module]]:
        r"""Return an iterable of the ModuleDict key/value pairs."""
        return self._modules.items()

    @_copy_to_script_wrapper
    def values(self) -> Iterable[nn.Module]:
        r"""Return an iterable of the ModuleDict values."""
        return self._modules.values()

    def update(self, modules: Mapping[str, nn.Module]) -> None:
        r"""Update the :class:`~torch.nn.ModuleDict` with key-value pairs from a
        mapping, overwriting existing keys.

        .. note::
            If :attr:`modules` is an ``OrderedDict``, a :class:`~torch.nn.ModuleDict`, or
            an iterable of key-value pairs, the order of new elements in it is preserved.

        Args:
            modules (iterable): a mapping (dictionary) from string to :class:`~torch.nn.Module`,
                or an iterable of key-value pairs of type (string, :class:`~torch.nn.Module`)
        """
        if not isinstance(modules, container_abcs.Iterable):
            raise TypeError('ModuleDict.update should be called with an ' 'iterable of key/value pairs, but got ' + type(modules).__name__)

        if isinstance(modules, (OrderedDict, ModuleDict, container_abcs.Mapping)):
            for key, module in modules.items():
                self[key] = module
        else:
            # modules here can be a list with two items
            for j, m in enumerate(modules):
                if not isinstance(m, container_abcs.Iterable):
                    raise TypeError('ModuleDict update sequence element ' '#' + str(j) + ' should be Iterable; is' + type(m).__name__)
                if not len(m) == 2:
                    raise ValueError('ModuleDict update sequence element ' '#' + str(j) + ' has length ' + str(len(m)) + '; 2 is required')
                # modules can be Mapping (what it's typed at), or a list: [(name1, module1), (name2, module2)]
                # that's too cumbersome to type correctly with overloads, so we add an ignore here
                self[m[0]] = m[1]  # type: ignore[assignment]

    @property
    def dtype(self):
        return list(self.values())[0].dtype

    def forward(self, key: str, *args, **kwargs):
        return self[key](*args, **kwargs)

    def load_state_dict(self, state_dict, strict: bool = True, **kwargs):
        if self.load_state_dict_mode == 'each':
            incompatible_keys = None

            model_names = list(self.keys())
            for model_name in model_names:
                keys = list(state_dict.keys())
                sub_model_name = model_name + '.'
                sub_state_dict = OrderedDict()
                for key in keys:
                    if key.startswith(sub_model_name):
                        sub_key = key[len(sub_model_name) :]
                        sub_state_dict[sub_key] = state_dict.pop(key)
                ret_keys = self[model_name].load_state_dict(sub_state_dict, strict=strict, **kwargs)
                if incompatible_keys is None:
                    incompatible_keys = ret_keys
                else:
                    incompatible_keys.missing_keys.extend(ret_keys.missing_keys)
                    incompatible_keys.unexpected_keys.extend(ret_keys.unexpected_keys)

            error_msgs: List[str] = []
            missing_keys = list(state_dict.keys())
            if len(missing_keys) > 0:
                incompatible_keys.missing_keys.extend(missing_keys)

            if strict:
                if len(incompatible_keys.unexpected_keys) > 0:
                    error_msgs.insert(
                        0,
                        'Unexpected key(s) in state_dict: {}. '.format(', '.join(f'"{k}"' for k in incompatible_keys.unexpected_keys)),
                    )
                if len(incompatible_keys.missing_keys) > 0:
                    error_msgs.insert(
                        0,
                        'Missing key(s) in state_dict: {}. '.format(', '.join(f'"{k}"' for k in incompatible_keys.missing_keys)),
                    )

            if len(error_msgs) > 0:
                raise RuntimeError('Error(s) in loading state_dict for {}:\n\t{}'.format(self.__class__.__name__, '\n\t'.join(error_msgs)))
            return incompatible_keys

        else:
            assert self.load_state_dict_mode is None
            return super().load_state_dict(state_dict, strict=strict, **kwargs)
