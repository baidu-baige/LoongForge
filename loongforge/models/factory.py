# Copyright 2026 The LoongForge Authors.
# SPDX-License-Identifier: Apache-2.0

"""model factory"""

from typing import Union, List

MODEL_ARCH_CONFIGS = {}
MODEL_ARCH_TO_FAMILY = {}
MODEL_FAMILY_TO_ARCHS = {}
MODEL_FAMILY_TO_PROVIDER = {}


def get_model_family(name: str):
    """get model family name"""
    _name = name.lower()

    if _name in MODEL_FAMILY_TO_ARCHS:
        return _name
    elif _name in MODEL_ARCH_TO_FAMILY:
        return MODEL_ARCH_TO_FAMILY[_name]
    else:
        raise ValueError(f"Unknown model family or arch: {_name}")


def get_support_model_family_and_archs():
    """get support model family and archs"""
    model_family_and_archs = []
    for family in MODEL_FAMILY_TO_ARCHS:
        model_family_and_archs.append(family)
        model_family_and_archs.extend(MODEL_FAMILY_TO_ARCHS[family])

    return model_family_and_archs


def get_support_model_archs(families: Union[List[str], str]):
    """get support model archs by model families"""
    assert isinstance(families, (list, str)), "families must be list or str"
    if isinstance(families, str):
        families = [families]

    archs = []
    for family in families:
        archs.extend(MODEL_FAMILY_TO_ARCHS.get(family, []))

    return archs


def register_model_config(model_family: str, model_arch: str):
    """
    register new model config to loongforge.models.factory。

    Args:
        model_family (str): model family name, e.g., "llama2". (Case-insensitive)
        model_arch (str): model architecture name, e.g., "llama2-7b". (Case-insensitive)
    """

    def _register_function(fn):
        _family = model_family.lower()
        _arch = model_arch.lower()

        if _arch in MODEL_ARCH_CONFIGS:
            raise ValueError(
                f"Cannot register duplicate model, family ({_family}), arch ({_arch})"
            )

        if not callable(fn):
            raise ValueError(
                f"Model arch register must be callable, family ({_family}), arch ({_arch})"
            )

        MODEL_ARCH_CONFIGS[_arch] = fn
        MODEL_ARCH_TO_FAMILY[_arch] = _family

        if _family not in MODEL_FAMILY_TO_ARCHS:
            MODEL_FAMILY_TO_ARCHS[_family] = []

        MODEL_FAMILY_TO_ARCHS[_family].append(_arch)

        return fn

    return _register_function


def get_model_config(arch_name: str):
    """get model config from model factory"""
    func = MODEL_ARCH_CONFIGS.get(arch_name, None)
    return func() if func is not None else None


def register_model_provider(model_family: Union[str, List[str]]):
    """
    register model_provider func to loongforge.models.factory。

    Args:
        model_family (Union[str, List[str]]): model family name, e.g., "llama2". (Case-insensitive)
    """

    def _register_function(fn):
        families = model_family

        if not isinstance(families, list):
            families = [families]

        for family in families:
            family = str(family).lower()

            if family in MODEL_FAMILY_TO_PROVIDER:
                raise ValueError(
                    f"Cannot register duplicate model provider, family ({family})"
                )

            if not callable(fn):
                raise ValueError(f"Model provider must be callable, family ({family})")

            MODEL_FAMILY_TO_PROVIDER[family] = fn
        return fn

    return _register_function


def get_model_provider(family: str):
    """get model provider from model factory"""
    return MODEL_FAMILY_TO_PROVIDER.get(family, None)
