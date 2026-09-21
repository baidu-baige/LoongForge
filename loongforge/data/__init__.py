# Copyright 2026 The LoongForge Authors.
# SPDX-License-Identifier: Apache-2.0

"""Load data exports on demand so embodied imports do not require MCore."""

from importlib import import_module

_EXPORTS = {
    "BlendedHuggingFaceDatasetConfig": "blended_hf_dataset_config",
    "BlendedHuggingFaceDatasetBuilder": "blended_hf_dataset_builder",
    "SFTDataset": "sft_dataset",
    "SFTDatasetConfig": "sft_dataset",
    "ChatTemplate": "chat_template",
    "HFChatTemplate": "chat_template",
    "get_support_templates": "chat_template",
    "load_chat_template_kwargs": "chat_template",
    "MMPlugin": "mm_plugin",
    "DataCollatorForSupervisedDataset": "sft_data_collator",
    "MultiModalDataCollatorForSupervisedDataset": "sft_data_collator",
}

__all__ = list(_EXPORTS)


def __getattr__(name):
    if name not in _EXPORTS:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    value = getattr(import_module(f".{_EXPORTS[name]}", __name__), name)
    globals()[name] = value
    return value
