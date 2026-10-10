# Copyright 2026 The LoongForge Authors.
# SPDX-License-Identifier: Apache-2.0

"""
global_vars.py - Embodied training global state (three typed singletons).

  - set_training_args() / get_training_args(): generic CLI args (TrainingArgs).
  - set_model_config()  / get_model_config():  model-structure config (ModelConfig).
  - set_data_config()   / get_data_config():   data-processing config (DataConfig).

All three are frozen dataclass instances produced by parse_train_args().
"""

_EMBODIED_TRAINING_ARGS = None
_EMBODIED_MODEL_CONFIG = None
_EMBODIED_DATA_CONFIG = None


def set_training_args(training_args):
    """Store generic training args globally. Call exactly once per process."""
    global _EMBODIED_TRAINING_ARGS
    assert _EMBODIED_TRAINING_ARGS is None, (
        "training args already set; set_training_args() should only be called once per process"
    )
    _EMBODIED_TRAINING_ARGS = training_args


def get_training_args():
    """Retrieve the globally stored training args (TrainingArgs)."""
    assert _EMBODIED_TRAINING_ARGS is not None, (
        "training args not initialized; call parse_train_args() first"
    )
    return _EMBODIED_TRAINING_ARGS


def set_model_config(model_cfg):
    """Store model-structure config globally. Call exactly once per process."""
    global _EMBODIED_MODEL_CONFIG
    assert _EMBODIED_MODEL_CONFIG is None, (
        "model config already set; set_model_config() should only be called once per process"
    )
    _EMBODIED_MODEL_CONFIG = model_cfg


def get_model_config():
    """Retrieve the globally stored model-structure config (ModelConfig)."""
    assert _EMBODIED_MODEL_CONFIG is not None, (
        "model config not initialized; call parse_train_args() first"
    )
    return _EMBODIED_MODEL_CONFIG


def set_data_config(data_cfg):
    """Store data-processing config globally. Call exactly once per process."""
    global _EMBODIED_DATA_CONFIG
    assert _EMBODIED_DATA_CONFIG is None, (
        "data config already set; set_data_config() should only be called once per process"
    )
    _EMBODIED_DATA_CONFIG = data_cfg


def get_data_config():
    """Retrieve the globally stored data-processing config (DataConfig)."""
    assert _EMBODIED_DATA_CONFIG is not None, (
        "data config not initialized; call parse_train_args() first"
    )
    return _EMBODIED_DATA_CONFIG
