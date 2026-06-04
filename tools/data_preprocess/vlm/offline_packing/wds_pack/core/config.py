# Copyright 2026 The LoongForge Authors.
# SPDX-License-Identifier: Apache-2.0

"""Configuration parsing helpers."""

import argparse
from pathlib import Path
from typing import Dict, Union

import yaml


def parse_args() -> argparse.Namespace:
    """Parse common command-line arguments."""
    parser = argparse.ArgumentParser(
        description="WDS-native offline packing"
    )
    parser.add_argument(
        "--config",
        type=str,
        default="configs/config.yaml",
        help="Path to the YAML configuration file",
    )
    return parser.parse_args()


def get_cfg(yaml_path: Union[str, Path]) -> Dict:
    """Load YAML configuration file and return it as a dictionary."""
    yaml_path = Path(yaml_path)
    if not yaml_path.is_file():
        raise FileNotFoundError(f"Configuration file not found: {yaml_path}")

    try:
        with yaml_path.open("r", encoding="utf-8") as f:
            cfg = yaml.safe_load(f)
        if not isinstance(cfg, dict):
            raise ValueError(
                f"Invalid configuration format, expected a dictionary: {yaml_path}"
            )
        return cfg
    except yaml.YAMLError as e:
        raise ValueError(f"Failed to parse YAML file: {yaml_path}\nError: {e}")
