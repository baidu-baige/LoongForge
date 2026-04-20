# Copyright 2026 The LoongForge Authors.
# SPDX-License-Identifier: Apache-2.0

"""Common utilities for offline data packing including config parsing, path helpers, and templates."""

import yaml
import os
from pathlib import Path
import pickle
from tqdm import tqdm
import argparse
from typing import Union, Dict

# TODO TEMPLATES needs to be expanded with more templates
TEMPLATES = {
    "packed_captioning": "<|vision_start|><|image_pad|><|vision_end|>{{ captions[0].content }}<|im_end|>",
    "packed_vqa": {
        "qwenvl": """
                    {% set image_count = namespace(value=0) %}
                    {% set video_count = namespace(value=0) %}

                    {% for message in messages %}
                        {% if loop.first and message['from'] != 'system' %}
                    <|im_start|>system
                    You are a helpful assistant.<|im_end|>
                        {% endif %}
                    <|im_start|>{{ message['from'] }}
                    {{ message['value'] | replace('<image>', '<|vision_start|><|image_pad|><|vision_end|>') }}<|im_end|>
                    {% endfor %}

                    {% if add_generation_prompt %}
                    <|im_start|>assistant
                    {% endif %}
                """
    },
    "packed_multi_mix_qa": {
        "qwenvl": """
                    {% set image_count = namespace(value=0) %}
                    {% set video_count = namespace(value=0) %}

                    {% for message in messages %}
                        {% if loop.first and message['from'] != 'system' %}
                    <|im_start|>system
                    You are a helpful assistant.<|im_end|>
                        {% endif %}
                    <|im_start|>{{ message['from'] }}
                    {{ message['value'] | replace('<image>', '<|vision_start|><|image_pad|><|vision_end|>') }}<|im_end|>
                    {% endfor %}

                    {% if add_generation_prompt %}
                    <|im_start|>assistant
                    {% endif %}
                """
    },
}
VALID_MEDIA_EXT = {"image": [".jpg", ".jpeg", ".png"], "video": [".mp4", ".avi"]}

# Supported media types, additional modalities may be added in the future
# mix indicates the dataset contains heterogeneous sample modalities (pure-text / text-image / text-video)
VALID_MEDIA_TYPE = ["text", "image", "video", "mix"]


def get_temp_dir(wds_dir: Union[str, Path]) -> Path:
    """.temp folder"""
    wds_dir = Path(wds_dir)
    temp_dir = wds_dir / ".temp"
    temp_dir.mkdir(parents=True, exist_ok=True)
    return temp_dir


def get_sample_record_path(wds_dir: Union[str, Path]) -> Path:
    """.temp/sample_record.txt"""
    temp_dir = get_temp_dir(wds_dir)
    return temp_dir / "sample_record.txt"

def get_token_info_report_path(wds_dir: Union[str, Path]) -> Path:
    """.temp/sample_len_report.txt"""
    temp_dir = get_temp_dir(wds_dir)
    return temp_dir / "sample_len_report.txt"


def get_log_file_path(wds_dir: Union[str, Path]) -> Path:
    """.temp/log.txt"""
    temp_dir = get_temp_dir(wds_dir)
    return temp_dir / "log.txt"

def get_packed_output_dir(cfg: Dict) -> Path:
    """get packed out dir"""
    wds_dir = Path(cfg["data"]["wds_dir"])
    custom_output = cfg["data"].get("packed_json_dir")
    
    if custom_output:
        packed_dir = Path(custom_output)
    else:
        packed_dir = wds_dir / "packed_json"
    
    packed_dir.mkdir(parents=True, exist_ok=True)
    return packed_dir



def parse_args() -> argparse.Namespace:
    """
    Parse command-line arguments
    """
    parser = argparse.ArgumentParser(
        description="Token Length Processor - Get token length from all samples"
    )
    parser.add_argument(
        "--config",
        type=str,
        default="config.yaml",
        help="Path to the YAML configuration file",
    )
    return parser.parse_args()


def get_cfg(yaml_path: Union[str, Path]) -> Dict:
    """
    Load YAML configuration file and return it as a dictionary
    """
    yaml_path = Path(yaml_path)
    if not yaml_path.is_file():
        raise FileNotFoundError(f"Configuration file not found: {yaml_path}")

    try:
        with open(yaml_path, "r", encoding="utf-8") as f:
            cfg = yaml.safe_load(f)
        if not isinstance(cfg, dict):
            raise ValueError(
                f"Invalid configuration format, expected a dictionary: {yaml_path}"
            )
        return cfg
    except yaml.YAMLError as e:
        raise ValueError(f"Failed to parse YAML file: {yaml_path}\nError: {e}")


def get_init_file(cfg):
    max_token_len = cfg["sample"]["max_token_len"]
    wds_dir = Path(cfg["data"]["wds_dir"])
    packed_files_dir = get_packed_output_dir(cfg)
    os.makedirs(packed_files_dir, exist_ok=True)
    token_info_file = get_token_info_report_path(wds_dir)
    return str(token_info_file), max_token_len, str(packed_files_dir), str(wds_dir)
