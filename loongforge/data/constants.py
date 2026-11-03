# Copyright 2026 The LoongForge Authors.
# SPDX-License-Identifier: Apache-2.0

"""constants"""

IGNORE_INDEX = -100

DEFAULT_DATASET_NAME = "default"

SFT_SUPPORT_DATA_TYPE = {
    "arrow": "arrow",
    "csv": "csv",
    "json": "json",
    "jsonl": "json",
    "parquet": "parquet",
    "txt": "text",
}


class TrainingPhase(object):
    """Training phase"""

    PRETRAIN = "pretrain"
    SFT = "sft"


class SFTDataFormats(object):
    """sft data formats"""

    ALPACA = "alpaca"
    SHAREGPT = "sharegpt"
    OPENAI = "openai"


class DataRoles(object):
    """data roles"""

    USER = "user"
    ASSISTANT = "assistant"
    OBSERVATION = "observation"
    FUNCTION = "function"
    SYSTEM = "system"


class Placeholder(object):
    """Placeholders"""

    IMAGE = "<image>"
    VIDEO = "<video>"
