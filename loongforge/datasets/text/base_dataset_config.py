# Copyright 2026 The LoongForge Authors.
# SPDX-License-Identifier: Apache-2.0

"""base dataset config"""

from dataclasses import dataclass, fields, field
from typing import List, Optional, Literal


@dataclass
class DataConfig:
    """config for common dataset"""

    tokenizer_type: Literal["NullTokenizer", "HFTokenizer"] = "HFTokenizer"
    task_encoder: Optional[str] = "VLMTaskEncoder"
    hf_tokenizer_path: Optional[str] = None
    data_path: str = None
    dataloader_type: str = None
    split: List[int] = field(default_factory=lambda: [100, 0, 0])
    add_question_in_pretrain: bool = True
    enable_discard_sample: bool = True
    num_workers: int = None
    additional_special_tokens = None
    use_fast_tokenizer = None
    padding_side = "right"
    seq_length = 512
    split_special_tokens = True
    is_tokenized_data = True

    def __init__(self, **kwargs):
        # 1. Only keep fields declared in the class
        names = {f.name for f in fields(self)}
        for k, v in kwargs.items():
            if k in names:
                setattr(self, k, v)
        if self.additional_special_tokens is not None:
            self.additional_special_tokens = [
                token.strip() for token in self.additional_special_tokens.split(",")
            ]
