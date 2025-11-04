"""base dataset config"""
from aiak_training_omni.models.common.base_config import BaseModelConfig
from dataclasses import dataclass, field
from typing import List, Optional, Literal


class DataConfig(BaseModelConfig):
    """config for common dataset"""
    tokenizer_type: Literal["NullTokenizer", "HFTokenizer"] = "HFTokenizer"
    hf_tokenizer_path: Optional[str] = None
    data_path: str = None
    dataloader_type: str = None
    split: List[int] = field(default_factory=lambda: [100, 0, 0])
    add_question_in_pretrain: bool = True
    enable_discard_sample: bool = True
    num_workers: int = None
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
