"""base train config"""
from aiak_training_omni.models.common.base_config import BaseModelConfig
from dataclasses import dataclass, field
from typing import List, Optional, Literal

class TrainConfig(BaseModelConfig):
    """config for training"""
    training_phase: str = None
    seq_length: int = None
    max_position_embeddings: int = None
    init_method_std: float = None

    micro_batch_size: int = None
    global_batch_size: int = None
    lr: float = None
    min_lr: float = None
    clip_grad: float = None
    weight_decay: float = None

    optimizer: Literal["adam", "muon"] = "adam"
    adam_beta1: float = None
    adam_beta2: float = None
    adam_eps: float = None

    train_iters: int = None
    lr_decay_iters: int = None
    lr_decay_style: str = "cosine"
    lr_warmup_fraction: float = None

    initial_loss_scale: int = None
    bf16: bool = True

    load: Optional[str] = None
    save: Optional[str] = None
    save_interval: int = None
    ckpt_format: str = None
    dataloader_save: Optional[str] = None

    attention_backend: str = None
    pipeline_model_parallel_size: int = 1
    tensor_model_parallel_size: int = 1

    use_distributed_optimizer: bool = True
    overlap_grad_reduce: bool = True
    overlap_param_gather: bool = True
    distributed_backend: str = None

    log_interval: int = None
    tensorboard_dir: Optional[str] = None
    log_timers_to_tensorboard: bool = True

    norm_epsilon: float = None
    def __init__(self, **kwargs):
        super().__init__(**kwargs)