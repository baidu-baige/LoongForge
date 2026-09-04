# Overview

This document is for users training embodied models with the LoongForge framework. It focuses on framework-level general capabilities and describes the features each module supports, along with configuration entry points and usage. For model-specific training recipes, data preprocessing scripts, and performance tuning items, see the corresponding **Quick Start** documents.

## 1. Entry Points and Configuration Conventions

This chapter describes the directory layout and launch script convention under embodied, and introduces the three configuration objects `TrainingArgs` / `ModelConfig` / `DataConfig` along with their override priority.

### 1.1 Directory Layout

| Path | Description |
| --- | --- |
| `examples/embodied/` | Model-level launch scripts; default configs can be overridden via the trailing pass-through arguments at the end of each script |
| `configs/models/embodied/` | Default YAML configs per model, containing two top-level sections: `model:` and `data:` |
| `loongforge/embodied/train.py` | Training entry point; parses config, builds the Trainer, and starts training |
| `loongforge/embodied/train/training_args.py` | Definition file for common training arguments; generates the shell CLI |
| `loongforge/embodied/train/config_map.py` | Model config routing table; binds `--model-name` to a YAML, `ModelConfig`, and `DataConfig` |
| `loongforge/embodied/model/` | Model architecture and model registration |
| `loongforge/embodied/data/datasets/` | Data processing components |

The training pipeline is:

```text
examples/embodied/<model>/run_*.sh
    ↓
loongforge/embodied/train.py
    ↓
parse_train_args()
    ↓
build_model_trainer()
    ↓
trainer.train()
```

### 1.2 Launch Script Convention

Launch scripts typically set environment variables, paths, distributed arguments, and default training arguments for the model, and keep `"$@"` at the end of the command to pass through additional shell flags or YAML dotlist overrides supplied by the user:

```bash
PYTHONPATH=$LOONGFORGE_PATH:${PYTHONPATH:-} \
torchrun "${DISTRIBUTED_ARGS[@]}" \
    "$LOONGFORGE_PATH/loongforge/embodied/train.py" \
    "${MODEL_CONFIG_ARGS[@]}" \
    "${DATA_ARGS[@]}" \
    "${TRAINING_ARGS[@]}" \
    "${DISTRIBUTED_TRAINING_ARGS[@]}" \
    "${LOGGING_ARGS[@]}" \
    "$@"
```

Example:

```bash
bash examples/embodied/pi05/run_pi05_ddp_finetune.sh \
    --train-iters 10000 \
    --per-device-batch-size 8 \
    model.action_horizon=64 \
    data.image_size=256
```

### 1.3 Configuration Layering

Configuration is split into three objects:

| Config Object | Entry Point | Scope | Example |
| --- | --- | --- | --- |
| `TrainingArgs` | Shell flag | Training pipeline arguments | `--train-iters`, `--lr-base`|
| `ModelConfig` | YAML or `model.xxx=...` | Model architecture and training strategy | `model.action_horizon=64` |
| `DataConfig` | YAML or `data.xxx=...` | Data loading and preprocessing | `data.image_size=256` |

Override priority:

```text
dataclass defaults  <  YAML config  <  shell flag / dotlist override
```

## 2. Data Processing

The data processing module converts dataset samples into a `PreparedBatch` that the model can directly consume. Users specify the dataset format and DataLoader behavior via `TrainingArgs`, and configure model-specific data processing logic via `DataConfig`.

Data processing pipeline:

```text
Dataset
    ↓
sample-level transform
    ↓
preprocessor / collate_fn
    ↓
PreparedBatch
    ↓
batch.to(device)
    ↓
model.forward(batch)
```

### 2.1 Dataset Format

Use `--dataset-format` to select the dataset reader. LeRobot, HDF5, and dummy formats are supported. `dummy_datasets` generates random samples when no real data is available, useful for debugging and validation.

| Feature | Argument | Default | Values / Type | Description |
| --- | --- | --- | --- | --- |
| Dataset format | `--dataset-format` | `lerobot_datasets` | `lerobot_datasets`, `hdf5_datasets`, `dummy_datasets` | Select dataset format |
| Dataset path | `--dataset-path` | `None` | Local path or dataset id | Training data path |
| Dataset split | `--split` | `train` | String | Dataset split |
| Dummy sample count | `--num-samples` | `100` | Positive integer | Number of samples generated under `dummy_datasets` |

### 2.2 LeRobot Data Strategy

When using the LeRobot format, the following arguments give finer control over data loading: pick the dataset format version (v2.0 / v2.1 / v3.0), select a build strategy tailored to different robots or tasks, configure the video decoding backend, and specify the robot type to match the embodiment's action-state layout.

| Feature | Argument | Default | Values / Type | Description |
| --- | --- | --- | --- | --- |
| LeRobotDataset version | `--lerobotdataset-version` | `v3.0` | `v2.0`, `v2.1`, `v3.0` | Parse different LeRobot on-disk formats |
| Data strategy | `--dataset-strategy` | `default` | `default`, `fastwam`, `groot_n1_7`, `cosmos3_droid`, `dreamzero` | Select LeRobot build strategy |
| Video backend | `--video-backend` | `torchcodec` | `torchcodec`, `decord`, `opencv`, `pyav`, `torchvision_av` | Video decoding implementation |
| Robot type | `--robot-type` | `None` | String | Select embodiment / action-state layout |

> **Note:** `--video-backend` selects the reading backend for `--lerobotdataset-version` v2.x and its variants. The v3.0 series supports `torchcodec` and `pyav` by default.

### 2.3 DataLoader Behavior

The arguments in this section control DataLoader worker parallelism, the multiprocessing start method, distributed index sharding, and streaming reads, covering everything from data prefetching to loading extremely large datasets.

| Feature | Argument | Default | Values / Type | Description |
| --- | --- | --- | --- | --- |
| Worker count | `--num-workers` | `4` | Non-negative integer | DataLoader workers per rank |
| Prefetch factor | `--dataloader-prefetch-factor` | `2` | Positive integer | Number of batches prefetched by each worker; only applies when `--num-workers` is greater than 0 |
| Worker seed | `--dataloader-seed-workers` | `False` | Bool flag | Whether to seed workers from `--seed` |
| Multiprocessing context | `--dataloader-multiprocessing-context` | `None` | `fork`, `spawn`, `forkserver` | DataLoader worker start method |
| Distributed sampler mode | `--distributed-sampler-mode` | `cyclic` | `cyclic`, `block` | Index sharding scheme for the distributed sampler (extensible) |
| Streaming | `--streaming` | `False` | Bool flag | Use a streaming / iterable dataset |

## 3. Training Configuration

The training configuration module parses shell arguments and YAML files, and produces three typed config objects: `TrainingArgs`, `ModelConfig`, and `DataConfig`. All general training features are exposed as shell flags via `TrainingArgs`.

### 3.1 Model Configuration Selection

Supported capabilities:

- Select a pre-registered model with `--model-name`; the corresponding YAML, `ModelConfig`, and `DataConfig` are bound automatically
- Point to a custom default YAML with `--config-file`
- Specify the tokenizer path (local path or HF repo id) with `--tokenizer-path`

| Feature | Argument | Default | Values / Type | Description |
| --- | --- | --- | --- | --- |
| Select model | `--model-name` | `None` | Model name registered in `config_map.py` | Selects the model schema, default YAML, `ModelConfig`, and `DataConfig` |
| Specify YAML | `--config-file` | `None` | YAML file path | Overrides the default YAML bound to `--model-name` |
| Specify tokenizer | `--tokenizer-path` | `None` | Local path or HF repo id | Sets the tokenizer path and syncs it to the `TOKENIZER_PATH` environment variable |

> **Note:** `--model-name` is still required even when `--config-file` is used, since it selects the structured config classes.

### 3.2 Basic Training Arguments

The arguments in this section control training scale (iteration count and batch size), reproducibility (random seed), and output directory. Gradient accumulation lets you scale the global batch elastically without increasing per-GPU memory.

| Feature | Argument | Default | Values / Type | Description |
| --- | --- | --- | --- | --- |
| Total training steps | `--train-iters` | `150000` | Positive integer | Number of optimizer update steps |
| Per-device batch | `--per-device-batch-size` | `4` | Positive integer | Micro-batch size for a single rank forward |
| Gradient accumulation | `--gradient-accumulation-steps` | `1` | Positive integer | Number of micro-batches accumulated before each optimizer step |
| Random seed | `--seed` | `3047` | Integer | Controls training initialization and data randomness |
| Output directory | `--output-dir` | `outputs/default` | Path | Saves logs, checkpoints, and run artifacts |

Global batch is computed as:

```text
global_batch_size = per_device_batch_size * world_size * gradient_accumulation_steps
```

### 3.3 Learning Rate and Optimizer

The learning rate and optimizer features are substantial. This section is organized as: base schedule → optimizer implementations → parameter-group LR.

Supported capabilities:

- Base learning rate and per-module independent learning rates
- 10 learning rate schedules covering linear, cosine, polynomial, constant, and their warmup / min_lr variants
- 6 optimizer implementations, including standard AdamW and several CUDA-fused accelerated variants
- Gradient clipping and weight decay, with the option to place bias / norm parameters in a separate group

#### 3.3.1 Base Learning Rate and Scheduler

| Feature | Argument | Default | Values / Type | Description |
| --- | --- | --- | --- | --- |
| Base learning rate | `--lr-base` | `2.5e-5` | float | LR for the default parameter group |
| LR schedule | `--lr-decay-style` | `cosine_with_min_lr` | See table below | Scheduler type |
| Warmup steps | `--lr-warmup-iters` | `2000` | Non-negative integer | Linear warmup steps |
| Decay steps | `--lr-decay-iters` | `None` | Positive integer or unset | Falls back to `--train-iters` when unset |
| Minimum LR | `--min-lr` | `1e-6` | float | Decay floor |
| Gradient clipping | `--clip-grad` | `1.0` | float, `<=0` disables | Max gradient norm |
| Weight decay | `--weight-decay` | `0.01` | float | Decoupled weight decay coefficient |
| Weight decay grouping | `--weight-decay-grouping` | `all` | `all`, `bias_norm` | Whether to disable weight decay on bias / norm parameters |

`--lr-decay-style` supports 10 schedules, covering linear, cosine, polynomial, and more.

| Value | Description |
| --- | --- |
| `linear` | Linear decay after warmup |
| `cosine` | cosine decay after warmup |
| `cosine_with_restarts` | cosine decay with hard restarts |
| `polynomial` | polynomial decay |
| `constant` | Constant learning rate |
| `constant_with_warmup` | Constant after warmup |
| `inverse_sqrt` | inverse sqrt decay |
| `cosine_with_min_lr` | cosine decay to `--min-lr` |
| `cosine_warmup_with_min_lr` | cosine warmup with a minimum LR |
| `lambda_linear` | Framework-custom cycle linear scheduler |

#### 3.3.2 Optimizer Implementations

Select the optimizer with `--optimizer`. Standard AdamW, several fused accelerated implementations (PyTorch / Transformer Engine / Apex), plain Adam, and SGD are supported:

| Value | Description |
| --- | --- |
| `AdamW` (default) | Standard AdamW |
| `TorchFusedAdamW` | PyTorch fused AdamW |
| `TEFusedAdamW` | Transformer Engine FusedAdam |
| `ApexFusedAdamW` | Apex FusedAdam |
| `Adam` | torch Adam |
| `SGD` | torch SGD |

Notes on fused Adam implementations:

- **TEFusedAdamW**: fuses the parameter update into a single CUDA kernel, significantly reducing memory bandwidth pressure; requires `Transformer-engine`
- **ApexFusedAdamW**: fuses updates across multiple parameter groups, lowering optimizer step latency for large models; requires `apex`

#### 3.3.3 Module-level Parameter-group LR

Use `--lr-group` to assign independent learning rates to different modules. A common pattern during fine-tuning is a smaller LR for the backbone and a larger LR for the action head; unmatched parameters fall back to `--lr-base`.

- **Argument**: `--lr-group`
- **Default**: `None` (disabled; all parameters use `--lr-base`)
- **Format**: `module.path=lr,module.path=lr`

Example:

```bash
bash examples/embodied/pi05/run_pi05_ddp_finetune.sh \
    --lr-base 1.0e-4 \
    --lr-group "model.backbone=1.0e-5,model.action_head=1.0e-4"
```

Rules:

- Path matching is order sensitive; submodule paths must come before their parent module paths
- Unmatched parameters use `--lr-base`
- Module paths must match the actual attribute paths in the model implementation

### 3.4 Checkpoint

The checkpoint module offers:

- **Weight Saving**: supports safetensors, pt, and dcp formats; DCP additionally supports async saving
- **Training State Saving**: persists optimizer, scheduler, RNG, and DataLoader state for resuming training
- **Pretrained Loading**: initialize model parameters from an external checkpoint

| Feature | Argument | Default | Values / Type | Description |
| --- | --- | --- | --- | --- |
| Load pretrained weights | `--pretrained-checkpoint` | `None` | Checkpoint path | Used to initialize model parameters |
| Resume | `--resume` | `False` | Bool flag | Locate the latest checkpoint under `output_dir/checkpoints` and resume from it |
| Save interval | `--save-interval` | `10000` | Non-negative integer; `0` disables | Save a checkpoint every N update steps |
| Save format | `--save-format` | `safetensors` | `safetensors`, `pt`, `dcp` | Checkpoint file format |
| Save training state | `--save-training-state` | `True` | Bool flag | Save optimizer, scheduler, RNG, and DataLoader state |
| Async save | `--async-save` | `False` | Bool flag | Enable async saving for DCP format |

Resume example:

```bash
bash examples/embodied/pi05/run_pi05_ddp_finetune.sh \
    --output-dir /path/to/previous_run \
    --resume
```

### 3.5 Logging and Monitoring

Supported monitoring backends:

- Console metrics logging, with configurable interval and per-stage timing verbosity
- W&B integration, supporting online / offline / disabled modes
- TensorBoard integration; enabled by specifying a directory
- Per-rank control over loss aggregation and log source

| Feature | Argument | Default | Values / Type | Description |
| --- | --- | --- | --- | --- |
| Log interval | `--log-interval` | `1` | Positive integer | Log metrics every N steps |
| Detailed timing interval | `--detail-log-interval` | `20` | Non-negative integer | Log per-stage timing every N steps |
| Timing log level | `--timing-log-level` | `0` | `0`, `1` | Verbosity of per-stage timing logs |
| W&B project | `--wandb-project` | `loongforge` | String | W&B project name |
| W&B mode | `--wandb-mode` | `disabled` | `online`, `offline`, `disabled` | W&B mode |
| TensorBoard directory | `--tensorboard-dir` | `None` | Path | Leave unset to disable TensorBoard |
| Loss log rank | `--loss-log-rank` | `[-1]` | List of ranks; `-1` means global average | Controls loss aggregation and log source |

### 3.6 Module Freezing

Use `--freeze-modules` to freeze parameters of the specified modules. This is commonly used during fine-tuning to fix the vision encoder or LLM backbone while only updating target modules such as the action head. Module paths follow `named_modules()` on the model implementation; per-model recommended freeze paths are documented in the corresponding Quick Start.

| Feature | Argument | Default | Values / Type | Description |
| --- | --- | --- | --- | --- |
| Freeze modules | `--freeze-modules` | Empty string (nothing frozen) | Comma-separated module paths | Sets `requires_grad=False` on matched module parameters |

## 4. Distributed Trainer

The Trainer module orchestrates the training lifecycle, including distributed context initialization, model construction, weight loading, model wrapping, optimizer and scheduler construction, DataLoader construction, the training loop, logging, checkpointing, and resource cleanup.

### 4.1 Trainer Selection

Select the trainer with `--trainer-type`. Valid values are the Trainer class names registered in `trainer_builder.py`. The default `FinetuneTrainer` covers standard single-data-stream supervised fine-tuning; for multiple data streams, unusual loss combinations, or non-standard step scheduling, register a custom Trainer class in `trainer_builder.py`.

### 4.2 Distributed Strategy

Two distributed parallelism strategies are supported:

- **DDP**: standard data parallelism, suitable when the model and optimizer state fit on a single GPU; can be combined with ZeRO-1 to shard optimizer state and save memory
- **FSDP**: fully sharded parallelism, suitable when the model, gradients, or optimizer state exceed single-GPU memory; supports HSDP (2D mesh sharding)

Training precision supports `bfloat16` (default), `float16`, and `float32`.

| Feature | Argument | Default | Values / Type | Description |
| --- | --- | --- | --- | --- |
| Distributed strategy | `--distributed-strategy` | `fsdp` | `ddp`, `fsdp` | Select DDP or FSDP |
| Training precision | `--dtype` | `bfloat16` | `bfloat16`, `float16`, `float32` | Model training dtype |
| DDP ZeRO-1 | `--zero-optimizer` | `False` | Bool flag | Shard optimizer state under DDP |
| HSDP shard size | `--hsdp-shard-size` | `None` | Positive integer | Enable HSDP under FSDP |

DDP example:

```bash
bash examples/embodied/pi05/run_pi05_ddp_finetune.sh \
    --distributed-strategy ddp \
    --dtype bfloat16
```

FSDP example:

```bash
bash examples/embodied/pi05/run_pi05_fsdp_finetune.sh \
    --distributed-strategy fsdp \
    --dtype bfloat16
```

DDP + ZeRO-1 example:

```bash
bash examples/embodied/pi05/run_pi05_ddp_finetune.sh \
    --distributed-strategy ddp \
    --zero-optimizer
```

Strategy selection guide:

| Scenario | Recommendation |
| --- | --- |
| Model parameters and optimizer state fit on a single GPU | DDP |
| Model fits on a single GPU, but optimizer state uses substantial memory | DDP + ZeRO-1 |
| Model, gradients, or optimizer state do not fit on a single GPU | FSDP |
| Multi-node training where parameter sharding should stay within a shard group to reduce cross-node FSDP communication | FSDP + HSDP |

#### 4.2.1 DDP / ZeRO Common Arguments

The following arguments give fine-grained control over DDP communication behavior and ZeRO-1, and take effect only under `--distributed-strategy ddp`:

- **DDP behavior**: tune unused-parameter detection, static graph optimization, bucket size, and bucket views to reduce communication overhead or save memory
- **ZeRO-1**: once enabled, optimizer state is sharded; you can further configure bucket views and fp32 master parameter maintenance

| Feature | Argument | Default | Values / Type | Description |
| --- | --- | --- | --- | --- |
| Unused parameter detection | `--ddp-find-unused-parameters` | `True` | Bool flag | Usually kept enabled when the model has conditional branches |
| Static graph optimization | `--ddp-static-graph` | `False` | Bool flag | Enable when the compute graph is stable across steps |
| Gradient bucket view | `--ddp-gradient-as-bucket-view` | `False` | Bool flag | Reuse DDP bucket memory |
| DDP bucket size | `--ddp-bucket-cap-mb` | `None` | Integer MB | Controls DDP all-reduce bucket size |
| ZeRO-1 | `--zero-optimizer` | `False` | Bool flag | Shard optimizer state |
| ZeRO bucket view | `--zero-parameters-as-bucket-view` | `False` | Bool flag | Reuse bucket memory under ZeRO |
| ZeRO master parameter dtype | `--zero-master-param-dtype` | `none` | `none`, `fp32` | Whether to maintain fp32 master parameters |

Example:

```bash
bash examples/embodied/pi05/run_pi05_ddp_finetune.sh \
    --distributed-strategy ddp \
    --no-ddp-find-unused-parameters \
    --ddp-static-graph
```

#### 4.2.2 FSDP Common Arguments

The following arguments give fine-grained control over FSDP sharding, wrap policy, dtypes, prefetching, and communication, and take effect only under `--distributed-strategy fsdp`:

- **Sharding and reshard**: control whether to reshard immediately after forward, with optional per-module-class overrides; the FSDP2 root group remains unsharded after forward
- **Wrap policy**: manually specify or exclude FSDP unit classes, or auto-wrap by a parameter-count threshold
- **dtype control**: pre-shard parameter dtype, post-all-gather dtype, and gradient reduce dtype can each be configured independently
- **Execution and communication**: prefetch adjacent FSDP units and optionally compress BF16 AllGather traffic with Delta-FP8

| Feature | Argument | Default | Values / Type | Description |
| --- | --- | --- | --- | --- |
| HSDP | `--hsdp-shard-size` | `None` | Positive integer | Enable the shard dimension of the 2D mesh |
| Default reshard policy | `--fsdp-reshard-default` | `None` | `true`, `false`, `none`, integer > 1 | Controls parameter reshard after forward |
| Per-class reshard policy | `--fsdp-reshard-module-overrides` | `None` | Comma-separated `ClassName=value` pairs | Override `reshard_after_forward` for selected module classes |
| Wrap classes | `--fsdp-wrap-modules` | `None` | Comma-separated module class names | Specify FSDP units |
| Exclude wrap classes | `--fsdp-no-wrap-modules` | `None` | Comma-separated module class names | Exclude the given module classes |
| Exclude params from sharding | `--fsdp-ignored-param-names` | `[]` | Space-separated name substrings | Frozen params matching any substring stay replicated on every rank |
| Replicate frozen module classes | `--fsdp-ignore-frozen-module-classes` | `None` | Comma-separated module class names | Leave fully frozen matched modules outside FSDP sharding to avoid unused AllGathers; increases replicated parameter memory |
| Replicated frozen parameter dtype | `--fsdp-ignored-frozen-param-dtype` | `None` | `fp32`, `bf16`, `fp16` | Optional dtype for replicated frozen parameters; when set, it must match `--dtype` |
| Auto wrap threshold | `--fsdp-min-param-num` | `1000000` | Non-negative integer | Parameter threshold for auto-wrapping repeated layers |
| Original parameter dtype | `--fsdp-original-param-dtype` | `None` | `fp32`, `bf16`, `fp16` | Parameter dtype before FSDP sharding |
| Unsharded parameter dtype | `--fsdp-unshard-param-dtype` | `None` | `fp32`, `bf16`, `fp16` | Forward/backward dtype after AllGather |
| Reduce dtype | `--fsdp-reduce-dtype` | `fp32` | `fp32`, `bf16`, `fp16` | Gradient reduce dtype |
| Output dtype | `--fsdp-output-dtype` | `None` | `fp32`, `bf16`, `fp16` | Optional dtype for floating-point outputs of each FSDP unit |
| Cast forward inputs | `--fsdp-cast-forward-inputs` | `True` | Bool flag | Whether to cast inputs to the parameter dtype |
| Forward prefetch distance | `--fsdp-forward-prefetch-distance` | `0` | Non-negative integer | Number of subsequent configured FSDP units to prefetch during forward |
| Backward prefetch distance | `--fsdp-backward-prefetch-distance` | `0` | Non-negative integer | Number of preceding configured FSDP units to prefetch during backward |
| Delta-FP8 AllGather | `--fsdp-delta-fp8-allgather` | `False` | Bool flag | Compress BF16 FSDP2 AllGather deltas into blockwise FP8; see the [usage guide](../features/delta_fp8_allgather.md) |

When replicating frozen module classes, every matched parameter must have `requires_grad=False`. This mode is incompatible with `--init-on-meta`; if `--fsdp-ignored-frozen-param-dtype` is specified, it must match the training compute dtype selected by `--dtype`.

Delta-FP8 changes only FSDP parameter communication precision; model computation remains in the dtype selected by the FSDP mixed-precision policy. See [Delta-FP8 AllGather](../features/delta_fp8_allgather.md) for prerequisites, usage, parameter reference, and validation guidance.

### 4.3 Stability and Runtime Control

If you encounter erratic loss spikes, numerical instability, or GPU OOM during training, the following knobs help:

- Loss spike protection and NaN/Inf checks for training robustness
- Deterministic mode and TF32 disable, useful for reproducibility alignment
- Manual GC mode to reduce step-to-step jitter introduced by garbage collection

| Feature | Argument | Default | Values / Type | Description |
| --- | --- | --- | --- | --- |
| Loss spike protection | `--loss-spike-threshold` | `100.0` | float | When loss exceeds the threshold or is NaN/Inf, its contribution is zeroed out for that step |
| NaN / Inf check | `--check-for-nan-in-loss-and-grad` | `True` | Bool flag | Check loss and gradients for abnormal values |
| Deterministic mode | `--deterministic-mode` | `False` | Bool flag | Enable deterministic algorithms |
| Disable TF32 | `--disable-tf32` | `False` | Bool flag | Disable CUDA TF32 |
| Manual GC | `--manual-gc` | `False` | Bool flag | Disable automatic GC and trigger it explicitly |
| Manual GC interval | `--manual-gc-interval` | `0` | Non-negative integer | With `--manual-gc` enabled, run GC every N steps |
