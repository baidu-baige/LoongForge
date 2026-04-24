# LoongForge Automated Test Guide

This document explains how to use the automated test scripts and configurations for LoongForge CI and functional validation.

## Directory Structure

- `tests/configs/`: Model configs run by default in CI (automatically tested in the main flow).
- `tests/optional_configs/`: Optional test configs (enabled manually or via parameters).
- `tests/main_start.sh`: Main test entry script.
- `tests/download_datasets.sh`: Dataset download script.
- Other helper scripts and directories: `ipipe_start.sh`, `main.py`, etc.

## Test Workflow

The workflow has three phases: **Data Preparation**, **Configuration**, and **Execution**.

### 1. Data Preparation

Datasets, HuggingFace base models, and pre-converted Megatron checkpoints (via Step1) are stored on BOS. Download them before each run.

**Download data**:

```bash
# Default mode: download 8 base cases (DeepSeek, LLaMA, Qwen, LLaVA, etc.)
bash tests/download_datasets.sh

# Optional mode: download optional regression cases (e.g., Qwen3_vl)
bash tests/download_datasets.sh --optional
```

**Run directly (if data already synced)**:

```bash
# Default mode
bash main_start.sh

# Optional mode
bash main_start.sh --optional
```

### 2. Configuration

Configure in `tests/main_start.sh`. **Uncomment** the desired mode and set variables as needed.

#### 2.1 Run Modes (Mode 1–5)

**Mode 1 (Default)**: Run models in `configs/` only

- Scenario: default CI flow
- Config:

```bash
# Scenario A: run all default CI models
model_names=""
optional_subdir=""
include_optional=false

# Scenario B: run specific default models
# model_names="llama3_8b qwen2.5_vl_7b"
# optional_subdir=""
# include_optional=false
```

**Mode 2 (Mixed)**: Run `configs/` and `optional_configs/` together

- Scenario: test core models and optional models together
- Config:

```bash
model_names="internvl2.5_8b"                # model under configs/
extra_models="internvl2.5/internvl2.5_8b"   # extra model under optional_configs/
include_optional=true
optional_subdir=""
```

**Mode 3 (Optional subdir)**: Run a specific subdirectory under `optional_configs/`

- Scenario: regression for a model family (e.g., InternVL 2.5)
- Config:

```bash
model_names="NONE"            # disable configs/ models
optional_subdir="internvl2.5"
include_optional=true
```

**Mode 4 (Optional specific)**: Run a specific model under `optional_configs/`

- Scenario: develop/debug a specific optional model
- Config:

```bash
model_names="internvl2.5/internvl2.5_8b"
include_optional=true
optional_subdir=""
```

**Mode 5 (All optional)**: Run all models under `optional_configs/`

- Scenario: full regression for optional configs
- Config:

```bash
model_names="NONE"
include_optional=true
optional_subdir=""
extra_models=""
```

#### 2.2 Other Parameters

- `tasks`: test tasks (`check_correctness_task`, `check_precess_data_task`).
- `training_type`: stage (`pretrain`, `sft`).
- `chip`: GPU type.
- `auto_collect_baseline`: baseline collection switch.

### 3. Execution

```bash
bash tests/main_start.sh
```

After completion, check regression results under `/workspace/E2E/diff`.

## 4. Add New Test Cases

### 4.1 Prepare Data and HF Weights

Upload HF weights and test datasets to BOS. Add `bcecmd sync` in `download_datasets.sh` based on case type (Default/Optional). Example for `qwen3_vl_30b_a3b`:

### 4.2 Write Test YAML

#### 4.2.1 Conventions

- Use `#` for comments, but do not use `#` inside Step `args`.
- Prefer quoted strings.
- Key params:
  - `--train-iters` should be set to `20`.
  - Must set `--load $CHECKPOINT_PATH`.
  - Add `--log-memory-stats` to output memory stats.

**Example**:

```yaml
Step2:
   TRAINING_ARGS: '
      --train-iters ${train_iters}
      --lr-decay-style cosine
      --load $CHECKPOINT_PATH
      --save-interval 2000
      --log-memory-stats
   '
```

**Do not write** (`#` is treated as a comment in YAML):

```yaml
Step2:
   TRAINING_ARGS: '
      --train-iters ${train_iters}
      --lr-decay-style cosine
      --load $CHECKPOINT_PATH
      #--save $CHECKPOINT_PATH
      --save-interval 2000
      --log-memory-stats
   '
```

#### 4.2.2 Key Control Fields

1. `MODEL_RUNNABLE` (global): set to `False` to skip the model.
2. `RUNNABLE_FLAG` (Step): Step1 is HF -> Megatron conversion; if a checkpoint already exists, set Step1 to `"False"` and start from Step2.

#### 4.2.3 Weight Conversion (Step1)

1. Under `scenarios`, `function` means functional tests: Step1 is conversion, Step2 is training launch config.
2. Copy configs from `examples/$model_name/checkpoint_convert` into Step1.
3. Converted checkpoints are saved in `CHECKPOINT_PATH`, and `download_datasets.sh` should be updated accordingly.
4. If conversion is not needed, set `RUNNABLE_FLAG: False` in Step1.

#### 4.2.4 Training Script (Step2)

1. Add `DATA_ARGS`, `TRAINING_ARGS`, `MOE_ARGS`, etc. in Step2.
2. Add `--log-memory-stats` to output memory usage.
3. If `--split 100,0,0`, set `--eval-iters 0`.
4. `--tp-comm-overlap-bootstrap-backend nccl` is deprecated in 0.15.0.
5. `tasks: check_correctness_task: True`.

### 4.3 Baseline Collection

1. Update `main_start.sh`: set `model_names`, `optional_subdir`, `include_optional`.
2. Set chip and enable baseline collection:

```bash
chip="A800"
auto_collect_baseline=true
```

3. Run `bash main_start.sh`. Baselines will be saved under `tests/baseline`.

Baseline fields:

<div align="center">

<table style="border-collapse:collapse;width:100%;border-top:2px solid #000;border-bottom:2px solid #000;">
  <thead>
    <tr style="border-bottom:1px solid #000;">
      <th style="text-align:center;padding:6px 8px;">Field</th>
      <th style="text-align:center;padding:6px 8px;">Description</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td style="text-align:center;padding:6px 8px;">Training type</td>
      <td style="text-align:center;padding:6px 8px;"><code>pretrain</code> / <code>sft</code></td>
    </tr>
    <tr>
      <td style="text-align:center;padding:6px 8px;">Iteration</td>
      <td style="text-align:center;padding:6px 8px;"><code>iteration</code> (default 20 iters)</td>
    </tr>
    <tr>
      <td style="text-align:center;padding:6px 8px;">Elapsed time</td>
      <td style="text-align:center;padding:6px 8px;"><code>elapsed_time_ms</code></td>
    </tr>
    <tr>
      <td style="text-align:center;padding:6px 8px;">Throughput</td>
      <td style="text-align:center;padding:6px 8px;"><code>throughput</code></td>
    </tr>
    <tr>
      <td style="text-align:center;padding:6px 8px;">Loss</td>
      <td style="text-align:center;padding:6px 8px;"><code>lm_loss</code></td>
    </tr>
    <tr>
      <td style="text-align:center;padding:6px 8px;">Activation</td>
      <td style="text-align:center;padding:6px 8px;"><code>grad_norm</code></td>
    </tr>
    <tr>
      <td style="text-align:center;padding:6px 8px;">Memory</td>
      <td style="text-align:center;padding:6px 8px;">
        <div><code>mem_allocated_avg_MB</code></div>
        <div><code>mem_max_allocated_avg_MB</code></div>
        <div>(requires <code>--log-memory-stats</code>)</div>
      </td>
    </tr>
  </tbody>
</table>

</div>

**Baseline Example**:

```json
{
  "pretrain": [
    {
      "iteration": 1,
      "elapsed_time_ms": 7728.5,
      "throughput": 45.5,
      "lm_loss": 4.230574,
      "grad_norm": 73.796051,
      "mem_allocated_avg_MB": 61018.83,
      "mem_max_allocated_avg_MB": 63323.1
    },
    ......
    {
      "iteration": 20,
      "elapsed_time_ms": 1604.1,
      "throughput": 221.0,
      "lm_loss": 0.07428951,
      "grad_norm": 6.447771,
      "mem_allocated_avg_MB": 61018.83,
      "mem_max_allocated_avg_MB": 63323.1
    }
  ],
  "sft": [
    {
      "iteration": 1,
      "elapsed_time_ms": 8520.8,
      "throughput": 50.2,
      "lm_loss": 3.341707,
      "grad_norm": 64.709526,
      "mem_allocated_avg_MB": 61017.01,
      "mem_max_allocated_avg_MB": 63323.1
    },
    ......
    {
      "iteration": 20,
      "elapsed_time_ms": 1634.8,
      "throughput": 263.3,
      "lm_loss": 0.000125067,
      "grad_norm": 0.098754,
      "mem_allocated_avg_MB": 61017.01,
      "mem_max_allocated_avg_MB": 63323.1
    }
  ]
}
```
