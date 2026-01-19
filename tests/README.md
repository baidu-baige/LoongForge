# AIAK-Training-Omni Automated Testing Instructions

The automated test scripts and configurations in this directory are used for Continuous Integration (CI) and functional verification of the AIAK-Training-Omni codebase. With flexible configuration of models and test types, it supports various test scenarios, facilitating extension and maintenance.

## Directory Structure Description

- `tests/configs/`: Configuration for models run by default in CI. Models in this directory are automatically tested in the main pipeline.
- `tests/optional_configs/`: Optional test case models. Used for supplementary or customized testing, not automatically executed in the default CI pipeline, and must be enabled manually or via parameters.
- `tests/main_start.sh`: Main test startup script. Automatically selects test models, types, and tasks based on parameters.
- `tests/download_datasets.sh`: Dataset download script, supports downloading default CI datasets or optional full regression test datasets.
- Other auxiliary scripts and directories: such as `ipipe_start.sh`, `main.py`, etc.

## Test Workflow

The test process is mainly divided into three stages: **Data Preparation**, **Configuration**, and **Execution**.

### 1. Data Preparation

To support flexible automated testing on different cluster resources, we store required datasets, HuggingFace original models, and pre-converted Megatron format Checkpoints (via Step1 conversion) on BOS. Before each test run, the download script must be executed first to pull the necessary resources to the local environment.

**Prerequisites**: Prepare a text file containing AK/SK information (e.g., `aksk.txt`) to configure `bcecmd` permissions.

**Run Command**:

```bash
# 1. Default Mode: Download 8 base case data for CI pipeline (DeepSeek, LLaMA, Qwen, LLaVA, etc.)
bash tests/download_datasets.sh /path/to/aksk.txt

# 2. Optional Mode: Download all regression test case data (including InternVL and other Optional models)
bash tests/download_datasets.sh /path/to/aksk.txt --optional
```

*Note: `main_start.sh` will also attempt to call this script at startup (if an aksk file is provided), but it defaults to executing only the default download logic. If optional data download is needed, it is recommended to run the above command manually.*

### 2. Configuration

The test execution logic is configured in `tests/main_start.sh`. Please open this file and **uncomment** the corresponding mode and set variables as needed:

**Supported Run Modes (Mode 1-5) and Configuration Examples:**

#### Mode 1 (Default): Run only models in configs/

*   **Scenario**: Default CI pipeline task.
*   **Configuration**:
    ```bash
    # Scenario A: Run all default CI models
    model_names=""
    optional_subdir=""
    include_optional=false

    # Scenario B: Run only specified default models
    # model_names="llama3_8b qwen2.5_vl_7b"
    # optional_subdir=""
    # include_optional=false
    ```

#### Mode 2 (Mixed): Mix run of models in configs/ and optional_configs/

*   **Scenario**: Test core models and newly added optional models simultaneously.
*   **Configuration**:
    ```bash
    model_names="internvl2.5_8b"                # Models in configs/
    extra_models="internvl2.5/internvl2.5_8b"   # Extra models in optional_configs/
    include_optional=true                       # Must be enabled
    optional_subdir=""
    ```

#### Mode 3 (Optional Subdir): Run all models in a specific subdirectory of optional_configs/

*   **Scenario**: Regression testing for a specific series (e.g., InternVL 2.5 series).
*   **Configuration**:
    ```bash
    model_names="NONE"                          # Disable models in configs/
    optional_subdir="internvl2.5"               # Specify subdirectory
    include_optional=true
    ```

#### Mode 4 (Optional Specific): Run specific models in optional_configs/

*   **Scenario**: Develop and debug a specific optional model.
*   **Configuration**:
    ```bash
    model_names="NONE"
    extra_models="internvl2.5/internvl2.5_8b"   # Specify model path
    include_optional=true
    optional_subdir=""
    ```

#### Mode 5 (All Optional): Run all models in optional_configs/

*   **Scenario**: Full regression test of all optional configurations.
*   **Configuration**:
    ```bash
    model_names="NONE"
    include_optional=true
    optional_subdir=""                          # Do not specify subdirectory
    extra_models=""                             # Do not specify extra models
    ```

**Other Configuration Parameters:**
*   `tasks`: Test tasks (`check_correctness_task`, `check_perfness_task`, `check_precess_data_task`).
*   `training_type`: Training phase (`pretrain`, `sft`).

### 3. Execution

After configuration is complete, start the test using `main_start.sh`:

```bash
# Start test (if default data preparation is needed, pass aksk file path)
bash tests/main_start.sh /path/to/aksk.txt

# If data is already prepared, run directly (some environments may not require aksk)
bash tests/main_start.sh
```

## 4. Adding New Test Cases

### 4.1 YAML Test Case Specification

Test cases are usually written in YAML format and placed under `configs/` or `optional_configs/`. When writing, please note:

- **Use `#` for comments, but avoid `#` within args in a Step, otherwise it will be parsed as a comment, causing args after `#` to be lost.**
- **Strings should use quotes, especially when containing special characters.**
- **Key Parameter Settings**:
    - `--train-iters` is recommended to be set to `20`.
    - Must set `--load $CHECKPOINT_PATH`.
    - **Do not set** `--save $CHECKPOINT_PATH`.
    - Need to add `--log-memory-stats` flag to output memory usage statistics.

Example:

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
But do not write:
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

Reason:
In YAML syntax, `#` denotes a comment. If you write `#--save $CHECKPOINT_PATH` directly inside a value string, YAML parsing will treat `#` and everything after it as a comment, causing parameter parsing interruption.

#### Key Execution Control Fields

In the YAML configuration file, you can flexibly control the test execution via the following fields:

1.  **`MODEL_RUNNABLE`** (Global Level)
    *   **Location**: Top level of YAML.
    *   **Function**: Controls whether the model Case is enabled globally.
    *   **Example**: `MODEL_RUNNABLE: True`. If set to `False`, the model will be skipped even if it is in the test list. Commonly used to temporarily disable unstable test cases.

2.  **`RUNNABLE_FLAG`** (Step Level)
    *   **Location**: Under specific Step configuration (e.g., `Step1`, `Step2`).
    *   **Function**: Controls whether that specific step is executed.
    *   **Example**: `RUNNABLE_FLAG: "False"`.
    *   **Scenario**: `Step1` is usually Checkpoint format conversion (HF -> Megatron). If the converted Checkpoint already exists on BOS or locally and does not need re-conversion, you can set `Step1`'s `RUNNABLE_FLAG` to `"False"` to start training directly from `Step2`, saving time.

### 4.2 Adding CI Test Cases (tests/configs)

If this model needs to be added to the default CI gatekeeping process (run on every code commit), follow these steps:

1.  **Create Configuration**: Create a new YAML file in `tests/configs/` directory (e.g., `my_new_model.yaml`).
2.  **Configuration Reference**: Recommended to reference general configuration in `common.yaml`.
3.  **Verification**: Manually run `main_start.sh` to ensure it runs correctly.

### 4.3 Adding Optional Test Cases (tests/optional_configs)

If this model is used only for regression testing, non-default CI processes, or depends on special large models/datasets, it should be placed in optional configurations:

1.  **Determine Directory & Structure**: Create a subdirectory in `tests/optional_configs/` by model series, e.g., `tests/optional_configs/internvl3.5/`.
2.  **Create Configuration**: Create a YAML file in the subdirectory (e.g., `internvl3.5.yaml`).
3.  **Path Isolation**: The system automatically adds suffixes to model names under `optional_configs` to isolate Checkpoint and log paths (e.g., `model_name` becomes `internvl2.5/internvl2.5_8b`).

### 4.4 Update Data Download Script

After adding a new Case, `tests/download_datasets.sh` must be updated to ensure the runtime environment can automatically pull required data.

1.  Open `tests/download_datasets.sh`.
2.  Depending on Case type (Default or Optional), add `bcecmd` sync command under the corresponding `if/elif` branch.
3.  **Resources to add typically include**:
    *   **HuggingFace Weights**: `bos:/ai-data/...` -> `${huggingface_dir}/...`
    *   **Dataset**: `bos:/aihc-ai-datasets-bj/...` -> `${datasets_dir}/...`
    *   **Checkpoint**: `bos:/aihc-ai-datasets-bj/...` -> `${checkpoint_dir}/...`

Example Code:

```bash
# Add to download_datasets.sh

# If CI default model, add to "default" branch
# If Optional model, add to "optional" branch

# my_new_model
bcecmd bos sync bos:/ai-data/my_new_model ${huggingface_dir}/my-org/my_new_model
bcecmd bos sync bos:/path/to/my_dataset ${datasets_dir}/my_new_model/dataset
```

### 4.5 Generate and Add Baseline Data

After adding a Case, run the model once to obtain baseline performance data (Baseline) for subsequent correctness and performance checks.

1.  **Run Model**: Use `main_start.sh` to run the newly added Case, expecting training to start normally and run for at least **20** Iterations.
    *   Result logs are typically saved in `/workspace/logs` or the log file specified by `TENSORBOARD_PATH` in the config.

2.  **Extract Baseline**: Use `tests/tools/log2json.py` tool to extract key metrics (Loss, Throughput, etc.) from the log.
    ```bash
    # Example usage
    python3 tests/tools/log2json.py /workspace/logs/my_new_model.log /tmp/my_new_model.json
    ```

3.  **Add Baseline File**:
    *   Move the generated JSON file to the `tests/baseline/` directory.
    *   **CI Default Models**: Put in `tests/baseline/default/` directory.
    *   **Optional Models**: Put in `tests/baseline/optional/` directory.
    *   Filename should be consistent with `model_name` (e.g., `internvl2.5_8b.json`).

Ensure Baseline data accurately reflects the model's convergence and performance metrics in a standard environment.


