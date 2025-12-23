# AIAK-Training-Omni 自动化测试说明

本目录下的自动化测试脚本和配置用于对 AIAK-Training-Omni 代码库进行持续集成（CI）和功能验证。通过灵活配置模型和测试类型，支持多种测试场景，便于扩展和维护。

## 目录结构说明

- `tests/configs/`：CI 默认执行的模型测试配置。此目录下的模型会在主流程自动测试。
- `tests/optional_configs/`：可选测试用例模型。用于补充性或定制化测试，不会在 CI 默认流程中自动执行，需手动指定或通过参数启用。
- `tests/main_start.sh`：主测试启动脚本。根据参数自动选择测试模型、类型和任务。
- 其他辅助脚本和目录：如 `download_datasets.sh`、`ipipe_start.sh`、`main.py` 等。

## 测试流程与 Step 说明

测试流程由 `main_start.sh` 脚本驱动，主要分为以下步骤：

1. **模型选择**
   - 通过设置 `model_names`、`optional_subdir`、`include_optional` 等参数，决定本次测试涉及的模型。
   - 支持三种模式：
     - 仅运行 `configs/` 下模型（CI 默认）
     - 混合运行 `configs/` 和 `optional_configs/` 下模型
     - 仅运行 `optional_configs/` 下某子目录所有模型

2. **测试任务类型**
   - `tasks` 参数决定测试内容：
     - `check_correctness_task`：正确性验证
     - `check_perfness_task`：性能测试
     - `check_precess_data_task`：数据处理流程测试

3. **训练类型**
   - `training_type` 参数支持 `pretrain`、`sft` 或两者组合，分别对应预训练和监督微调流程。

4. **容差与超时设置**
   - `accuracy_relative_tolerance`、`performance_relative_tolerance` 控制测试容忍度。
   - `TIMEOUT` 设置单次测试最大允许时长。

5. **参数构建与执行**
   - 脚本自动拼接参数并调用 `main.py` 执行测试。

## YAML 测试用例编写规范

测试用例通常以 YAML 格式编写，放置于 `configs/` 或 `optional_configs/` 下。编写时需注意：

- **注释请用 `#`，但避免在Step下的args中出现 `#`，否则会被解析为注释，导致`#`后的args丢失。**
- **字符串建议加引号，尤其包含特殊字符时。**

示例：

```yaml
Step2:
   DATA_ARGS: '
   --tokenizer-type HFTokenizer
   --hf-tokenizer-path $TOKENIZER_PATH
   --data-path $DATA_PATH
   --split 100,0,0
   --chat-template empty'
   TRAINING_ARGS: '
   --training-phase sft
   --seq-length 16384
   --max-position-embeddings 32768
   --max-packed-tokens 16384
   --init-method-std 0.01
   --micro-batch-size 1
   --global-batch-size 32
   --lr 1e-5
   --min-lr 0.0
   --clip-grad 1.0
   --weight-decay 0.05
   --optimizer adam
   --adam-beta1 0.9
   --adam-beta2 0.999
   --adam-eps 1e-8
   --norm-epsilon 1e-6
   --attention-dropout 0
   --hidden-dropout 0
   --train-iters ${train_iters}
   --lr-decay-style cosine
   --lr-warmup-fraction 0.03
   --bf16
   --seed 42
   --no-gradient-accumulation-fusion
   #--load $CHECKPOINT_PATH
   --save-interval 2000
   --exit-interval 500
   --dataloader-type external
   --variable-seq-lengths  
   --min-num-frame 8
   --max-num-frame 32
   --max-buffer-size 20
   --energon-pack-algo sequential_max_images
   --allow-missing-adapter-checkpoint'
   # --save ${step1_output_path}
```

**注意：不能在 value 字符串中加 `#--load $CHECKPOINT_PATH` 这种注释。**

原因说明：

YAML 语法中，`#` 表示注释符号。如果你在 value 字符串中直接写 `#--load $CHECKPOINT_PATH`，YAML 解析时会把 `#` 及其后面的内容全部当作注释，导致这一行后面的参数不会被传递到实际命令中。
