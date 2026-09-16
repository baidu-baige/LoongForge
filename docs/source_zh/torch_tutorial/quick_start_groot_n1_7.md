# 快速入门：GR00TN1.7 模型训练

本文档介绍如何在 LoongForge 框架下快速启动 **GR00TN1.7** SFT 训练。

## 0. 资源准备

### 0.1 模型权重

GR00T-N1.7 主权重由 `CHECKPOINT_PATH` 指定，并通过 `--pretrained-checkpoint` 加载。脚本默认路径为 `/workspace/huggingface.co/GR00T-N1.7-3B`：

```bash
hf download nvidia/GR00T-N1.7-3B --local-dir /workspace/huggingface.co/GR00T-N1.7-3B
```

如果 Hugging Face 仓库需要权限，先完成登录：

```bash
huggingface-cli login
```

### 0.2 Qwen3-VL / Cosmos Backbone

GR00T-N1.7 使用 Cosmos-Reason2-2B 作为 backbone，脚本通过 `COSMOS_LOCAL_PATH` 指向本地 backbone，并将 `TOKENIZER_PATH` 默认设置为同一路径：

```bash
hf download nvidia/Cosmos-Reason2-2B --local-dir /workspace/huggingface.co/nvidia/Cosmos-Reason2-2B
```

注意：`COSMOS_LOCAL_PATH` 不是 GR00T-N1.7 主权重路径，它负责 Qwen3-VL backbone、processor/tokenizer 和相关 config 的本地加载。

### 0.3 数据集

脚本默认使用官方预处理好的 LeRobot v2.1 格式示例数据集 `cube_to_bowl_5`。如需使用自定义数据训练，请参照 [NVIDIA Isaac-GR00T](https://github.com/NVIDIA/Isaac-GR00T) 完成离线数据预处理，将原始数据转换为 LeRobot v2.1 格式后再用于训练。

```bash
export DATA_PATH=/workspace/cube_to_bowl_5
```

训练入口中的数据参数为：

```bash
--dataset-format lerobot_datasets
--dataset-strategy groot_n1_7
--lerobotdataset-version v2.1
--video-backend torchcodec
```

数据目录须包含 LeRobot 元数据及视频/图像字段，若目录下存在 `meta/modality.json`、`meta/stats.json`、`meta/relative_stats.json`，N1.7 数据策略将在训练时自动读取，用于解析模态信息和归一化统计量。

## 1. 数据配置

训练时会在线完成：

- LeRobot episode/step 采样；
- GR00T-N1.7 sample transform；
- Qwen3-VL 文本和图像 batch collate；
- state/action 归一化、padding 和 action mask 构造。

默认数据配置如下：

```yaml
data:
  embodiment_tag: libero_sim
  groot_preprocess_mode: sharded
  max_token_len: null
  preprocess_action_horizon: null
  state_dropout_prob: 0.2
  exclude_state: false
  use_mean_std: false
  use_percentiles: true
  clip_outliers: true
  apply_sincos_state_encoding: false
  use_relative_action: true
  image_crop_size: [230, 230]
  image_target_size: [256, 256]
  random_rotation_angle: 0
  formalize_language: true
  shard_size: 1024
  episode_sampling_rate: 0.1
  num_shards_per_epoch: 100000
```

如果训练自定义 embodiment，可在启动命令末尾覆盖：

```bash
bash examples/vla/groot_n1_7/run_groot_n1_7_ddp_finetune.sh data.embodiment_tag=new_embodiment
```

## 2. 启动训练

先统一设置路径：

```bash
cd /workspace/LoongForge

export LOONGFORGE_PATH=/workspace/LoongForge/
export CHECKPOINT_PATH=/workspace/huggingface.co/GR00T-N1.7-3B
export COSMOS_LOCAL_PATH=/workspace/huggingface.co/nvidia/Cosmos-Reason2-2B/
export TOKENIZER_PATH=$COSMOS_LOCAL_PATH
export DATA_PATH=/workspace/cube_to_bowl_5
export OUTPUT_DIR=/workspace/outputs/groot_n1_7
export TENSORBOARD_PATH=/workspace/tensorboard-log/groot_n1_7
```

### 2.1 验证训练链路

先用最小配置跑通数据加载、权重加载、DDP 通信和 forward/backward 链路。此模式关闭所有性能优化，使用标准 `AdamW`：

```bash
bash /workspace/LoongForge/examples/vla/groot_n1_7/run_groot_n1_7_ddp_finetune.sh \
    --optimizer AdamW \
    --cuda-graph-impl none \
    --cuda-graph-pad-length 0 \
    --no-ddp-static-graph \
    --train-iters 10
```

### 2.2 默认训练配置

脚本默认同时开启以下优化项，可直接运行：

```bash
bash /workspace/LoongForge/examples/vla/groot_n1_7/run_groot_n1_7_ddp_finetune.sh
```

默认启用的优化包括：

- **TEFusedAdamW** — 融合优化器，减少优化器 step 的 kernel launch 开销；
- **CUDA Graph** — `per_microbatch` 模式，对每个 microbatch 捕获 CUDA Graph 以减少 host 侧调度开销；
- **DDP Static Graph** — 启用 DDP 静态图模式，配合 CUDA Graph 使用；

如需关闭某项优化，可在命令末尾覆盖对应参数，例如回退到普通 AdamW：

```bash
--optimizer AdamW
```

或关闭 CUDA Graph：

```bash
--cuda-graph-impl none
```

CUDA Graph 补充说明：

- `--cuda-graph-pad-length 0` 表示不强制固定 token 长度，实际 padding 使用数据配置中的 `max_token_len` 或动态 batch padding；
- 如需固定文本长度以提高 graph 复用率，可设置非零 `--cuda-graph-pad-length`。

### 2.3 正确性验证

为保证训练精度不受优化手段影响，我们在相同数据、权重和训练配置下，对 LoongForge 适配的 GR00T-N1.7 与官方实现进行了逐 step 的 action loss 对比验证。结果表明，LoongForge 的各项性能优化（如 TEFusedAdamW、CUDA Graph、DDP static graph 等）对训练精度无损：
![alt text](../../assets/images/precision/GR00TN1.7.png)
