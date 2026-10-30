# 快速入门：FastWAM模型训练

本文档介绍如何在 LoongForge 框架下快速启动 **FastWAM** SFT（监督微调）训练

## 0. 资源准备
### 0.1 模型权重
#### 0.1.1 抽取 Action DiT 主干
从 Wan2.2 视频 DiT checkpoint 中抽取并线性插值出 Action DiT 头权重，得到一份 `.pt` 文件供训练直接加载：

```bash
LOCAL_MODEL_PATH=/data/models \
OUTPUT=$LOONGFORGE_PATH/checkpoints/ActionDiT_linear_interp_Wan22_alphascale_1024hdim.pt \
    bash examples/embodied/fastwam/preprocess_action_dit_backbone.sh
```

#### 0.1.2 Wan-AI/Wan2.2-TI2V-5B 权重

```bash
hf download Wan-AI/Wan2.2-TI2V-5B --local-dir /workspace/huggingface.co/Wan-AI/Wan2.2-TI2V-5B
```

### 0.2 Tokenizer / Processor
将数据集中所有任务指令一次性编码为 text embed，训练时按需读取：

```bash
DATASET_PATH=/data/libero \
TEXT_EMBEDDING_CACHE_DIR=/data/cache/fastwam_text_embeds \
    bash examples/embodied/fastwam/precompute_text_embeds.sh
```

### 0.3 数据集

```bash
hf download yuanty/LIBERO-fastwam --local-dir /workspace/data/LIBERO-fastwam
cd /workspace/data/LIBERO-fastwam
tar -zxvf libero_spatial_no_noops_lerobot.tar.gz
```

## 1. 数据配置

数据相关字段主要通过 YAML 与环境变量配置。

**模型与数据 YAML：**

```yaml
model:
  action_dit_pretrained_path: checkpoints/ActionDiT_linear_interp_Wan22_alphascale_1024hdim.pt  # 前面生成好的 action dit 路径
  redirect_common_files: true
  dtype: bfloat16

data:
  text_embedding_cache_dir: data/text_embeds_cache/libero  # 前面生成好的 text embed 路径
```

**环境变量：**

```bash
export LOONGFORGE_PATH=/workspace/LoongForge                                                    # 仓库根目录
export DATASET_PATH=/workspace/data/LIBERO-fastwam/libero_spatial_no_noops_lerobot              # LeRobot 数据集根目录
export DIFFSYNTH_MODEL_BASE_PATH=/workspace/huggingface.co/Wan-AI/Wan2.2-TI2V-5B                # Wan2.2 5B 模型基础路径
```

## 2. 启动训练

### 2.1 启动脚本
单机 DDP 训练示例：

```bash
bash examples/embodied/fastwam/run_fastwam_sft_ddp_finetune.sh
```

### 2.2 正确性验证

为保证训练精度不受优化手段影响，我们在相同数据、权重和训练配置下，对 LoongForge 适配的 Fastwam 与官方实现进行了逐 step 的 action loss 对比验证。结果表明，LoongForge 的各项性能优化对训练精度无损：
![alt text](../../assets/images/precision/fastwam.png)

## 3. 已验证配方（LIBERO 四套件联合）

下面是参考 20k SFT checkpoint 使用的配方。8×A800-80GB、单卡 batch 16（全局 batch 128）下约 1.37 s/step、71 GB/卡、8.4 小时：

```bash
export DATASET_PATH=/path/to/LIBERO-fastwam/libero_spatial_no_noops_lerobot,/path/to/LIBERO-fastwam/libero_object_no_noops_lerobot,/path/to/LIBERO-fastwam/libero_goal_no_noops_lerobot,/path/to/LIBERO-fastwam/libero_10_no_noops_lerobot

LR_DECAY_STYLE=linear_warmup_cosine_annealing \
CLIP_GRAD=1.0 \
ZERO_MASTER_PARAM_DTYPE=fp32 \
SAVE_FORMAT=dcp ASYNC_SAVE=1 \
GPUS_PER_NODE=8 TRAIN_ITERS=20000 PER_DEVICE_BATCH_SIZE=16 SEED=42 \
EXTRA_MODEL_OVERRIDES="model.mot_compile_blocks=none" \
EXTRA_DATA_OVERRIDES="data.norm_stats_path=/path/to/dataset_stats.json" \
bash examples/embodied/fastwam/run_fastwam_sft_ddp_zero1_finetune.sh \
  --config-file configs/models/embodied/fastwam.yaml \
  --optimizer TorchFusedAdamW \
  --lr-base 1e-4 --lr-warmup-iters 1000 --min-lr 1e-6
```

说明：

- `ZERO_MASTER_PARAM_DTYPE=fp32` 会保留一份 fp32 权重副本和 fp32 Adam 状态，与参考实现的 `BF16_Optimizer` 一致。若直接用 bf16 参数更新，lr 衰减后大部分权重每步的更新量会被舍入成 0（实测 step ~18.5k 时 99.5% 的元素），训练尾部会「学不动」。代价是约 +1% 步时、checkpoint 变大。
- `CLIP_GRAD=1.0` 是默认值。关掉后该配方下 `libero_10` 掉 3.4pt（McNemar p=0.02），其余三个 suite 在 ±0.4pt 以内。
- `data.norm_stats_path` 必须指向与 checkpoint 匹配的 `dataset_stats.json`：`normalization_mode: q99` 需要其中的 `q01`/`q99`，否则数据管线会报 `KeyError: 'q01'`。
- 四个 suite 按帧数联合采样（不是每套件等权），`DATASET_PATH` 直接列四个 suite 即可。

### 评测结果（2000 episode，4 套件 × 500）

| suite | 成功率 |
|---|---|
| libero_spatial | 97.00%（485/500） |
| libero_object | 99.60%（498/500） |
| libero_goal | 97.20%（486/500） |
| libero_10 | 94.20%（471/500） |
| **合计** | **97.00%（1940/2000）** |

## 4. 评测

环境准备与协议说明见 [LIBERO 评测文档](https://github.com/baidu-baige/LoongForge/blob/master/docs/source/embodied_tutorial/eval_docs/benchmarks/libero.md)。运行前先把 suite YAML 里的 `/path/to/...` 占位符（checkpoint、Wan2.2 路径、`dataset_stats.json`）填好：

```bash
CONFIG=examples/embodied/fastwam/eval/configs/libero/libero_10.yaml \
REPO_ROOT=/path/to/LoongForge-VLA \
BENCHMARK_PYTHON=/path/to/libero/bin/python \
DIFFSYNTH_MODEL_BASE_PATH=/path/to/models_root \
bash examples/embodied/fastwam/eval/run_libero_eval.sh
```

随附 YAML 使用的是参考评测协议：`episodes_per_task: 50`，`max_steps` 400（`spatial`/`object`/`goal`）与 700（`libero_10`），action `sigma_shift: 1.0`，`num_steps_wait: 30`，`controller_force_update: false`，`prefer_controller_ee: false`，二值化夹爪，`noise_seed: 42`。
