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
