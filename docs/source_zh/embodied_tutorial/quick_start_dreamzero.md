# 快速入门：DreamZero模型训练

本文档介绍如何在 LoongForge 框架下快速启动 **DreamZero** 训练，包括权重与数据准备、Full / LoRA 训练、续训以及可选的 Feature Cache。支持 Wan2.2-TI2V-5B 和 Wan2.1-I2V-14B 两种模型规格，命令默认从 LoongForge 仓库根目录执行，示例使用 `/workspace/LoongForge`，请按实际环境替换。

## 0. 资源准备
### 0.1 环境变量
```bash
cd /workspace/LoongForge
export LOONGFORGE_PATH=$(pwd)
export DREAMZERO_CKPT_ROOT=/workspace/dreamzero/checkpoints
export DREAMZERO_DATA_ROOT=/workspace/dreamzero/data
export DREAMZERO_CACHE_ROOT=/workspace/dreamzero/cache

export WAN21_CKPT_DIR=$DREAMZERO_CKPT_ROOT/Wan2.1-I2V-14B-480P
export WAN22_CKPT_DIR=$DREAMZERO_CKPT_ROOT/Wan2.2-TI2V-5B
export DREAMZERO_AGIBOT_CKPT_DIR=$DREAMZERO_CKPT_ROOT/DreamZero-AgiBot

mkdir -p "$DREAMZERO_CKPT_ROOT" "$DREAMZERO_DATA_ROOT" "$DREAMZERO_CACHE_ROOT"
```

| 环境变量 | 含义 |
| --- | --- |
| `LOONGFORGE_PATH` | LoongForge 仓库根目录，用于从任意目录定位训练脚本和配置 |
| `DREAMZERO_CKPT_ROOT` | DreamZero 相关权重的统一存放目录 |
| `DREAMZERO_DATA_ROOT` | 各训练数据集的统一存放目录 |
| `DREAMZERO_CACHE_ROOT` | 离线 Feature Cache 的统一输出目录；不使用 cache 时不会影响在线训练 |
| `WAN21_CKPT_DIR` | Wan2.1-I2V-14B 权重目录，同时为 5B 训练提供 CLIP 图像编码器 |
| `WAN22_CKPT_DIR` | Wan2.2-TI2V-5B 权重目录 |
| `DREAMZERO_AGIBOT_CKPT_DIR` | AgiBot / YAM LoRA 训练使用的初始化权重目录；DROID 和 LIBERO 不需要 |

上述路径均可按实际存储位置调整。训练脚本会读取对应变量，不要求使用示例中的目录结构。

### 0.2 下载权重

| 场景 | 必需权重 |
| --- | --- |
| Wan2.2 5B DROID / LIBERO | 完整 Wan2.2 5B；Wan2.1 中的 CLIP image encoder |
| Wan2.1 14B DROID | 完整 Wan2.1 14B |
| Wan2.1 14B AgiBot / YAM LoRA | 完整 Wan2.1 14B；DreamZero-AgiBot 初始化权重 |

```bash
python -m pip install -U "huggingface_hub[cli]"

# Wan backbone
hf download Wan-AI/Wan2.1-I2V-14B-480P --local-dir "$WAN21_CKPT_DIR"
hf download Wan-AI/Wan2.2-TI2V-5B --local-dir "$WAN22_CKPT_DIR"

# 只训练 5B 时，Wan2.1 只需补充 CLIP（Wan2.2 仓库不含 CLIP，5B 仍用它编码首帧）
hf download Wan-AI/Wan2.1-I2V-14B-480P \
    models_clip_open-clip-xlm-roberta-large-vit-huge-14.pth \
    --local-dir "$WAN21_CKPT_DIR"

# AgiBot / YAM LoRA 初始化权重（DROID、LIBERO 不需要）
hf download GEAR-Dreams/DreamZero-AgiBot \
    --repo-type model --local-dir "$DREAMZERO_AGIBOT_CKPT_DIR"
```

Tokenizer 默认使用 Wan 仓库自带的 `$WAN22_CKPT_DIR/google/umt5-xxl`（5B）/ `$WAN21_CKPT_DIR/google/umt5-xxl`（14B），无需单独下载。若需要独立共享目录：

```bash
export TOKENIZER_PATH=$DREAMZERO_CKPT_ROOT/umt5-xxl
hf download google/umt5-xxl \
    special_tokens_map.json spiece.model tokenizer.json tokenizer_config.json \
    --local-dir "$TOKENIZER_PATH"
```

### 0.3 权重完整性检查
```bash
test -f "$WAN22_CKPT_DIR/diffusion_pytorch_model.safetensors.index.json"
test -f "$WAN22_CKPT_DIR/models_t5_umt5-xxl-enc-bf16.pth"
test -f "$WAN22_CKPT_DIR/Wan2.2_VAE.pth"
test -f "$WAN21_CKPT_DIR/models_clip_open-clip-xlm-roberta-large-vit-huge-14.pth"
test -f "$WAN22_CKPT_DIR/google/umt5-xxl/spiece.model"

# 训练 14B 时再检查：
test -f "$WAN21_CKPT_DIR/diffusion_pytorch_model.safetensors.index.json"
test -f "$WAN21_CKPT_DIR/models_t5_umt5-xxl-enc-bf16.pth"
test -f "$WAN21_CKPT_DIR/Wan2.1_VAE.pth"
echo "DreamZero weight check passed"
```

### 0.4 下载数据集
```bash
# DROID（约 131GB，已转换为 LeRobot v2）
export DROID_DATA_ROOT=$DREAMZERO_DATA_ROOT/droid_lerobot
hf download GEAR-Dreams/DreamZero-DROID-Data \
    --repo-type dataset --local-dir "$DROID_DATA_ROOT"

# LIBERO（用于仿真场景训练）
export LIBERO_DATA_ROOT=$DREAMZERO_DATA_ROOT/libero_lerobot
hf download physical-intelligence/libero \
    --repo-type dataset --local-dir "$LIBERO_DATA_ROOT"
```

AgiBot / YAM 暂无可直接使用的 LeRobot v2 训练数据，需按下节字段约定完成转换。

## 1. 数据配置
### 1.1 输入格式
```text
dataset_root/
├── data/**/episode_*.parquet
├── videos/                         # info.json 中使用 video dtype 时必需
└── meta/
    ├── info.json                   # codebase_version 须以 v2 开头
    ├── tasks.jsonl
    └── episodes.jsonl
```

| `EMBODIMENT_TAG` | State / Action | 视频字段 | 语言或任务字段 |
| --- | --- | --- | --- |
| `oxe_droid` | `observation.state` ≥14 维；`action` ≥21 维 | `exterior_image_1_left`、`exterior_image_2_left`、`wrist_image_left` | 三个 `annotation.language.*` 字段 |
| `libero_sim` | `state` 8 维；`actions` 7 维 | `image`、`wrist_image` | `task_index` |
| `agibot` | `observation.state` ≥20 维；`action` ≥22 维 | `top_head`、`hand_left`、`hand_right` | `task_index` |
| `yam` | packed `observation.state` / `action` ≥46 维 | top/left/right 三个 `*-camera-images-rgb` 字段 | `task_index` |

完整字段以 `tools/data_preprocess/embodied/dreamzero/prepare_dataset.py` 中的 `PRESETS` 为准。

### 1.2 生成 DreamZero metadata
```bash
EMBODIMENT_TAG=oxe_droid DATA_PATH="$DROID_DATA_ROOT" \
    bash examples/embodied/dreamzero/prepare_dreamzero_dataset.sh

EMBODIMENT_TAG=libero_sim DATA_PATH="$LIBERO_DATA_ROOT" \
    bash examples/embodied/dreamzero/prepare_dreamzero_dataset.sh

EMBODIMENT_TAG=agibot DATA_PATH=/path/to/agibot_lerobot \
    bash examples/embodied/dreamzero/prepare_dreamzero_dataset.sh

EMBODIMENT_TAG=yam DATA_PATH=/path/to/yam_lerobot \
    bash examples/embodied/dreamzero/prepare_dreamzero_dataset.sh
```

常用选项：

- `FORCE=1`：覆盖已有 metadata
- `SKIP_STATISTICS=1`：只生成 schema、跳过统计量；开始完整训练前仍需补齐统计量

处理成功后会输出 `DreamZero dataset preparation complete`。

## 2. 启动训练
### 2.1 Full / LoRA 训练
```bash
cd "$LOONGFORGE_PATH"
export DATA_PATH=$DROID_DATA_ROOT
export WANDB_MODE=disabled

# 5B Full（FSDP + Delta-FP8），默认 200000 步
bash examples/embodied/dreamzero/run_dreamzero_wan22_5b_full_fsdp_finetune.sh

# 5B LoRA
TRAIN_ITERS=10000 SAVE_INTERVAL=1000 \
    bash examples/embodied/dreamzero/run_dreamzero_wan22_5b_lora_fsdp_finetune.sh

# 14B Full
bash examples/embodied/dreamzero/run_dreamzero_wan21_14b_full_fsdp_finetune.sh

# 14B LoRA，DROID
TRAIN_ITERS=10000 SAVE_INTERVAL=1000 \
    bash examples/embodied/dreamzero/run_dreamzero_wan21_14b_lora_fsdp_finetune.sh

# LIBERO 5B Full
MODEL_NAME=dreamzero_libero_wan22_5b DATA_PATH="$LIBERO_DATA_ROOT" \
    bash examples/embodied/dreamzero/run_dreamzero_wan22_5b_full_fsdp_finetune.sh

# AgiBot / YAM 14B LoRA（从 DreamZero-AgiBot 初始化）
MODEL_NAME=dreamzero_agibot_wan21_14b DATA_PATH=/path/to/agibot_lerobot \
    bash examples/embodied/dreamzero/run_dreamzero_wan21_14b_lora_fsdp_finetune.sh

MODEL_NAME=dreamzero_yam_wan21_14b DATA_PATH=/path/to/yam_lerobot \
    bash examples/embodied/dreamzero/run_dreamzero_wan21_14b_lora_fsdp_finetune.sh
```

DROID 场景无需设置 `MODEL_NAME`，训练脚本会自动选择对应配置；仅切换到 LIBERO、AgiBot 或 YAM 时需要按示例覆盖。

默认配置：

| 场景 | 分布式策略 | 每卡 batch | 默认 global batch | 默认 LR |
| --- | --- | --- | --- | --- |
| 5B Full | 8 卡 FSDP + Delta-FP8 | 1 | 8 | `1e-5` |
| 5B LoRA | 8 卡 FSDP | 1 | 8 | `1e-5` |
| 14B Full | 8 卡 FSDP | 1 | 8 | `1e-5` |
| 14B DROID LoRA | 8 卡 FSDP | 1 | 8 | `1e-4` |
| 14B AgiBot / YAM LoRA | 8 卡 FSDP | 1 | 8 | `1e-5` |

可通过 `PER_DEVICE_BATCH_SIZE` 和 `GLOBAL_BATCH_SIZE` 调整批量大小，但需满足 `GLOBAL_BATCH_SIZE % (GPUS_PER_NODE * NNODES * PER_DEVICE_BATCH_SIZE) == 0`。机器不足 8 卡时可设置 `GPUS_PER_NODE`，默认 global batch size 会随之变化。

### 2.2 续训
```bash
TRAIN_ITERS=20000 SAVE_INTERVAL=1000 \
OUTPUT_DIR=/workspace/outputs/dreamzero/droid_wan22_5b_lora \
    bash examples/embodied/dreamzero/run_dreamzero_wan22_5b_lora_fsdp_finetune.sh \
    --resume
```

`--resume` 会从 `$OUTPUT_DIR/checkpoints` 加载最新 checkpoint。续训时须保持与保存时相同的 world size 和 batch 配置；未找到 checkpoint 时会直接报错。

### 2.3 离线 Feature Cache（可选）
```bash
# 生成完整 cache（5B 默认 video_latents + prompt_embs；14B 默认不含 prompt_embs）
MODEL_NAME=dreamzero_full_wan22_5b DATA_PATH="$DROID_DATA_ROOT" \
CACHE_OUTPUT_DIR=$DREAMZERO_CACHE_ROOT/dreamzero_full_wan22_5b \
GPUS_PER_NODE=8 SAMPLE_TRANSFORM_SEED=0 VALIDATION_REQUIRE_FULL_COVERAGE=1 \
    bash examples/embodied/dreamzero/precompute_dreamzero_cache.sh

# 使用 cache 训练（须与生成时 SAMPLE_TRANSFORM_SEED、分辨率、语言 chunk 配置一致）
CACHE_DIR=$DREAMZERO_CACHE_ROOT/dreamzero_full_wan22_5b SAMPLE_TRANSFORM_SEED=0 \
    bash examples/embodied/dreamzero/run_dreamzero_wan22_5b_full_fsdp_finetune.sh
```

训练脚本会在启动阶段严格校验 cache。manifest、`_SUCCESS` 或 transform 配置不匹配时，训练会在执行首步前终止。

### 2.4 性能优化开关
DROID 5B / 14B 配置已启用经过验证的默认性能优化。5B Full FSDP recipe 在支持的 CUDA/NCCL 设备上默认开启 Delta-FP8 AllGather；可传入 `--no-fsdp-delta-fp8-allgather` 与原生 BF16 通信做 A/B 对比。LIBERO、AgiBot 和 YAM 默认采用保守配置；调整性能开关后，建议先运行 3～10 步，检查 loss、grad norm 和显存占用。

对照组（关闭全部内核优化，用于定位 loss / 性能问题）：

```bash
# Wan2.2 5B
bash examples/embodied/dreamzero/run_dreamzero_wan22_5b_full_fsdp_finetune.sh \
    model.flash_attention_dense=false model.compile_causal_attention_block=false \
    model.batch_vae_encode=false 'model.prompt_emb_cache=""'

# Wan2.1 14B
bash examples/embodied/dreamzero/run_dreamzero_wan21_14b_full_fsdp_finetune.sh \
    model.skip_single_state_attention=false model.compile_causal_cross_attention=false \
    model.compile_cross_attention_emulate_precision_casts=false model.compile_block_norm_modulate=false \
    model.qk_rmsnorm_impl=wan model.manual_self_attn_linear_backward=false model.fused_rope=false \
    'model.prompt_emb_cache=""' model.cache_fa_lens=false model.cache_fa_lens_clone=false
```

空字符串须写成带引号的 `'model.prompt_emb_cache=""'`。

主要开关：

| 开关 | 默认值 | 作用 |
| --- | --- | --- |
| `model.flash_attention_dense` | `true` | dense attention 直接用 FlashAttention |
| `model.compile_causal_attention_block` | `true` | 编译 causal attention block |
| `model.batch_vae_encode` | `true` | 批量 VAE 编码，吞吐更高但显存峰值更大，OOM 时关闭 |
| `model.prompt_emb_cache` | `gpu` | 缓存冻结 Text Encoder 输出，可设 `cpu` 或空字符串关闭 |
| `model.compile_causal_cross_attention`（14B） | `true` | `torch.compile` 编译 causal cross-attention |
| `model.fused_rope`（14B） | `true` | 融合 RoPE kernel |
| `model.cache_fa_lens`（14B） | `true` | 缓存 FlashAttention cumulative sequence lengths |

### 2.5 正确性验证

为保证训练精度不受优化手段影响，我们在相同数据、权重和训练配置下，对 LoongForge 适配的 DreamZero 与官方实现进行了逐 step 的 loss 对比验证。结果表明，LoongForge 的各项性能优化对训练精度无损：

![DreamZero 训练流程概览](../../assets/images/precision/dreamzero.png)
