# 快速入门：GR00T-N1.7 模型训练

本文档以 LIBERO 四个 suite 为例，介绍如何在 LoongForge 框架下微调 **GR00T-N1.7**，并对得到的 checkpoint 进行评测。

- [1. 训练准备](#1-训练准备)
- [2. 微调训练](#2-微调训练)
- [3. 评测准备](#3-评测准备)
- [4. 评测](#4-评测)

---

## 1. 训练准备

### 1.1 环境

参照 LoongForge 安装指南，拉取最新 GPU 镜像并启动容器：

```bash
docker pull loongforge/loongforge:latest
mkdir -p workspace
docker run --gpus all --ipc=host -it --rm \
  -v "$(pwd)/workspace:/workspace/data" \
  -w /workspace/LoongForge \
  loongforge/loongforge:latest bash
```

镜像开箱即可训练 GR00T，无需额外安装依赖。

### 1.2 预训练权重

从 Hugging Face 下载以下权重：

| 权重 | 用途 |
|---|---|
| [GR00T-N1.7-3B](https://huggingface.co/nvidia/GR00T-N1.7-3B) | 微调实际使用的预训练权重 |
| [Cosmos-Reason2-2B](https://huggingface.co/nvidia/Cosmos-Reason2-2B) | 仅提供 VLM backbone 结构及 tokenizer / processor |

> **注意**：Cosmos-Reason2-2B 的权重在加载后会被 GR00T-N1.7-3B 的权重覆盖。

### 1.3 训练数据

脚本默认使用官方预处理好的 LeRobot v2.1 格式数据集。如需使用自定义数据训练，请参照 [NVIDIA Isaac-GR00T](https://github.com/NVIDIA/Isaac-GR00T) 完成离线数据预处理，将原始数据转换为 LeRobot v2.1 格式后再用于训练。

LIBERO 包含四个 suite，数据集托管在 Hugging Face 的 IPEC-COMMUNITY 组织下：

| Suite | 侧重点 | 数据集 |
|---|---|---|
| Spatial | 空间推理 | [libero_spatial_no_noops_1.0.0_lerobot](https://huggingface.co/datasets/IPEC-COMMUNITY/libero_spatial_no_noops_1.0.0_lerobot) |
| Object | 物体泛化 | [libero_object_no_noops_1.0.0_lerobot](https://huggingface.co/datasets/IPEC-COMMUNITY/libero_object_no_noops_1.0.0_lerobot) |
| Goal | 目标条件学习 | [libero_goal_no_noops_1.0.0_lerobot](https://huggingface.co/datasets/IPEC-COMMUNITY/libero_goal_no_noops_1.0.0_lerobot) |
| 10 (Long) | 长程多步任务 | [libero_10_no_noops_1.0.0_lerobot](https://huggingface.co/datasets/IPEC-COMMUNITY/libero_10_no_noops_1.0.0_lerobot) |

每个数据集的目录结构如下：

```text
my_dataset/
├── meta/
│   ├── info.json          # 数据集元信息
│   ├── episodes.jsonl     # episode 索引与长度
│   ├── tasks.jsonl        # 语言任务描述
│   ├── stats.json         # 归一化统计量（须包含 q01/q99，由下文脚本生成）
│   └── modality.json      # state/action/video 字段映射（GR00T 专用）
├── data/chunk-000/        # parquet 文件（每个时间步的 state、action）
└── videos/chunk-000/      # 每个 episode 的 mp4 视频
```

下载完成后，在 LoongForge 根目录下执行以下步骤（将 `{suite}` 替换为各 suite 的数据集目录）：

```bash
# 1. 覆盖 modality.json（四个 suite 都需要）
cp examples/embodied/gr00t_n1_7/LIBERO/dataset/modality.json \
   /path/to/IPEC-COMMUNITY/{suite}/meta/modality.json

# 2. 为四个 suite 生成 meta/stats.json（含 q01/q99 的归一化统计量），
#    脚本会原地覆盖各数据集的 meta/stats.json
python examples/embodied/gr00t_n1_7/gen_stats.py \
   /path/to/IPEC-COMMUNITY/libero_spatial_no_noops_1.0.0_lerobot \
   /path/to/IPEC-COMMUNITY/libero_object_no_noops_1.0.0_lerobot \
   /path/to/IPEC-COMMUNITY/libero_goal_no_noops_1.0.0_lerobot \
   /path/to/IPEC-COMMUNITY/libero_10_no_noops_1.0.0_lerobot

# 3. 仅 Goal suite：替换一个损坏的视频
cp examples/embodied/gr00t_n1_7/LIBERO/dataset/Goal/patches/episode_000082.mp4 \
   /path/to/IPEC-COMMUNITY/libero_goal_no_noops_1.0.0_lerobot/videos/chunk-000/observation.images.wrist_image/
```

---

## 2. 微调训练

### 2.1 启动训练

训练脚本：`examples/embodied/gr00t_n1_7/LIBERO/run_libero_sft.sh`

在脚本中设置 LoongForge、预训练权重和数据集路径（也可以通过环境变量传入），然后启动训练：

> **注意**：
> - `data.embodiment_tag=libero_sim` 用于告诉 GR00T-N1.7 数据来自 LIBERO 仿真器中的 Panda 机械臂，**请勿修改**。
> - 如果 CUDA Graph 重放失败，训练会自动回退到 eager 模式并正常进行。

```bash
cd examples/embodied/gr00t_n1_7/LIBERO

# 微调单个 suite，例如 Spatial
LOONGFORGE_PATH=/path/to/LoongForge \
CHECKPOINT_PATH=/path/to/GR00T-N1.7-3B \
COSMOS_LOCAL_PATH=/path/to/Cosmos-Reason2-2B \
DATASET_ROOT=/path/to/IPEC-COMMUNITY \
OUTPUT_ROOT=/path/to/outputs/gr00t_n1_7_libero \
bash run_libero_sft.sh spatial

# 依次微调全部 suite（spatial、object、goal、10）
bash run_libero_sft.sh all

# 冒烟测试：只跑 20 步验证训练链路
TRAIN_ITERS=20 SAVE_INTERVAL=20 bash run_libero_sft.sh goal
```

每个 suite 单独训练，checkpoint 保存在 `$OUTPUT_ROOT/libero_<suite>/checkpoints/steps_<N>`，训练日志保存在 `$OUTPUT_ROOT/libero_<suite>.log`。

### 2.2 关键参数

**脚本参数**（均可通过环境变量覆盖）：

| 参数 | 默认值 | 说明 |
|---|:---:|---|
| `LOONGFORGE_PATH` | - | LoongForge 仓库根目录 |
| `CHECKPOINT_PATH` | - | GR00T-N1.7-3B 预训练权重目录（须包含 `config.json`） |
| `COSMOS_LOCAL_PATH` | - | Cosmos-Reason2-2B 目录，同时作为 `TOKENIZER_PATH` 的默认值 |
| `DATASET_ROOT` | - | 存放各 suite 数据集 `libero_<suite>_no_noops_1.0.0_lerobot/` 的根目录 |
| `OUTPUT_ROOT` | - | 输出根目录，每个 suite 输出到 `libero_<suite>/` 子目录 |
| `GPUS_PER_NODE` / `NNODES` | 8 / 1 | 每节点 GPU 数 / 节点数 |
| `TRAIN_ITERS` | 20000 | 总训练步数 |
| `GLOBAL_BATCH_SIZE` | 640 | 全局 batch size，须能被 GPU 数 × 梯度累积步数整除 |
| `GRADIENT_ACCUMULATION_STEPS` | 1 | 梯度累积步数，每卡 batch size 由前两项自动计算 |
| `SAVE_INTERVAL` | 1000 | checkpoint 保存间隔（步） |
| `LR_WARMUP_ITERS` | `TRAIN_ITERS / 20` | 学习率 warmup 步数（5%），之后按 cosine 衰减到 0，峰值学习率 1e-4 |

**模型与数据配置**：模型和数据相关参数在 `configs/models/embodied/gr00t_n1_7.yaml` 中配置，常用字段如下：

| 字段 | 默认值 | 说明 |
|---|:---:|---|
| `data.embodiment_tag` | `new_embodiment` | 本体标签，决定使用的 projector 和 modality 布局；LIBERO 脚本中设置为 `libero_sim` |
| `data.preprocess_action_horizon` | 16 | 每个样本的 action chunk 长度 |
| `data.state_dropout_prob` / `model.state_dropout_prob` | 0.2 | 训练时丢弃 state 输入的概率 |
| `data.use_relative_action` | true | 是否使用相对动作 |
| `data.use_percentiles` / `data.clip_outliers` | true / true | 使用 `stats.json` 中的 q01/q99 进行归一化并截断离群值 |
| `model.tune_llm` / `model.tune_visual` | false / false | 是否训练 LLM / 视觉编码器 |
| `model.tune_projector` / `model.tune_diffusion_model` | true / true | 是否训练 projector / action head（DiT） |

推荐在脚本的 `OVERRIDES` 中以命令行参数覆盖这些字段，无需修改配置文件。例如：

```bash
OVERRIDES=(
    data.embodiment_tag=libero_sim
    model.state_dropout_prob=0.5
    data.state_dropout_prob=0.5
    data.preprocess_action_horizon=8
)
```

### 2.3 训练加速

LIBERO 脚本默认开启 full-iteration CUDA Graph：前几步以 eager 模式 warmup，之后把一整步训练（forward、backward、DDP 梯度同步、优化器更新）捕获为一张 CUDA Graph，后续每步直接重放，消除 host 侧调度和 kernel launch 开销。相关参数位于脚本的 `DISTRIBUTED_TRAINING_ARGS` 中：

| 参数 | 脚本取值 | 说明 |
|---|:---:|---|
| `--cuda-graph-impl` | `local` | CUDA Graph 后端，`none` 表示关闭 |
| `--cuda-graph-scope` | `full_iteration` | 捕获范围；GR00T-N1.7 仅支持 `full_iteration`，即整步捕获 |
| `--cuda-graph-warmup-steps` | 3 | 捕获前以 eager 模式运行的 warmup 步数 |
| `--cuda-graph-pad-length` | 0 | 文本 token 的固定 padding 长度，0 表示不强制固定长度 |
| `--cuda-graph-ddp-sync-in-graph` | 开启 | 将 DDP 梯度 all-reduce 一并捕获进 graph，full-iteration 模式下必须开启 |
| `--ddp-static-graph` | 开启 | 声明各步计算图不变，启用 DDP 静态图优化，配合 CUDA Graph 使用 |
| `--no-ddp-find-unused-parameters` | 开启 | 跳过未使用参数的检测，减少每步开销 |
| `--ddp-bucket-cap-mb` | 100 | DDP 梯度 all-reduce 的 bucket 大小（MiB） |
| `--no-check-for-nan-in-loss-and-grad` | 开启 | 关闭每步 host 侧 loss/梯度 NaN 检查，避免 device→host 同步打断 graph 重放 |

数据加载侧的相关参数（`DATA_ARGS`）：

| 参数 | 脚本取值 | 说明 |
|---|:---:|---|
| `--num-workers` | 4 | 每卡 DataLoader worker 数，可通过 `NUM_WORKERS` 环境变量调整 |
| `--dataloader-prefetch-factor` | 8 | 每个 worker 预取的 batch 数（框架默认 2） |
| `--dataloader-multiprocessing-context` | `fork` | DataLoader worker 的进程启动方式 |
| `--distributed-sampler-mode` | `block` | 按连续分片（而非轮转）为各 rank 划分样本 |

**融合算子**：GR00T-N1.7 依赖一组自定义 CUDA 融合算子 `gr00t_n1_7_op`（源码位于 `ops/cuda_source/gr00t_n1_7_op`），用于减少 kernel 数量和显存读写：

| 算子 | 说明 |
|---|---|
| Qwen3-VL 融合算子 | 视觉 / 文本 RoPE、RMSNorm、SiLU-multiply 融合 kernel，用于 backbone 前向 |
| Fused AdamW | 基于 `MultiTensorApply` 的融合 AdamW，与 PyTorch AdamW 数值对齐；提供 capturable 版本，可被 CUDA Graph 捕获。仅在 `--optimizer TEFusedAdamW` 时使用，LIBERO 脚本默认的 `AdamW` 不会用到 |
| DDP reducer bucket control | 控制 DDP `c10d::Reducer` 的梯度 bucket 初始化，配合 graph 内梯度同步使用 |

训练脚本启动时会检查 `gr00t_n1_7_op` 是否可导入，未安装时自动从源码编译安装（仅首次需要，耗时数分钟）。也可以提前手动安装：

```bash
cd ops/cuda_source/gr00t_n1_7_op
pip install --no-build-isolation -e .
```

> **注意**：
> - 默认只编译 sm_80 和 sm_120 架构。其他 GPU 需通过 `TORCH_CUDA_ARCH_LIST` 指定，例如 `TORCH_CUDA_ARCH_LIST="9.0" pip install --no-build-isolation -e .`。
> - 编译并行度由 `NVCC_THREADS`（默认取 `MAX_JOBS`，再默认 8）控制。
> - DDP reducer bucket control 依赖 PyTorch 内部 ABI，须与运行时的 PyTorch 版本一同编译；升级 PyTorch 后需重新安装。

开启 CUDA Graph 后，训练日志末尾会打印 `Full-iteration CUDA graph summary`，其中 `replays` 为重放步数、`eager_fallbacks` 为回退到 eager 的次数，可用于确认加速是否生效。

### 2.4 Loss 曲线

| libero_spatial | libero_object |
|:---:|:---:|
| ![libero_spatial loss](../../assets/images/gr00t_n1_7/loss_libero_spatial.png) | ![libero_object loss](../../assets/images/gr00t_n1_7/loss_libero_object.png) |
| **libero_goal** | **libero_10** |
| ![libero_goal loss](../../assets/images/gr00t_n1_7/loss_libero_goal.png) | ![libero_10 loss](../../assets/images/gr00t_n1_7/loss_libero_10.png) |

### 2.5 正确性验证

为保证训练精度不受优化手段影响，我们在相同数据、权重和训练配置下，对 LoongForge 适配的 GR00T-N1.7 与官方实现进行了逐 step 的 action loss 对比验证。结果表明，LoongForge 的各项性能优化（如 TEFusedAdamW、CUDA Graph、DDP static graph 等）对训练精度无损：

![GR00T-N1.7 precision](../../assets/images/precision/GR00TN1.7.png)

---

## 3. 评测准备

参照官方 [LIBERO](https://github.com/Lifelong-Robot-Learning/LIBERO) 仓库安装基础 LIBERO 环境，再安装评测 client 所需的额外依赖：

```bash
pip install websockets msgpack pyyaml
```

也可以参照 [Isaac-GR00T](https://github.com/NVIDIA/Isaac-GR00T) 搭建评测环境：

```bash
git clone https://github.com/NVIDIA/Isaac-GR00T.git
cd Isaac-GR00T
bash gr00t/eval/sim/LIBERO/setup_libero.sh
pip install websockets
```

评测采用 server / client 架构：LoongForge policy server 加载 checkpoint 进行推理，LIBERO 仿真器作为 client 与其交互。两者可以使用不同的 Python 环境，分别由下文的 `SERVER_PY` 和 `BENCHMARK_PYTHON` 指定。

---

## 4. 评测

### 4.1 启动评测

`examples/embodied/gr00t_n1_7/eval/run_libero_eval.sh` 会依次评测四个 suite，每个 suite 200 个 episode（10 个任务 × 20 个 episode）。脚本会自动启动和关闭 policy server，无需手动部署：

```bash
cd examples/embodied/gr00t_n1_7/eval

COSMOS_PATH=/path/to/Cosmos-Reason2-2B \
CKPT_PATH=/path/to/outputs/gr00t_n1_7_libero/libero_{suite}/checkpoints/steps_20000 \
STATS_PATH=/path/to/IPEC-COMMUNITY/libero_{suite}_no_noops_1.0.0_lerobot/meta/stats.json \
LOONGFORGE_PATH=/path/to/LoongForge \
OUT_ROOT=/path/to/outputs \
SERVER_PY=/path/to/policy/bin/python \
BENCHMARK_PYTHON=/path/to/libero/bin/python \
GPU=0 \
bash run_libero_eval.sh

# 冒烟测试：只评测 libero_10 的 1 个任务 × 2 个 episode
SUITES=libero_10 EPISODES=2 MAX_TASKS=1 bash run_libero_eval.sh
```

### 4.2 关键参数

| 参数 | 默认值 | 说明 |
|---|:---:|---|
| `COSMOS_PATH` | - | 本地 Cosmos-Reason2-2B 目录 |
| `CKPT_PATH` | - | 微调得到的 checkpoint 路径，`{suite}` 会依次替换为 `spatial` / `object` / `goal` / `10` |
| `STATS_PATH` | - | 对应 suite 的统计量文件（`stats.json`），同样支持 `{suite}` |
| `LOONGFORGE_PATH` | - | LoongForge 仓库根目录 |
| `OUT_ROOT` | - | 评测输出根目录 |
| `SERVER_PY` | `/usr/bin/python` | LoongForge policy server 使用的 Python 解释器 |
| `BENCHMARK_PYTHON` | - | LIBERO 仿真器使用的 Python 解释器（须已安装 LIBERO） |
| `GPU` | 0 | 使用的 GPU 编号 |
| `SUITES` | 全部四个 suite | 要评测的 suite，以空格分隔，例如 `"libero_spatial libero_goal"` |
| `EPISODES` | 20 | 每个任务评测的 episode 数；10 × 20 = 200 与官方口径一致 |
| `MAX_TASKS` | 0 | 每个 suite 评测的任务数，0 表示全部 10 个任务 |
| `SEED` | 42 | 随机种子 |
| `MUJOCO_GL` | `egl` | MuJoCo 渲染后端，可按环境改为 `osmesa` |

评测协议（每个 episode 最多 720 步、action chunk 为 8 等）固定在 `eval/configs/libero/libero_<suite>.yaml` 中，路径类参数由脚本在运行时注入，无需修改 YAML。

### 4.3 评测结果

四个 suite 的成功率（每个 suite 10 个任务 × 20 个 episode = 200 个 episode）：

| Suite | 成功 episode 数 | LoongForge | 官方 N1.7 |
|---|:---:|:---:|:---:|
| libero_spatial | 196 / 200 | 98.0% | 97.65% |
| libero_object | 197 / 200 | 98.5% | 98.45% |
| libero_goal | 193 / 200 | 96.5% | 97.5% |
| libero_10 (Long) | 191 / 200 | 95.5% | 94.35% |
| **平均** | **777 / 800** | **97.13%** | **96.99%** |

> **注意**：
> - “官方 N1.7” 一列取自 Isaac-GR00T 的 `examples/LIBERO/README.md`。
> - 评测存在随机性，每次运行的结果会略有差异。

### 4.4 Rollout 示例

| libero_spatial | libero_object |
|:---:|:---:|
| ![libero_spatial](../../assets/images/gr00t_n1_7/replay_libero_spatial.gif) | ![libero_object](../../assets/images/gr00t_n1_7/replay_libero_object.gif) |
| *pick up the black bowl on the wooden cabinet and place it on the plate* | *pick up the orange juice and place it in the basket* |
| **libero_goal** | **libero_10** |
| ![libero_goal](../../assets/images/gr00t_n1_7/replay_libero_goal.gif) | ![libero_10](../../assets/images/gr00t_n1_7/replay_libero_10.gif) |
| *put the bowl on the stove* | *put both the cream cheese box and the butter in the basket* |