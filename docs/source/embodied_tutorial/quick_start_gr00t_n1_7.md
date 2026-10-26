# Quick Start: GR00T-N1.7 Training

Using the four LIBERO suites as an example, this guide shows how to finetune **GR00T-N1.7** with LoongForge and evaluate the resulting checkpoints.

- [1. Training Setup](#1-training-setup)
- [2. Finetuning](#2-finetuning)
- [3. Evaluation Setup](#3-evaluation-setup)
- [4. Evaluation](#4-evaluation)

---

## 1. Training Setup

### 1.1 Environment

Following the LoongForge installation guide, pull the latest GPU image and start a container:

```bash
docker pull loongforge/loongforge:latest
mkdir -p workspace
docker run --gpus all --ipc=host -it --rm \
  -v "$(pwd)/workspace:/workspace/data" \
  -w /workspace/LoongForge \
  loongforge/loongforge:latest bash
```

The image can train GR00T out of the box; no extra dependencies are required.

### 1.2 Pretrained Weights

Download the following weights from Hugging Face:

| Weights | Purpose |
|---|---|
| [GR00T-N1.7-3B](https://huggingface.co/nvidia/GR00T-N1.7-3B) | The pretrained weights actually used for finetuning |
| [Cosmos-Reason2-2B](https://huggingface.co/nvidia/Cosmos-Reason2-2B) | Only provides the VLM backbone architecture and the tokenizer / processor |

> **Note**: the Cosmos-Reason2-2B weights are overwritten by the GR00T-N1.7-3B weights after loading.

### 1.3 Training Data

The script uses the officially preprocessed datasets in LeRobot v2.1 format by default. To train on custom data, follow [NVIDIA Isaac-GR00T](https://github.com/NVIDIA/Isaac-GR00T) to preprocess it offline and convert the raw data into LeRobot v2.1 format first.

LIBERO has four suites. The datasets are hosted under the IPEC-COMMUNITY organization on Hugging Face:

| Suite | Focus | Dataset |
|---|---|---|
| Spatial | Spatial reasoning | [libero_spatial_no_noops_1.0.0_lerobot](https://huggingface.co/datasets/IPEC-COMMUNITY/libero_spatial_no_noops_1.0.0_lerobot) |
| Object | Object generalization | [libero_object_no_noops_1.0.0_lerobot](https://huggingface.co/datasets/IPEC-COMMUNITY/libero_object_no_noops_1.0.0_lerobot) |
| Goal | Goal-conditioned learning | [libero_goal_no_noops_1.0.0_lerobot](https://huggingface.co/datasets/IPEC-COMMUNITY/libero_goal_no_noops_1.0.0_lerobot) |
| 10 (Long) | Long-horizon multi-step tasks | [libero_10_no_noops_1.0.0_lerobot](https://huggingface.co/datasets/IPEC-COMMUNITY/libero_10_no_noops_1.0.0_lerobot) |

Each dataset has the following layout:

```text
my_dataset/
├── meta/
│   ├── info.json          # dataset metadata
│   ├── episodes.jsonl     # episode index and lengths
│   ├── tasks.jsonl        # language task descriptions
│   ├── stats.json         # normalization stats (must contain q01/q99; generated below)
│   └── modality.json      # state/action/video key mapping (GR00T-specific)
├── data/chunk-000/        # parquet files (state, action per timestep)
└── videos/chunk-000/      # mp4 video files per episode
```

After downloading, run the following steps from the LoongForge root (replace `{suite}` with the dataset directory of each suite):

```bash
# 1. Overwrite modality.json (all four suites)
cp examples/embodied/gr00t_n1_7/LIBERO/dataset/modality.json \
   /path/to/IPEC-COMMUNITY/{suite}/meta/modality.json

# 2. Generate meta/stats.json (normalization stats with q01/q99) for all four suites;
#    the script overwrites each dataset's meta/stats.json in place
python examples/embodied/gr00t_n1_7/gen_stats.py \
   /path/to/IPEC-COMMUNITY/libero_spatial_no_noops_1.0.0_lerobot \
   /path/to/IPEC-COMMUNITY/libero_object_no_noops_1.0.0_lerobot \
   /path/to/IPEC-COMMUNITY/libero_goal_no_noops_1.0.0_lerobot \
   /path/to/IPEC-COMMUNITY/libero_10_no_noops_1.0.0_lerobot

# 3. Goal suite only: replace one corrupted video
cp examples/embodied/gr00t_n1_7/LIBERO/dataset/Goal/patches/episode_000082.mp4 \
   /path/to/IPEC-COMMUNITY/libero_goal_no_noops_1.0.0_lerobot/videos/chunk-000/observation.images.wrist_image/
```

---

## 2. Finetuning

### 2.1 Launch Training

Training script: `examples/embodied/gr00t_n1_7/LIBERO/run_libero_sft.sh`

Set the LoongForge, pretrained-weight and dataset paths in the script (or pass them as environment variables), then start training:

> **Notes**:
> - `data.embodiment_tag=libero_sim` tells GR00T-N1.7 that the data comes from the Panda arm in the LIBERO simulator. **Do not change it.**
> - If a CUDA Graph replay fails, training automatically falls back to eager mode and continues normally.

```bash
cd examples/embodied/gr00t_n1_7/LIBERO

# Finetune a single suite, e.g. Spatial
LOONGFORGE_PATH=/path/to/LoongForge \
CHECKPOINT_PATH=/path/to/GR00T-N1.7-3B \
COSMOS_LOCAL_PATH=/path/to/Cosmos-Reason2-2B \
DATASET_ROOT=/path/to/IPEC-COMMUNITY \
OUTPUT_ROOT=/path/to/outputs/gr00t_n1_7_libero \
bash run_libero_sft.sh spatial

# Finetune all suites in turn (spatial, object, goal, 10)
bash run_libero_sft.sh all

# Smoke test: run only 20 iterations to validate the training pipeline
TRAIN_ITERS=20 SAVE_INTERVAL=20 bash run_libero_sft.sh goal
```

Each suite is trained separately. Checkpoints are saved to `$OUTPUT_ROOT/libero_<suite>/checkpoints/steps_<N>` and training logs to `$OUTPUT_ROOT/libero_<suite>.log`.

### 2.2 Key Parameters

**Script parameters** (all can be overridden via environment variables):

| Parameter | Default | Description |
|---|:---:|---|
| `LOONGFORGE_PATH` | - | LoongForge repository root |
| `CHECKPOINT_PATH` | - | GR00T-N1.7-3B pretrained weight directory (must contain `config.json`) |
| `COSMOS_LOCAL_PATH` | - | Cosmos-Reason2-2B directory; also the default value of `TOKENIZER_PATH` |
| `DATASET_ROOT` | - | Root directory holding each suite's `libero_<suite>_no_noops_1.0.0_lerobot/` dataset |
| `OUTPUT_ROOT` | - | Output root; each suite writes to its own `libero_<suite>/` subdirectory |
| `GPUS_PER_NODE` / `NNODES` | 8 / 1 | GPUs per node / number of nodes |
| `TRAIN_ITERS` | 20000 | Total training iterations |
| `GLOBAL_BATCH_SIZE` | 640 | Global batch size; must be divisible by number of GPUs × gradient accumulation steps |
| `GRADIENT_ACCUMULATION_STEPS` | 1 | Gradient accumulation steps; the per-device batch size is derived from the two values above |
| `SAVE_INTERVAL` | 1000 | Checkpoint save interval (iterations) |
| `LR_WARMUP_ITERS` | `TRAIN_ITERS / 20` | Learning-rate warmup iterations (5%), followed by cosine decay to 0; peak learning rate is 1e-4 |

**Model and data configuration**: model and data parameters are configured in `configs/models/embodied/gr00t_n1_7.yaml`. Commonly used fields:

| Field | Default | Description |
|---|:---:|---|
| `data.embodiment_tag` | `new_embodiment` | Embodiment tag that selects the projector and modality layout; set to `libero_sim` in the LIBERO script |
| `data.preprocess_action_horizon` | 16 | Action chunk length of each sample |
| `data.state_dropout_prob` / `model.state_dropout_prob` | 0.2 | Probability of dropping the state input during training |
| `data.use_relative_action` | true | Whether to use relative actions |
| `data.use_percentiles` / `data.clip_outliers` | true / true | Normalize with q01/q99 from `stats.json` and clip outliers |
| `model.tune_llm` / `model.tune_visual` | false / false | Whether to train the LLM / vision encoder |
| `model.tune_projector` / `model.tune_diffusion_model` | true / true | Whether to train the projector / action head (DiT) |

We recommend overriding these fields as command-line arguments in the script's `OVERRIDES` instead of editing the config file. For example:

```bash
OVERRIDES=(
    data.embodiment_tag=libero_sim
    model.state_dropout_prob=0.5
    data.state_dropout_prob=0.5
    data.preprocess_action_horizon=8
)
```

### 2.3 Training Acceleration

The LIBERO script enables full-iteration CUDA Graph by default: the first few iterations run eagerly as warmup, then one whole training step (forward, backward, DDP gradient sync and optimizer update) is captured into a single CUDA Graph and replayed on every subsequent step, eliminating host-side scheduling and kernel-launch overhead. The related arguments live in the script's `DISTRIBUTED_TRAINING_ARGS`:

| Argument | Script value | Description |
|---|:---:|---|
| `--cuda-graph-impl` | `local` | CUDA Graph backend; `none` disables it |
| `--cuda-graph-scope` | `full_iteration` | Capture scope; GR00T-N1.7 only supports `full_iteration`, i.e. capturing the whole step |
| `--cuda-graph-warmup-steps` | 3 | Number of eager warmup iterations before capture |
| `--cuda-graph-pad-length` | 0 | Fixed padding length for text tokens; 0 means no fixed length is enforced |
| `--cuda-graph-ddp-sync-in-graph` | on | Captures the DDP gradient all-reduce inside the graph; required in full-iteration mode |
| `--ddp-static-graph` | on | Declares the graph static across iterations to enable DDP static-graph optimizations; used together with CUDA Graph |
| `--no-ddp-find-unused-parameters` | on | Skips unused-parameter detection to reduce per-step overhead |
| `--ddp-bucket-cap-mb` | 100 | Bucket size (MiB) of the DDP gradient all-reduce |
| `--no-check-for-nan-in-loss-and-grad` | on | Disables the per-step host-side NaN check on loss/gradients, avoiding device-to-host syncs that would interrupt graph replay |

Data-loading arguments (`DATA_ARGS`):

| Argument | Script value | Description |
|---|:---:|---|
| `--num-workers` | 4 | DataLoader workers per device; adjustable via the `NUM_WORKERS` environment variable |
| `--dataloader-prefetch-factor` | 8 | Batches prefetched per worker (framework default: 2) |
| `--dataloader-multiprocessing-context` | `fork` | Start method of DataLoader worker processes |
| `--distributed-sampler-mode` | `block` | Partitions samples across ranks in contiguous shards (instead of round-robin) |

**Fused operators**: GR00T-N1.7 depends on a set of custom fused CUDA operators, `gr00t_n1_7_op` (sources under `ops/cuda_source/gr00t_n1_7_op`), which reduce kernel count and memory traffic:

| Operator | Description |
|---|---|
| Qwen3-VL fused ops | Fused vision / text RoPE, RMSNorm and SiLU-multiply kernels used in the backbone forward pass |
| Fused AdamW | Fused AdamW built on `MultiTensorApply`, numerically aligned with PyTorch AdamW; provides a capturable variant that can be captured by CUDA Graph. Only used with `--optimizer TEFusedAdamW`; the LIBERO script's default `AdamW` does not use it |
| DDP reducer bucket control | Controls gradient bucket initialization of the DDP `c10d::Reducer`, used together with in-graph gradient sync |

On startup, the training script checks whether `gr00t_n1_7_op` can be imported and, if not, builds and installs it from source automatically (only needed once; takes a few minutes). You can also install it manually beforehand:

```bash
cd ops/cuda_source/gr00t_n1_7_op
pip install --no-build-isolation -e .
```

> **Notes**:
> - Only the sm_80 and sm_120 architectures are compiled by default. For other GPUs, specify the architecture via `TORCH_CUDA_ARCH_LIST`, e.g. `TORCH_CUDA_ARCH_LIST="9.0" pip install --no-build-isolation -e .`.
> - Build parallelism is controlled by `NVCC_THREADS` (defaults to `MAX_JOBS`, then to 8).
> - DDP reducer bucket control depends on PyTorch's internal ABI and must be built against the runtime PyTorch version; reinstall it after upgrading PyTorch.

With CUDA Graph enabled, the end of the training log prints a `Full-iteration CUDA graph summary`, where `replays` is the number of replayed steps and `eager_fallbacks` the number of fallbacks to eager mode; use it to confirm that acceleration is in effect.

### 2.4 Loss Curves

| libero_spatial | libero_object |
|:---:|:---:|
| ![libero_spatial loss](../../assets/images/gr00t_n1_7/loss_libero_spatial.png) | ![libero_object loss](../../assets/images/gr00t_n1_7/loss_libero_object.png) |
| **libero_goal** | **libero_10** |
| ![libero_goal loss](../../assets/images/gr00t_n1_7/loss_libero_goal.png) | ![libero_10 loss](../../assets/images/gr00t_n1_7/loss_libero_10.png) |

### 2.5 Correctness Verification

To ensure training accuracy is not affected by the optimizations, we compared the step-by-step action loss of LoongForge's GR00T-N1.7 implementation against the official one, using identical data, weights and training configuration. The results show that LoongForge's performance optimizations (TEFusedAdamW, CUDA Graph, DDP static graph, etc.) preserve training accuracy:

![GR00T-N1.7 precision](../../assets/images/precision/GR00TN1.7.png)

---

## 3. Evaluation Setup

Install the base LIBERO environment following the official [LIBERO](https://github.com/Lifelong-Robot-Learning/LIBERO) repository, then install the extra dependencies the eval client needs:

```bash
pip install websockets msgpack pyyaml
```

Alternatively, build the evaluation environment following [Isaac-GR00T](https://github.com/NVIDIA/Isaac-GR00T):

```bash
git clone https://github.com/NVIDIA/Isaac-GR00T.git
cd Isaac-GR00T
bash gr00t/eval/sim/LIBERO/setup_libero.sh
pip install websockets
```

Evaluation uses a server / client architecture: the LoongForge policy server loads the checkpoint and runs inference, while the LIBERO simulator acts as the client. The two can use different Python environments, specified by `SERVER_PY` and `BENCHMARK_PYTHON` below.

---

## 4. Evaluation

### 4.1 Launch

`examples/embodied/gr00t_n1_7/eval/run_libero_eval.sh` evaluates the four suites serially, 200 episodes per suite (10 tasks × 20 episodes). The script starts and stops the policy server itself; no manual deployment is needed:

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

# Smoke test: evaluate only 1 task × 2 episodes of libero_10
SUITES=libero_10 EPISODES=2 MAX_TASKS=1 bash run_libero_eval.sh
```

### 4.2 Key Parameters

| Parameter | Default | Description |
|---|:---:|---|
| `COSMOS_PATH` | - | Local Cosmos-Reason2-2B directory |
| `CKPT_PATH` | - | Finetuned checkpoint path; `{suite}` is replaced in turn with `spatial` / `object` / `goal` / `10` |
| `STATS_PATH` | - | Statistics file (`stats.json`) of the suite; also supports `{suite}` |
| `LOONGFORGE_PATH` | - | LoongForge repository root |
| `OUT_ROOT` | - | Root directory for evaluation outputs |
| `SERVER_PY` | `/usr/bin/python` | Python interpreter of the LoongForge policy server |
| `BENCHMARK_PYTHON` | - | Python interpreter of the LIBERO simulator (LIBERO must be installed) |
| `GPU` | 0 | GPU index to use |
| `SUITES` | all four suites | Space-separated suites to evaluate, e.g. `"libero_spatial libero_goal"` |
| `EPISODES` | 20 | Episodes per task; 10 × 20 = 200 matches the official protocol |
| `MAX_TASKS` | 0 | Tasks evaluated per suite; 0 means all 10 tasks |
| `SEED` | 42 | Random seed |
| `MUJOCO_GL` | `egl` | MuJoCo rendering backend; can be switched to `osmesa` depending on the environment |

The evaluation protocol (at most 720 steps per episode, action chunk of 8, etc.) is fixed in `eval/configs/libero/libero_<suite>.yaml`; path parameters are injected by the script at runtime, so no YAML needs editing.

### 4.3 Results

Success rates on the four suites (10 tasks × 20 episodes = 200 episodes per suite):

| Suite | Successful episodes | LoongForge | Official N1.7 |
|---|:---:|:---:|:---:|
| libero_spatial | 196 / 200 | 98.0% | 97.65% |
| libero_object | 197 / 200 | 98.5% | 98.45% |
| libero_goal | 193 / 200 | 96.5% | 97.5% |
| libero_10 (Long) | 191 / 200 | 95.5% | 94.35% |
| **Average** | **777 / 800** | **97.13%** | **96.99%** |

> **Notes**:
> - The "Official N1.7" column is taken from Isaac-GR00T `examples/LIBERO/README.md`.
> - Evaluation is stochastic, so results vary slightly from run to run.

### 4.4 Rollout Examples

| libero_spatial | libero_object |
|:---:|:---:|
| ![libero_spatial](../../assets/images/gr00t_n1_7/replay_libero_spatial.gif) | ![libero_object](../../assets/images/gr00t_n1_7/replay_libero_object.gif) |
| *pick up the black bowl on the wooden cabinet and place it on the plate* | *pick up the orange juice and place it in the basket* |
| **libero_goal** | **libero_10** |
| ![libero_goal](../../assets/images/gr00t_n1_7/replay_libero_goal.gif) | ![libero_10](../../assets/images/gr00t_n1_7/replay_libero_10.gif) |
| *put the bowl on the stove* | *put both the cream cheese box and the butter in the basket* |
