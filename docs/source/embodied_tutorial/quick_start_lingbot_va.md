# Quick Start: LingBot-VA Training

This guide walks you through launching **LingBot-VA** post-training in the LoongForge framework. The examples provide two 8-GPU training entrypoints for RoboTwin and LIBERO.

## 0. Resource Preparation
The LingBot-VA post-training weights and LeRobot datasets are published at the following locations:

| Resource | Link |
|-|-|
| RoboTwin Post-training Weights | [https://huggingface.co/robbyant/lingbot-va-posttrain-robotwin](https://huggingface.co/robbyant/lingbot-va-posttrain-robotwin) |
| LIBERO-Long Post-training Weights | [https://huggingface.co/robbyant/lingbot-va-posttrain-libero-long](https://huggingface.co/robbyant/lingbot-va-posttrain-libero-long) |
| RoboTwin LeRobot Dataset | [https://huggingface.co/datasets/robbyant/robotwin-clean-and-aug-lerobot](https://huggingface.co/datasets/robbyant/robotwin-clean-and-aug-lerobot) |
| LIBERO-Long LeRobot Dataset | [https://huggingface.co/datasets/robbyant/libero-long-lerobot](https://huggingface.co/datasets/robbyant/libero-long-lerobot) |

This project follows the same offline preprocessing scheme as the official LingBot-VA community. Please stage the weights and data under the local `/workspace` directory in advance. The data directory must contain pre-generated latents and the corresponding `empty_emb.pt`. When using your own dataset, first convert the raw robot data to LeRobot format, then augment each episode in `meta/episodes.jsonl` with `action_config` (start/end frames of action clips and text descriptions), map the actions to the standard 30-dimensional format (pad missing dimensions with 0), and extract video latents with Wan2.2 VAE, placing them under a `latents/` directory that mirrors `videos/`. Videos should be preprocessed according to the official data spec. For detailed fields, directory layout, and preprocessing commands, refer to the **Custom Dataset Preparation** section of the official [README.md](https://github.com/Robbyant/lingbot-va/blob/main/README.md).

## 1. Launch Training
```bash
cd /workspace/LoongForge

export LOONGFORGE_PATH=/workspace/LoongForge
export NCCL_DEBUG=WARN
export GPUS_PER_NODE=8
```
RoboTwin:

```bash
export CHECKPOINT_PATH=/workspace/models/lingbot-va-posttrain-robotwin
export DATA_PATH=/workspace/datasets/robotwin-clean-and-aug-lerobot
export OUTPUT_DIR=/workspace/outputs/lingbot_va_robotwin

bash examples/embodied/lingbot_va/run_lingbot_va_robotwin_fsdp_finetune.sh
```
LIBERO-Long:

```bash
export CHECKPOINT_PATH=/workspace/models/lingbot-va-posttrain-libero-long
export DATA_PATH=/workspace/datasets/libero-long-lerobot
export OUTPUT_DIR=/workspace/outputs/lingbot_va_libero

bash examples/embodied/lingbot_va/run_lingbot_va_libero_fsdp_finetune.sh
```
## 2. Performance Optimization Options
### 2.1 Baked-in Optimizations
The following optimizations are baked into the LingBot implementation and have no toggle:

| Optimization | Effect | Bottleneck Addressed |
|-|-|-|
| Self-Flex mask cache | Reuse pre-computed attention masks. | Avoid re-constructing masks and the associated CPU/GPU overhead. |
| LeRobot repo discovery cache | Cache the list of repos in the data directory. | Avoid repeatedly scanning large volumes of metadata each epoch, reducing startup and dataloader initialization overhead. |
| Transformer Engine Q/K RMSNorm | Use the fused Q/K RMSNorm implementation. | Reduce normalization kernel launches, memory access, and intermediate tensor overhead. |
| Self-Flex block64 | Fix the block64 configuration for Flex Attention's Triton fused forward/backward kernels (FWD=`64,64,4,3`, BWD=`32,32,4,1`). | Avoid falling back to dense attention, reducing attention compute cost and memory pressure. |
| Cross attention optimized implementation | Use an efficient cross-attention path. | Reduce kernel and memory overhead of cross-attention. |

### 2.2 Optional Optimizations
The following performance optimizations are user-controllable and are enabled by default in the example scripts:

| Switch | Default | Effect |
|-|-|-|
| `LINGBOT_BALANCED_SAMPLER` | `1` | Perform deterministic load balancing across DP ranks based on per-sample compute cost, reducing slow-rank stragglers; this also changes sample loading order, so disable it if you need step-by-step alignment of loss with the official community. |
| `LINGBOT_LAYERWISE_COMPILE` | `1` | Compile the layerwise norm, modulation, and residual gate paths. |

For example, to disable the optional optimizations:

The command below uses RoboTwin as an example; for LIBERO, use `run_lingbot_va_libero_fsdp_finetune.sh`.

```bash
LINGBOT_BALANCED_SAMPLER=0 \
LINGBOT_LAYERWISE_COMPILE=0 \
bash examples/embodied/lingbot_va/run_lingbot_va_robotwin_fsdp_finetune.sh
```
The following switches control logging, diagnostics, or artifact management and do not change model training semantics:

| Switch | Default | Effect |
|-|-|-|
| `LINGBOT_BASELINE_LOSS_LOG` | `1` | Emit loss logs compatible with the upstream baseline, which facilitates comparison of training curves. |
| `LINGBOT_SAMPLE_META_EXPORT` | `0` | Export sample/frame/CFG metadata, for diagnosing sample order and reproducing experiments. |
| `LINGBOT_SKIP_FINAL_CHECKPOINT` | `0` | Skip final checkpoint saving to reduce disk footprint and shutdown time. |

### 2.3 Correctness Verification

To ensure the optimizations do not affect training accuracy, we performed a step-by-step action-loss comparison between LoongForge's LingBot-VA adaptation and the official implementation under identical data, weights, and training configurations. The results show that LoongForge's performance optimizations do not affect training accuracy:
![alt text](../../assets/images/precision/Lingbot-va.png)
