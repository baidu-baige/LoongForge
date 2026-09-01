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
| Self-Flex BlockMask cache | Cache the BlockMask under a key built from the latent/action shapes, chunk, window and patch sizes, keeping up to 256 entries. | Avoid rebuilding CPU mask metadata and repeating H2D copies for a shape already seen. |
| Self-Flex block64 kernel configuration | Fix the block64 configuration of FlexAttention's Triton fused kernels (FWD=`64,64,4,1`, BWD=`32,32,4,1`) and take the `IS_DIVISIBLE` fast path (q/k/v sequence lengths must be multiples of 128). | Avoid falling back to dense attention, cutting attention compute and memory pressure. |
| Compiled FlexAttention and BlockMask construction | Both self-attention and `create_block_mask` run through `torch.compile(dynamic=True)`; if compilation is unavailable the code falls back to eager and warns once. | Reduce kernel launches and Python dispatch on the attention path. |
| Compiled layerwise fused paths | Six BF16 per-layer segments are compiled: modulation prologue, self residual + cross norm, cross residual + FF norm, residual gate, and output modulation norm. | Fuse the small per-layer operators, cutting kernel count and intermediate tensors. |
| Paired Triton RoPE | Apply rotary embeddings to q and k in one Triton kernel (`apply_triton_rope_pair`). | Replace repeated elementwise ops and transposes, lowering memory traffic and launch cost. |
| RoPE frequency and timestep frequency caches | Rotary frequencies are cached per grid key (up to 16 entries), timestep frequencies per device and dimension, and the time/text embeddings run compiled implementations. | Avoid recomputing trigonometric terms and projections every step, shortening host-side forward preparation. |
| Transformer Engine Q/K RMSNorm | Q/K normalization uses the fused `te.RMSNorm` implementation. | Reduce normalization kernel launches, memory access, and intermediate tensors. |
| Cross attention through SDPA | Cross-attention calls `F.scaled_dot_product_attention` directly instead of entering the flex path. | Cross-attention needs no block-sparse mask, so the flex path's extra overhead is avoided. |
| LeRobot repo discovery cache | The repo list and per-sample costs are cached on disk under a data signature (default `<dataset_path>/.lingbot_cost_cache`), coordinated across ranks by a file lock with a 1800 s wait and 2 s polling. | Avoid every rank rescanning large volumes of metadata, shrinking startup and dataloader initialization time. |
| Nested FSDP2 wrapping with post-step reshard | Apply nested `fully_shard` in block + root order and reshard after the optimizer step (active when `LINGBOT_FSDP_RESHARD=0`). | Remove the redundant all-gather between forward and backward, reducing communication volume. |
| Compiled device-side loss guard | NaN/inf and loss-scaling checks run in a `fullgraph`-compiled device-side operator. | Avoid a device-to-host synchronization per micro-batch just to run the checks. |

> Note: with the framework's `--manual-gc` enabled, LingBot keeps young-generation collection and suppresses generation 2 (threshold `GC_GENERATION2_THRESHOLD`) to remove periodic GC jitter. This follows the framework switch and is off by default in the example scripts.

### 2.2 Optional Optimizations
The following performance optimizations are user-controllable and are exported at their recommended values in the example scripts:

| Switch | Default | Effect |
|-|-|-|
| `LINGBOT_BALANCED_SAMPLER` | `1` | Deterministically balance load across DP ranks by per-sample compute cost and align each rank's microbatch costs, reducing slow-rank stragglers; it also changes the sample loading order, so disable it if you need step-by-step loss alignment with the official community (disabling falls back to the public `DistributedSampler` partitioning). |
| `LINGBOT_FSDP_RESHARD` | `0` | `0` keeps parameters unsharded between forward and backward (the accepted configuration); `1` restores the framework-default reshard behavior. |
| `LINGBOT_FSDP_BF16_REDUCE` | `1` | `1` reduces gradients in the parameter dtype (BF16), removing one FP32 reduce-scatter per step; `0` reduces in FP32. This switch overrides `--fsdp-reduce-dtype` and is a numerics choice, not only a speed one. |

For example, to disable the optional optimizations:

The command below uses RoboTwin as an example; for LIBERO, use `run_lingbot_va_libero_fsdp_finetune.sh`.

```bash
LINGBOT_BALANCED_SAMPLER=0 \
LINGBOT_FSDP_RESHARD=1 \
LINGBOT_FSDP_BF16_REDUCE=0 \
bash examples/embodied/lingbot_va/run_lingbot_va_robotwin_fsdp_finetune.sh
```

### 2.3 Functional Switches
The following environment variables serve diagnostics, reproducibility and artifact management, and do not change model training semantics:

| Switch | Default | Effect |
|-|-|-|
| `LINGBOT_SAMPLE_ORDER_EXPORT_DIR` | empty (no export) | Export the per-rank, per-epoch sample order as JSON (seed, balance group size, index sequence), for reproducibility and step-by-step alignment checks. |
| `LINGBOT_REPO_DISCOVERY_CACHE_DIR` | `<dataset_path>/.lingbot_cost_cache` | Choose the repo discovery cache directory; point it at a writable path when the dataset directory is read-only or shared across jobs. |

### 2.4 Correctness Verification

To ensure the optimizations do not affect training accuracy, we performed a step-by-step action-loss comparison between LoongForge's LingBot-VA adaptation and the official implementation under identical data, weights, and training configurations. The results show that LoongForge's performance optimizations do not affect training accuracy:
![alt text](../../assets/images/precision/Lingbot-va.png)
