#!/usr/bin/env bash
# Copyright 2026 The LoongForge Authors.
# SPDX-License-Identifier: Apache-2.0

# ═══════════════════════════════════════════════════════════════
# run_fastwam_sft_ddp_zero1_finetune_official_aligned.sh
# FastWAM SFT — official-aligned launch (LIBERO 4-suite joint, first-frame).
#
# Aligns the LoongForge SFT run to standalone FastWAM
# (/path/to/FastWAM). The alignment sits in two places:
#   (A) configs/models/embodied/fastwam_aligned.yaml  — passed via --config-file
#   (B) the command line at the bottom of this file    — flags this script adds
#
# ── (A) YAML alignment  [ --config-file configs/models/embodied/fastwam_aligned.yaml ]
#
#   data / preprocessing — matches the official data pipeline
#     data.normalization_mode          q99         -> min_max      (official: min/max)
#     data.image_normalize_mode        siglip      -> identity     (official: ToTensor, stays [0,1])
#     data.image_resize_strategy       center_crop -> resize_only  (official: Resize(224,224))
#     model.action_scheduler.*_shift   class default                 (1.0/1.0 in modeling_configuration_fastwam.py; official action scheduler)
#
#   *** ACCELERATION SWITCHES TURNED OFF ***  — the training-side speed optimizations
#     model.disable_train_autocast         true -> false
#         Official trains under `with self.accelerator.autocast()` (trainer.py:696).
#         Not redundant even with bf16 params: autocast still promotes
#         softmax / layernorm-class ops to fp32.
#     model.mot_compile_blocks             both -> none
#         Official compile_training_denoise=false (configs/model/fastwam.yaml:9) and
#         mot.py:679 gates compilation on it -> official never compiles the MoT
#         training layers. This is the biggest training-speed knob turned off here.
#     model.drop_all_true_cross_attn_mask  true -> false
#         Official always passes the cross-attn mask; report §5.6 records False as the
#         official semantics. Costs the SDPA flash kernel, so this one is a
#         speed/faithfulness trade-off — set it back to true if throughput matters more.
#
#   KEPT ON PURPOSE (verified to already match official — do NOT "fix" these)
#     model.compile_vae_encode         true      Official ALSO compiles the VAE encode on the
#                                                TRAINING path: fastwam.py:252 compiles
#                                                unconditionally and build_inputs:340 calls
#                                                _encode_video_latents. Turning it off would ADD
#                                                a deviation. Only the backend differs
#                                                (official cudagraphs+fullgraph vs LF inductor)
#                                                and that is not configurable.
#     model.rmsnorm_impl               wan       Upstream Wan RMSNorm. The residual vs official
#                                                is a fused-kernel rounding difference
#                                                (F.rms_norm vs explicit fp32 upcast), not semantic.
#     model.dtype                      bfloat16  Official's loader default is also bf16.
#     model.mot_checkpoint_mixed_attn  false     Official task config sets the same.
#     data.action_padding_strategy=zero, data.action_video_freq_ratio=4, data.image_size=224,
#     data.num_video_frames=9, model.action_dim=7, model.proprio_dim=8, model.action_horizon=32
#                                                All already matched before this pass.
#
# ── (B) Command-line alignment added by this script
#
#   --config-file "$CONFIG_FILE"
#       (A) above. Default $CONFIG_FILE = fastwam_aligned.yaml; override via env.
#   --sampler-epoch-mode fixed
#       Official ResumableEpochSampler seeds with `seed + epoch + epoch_offset` and never
#       advances the epoch during normal training -> every epoch reuses ONE permutation.
#       SAMPLER_EPOCH_MODE=advance restores LoongForge's per-epoch reshuffle.
#   --optimizer AdamW
#       Official builds plain `torch.optim.AdamW(...)` with no `fused` kwarg
#       (trainer.py:89-94). LoongForge's "TorchFusedAdamW" maps to the SAME class
#       (optimizer.py:29-30) but adds `fused=True` (optimizer.py:292-293), i.e. a
#       different CUDA kernel. "AdamW" selects the same class without the fused flag,
#       matching official's code path. OPTIMIZER=TorchFusedAdamW restores the fused kernel.
#   --lr-base 1e-4 --lr-warmup-iters 1000 --min-lr 1e-6
#       Official: lr 1e-4, warmup = 5% of steps (=1000 @20k), CosineAnnealingLR
#       eta_min = lr*0.01 = 1e-6. Seed 42 via $SEED.
#   --clip-grad 0   (official parity; CLIP_GRAD=1.0 keeps the LoongForge default)
#       Official FastWAM runs WITHOUT gradient clipping even though configs/train.yaml
#       sets `max_grad_norm: 1.0`:
#         - under DeepSpeed, `Accelerator.clip_grad_norm_` only returns
#           `engine.get_global_grad_norm()` and IGNORES `max_norm` (accelerate
#           accelerator.py), so clipping is delegated to DeepSpeed;
#         - `scripts/ds_configs/ds_zero1_config.json` has no `gradient_clipping` key
#           and DeepSpeed's `GRADIENT_CLIPPING_DEFAULT = 0.`
#           (deepspeed/runtime/constants.py), so nothing is injected either.
#       MEASURED on the real FastWAM trainer (8x A800, 4-suite LIBERO, 300 steps,
#       temporary `[gradnorm]` probe on its `train/grad_norm`): its global grad norm
#       is ~0.3-7.1 (mean 1.24, median 0.92, >1.0 on 44% of steps). LoongForge's is
#       the same order -- step 1 is 7.0985 (official) vs 7.0938 (LF). So the base
#       script's `--clip-grad 1.0` rescales most steps by roughly 1-7x.
#       MEASURED effect of removing it (20k SFT, aligned protocol, 2000 episodes,
#       everything else identical): libero_10 94.20% -> 90.80% (-3.40pt, McNemar
#       p=0.020); TOTAL -0.65pt; the other three suites unchanged (<=0.4pt).
#       The LoongForge default (1.0) is therefore kept; CLIP_GRAD=0 gives exact
#       official parity at the cost measured above.
#
# ── NOT aligned by default — pass explicitly when you want it
#
#   --zero-master-param-dtype fp32
#       Official DeepSpeed BF16_Optimizer keeps fp32 master params + fp32 Adam state;
#       LoongForge's framework default (--zero-master-param-dtype none) updates bf16
#       params directly instead. Measured on 20k LIBERO SFT (libero_10, sigma_shift 1.0):
#       bf16 89.60% -> fp32 93.20% (+3.6pt) at +0.9% step time (1360.7 -> 1372.6 ms/step).
#       Pass this flag to reproduce the official optimizer-state dtype; the framework
#       default is left unchanged for every other model.
#
# ── CANNOT be aligned from config (structural; measured residual rel-L2 ~1.4% on actions)
#
#   1. ZeRO-1 implementation   LF torch ZeroRedundancyOptimizer + _FP32MasterOptimizerAdapter
#                              vs DeepSpeed ZeRO-1 + BF16_Optimizer: shard placement, reduce
#                              order and comm buckets differ. NOTE: the Adam *kernel*
#                              (--optimizer AdamW, no `fused`) and the fp32-master / fp32-Adam
#                              dtype semantics (--zero-master-param-dtype fp32) ARE aligned.
#   2. gradient reduce order   LF DDP bucket settings vs accelerate/DeepSpeed bucketing
#   3. VAE encode compile      official backend="cudagraphs", fullgraph=True; LF inductor
#   4. RMSNorm rounding        LF F.rms_norm fused vs official x.float() -> .to(bf16) * weight
#   5. sampler class           LF StatefulDistributedSampler vs official ResumableEpochSampler
#                              (--sampler-epoch-mode fixed only aligns "epoch does not
#                               advance", not the per-rank permutation itself)
#
# Usage:
#   bash run_fastwam_sft_ddp_zero1_finetune_official_aligned.sh \
#       [extra args passed to train.py]
#
# Key env vars (override as needed):
#   LOONGFORGE_PATH, DIFFSYNTH_MODEL_BASE_PATH, TOKENIZER_PATH, DATASET_PATH
#   ACTION_DIT_PRETRAINED_PATH, TEXT_EMBEDDING_CACHE_DIR, OUTPUT_DIR
#   TRAIN_ITERS, GPUS_PER_NODE, SEED, CONFIG_FILE, SAMPLER_EPOCH_MODE, OPTIMIZER
# ═══════════════════════════════════════════════════════════════
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# ── Paths (defaults point at the asset root; override via env) ──
export LOONGFORGE_PATH=${LOONGFORGE_PATH:-"$(cd "$SCRIPT_DIR/../../.." && pwd)"}
export DIFFSYNTH_MODEL_BASE_PATH=${DIFFSYNTH_MODEL_BASE_PATH:-"/path/to/models"}
export TOKENIZER_PATH=${TOKENIZER_PATH:-"$DIFFSYNTH_MODEL_BASE_PATH/Wan-AI/Wan2.2-TI2V-5B"}
export DATASET_PATH=${DATASET_PATH:-"$DIFFSYNTH_MODEL_BASE_PATH/LIBERO-fastwam/libero_spatial_no_noops_lerobot,$DIFFSYNTH_MODEL_BASE_PATH/LIBERO-fastwam/libero_object_no_noops_lerobot,$DIFFSYNTH_MODEL_BASE_PATH/LIBERO-fastwam/libero_goal_no_noops_lerobot,$DIFFSYNTH_MODEL_BASE_PATH/LIBERO-fastwam/libero_10_no_noops_lerobot"}
export ACTION_DIT_PRETRAINED_PATH=${ACTION_DIT_PRETRAINED_PATH:-"$DIFFSYNTH_MODEL_BASE_PATH/checkpoints/ActionDiT_linear_interp_Wan22_alphascale_1024hdim.pt"}
export TEXT_EMBEDDING_CACHE_DIR=${TEXT_EMBEDDING_CACHE_DIR:-"$DIFFSYNTH_MODEL_BASE_PATH/data/text_embeds_cache/libero"}
export OUTPUT_DIR=${OUTPUT_DIR:-"$LOONGFORGE_PATH/outputs/fastwam_sft_official_aligned"}
export TRAIN_ITERS=${TRAIN_ITERS:-20000}
export GPUS_PER_NODE=${GPUS_PER_NODE:-8}
export SEED=${SEED:-42}

# Official-aligned YAML (model: + data: sections). --model-name (set by the base
# script, default "fastwam") is still required: it selects the ModelConfig /
# DataConfig classes.
CONFIG_FILE=${CONFIG_FILE:-"$LOONGFORGE_PATH/configs/models/embodied/fastwam_aligned.yaml"}
if [[ ! -f "$CONFIG_FILE" ]]; then
    echo "ERROR: aligned config not found: $CONFIG_FILE" >&2
    exit 1
fi

# Pin the sampler epoch to 0: standalone FastWAM's ResumableEpochSampler seeds with
# `seed + epoch + epoch_offset` and never advances the epoch during normal training,
# so every epoch reuses ONE permutation. Set SAMPLER_EPOCH_MODE=advance to restore
# LoongForge's per-epoch reshuffle.
SAMPLER_EPOCH_MODE=${SAMPLER_EPOCH_MODE:-fixed}

# Plain AdamW (no `fused` kwarg) = official's code path. "TorchFusedAdamW" is the same
# class with fused=True, which is a different CUDA kernel. See the header note.
OPTIMIZER=${OPTIMIZER:-AdamW}

# Official FastWAM effectively runs without gradient clipping (see the header note).
# Its grad norm is ~0.3-7.1, so `--clip-grad 0` is the exact official parity; but we
# measured that it costs -3.4pt on libero_10, so the LoongForge default (1.0) is kept.
# Set CLIP_GRAD=0 for exact official parity.
CLIP_GRAD=${CLIP_GRAD:-1.0}

echo "════════════════════════════════════════════════════════════"
echo "  LoongForge FastWAM SFT (OFFICIAL-ALIGNED, 4-suite joint)"
echo "  Config:    $CONFIG_FILE"
echo "  Data:      $DATASET_PATH"
echo "  Output:    $OUTPUT_DIR"
echo "  Iters:     $TRAIN_ITERS   GPUs: $GPUS_PER_NODE   Seed: $SEED"
echo "  Sampler:   epoch-mode=$SAMPLER_EPOCH_MODE"
echo "  Optimizer: $OPTIMIZER   clip-grad: $CLIP_GRAD"
echo "════════════════════════════════════════════════════════════"

# NOTE: "$@" is appended AFTER the optionals below. Flags are safe there (a repeated
# flag simply wins, e.g. --sampler-epoch-mode advance). Bare dotlist overrides are
# NOT: argparse's positional `overrides` (nargs="*") fails with "unrecognized
# arguments" when it appears after an optional that takes a value.
bash "$LOONGFORGE_PATH/examples/embodied/fastwam/run_fastwam_sft_ddp_zero1_finetune.sh" \
    --config-file "$CONFIG_FILE" \
    --optimizer "$OPTIMIZER" \
    --lr-base 1e-4 --lr-warmup-iters 1000 --min-lr 1e-6 \
    --clip-grad "$CLIP_GRAD" \
    --sampler-epoch-mode "$SAMPLER_EPOCH_MODE" \
    "$@"
