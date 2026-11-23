# FastWAM arch training

The `fastwam_arch` model name selects the restructured arch training path.
The existing `fastwam` name continues to select `FastWAMPolicy`.
Both use the native FastWAM dataset strategy, transforms and preprocessor.

Use `--model-name fastwam_arch --dataset-strategy fastwam` with the trainer's
normal dataset and distributed-training arguments. Its typed configuration is
`configs/models/embodied/arch/fastwam.yaml`.

## Initialization

Set `FASTWAM_ASSETS_ROOT` to a directory containing `Wan2.2-TI2V-5B`,
including `Wan2.2_VAE.pth`. Set `FASTWAM_VIDEO_CHECKPOINT` and
`FASTWAM_ACTION_CHECKPOINT` to the corresponding video and ActionDiT weights.
The video initializer accepts a DiT file, an indexed safetensors directory,
or its `diffusion_pytorch_model.safetensors.index.json`. The action initializer
expects the preprocessed ActionDiT checkpoint file.
Set `FASTWAM_TEXT_CACHE` to the precomputed context cache.

The arch core is assembled independently from the provider. It does not execute
the legacy model's training forward. Checkpoint namespaces are transferred to
the arch stream, head and condition names with shape and coverage checks.
Implicit random initialization is rejected. Explicit scratch experiments may
set `allow_random_initialization`, but must still supply the frozen VAE weights.

## Numerical contract

- Wan2.2 latents: 48 channels, spatial stride 16 and temporal stride 4.
- Video/action scheduler train shifts: 5 / 1.
- Sampling order: video noise, video timestep, action noise, action timestep.
- The clean first latent frame replaces the noised first video frame.
- Video timestep embeddings are token-separated with timestep zero on the
  first latent frame.
- First-frame video queries read first-frame keys; later video queries read
  all video keys. Action queries read action and the first video frame.
- Video flow loss excludes the conditioned first frame. Pixel-frame padding
  is reduced into latent padding with `all` over each four-frame temporal group.
- Action padding is excluded using a per-sample valid-token denominator.
- Both losses are weighted by their scheduler's per-sample timestep weight.
- The existing head/loss configuration controls the final loss weights.
- FastWAM-only Q/K RMSNorm uses the master's ATen implementation. RoPE uses
  input-dtype cos/sin tables and the same conjugate-rotation backward; CUDA uses
  the FlashAttention Triton rotary kernel. Shared VPP2 normalization and RoPE
  implementations are unchanged.
- CUDA attention/FFN input normalization and modulation call FastWAM's
  original norm-modulation wrapper with contiguous hidden inputs, preserving
  the master's layout-dependent fused-kernel dispatch. Without this copy,
  transposed hidden views select the eager fallback and introduce BF16
  intermediate rounding. Kernel binding identity alone is not evidence that
  both branches execute the fused kernel.

## Validation scope

Configuration and registry checks cover both the new alias and legacy routing.
Loss alignment must compare identical full initialization, latents, text,
proprioception, padding, noise and timesteps. Attention backend and precision
must also be shared throughout forward and backward.

A CPU latent-level test is not a substitute for a full-size pretrained
pixel/VAE/DDP training comparison. Report these results separately.

The tiny CPU master comparison covers eager and checkpointed execution with
both all-valid and mixed text masks. It uses independently constructed models
and identical transferred parameters; two independent AdamW updates compare
predictions, video/action/total losses, every gradient, updated parameters and
optimizer moments with strict equality. CPU bindings replace unavailable CUDA
rotary and norm-modulation kernels; the VAE fixture supplies common latents.
This verifies the arch training recipe, not pretrained pixel or DDP parity.

Full-size BF16 short-trajectory validation has completed. With identical initialization,
noise/timesteps, shared real VAE latents and deterministic MATH SDPA, the
pre-fix three-step independent CPU AdamW comparison does not have exact
prediction/loss/gradient parity. Successful tiny CPU tests or a finite
full-size backward must not be interpreted as full-size training equivalence.

An actual full-size first-layer norm-modulation replay confirms that changing
only the arch hidden input to contiguous reproduces the master's output
exactly. This establishes the cause of that local divergence, not whole-model
or optimizer-trajectory parity. A fresh post-fix full-size first forward
(BF16, batch size 1, deterministic MATH SDPA, common real VAE latents)
has exact video/action predictions and all three losses, with all 74 observed
semantic boundaries and RNG states exact. It runs without backward or
optimizer updates; post-fix training-trajectory validation is separate.

The fresh post-fix three-update comparison is strictly exact for predictions,
video/action/total losses, all 1,651 trainable-tensor gradient and updated-weight
fingerprints, AdamW moments/counters, and forward/backward RNG at every step.
Both branches update independently with CPU AdamW (foreach=False, fused=False),
checkpointing enabled, BF16 parameters and deterministic MATH SDPA. The shared
latents come from one actual pretrained VAE encoding; text/action/proprio/padding
are synthetic. This verifies that specific short training trajectory, not a
real-dataset/T5, independent-VAE, production GPU optimizer, DDP, batch>1 or
100-update comparison.
