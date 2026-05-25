# Wan2.2 Packing Training

Wan2.2 packing concatenates multiple variable-length video samples into one packed training sequence. It keeps per-sample attention and loss boundaries through THD `PackedSeqParams`, so packed samples do not attend to each other and padding tokens do not contribute to loss.

## When to Use

Enable packing when your Wan2.2 training set contains videos or text prompts with noticeably different lengths. Packing reduces wasted computation from padding and is compatible with context parallel training.

Supported context-parallel modes:
- No CP: `CP_SIZE=1`
- Ring CP: `CP_SIZE>1`, `CP_ULYSSES_DEGREE=1`
- Ulysses CP: `CP_SIZE=CP_ULYSSES_DEGREE`
- Hybrid Ring + Ulysses: `CP_SIZE>CP_ULYSSES_DEGREE>1`

## Data Requirements

Use the same preprocessed Wan dataset format as normal training. Each sample should provide:
- `input_latents`: video latent tensor
- `y`: optional image-conditioning latent
- `context`: text embedding
- `seed`: sample seed used for deterministic noise and timestep generation
- `grid_sizes`: optional latent patch grid; when missing, LoongForge derives it from `input_latents`

Packing supports variable-length samples. For CP training, each sample in a packed bin is padded to the per-sample CP split boundary before the bin is concatenated.

## How to Enable

Add the packing flags to the Wan pretrain script:

```bash
--packing-sft-data
--packing-buffer-size 512
```

Example launch:

```bash
cd examples/wan
CUDA_VISIBLE_DEVICES=0,1,2,3 \
CP_SIZE=4 \
CP_ULYSSES_DEGREE=2 \
bash pretrain_wan2.2_i2v_a14b.sh
```

`--packing-buffer-size` controls how many samples are buffered before forming packed bins. Larger buffers can improve packing density but use more host memory.

## Notes and Limitations

- Packing currently uses `micro_batch_size=1`; the validator enforces this when packing is enabled.
- The packed attention path uses THD metadata and may not be bitwise identical to non-packed dense attention, but loss should stay numerically close.
- Keep `seq_length` large enough for one packed bin. In CP mode, LoongForge aligns the effective sequence length to the required CP split boundary.
- For accuracy checks, compare the first several training iterations against a packing-off run with the same data order and do not change `train-iters` between the two runs.
