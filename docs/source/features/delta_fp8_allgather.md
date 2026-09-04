# Delta-FP8 AllGather

Delta-FP8 AllGather is an opt-in FSDP2 communication optimization for the LoongForge embodied training stack. It reduces parameter AllGather traffic by communicating blockwise FP8 deltas instead of complete BF16 parameters. It changes communication precision only; forward and backward computation continue to use the dtype configured by the FSDP mixed-precision policy.

Delta-FP8 AllGather is independent of LoongForge's end-to-end [FP8 training](fp8_training.md). Enabling this feature does not convert model weights, activations, or GEMMs to FP8.

## 1. Requirements

Delta-FP8 requires all of the following:

- FSDP2 selected with `--distributed-strategy fsdp`
- A CUDA device and the NCCL distributed backend
- An NVIDIA GPU with compute capability 8.9 or later
- PyTorch FP8 E4M3 support
- A CUDA Triton backend exposing `tl.float8e4nv`
- BF16 FSDP parameter communication using the default AllGather implementation

The runtime capability check executes while FSDP groups are registered. Unsupported devices or backends fail at startup with an error containing the detected device and backend. Non-BF16 parameter groups and custom AllGather implementations retain native FSDP behavior.

## 2. Usage

### 2.1 DreamZero Wan2.2-5B

The DreamZero Wan2.2-5B full FSDP recipe enables Delta-FP8 with its validated settings. No additional Delta-FP8 argument is required:

```bash
bash examples/embodied/dreamzero/run_dreamzero_wan22_5b_full_fsdp_finetune.sh
```

To use native BF16 AllGather for an A/B comparison:

```bash
bash examples/embodied/dreamzero/run_dreamzero_wan22_5b_full_fsdp_finetune.sh \
    --no-fsdp-delta-fp8-allgather
```

### 2.2 Other Embodied Models

The framework-level default is disabled. Append the following flag to an FSDP launcher to enable Delta-FP8 for eligible parameter groups:

```bash
bash path/to/fsdp_launcher.sh \
    --fsdp-delta-fp8-allgather
```

Support in the communication path does not by itself establish numerical or performance suitability for every model. Validate loss, step time, and peak memory against the same model's BF16 FSDP baseline before adopting it in another recipe.

## 3. Configuration

| Argument | Default | Description |
| --- | --- | --- |
| `--fsdp-delta-fp8-allgather` | `False` | Enable Delta-FP8 for the current model's eligible FSDP groups |
| `--fsdp-delta-fp8-block` | `256` | Number of elements sharing one FP32 scale; must be a positive power of two no greater than 1,048,576 |
| `--fsdp-delta-fp8-prime-steps` | `1` | Number of full BF16 AllGathers used to initialize each FSDP unit before delta communication |
| `--fsdp-delta-fp8-reprime-interval` | `0` | Perform a full BF16 AllGather every N unshards; `0` disables periodic re-priming |

Keep the default block size and priming settings unless a matched loss and performance validation supports changing them. A smaller block gives finer quantization scales but communicates more scale data. Periodic re-priming can re-anchor the reference after discontinuous parameter changes, but it also adds full BF16 collectives.

## 4. How It Works

For each eligible FSDP unit:

1. Full BF16 AllGather initializes a persistent BF16 reference.
2. The local parameter shard is compared with the corresponding reference shard.
3. The delta is quantized per block into an FP8 E4M3 payload with an FP32 scale.
4. FP8 payloads and scales are gathered across ranks while preserving the caller's asynchronous collective semantics.
5. Gathered deltas are dequantized and accumulated into the BF16 reference used by FSDP.

Accumulating reconstructed deltas into the reference feeds quantization residuals into later updates instead of repeatedly quantizing against a fixed initial value. The reference aliases FSDP's unsharded parameter storage, and quantization scratch buffers are shared across FSDP units to avoid persistent per-unit staging allocations.

## 5. Validation

Delta-FP8 uses lossy communication. For a new model, optimizer, block size, or re-priming policy, compare it with a canonical native BF16 FSDP run under the same weights, data order, batch configuration, precision, hardware, and measurement window. At minimum, check:

- Step-by-step loss and gradient norm over a representative training window
- Stable step time after excluding initialization and compilation warmup
- Per-rank peak GPU memory
- NaN, Inf, communication, and distributed runtime errors

Do not infer support for a model solely from a successful short smoke test. Promote the setting into a default recipe only after its numerical and performance behavior has been validated for that workload.

## 6. Troubleshooting

| Symptom | Action |
| --- | --- |
| Startup reports an unsupported device, backend, or Triton FP8 type | Use native BF16 AllGather or run on a supported CUDA/NCCL environment |
| Performance does not improve | Confirm that communication uses BF16 default FSDP AllGather and that AllGather is a material bottleneck |
| Loss diverges from the BF16 reference | Restore the default block and priming settings; if the difference remains unacceptable, disable Delta-FP8 for that workload |
| Behavior changes after a discontinuous parameter update | Validate a nonzero `--fsdp-delta-fp8-reprime-interval` or restart the run so references are initialized again |
