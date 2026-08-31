# Wall-OSS-0.5 Examples

Run these scripts from the repository root.

| Script | Purpose |
| --- | --- |
| `compute_norm_stats.sh` | Compute LeRobot action/state normalization statistics (e.g. `libero_norm_stats.json`). |
| `run_wall_oss_dmuon_fsdp8.sh` | Run Wall-OSS-0.5 fine-tuning on 8 GPUs with FSDP + the DMuon optimizer. |

## Paths

The scripts default to `/workspace/{LoongForge,outputs,datasets}` and
`/workspace/wall-oss-05` for the checkpoints. Set the shared roots when the
data, weights, or repo live elsewhere:

```bash
export LOONGFORGE_PATH=/path/to/LoongForge
export CKPT_ROOT=/path/to/wall-oss-05          # holds Qwen2.5-VL-3B-Instruct and wall-oss-0.5
export DATA_PATH=/path/to/libero               # source LeRobot dataset
export OUTPUT_ROOT=/path/to/outputs
export NORM_STATS_PATH=/path/to/libero_norm_stats.json
export WALL_OSS_OPS_SRC=/path/to/ops/cuda_source/wall_oss_05_op   # CUDA operator sources
```

## CUDA Operators

The fused operators (RoPE / M-RoPE / RMSNorm / SwiGLU / MoE permute / index
kernels) ship as the standalone `wall_oss_05_op` package, whose sources live in
`ops/cuda_source/wall_oss_05_op`. `run_wall_oss_dmuon_fsdp8.sh` imports the
package first and, when it is missing, installs it from `WALL_OSS_OPS_SRC`
(default `$LOONGFORGE_PATH/ops/cuda_source/wall_oss_05_op`):

```bash
pip install --no-build-isolation -e ops/cuda_source/wall_oss_05_op
```

Each operator falls back to pure PyTorch when the compiled extension is absent,
so training still runs — just slower. The resolved backends are logged once per
run as `[WallOpsBackendInventory]`; expect `cuda_inline` for all nine operators.

The tokenizer and pretrained weights resolve from `CKPT_ROOT` by default
(`$CKPT_ROOT/Qwen2.5-VL-3B-Instruct` and `$CKPT_ROOT/wall-oss-0.5`); override
`TOKENIZER_PATH` / `PRETRAINED_CHECKPOINT` to point elsewhere.

## Compute Norm Stats

Wall-OSS-0.5 normalizes actions/proprioception with per-dim `mean/std/q01/q99`
statistics. **This step is required before the fine-tuning below** — the
fine-tune script consumes the resulting JSON via `NORM_STATS_PATH`, so generate
it once from the source LeRobot dataset first. The JSON layout matches
`wall_oss_0_5/transforms/wall_oss_0_5_utils.py::load_norm_stats`.

```bash
DATASET_PATH=/path/to/libero \
OUTPUT_PATH=$LOONGFORGE_PATH/data/wall_oss_0_5_norm_stats/libero_norm_stats.json \
  bash examples/embodied/wall_oss_0_5/compute_norm_stats.sh
```

`STATE_KEY` (default `observation.state`) and `ACTION_KEY` (default `action`)
select the dataset columns; extra arguments are forwarded to the underlying
`compute_norm_stats.py`.

## Fine-Tuning (FSDP + DMuon)

Point `NORM_STATS_PATH` at the JSON produced above, then launch 8-GPU training.
The script wraps the model with FSDP2 and routes matrix parameters through the
DMuon optimizer (AdamW for the rest), using `--custom-lr-lambda` for the
warmup + cosine-to-`--min-lr` schedule.

```bash
DATA_PATH=/path/to/libero \
NORM_STATS_PATH=/path/to/libero_norm_stats.json \
  bash examples/embodied/wall_oss_0_5/run_wall_oss_dmuon_fsdp8.sh
```

Common overrides (all optional): `GPUS_PER_NODE` (default 8), `CUDA_ID`,
`RUN_NAME`, `OUTPUT_ROOT`, `MASTER_ADDR` / `MASTER_PORT`. Any extra flags are
forwarded to `loongforge/embodied/train.py`, so schedule/optimizer settings such
as `--train-iters`, `--per-device-batch-size`, or the `--dmuon-*` group can be
tuned from the command line.
