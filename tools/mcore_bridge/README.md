# mcore_bridge

Online **HuggingFace ↔ Mcore** checkpoint bridge. It lets the training runtime
read HF safetensors directly at startup (converting to Mcore format on the fly)
and optionally export Mcore weights back to HF format after training — no
separate offline conversion step, and no need to keep both copies on disk.

Inspired by [NVIDIA Megatron-Bridge](https://github.com/NVIDIA-NeMo/Megatron-Bridge).

## How it's used

This is a **runtime-integrated** tool, not a standalone CLI. `loongforge/train.py`
imports it automatically; you enable it just by pointing `--load` at an HF model
directory:

```bash
--load  $HF_CKPT_PATH   # HF dir with config.json + *.safetensors
--save  $OUT_PATH       # where Mcore checkpoints are written (may equal --load)
--save-hf true          # optional: export HF weights when training finishes
```

On the first run (no `latest_checkpointed_iteration.txt` in `--save`) it loads HF
and converts online; on later runs it resumes from the saved Mcore shards. See the
full feature guide for loading/saving/resume semantics and VLM heterogeneous TP:

- `docs/source/features/mcore_bridge.md` (EN) / `docs/source_zh/features/mcore_bridge.md` (ZH)

Example training scripts: `examples/qwen2.5/pretrain/pretrain_qwen2.5_7b_bridge.sh`,
`examples/deepseek_v2/pretrain/pretrain_deepseek_v2_lite_group_bridge.sh`.

## Roundtrip test

`tests/mcore_bridge/hf_roundtrip_test.py` runs a zero-step `HF → Mcore → HF`
round-trip and compares weights, using the exact same build/load/save path as
training. The driver and its per-model launch scripts live outside this package,
under `tests/mcore_bridge/<family>/`:

```bash
bash tests/mcore_bridge/qwen2.5/0.5b_bridge_roundtrip.sh   # one model
bash tests/mcore_bridge/qwen3/all.sh                       # a whole family
```

A `roundtrip_comparison.json` is written to `--save-hf-path`; it passes when
`num_different == 0` with no missing/extra keys or shape mismatches.

## Layout

```text
mcore_bridge/
  checkpoint/   # HF converter + online loader/saver
  core/         # parser, topology sharder, TP gather
  config/       # parallel config
  utils/        # timing/comparison helpers
```

Roundtrip tests (driver + per-model launch scripts) live in the repo test
suite at `tests/mcore_bridge/`, not inside this package.

Modules import as `tools.mcore_bridge.*` (repo root on `sys.path`) or bare
`mcore_bridge.*` (with `tools/` on `sys.path`); both are set up by the entry
scripts and by `loongforge/train.py`.
