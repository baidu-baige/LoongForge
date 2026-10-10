# Torch DCP Convert

Turn a sharded Torch **DCP** (Distributed Checkpoint) into a single weights file
you can actually load elsewhere. Training writes DCP shards (a `dcp/` directory
of per-rank files); most consumers want one `model.safetensors` (or
`pytorch_model.pt`).

Reach for this when you need to:

- Feed a training checkpoint to `load_pretrained` (which only accepts
  single-file weights, not DCP shards — unlike resume, which reads DCP directly).
- Export a release artifact for downstream inference.

Runs in a single process — no `torchrun`, no distributed init.

## Prerequisites

Just `torch` (and `safetensors` if you output `.safetensors`, which is the
default). Nothing from LoongForge needs to be importable.

## Usage

```bash
python tools/torch_dcp_convert/dcp_to_safetensors.py \
    --ckpt   outputs/run/checkpoints/steps_10000 \
    --out    outputs/run/release/steps_10000.safetensors \
    --format safetensors
```

**Input** (`--ckpt`) must be a **DCP checkpoint** — either a `steps_N` directory
containing a `dcp/` subdirectory, or the `dcp/` directory itself (it must hold
`.metadata`). Only the `model.*` tensors are consolidated; optimizer state and
non-tensor entries are skipped.

| Argument | Required | Description |
|----------|:---:|-------------|
| `--ckpt` | ✅ | DCP checkpoint: a `steps_N` dir or its `dcp/` subdirectory |
| `--out` | ✅ | Output single-file path (e.g. `model.safetensors`) |
| `--format` | – | `safetensors` (default) or `pt` (`torch.save`) |

**Output** is one file at `--out`, ready to hand to `load_pretrained` or a
release directory.
