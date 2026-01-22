Below is the full English Markdown translation of your “start_training” section.

---

## Parameter Management
The framework combines Megatron-LM arguments with Hydra configs, keeping full CLI compatibility while enabling fine-grained module-level control for multimodal models.

### Arguments
All native Megatron flags are supported:

* **Parallelism**: `--tensor-model-parallel-size`, `--pipeline-model-parallel-size`, `--context-parallel-size`, etc.  
  Reference: [NVIDIA Megatron parallelism guide](https://github.com/NVIDIA/Megatron-LM/blob/main/docs/user-guide/parallelism-guide.md)
* **Training**: `--lr`, `--data-path`, `--fp16`, optimizer settings, dataset paths, precision, etc.  
  Reference: [Megatron training examples](https://github.com/NVIDIA/Megatron-LM/blob/main/docs/user-guide/training-examples.md)

**AIAK-Omni specific extras**:

| Category | Argument | Purpose | Notes |
|----------|----------|---------|-------|
| model_args | `--model-name` | Select model | Family name (e.g. *llama2*) or exact arch (e.g. *llama2-7b*). Family → you must specify all hyper-params; arch → AIAK auto-fills them to match open-source checkpoints. |
|  | `--config-path` | Model config file | Path to YAML/JSON config |
|  | `--specify-overwrite-model` | Override policy | Controls whether external config overwrites built-in defaults. Default: *foundation_model* |
|  | `--mtp-loss-coef` | DeepSeek-V3 | MTP auxiliary loss coefficient (float, default 0.1) |
|  | `--enable-fa-within-mla` | MLA | Deprecated; use `--attention-backend=flash` instead. When enabled, pads Q/K/V to allow FlashAttention inside MLA. |
| tokenizer_args | `--tokenizer-type` | Tokenizer class | *NullTokenizer*, *HFTokenizer*; auto-inferred if empty |
|  | `--hf-tokenizer-path` | HF model id or local path |  |
|  | `--use-fast-tokenizer` | Speed-up | Enable fast Rust tokenizer (default False) |
|  | `--split-special-tokens` | Splitting rule | Whether to split special tokens (default False) |
|  | `--padding-side` | Pad direction | *left* or *right* (default right) |
|  | `--additional-special-tokens` | User-defined specials | Comma list, e.g. `[TOK1,TOK2]` |
|  | `--vocab-size-in-config-file` | Vocab from HF config |  |
|  | `--padded-vocab-size` | Manually padded size |  |
| sft_args | `--chat-template` | Chat template | Pick from `get_support_templates()` |
|  | `--sft-dataset-config` | Dataset config JSON | Default: `configs/dataset_config.json`. **Required for SFT**; if omitted the framework still tries to load the default file. |
|  | `--sft-dataset` | Unified dataset list | Space-separated names matching `--data-path` order; mutually exclusive with separate `--sft-*-dataset` flags. |
|  | `--sft-train-dataset` | Stand-alone train set |  |
|  | `--sft-valid-dataset` | Stand-alone valid set |  |
|  | `--sft-test-dataset` | Stand-alone test set |  |
|  | `--sft-sort-batch` | Sort samples | Ascending length sort after packing (default False) |
|  | `--sft-data-streaming` | Streaming loader | Default False |
|  | `--streaming-buffer-size` | Buffer size | Default 16384 |
|  | `--sft-data-mix-strategy` | Mixing policy | *concat*, *interleave_under*, *interleave_over* (default concat) |
|  | `--sft-num-preprocess-workers` | CPU workers | Used in non-streaming mode |
|  | `--train-on-prompt` | Loss on prompt | Compute loss/grad on prompt tokens too (default False) |
|  | `--history-mask-loss` | Last-turn only | Mask loss to the final assistant response (default False) |
|  | `--is-tokenized-data` | Pre-tokenized inputs | Skip tokenization (default False) |
|  | `--packing-sft-data` | Pack samples | Fit multiple samples into one sequence (default False) |
|  | `--enable-discard-sample` | Drop long samples | Discard if > `--seq-length` (default False) |
|  | `--packing-batch-size` | Pack buffer | Default 10 000 |
|  | `--use-fixed-seq-lengths` | Fixed length | Pad every sample to `--seq-length`. **LLM only**. |
| training_args | `--training-phase` | Stage | *pretrain* or *sft* (default pretrain) |
|  | `--no-detail-log` | Verbose log | Disable detail-log-interval (default True) |
|  | `--detail-log-interval` | Log frequency | Iterations between detailed logs (default 20) |
|  | `--variable-seq-lengths` |  | Deprecated |
|  | `--enable-ema` | EMA | Enable exponential moving average |
|  | `--ema-decay` |  | Decay factor (default 0.9999) |
|  | `--save-ema` |  | Directory to save EMA checkpoints |
|  | `--load-ema` |  | Directory to load EMA checkpoints |
|  | `--ckpt-format` | Checkpoint format | *torch* or *torch_dist* (default torch) |
| multimodal_args | `--language-model-type` | MM config | Language model family |
|  | `--trainable-modules` | Freeze policy | *all*, *language_model*, *adapter*, *vision_model*, … (default all) |
|  | `--dataloader-save` | Energon state | Path to save dataloader state |
|  | `--packing-pretrain-data` | Pack pretrain | Enable packing for multimodal pre-training |
|  | `--add-question-in-pretrain` | Data aug | Append question to VQASample |
|  | `--image-resolution` | Qwen2-VL | Input image resolution |
|  | `--min-pixels` |  | Min pixels, default 4×28×28 |
|  | `--max-pixels` |  | Max pixels, default 16384×28×28 |
|  | `--frame-min-pixels` | Video | Per-frame min, default 128×28×28 |
|  | `--frame-max-pixels` |  | Per-frame max, default 768×28×28 |
|  | `--video-max-pixels` |  | Whole-video max, default 65536×28×28 |
|  | `--fps` |  | Frames per second, default 2.0 |
|  | `--fps-min-frames` |  | Min frames, default 4 |
|  | `--fps-max-frames` |  | Max frames, default 768 |
| parallel_args | `--context-parallel-ulysses-degree` | Ulysses CP | Degree of Ulysses attention context parallelism (default 1) |
| log_tensor_args | `--enable-log-tensor` | LLM-inspector | Enable tensor tracing |
|  | `--log-tensor-name-pattern` |  | Module name regex (default None → all grad tensors) |
|  | `--log-tensor-stage` |  | Stage to trace: *init*, *forward*, *backward* |
|  | `--log-tensor-iter-pattern` |  | Iterations to trace (default None) |
|  | `--log-tensor-mbs-pattern` |  | Micro-batch indices (default None) |
|  | `--log-tensor-layer-pattern` |  | Layer indices (default None) |
|  | `--log-tensor-rank` |  | TP/PP rank to trace (default 0) |
|  | `--save-tensor` |  | Dump tensors to disk |
|  | `--save-tensor-dir` |  | Output directory |

---

### Hydra Config
Model-specific Hydra configs live in `configs/`.  
Every model inherits from Megatron-Core’s [TransformerConfig](https://github.com/NVIDIA/Megatron-LM/blob/bcdd405f1cc31904cce6434110d4724b3119e0a5/megatron/core/transformer/transformer_config.py#L34), letting you override any submodule with fine-grained control.  
Typical CLI usage:

```bash
+model.image_encoder.freeze=True
+model.foundation.recompute_num_layers=28
```

#### Customising VLM Modules
The framework decomposes VLMs into:

* **image_encoder** – vision transformer that extracts image features  
* **image_projector** – adapter mapping visual features to text embedding dim  
* **foundation** – the LLM backbone  

You can freeze or reconfigure each part independently:

```bash
# partial activation checkpointing
+model.image_encoder.recompute_num_layers=10
+model.foundation.recompute_num_layers=28

# freeze vision components
+model.image_encoder.freeze=True
+model.image_projector.freeze=True
```

---

## Launch Training
Pick the script and run, e.g. pre-train Qwen3-VL-30B-A3B:

```bash
sh examples/qwen3_vl/pretrain/pretrain_qwen3_vl_30b_a3b.sh
```

The same pattern applies to all other models—just replace the script path.