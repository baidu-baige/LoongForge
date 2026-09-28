# LoongForge Tools

Standalone developer/CLI utilities for LoongForge — checkpoint conversion, data
preparation, benchmarking, and robot-data generation. These are dev tools, not
part of the importable `loongforge` runtime package.

## Layout convention

Every entry directly under `tools/` is **one atomic tool** in its own single-level
directory. A tool may contain many files and internal sub-packages, but the
`tools/` root itself stays flat: one directory == one tool capability.

**Do not** add a category/grouping directory (e.g. a `data_preprocess/` that nests
`llm/`, `vlm/`, ...). Such wrappers hide the tool inventory and invite deeper
nesting. If a new tool belongs to an existing "family", give it its own top-level
directory (e.g. `audio_data_preprocess/`), not a shared parent. The name of a
top-level directory may denote a class of tools, but only a class that is **not
further subdivided**.

## Capability index

| Tool | Purpose | Entry point |
|------|---------|-------------|
| [`convert_checkpoint/`](./convert_checkpoint) | HuggingFace ↔ Mcore checkpoint conversion (LLM/VLM, MoE expert merge, FP8) | `module_convertor/model.py` |
| [`dist_checkpoint/`](./dist_checkpoint) | Distributed checkpoint save/load and HF bridge | package |
| [`dcp_to_safetensors/`](./dcp_to_safetensors) | Consolidate a Torch DCP checkpoint into a single `.safetensors` file | `dcp_to_safetensors.py` |
| [`te_parallel_benchmark/`](./te_parallel_benchmark) | Benchmark TransformerEngine parallel layers under TP/EP and emit an adaptive-FP8 policy | `benchmark_te_parallel_layers.py` |
| [`llm_data_preprocess/`](./llm_data_preprocess) | Tokenize/pack LLM pretrain & SFT corpora into Megatron-indexed data | `preprocess_pretrain_data.py`, `preprocess_sft_data.py` |
| [`vlm_data_preprocess/`](./vlm_data_preprocess) | Convert VLM annotations + media to WebDataset, plus offline sequence packing | `convert_to_webdataset.py`, `offline_packing/` |
| [`dreamzero_precompute/`](./dreamzero_precompute) | Prepare DreamZero datasets and precompute/validate feature caches | `prepare_dataset.py`, `precompute_features.py` |
| [`ego2robot/`](./ego2robot) | Convert first-person human manipulation video into LeRobot v3.0 training data across dual-arm robot morphologies | `cli.py` (see its `README.md`) |
