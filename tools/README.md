# LoongForge Tools

Developer utilities for LoongForge — checkpoint conversion, data preparation,
benchmarking, and robot-data generation. They live outside the `loongforge`
package (under `tools/`); most are standalone CLIs you run directly, though a few
(e.g. `mcore_bridge`) are imported by the training runtime. Each directory below
is one tool; see its own README for usage.

| Tool | Purpose |
|------|---------|
| [`mcore_checkpoint_convert/`](./mcore_checkpoint_convert) | Offline HuggingFace ↔ Mcore checkpoint conversion (LLM/VLM, MoE expert merge, FP8) |
| [`mcore_bridge/`](./mcore_bridge) | Online HuggingFace ↔ Mcore checkpoint bridge (distributed save/load) used by the training runtime |
| [`torch_dcp_convert/`](./torch_dcp_convert) | Consolidate a Torch DCP (distributed checkpoint) into a single-file `.safetensors`/`.pt` |
| [`te_parallel_benchmark/`](./te_parallel_benchmark) | Benchmark TransformerEngine parallel layers under TP/EP and emit an adaptive-FP8 policy |
| [`llm_data_preprocess/`](./llm_data_preprocess) | Tokenize & pack LLM pretrain / SFT corpora into training-ready datasets (Megatron-indexed for pretrain, tokenized HF dataset for SFT) |
| [`vlm_data_preprocess/`](./vlm_data_preprocess) | Convert VLM annotations + media to WebDataset, plus offline sequence packing |
| [`dreamzero_data_preprocess/`](./dreamzero_data_preprocess) | Prepare DreamZero datasets and precompute/validate feature caches |
| [`ego2robot/`](./ego2robot) | Convert first-person human manipulation video into LeRobot v3.0 training data across dual-arm robot morphologies |
