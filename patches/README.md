# Maintenance Principles

* Patches are maintained against recent upstream release versions. Only one primary version is actively maintained on the main branch, and it is updated promptly as the upstream evolves.
* Modifications to the original codebase are added or removed in response to upstream changes. When an equivalent feature becomes available upstream and meets expectations, we default to adopting the upstream implementation.
* Changes to upstream code should be made conservatively. Whenever possible, modifications are preferred in the upper-level codebase to reduce maintenance complexity.
* Patches should primarily consist of bug fixes and new features; existing upstream capabilities should not be removed by default. Critical changes should be considered for contribution back to the upstream repository.

# Key Modifications (including but not limited to)

* **Megatron 0.15:**
  * Enable variable-length sequence support and introduce `--reduce-variable-seq-shape-p2p-comm` to reduce P2P communication overhead for variable-length data shapes
  * Enhance multi-head MTP (Multi-Token Prediction) with support for shared/independent head weights and cascaded/concatenated computation modes
  * Fix incorrect loss mask when `pad == eos`
  * Support per-module precision control (e.g., `expert_bias`, `output_layer`) via `--use-fp32-dtype-for-param-pattern`, useful for improving RL training precision
  * Improve FP8 training precision by avoiding redundant re-quantization during weight loading to ensure convergence
  * Refine All-to-All overlap under `CUDA_DEVICE_MAX_CONNECTIONS=1`
  * Add tensor-granularity offload with enhanced computation–All-to-All overlap, saving more memory compared to the upstream version
  * Extend selective recomputation with finer granularity: `mlp_act`, `router_expert`, and `pre_mlp`
  * Fix compatibility between MFSDP and precision-aware optimizer
  * Fix MFSDP parameter sharding edge case that could cause training errors
  * Fix gradient retrieval in MFSDP with precision-aware optimizer
  * Support optimizer offload when `--fp8-param-gather` is enabled
  * Fix precision issues when resuming training from a checkpoint with BF16 and optimizer offload enabled
  * Support BF16 precision optimizer storing BF16 checkpoints
  * Integrate DeepSpeed CPUAdam kernel to accelerate optimizer offload parameter updates (disable via `--no-use-deepspeed-cpu-adam`)
  * Cherry-pick Muon Optimizer support
  * Add detailed time logger that prints timing logs every 20 steps by default
  * Add MoE memory monitoring with log output

* **TE 2.9:**
  * Support customizable FP8 quantization `amax_eps` via environment variables: `FP8_QUANT_FWD_INP_AMAX_EPS`, `FP8_QUANT_FWD_WEIGHT_AMAX_EPS`, `FP8_QUANT_BWD_GRAD_AMAX_EPS`
  * Enhance BF16 precision optimizer performance with memory buffer

# Planned Upgrades

* Megatron 0.15 → Megatron 0.16
* TE 2.9 → TE 2.12