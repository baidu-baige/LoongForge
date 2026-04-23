# Maintenance Principles

* Patches are maintained against recent upstream release versions. Only one primary version is actively maintained on the main branch, and it is updated promptly as the upstream evolves.
* Modifications to the original codebase are added or removed in response to upstream changes. When an equivalent feature becomes available upstream and meets expectations, we default to adopting the upstream implementation.
* Changes to upstream code should be made conservatively. Whenever possible, modifications are preferred in the upper-level codebase to reduce maintenance complexity.
* Patches should primarily consist of bug fixes and new features; existing upstream capabilities should not be removed by default. Critical changes should be considered for contribution back to the upstream repository.

# Key Modifications (including but not limited to)

* **Megatron (Loong-Megatron):**
  All Megatron-LM modifications now live in the [Loong-Megatron](https://github.com/baidu-baige/Loong-Megatron) fork repository.
  See the Loong-Megatron README for a full list of changes.

* **TE 2.9:**
  * Support customizable FP8 quantization `amax_eps` via environment variables: `FP8_QUANT_FWD_INP_AMAX_EPS`, `FP8_QUANT_FWD_WEIGHT_AMAX_EPS`, `FP8_QUANT_BWD_GRAD_AMAX_EPS`
  * Enhance BF16 precision optimizer performance with memory buffer

# Planned Upgrades

* TE 2.9 → TE 2.12
