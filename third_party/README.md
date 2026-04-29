# Third Party
* Loong-Megatron: a fork of Megatron-LM serving as the backend training engine for LoongForge, with in-depth customizations including performance optimization strategies, XPU support, bug fixes, and other functional enhancements.

## Maintenance Principles
Loong-Megatron tracks recent upstream release versions of NVIDIA/Megatron-LM. Each primary version is maintained on a dedicated loong-main/core_v* branch and updated promptly as the upstream evolves.

* Modifications to the original codebase are added or removed in response to upstream changes. When an equivalent feature becomes available upstream and meets expectations, we default to adopting the upstream implementation.
* Changes to upstream code should be made conservatively. Whenever possible, modifications are preferred in the upper-level codebase to reduce maintenance complexity.
* Patches should primarily consist of bug fixes and new features; existing upstream capabilities should not be removed by default. Critical changes should be considered for contribution back to the upstream repository.
