# LoongForge Test Suites

The `tests/` directory hosts three independent, self-contained test suites. Each owns its own
scripts, configs, and baselines — they share nothing except being rooted under `tests/`.

| Suite | Directory | Entry | Model targets | Baselines |
|---|---|---|---|---|
| **LLM/VLM E2E** (config-driven) | [tests/llm_vlm/](llm_vlm/) | `tests/llm_vlm/main_start.sh` | YAML scenarios under `configs/` + `optional_configs/` | `tests/llm_vlm/baseline/{default,optional}/<chip>/` |
| **Embodied VLA regression** (manifest-driven) | [tests/embodied/](embodied/) | `tests/embodied/run.sh` | `examples/embodied/*.sh` via `tests/embodied/config/scripts.yaml` | `tests/embodied/baseline/<chip>/` |
| **Mcore-Bridge roundtrip** (checkpoint correctness) | [tests/mcore_bridge/](mcore_bridge/) | `tests/mcore_bridge/<family>/*_bridge_roundtrip.sh` (driver: `hf_roundtrip_test.py`) | per-model roundtrip scripts by family | `roundtrip_comparison.json` (no stored baseline) |

See each suite's own README for usage:
- [tests/llm_vlm/README.md](llm_vlm/README.md)
- [tests/embodied/README.md](embodied/README.md)
- Mcore-Bridge: [tools/mcore_bridge/README.md](../tools/mcore_bridge/README.md) and `docs/source/features/mcore_bridge.md`
