# LoongForge Test Suites

The `tests/` directory hosts two independent, self-contained test suites. Each owns its own
scripts, configs, and baselines — they share nothing except being rooted under `tests/`.

| Suite | Directory | Entry | Model targets | Baselines |
|---|---|---|---|---|
| **LLM/VLM E2E** (config-driven) | [tests/llm_vlm/](llm_vlm/) | `tests/llm_vlm/main_start.sh` | YAML scenarios under `configs/` + `optional_configs/` | `tests/llm_vlm/baseline/{default,optional}/<chip>/` |
| **Embodied VLA regression** (manifest-driven) | [tests/embodied/](embodied/) | `tests/embodied/run.sh` | `examples/embodied/*.sh` via `tests/embodied/config/scripts.yaml` | `tests/embodied/baseline/<chip>/` |

See each suite's own README for usage:
- [tests/llm_vlm/README.md](llm_vlm/README.md)
- [tests/embodied/README.md](embodied/README.md)
