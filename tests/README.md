# LoongForge Test Suites

The `tests/` directory hosts two independent, self-contained test suites. Each owns its own
scripts, configs, and baselines — they share nothing except being rooted under `tests/`.

| Suite | Directory | Entry | Model targets | Baselines |
|---|---|---|---|---|
| **LLM/VLM E2E** (config-driven) | [tests/llm_vlm/](llm_vlm/) | `tests/llm_vlm/main_start.sh` | YAML scenarios under `configs/` + `optional_configs/` | `tests/llm_vlm/baseline/{default,optional}/<chip>/` |
| **Native VLA regression** (manifest-driven) | [tests/native/](native/) | `tests/native/run.sh` | `examples/{vla,world}/*.sh` via `tests/native/config/scripts.yaml` | `tests/native/baseline/<chip>/` |

See each suite's own README for usage:
- [tests/llm_vlm/README.md](llm_vlm/README.md)
- [tests/native/README.md](native/README.md)
