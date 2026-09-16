# LoongForge Test Suites

The `tests/` directory hosts two independent, self-contained test suites. Each owns its own
scripts, configs, and baselines — they share nothing except being rooted under `tests/`.

| Suite | Directory | Entry | Model targets | Baselines |
|---|---|---|---|---|
| **MCore E2E** (config-driven) | [tests/mcore/](mcore/) | `tests/mcore/main_start.sh` | YAML scenarios under `configs/` + `optional_configs/` | `tests/mcore/baseline/{default,optional}/<chip>/` |
| **Torch VLA regression** (manifest-driven) | [tests/torch/](torch/) | `tests/torch/run.sh` | `examples/{vla,world}/*.sh` via `tests/torch/config/scripts.yaml` | `tests/torch/baseline/<chip>/` |

See each suite's own README for usage:
- [tests/mcore/README.md](mcore/README.md)
- [tests/torch/README.md](torch/README.md)
