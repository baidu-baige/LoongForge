# LoongForge Test Suites

The `tests/` directory hosts three independent, self-contained test suites. Each owns its own
scripts, configs, and baselines — they share nothing except being rooted under `tests/`.

The top level of `tests/` also holds standalone unit tests (`tests/test_*.py`), in two groups:

| Group | Files | Under test | How it runs |
|---|---|---|---|
| Repository rules and component units | the other `tests/test_*.py` | layout and dependency rules, model configs, data builders, dispatch | `python -m pytest tests/test_*.py`; `test_dp_balance_layout.py` needs multiple GPUs |
| CI and release contract tests | `test_ci_*.py`, `test_ci_helpers.js`, `test_release_*.py`, `test_sensitive_scan.py`, `test_doc_links.py`, `test_image_policy.py` | `ci/`, `.github/scripts/`, `docker/` | `.github/workflows/workflow-lint.yml` runs them on a CPU runner without torch; the JS file uses `node --test` |

CI contract tests stay in `tests/` on purpose: `tests/` is the single pytest root, and the workflow
lists the files it runs explicitly, so being next to the code they exercise adds nothing.

| Suite | Directory | Entry | Model targets | Baselines |
|---|---|---|---|---|
| **LLM/VLM E2E** (config-driven) | [tests/llm_vlm/](llm_vlm/) | `tests/llm_vlm/main_start.sh` | YAML scenarios under `configs/` + `optional_configs/` | `tests/llm_vlm/baseline/{default,optional}/<chip>/` |
| **Embodied VLA regression** (manifest-driven) | [tests/embodied/](embodied/) | `tests/embodied/run.sh` | `examples/*.sh` via `tests/embodied/config/scripts.yaml` | `tests/embodied/baseline/<chip>/` |
| **Mcore-Bridge roundtrip** (checkpoint correctness) | [tests/mcore_bridge_roundtrip/](mcore_bridge_roundtrip/) | `tests/mcore_bridge_roundtrip/<family>/*_bridge_roundtrip.sh` (driver: `hf_roundtrip_test.py`) | per-model roundtrip scripts by family | `roundtrip_comparison.json` (no stored baseline) |

See each suite's own README for usage:
- [tests/llm_vlm/README.md](llm_vlm/README.md)
- [tests/embodied/README.md](embodied/README.md)
- Mcore-Bridge: [tools/mcore_bridge/README.md](../tools/mcore_bridge/README.md) and `docs/source/features/mcore_bridge.md`
