# LoongForge Test Suites

The `tests/` directory hosts three independent, self-contained test suites. Each owns its own
scripts, configs, and baselines — they share nothing except being rooted under `tests/`.

The top level of `tests/` holds only the CI and release contract tests. Unit tests live with the suite
of the engine they exercise:

| Group | Location | Under test | How it runs |
|---|---|---|---|
| CI and release contract tests | `tests/test_*.py`, `tests/test_ci_helpers.js` | `ci/`, `.github/scripts/`, `docker/`, and the E2E harness | `.github/workflows/workflow-lint.yml` runs them on a CPU runner without torch; the JS file uses `node --test` |
| Megatron-side unit tests | `tests/llm_vlm/unit/` | models, data, and checkpoint conversion used by the Megatron engine | `python -m pytest tests/llm_vlm/unit`; needs torch and Megatron, some tests need a GPU. `e2e_dsa_sm80.py`, `parallel_precision_compare.py`, and `align_multimodal.py` are manual `torchrun` scripts |
| Torch-side unit tests | `tests/embodied/unit/`, `tests/embodied/evaluation/` | the Torch engine, embodied models, and embodied evaluation | `python -m pytest tests/embodied/unit tests/embodied/evaluation`; `unit/` needs torch, some tests need a GPU |

workflow-lint also runs the torch-free `tests/llm_vlm/unit/test_vlm_dataloader_read_order.py`.

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
