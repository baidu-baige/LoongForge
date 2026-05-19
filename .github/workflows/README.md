# GitHub Actions Workflows

This directory contains the CI/CD workflows for LoongForge.

## Workflows

| Workflow | Trigger | Purpose |
|---|---|---|
| `pr-title.yml` | PR open/edit | Validate PR title format: `[<modules>] <type>: <description>` |
| `license.yml` | PR | Check SPDX Apache-2.0 header on newly added source files |
| `secrets.yml` | PR + push to master | Scan for leaked secrets via gitleaks |
| `build.yml` | PR + push to master | Build sdist + wheel on Python 3.10 / 3.12 |
| `auto-label.yml` | Issue/PR open/edit | Auto-label issues and PRs by keyword matching |

All workflows support `workflow_dispatch` for manual re-runs from the Actions UI (except `auto-label.yml`).

## PR Title Convention

```
[<modules>] <type>: <description>
```

**Modules:** `llm, vlm, vla, diffusion, train, data, ops, ckpt, peft, docker, xpu, ci, docs, tests, scripts, release`

**Types:** `feat, fix, refactor, perf, docs, test, chore, ci`

**Example:** `[llm, ckpt] feat: support Qwen3-Next checkpoint conversion`

## Adding a New Workflow

1. Create a `.yml` file in this directory.
2. Set `permissions` to least-privilege (default: `contents: read`).
3. Add a `concurrency` block to cancel stale runs on PR branches.
4. Test locally where possible before pushing.
