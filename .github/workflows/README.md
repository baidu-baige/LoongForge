# GitHub Actions Workflows

This directory contains the CI/CD workflows for LoongForge.

## Workflows

| Workflow | Trigger | Purpose |
|---|---|---|
| `ci-gate.yml` | PR + workflow dispatch | Run and summarize all blocking CPU checks |
| `pr-title.yml` | Reusable workflow | Validate PR title format: `[<modules>] <type>: <description>` |
| `license.yml` | Reusable workflow | Check SPDX Apache-2.0 header on newly added source files |
| `secrets.yml` | Reusable workflow | Scan PR commits for leaked secrets via gitleaks |
| `lint.yml` | Reusable workflow | Run Ruff on changed Python files |
| `build.yml` | Reusable workflow | Build sdist + wheel and run Python 3.12 import smoke |
| `submodule-sync.yml` | repository dispatch / workflow dispatch + manual | Sync `third_party/Loong-Megatron` to its tracked branch and push the submodule pointer update |
| `auto-label.yml` | Issue/PR open/edit | Auto-label issues and PRs by keyword matching |
| `issue-notify.yml` | Issue opened | Notify Ruliu group when a new issue is opened |

The CPU checks are run manually through `ci-gate.yml` when needed. Operational workflows such as `submodule-sync.yml` may retain their own dispatch inputs.

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

## Submodule Sync

`submodule-sync.yml` updates `third_party/Loong-Megatron` to the branch configured in `.gitmodules` and commits the submodule pointer when it changes.

The workflow defaults to `master`. It can also receive `submodule_repository` from workflow inputs or `repository_dispatch` payloads to test against a forked Loong-Megatron without changing `.gitmodules`.

Required secrets:

- `SUBMODULE_SYNC_APP_ID`
- `SUBMODULE_SYNC_APP_PRIVATE_KEY`

The GitHub App behind those secrets must be able to push to the configured target branch. This workflow is separate from the PR `CI Gate` and is not a required merge check.

## Ruliu Issue Notifications

`issue-notify.yml` sends a Markdown message to a Ruliu group when a new GitHub Issue is opened. It runs on the self-hosted Linux runner because the Ruliu webhook host is only reachable from the internal network.

Required secret:

- `RULIU_ISSUE_WEBHOOK`: Ruliu group robot webhook URL for issue notifications.
