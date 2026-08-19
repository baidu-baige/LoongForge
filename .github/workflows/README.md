# GitHub Actions Workflows

This directory contains the CI/CD workflows for LoongForge.

## Workflows

| Workflow | Trigger | Purpose |
|---|---|---|
| `ci-gate.yml` | PR + workflow dispatch | Run and summarize all blocking CPU checks |
| `workflow-lint.yml` | Reusable workflow | Validate GitHub Actions expressions and YAML |
| `pr-title.yml` | Reusable workflow | Validate PR title format: `[<modules>] <type>: <description>` |
| `license.yml` | Reusable workflow | Check SPDX Apache-2.0 header on newly added source files |
| `secrets.yml` | Reusable workflow | Scan PR commits for leaked secrets via gitleaks |
| `lint.yml` | Reusable workflow | Run Ruff on changed Python files |
| `build.yml` | Reusable workflow | Build sdist + wheel and run Python 3.12 import smoke |
| `submodule-sync.yml` | repository dispatch / workflow dispatch + manual | Sync `third_party/Loong-Megatron` to its tracked branch and push the submodule pointer update |
| `auto-label.yml` | Issue/PR open/edit | Auto-label issues and PRs by keyword matching |
| `issue-notify.yml` | Issue opened | Notify Ruliu group when a new issue is opened |
| `ok-to-test.yml` | Maintainer issue comment | Validate and dispatch `/ok-to-test` GPU regression |
| `gpu-regression.yml` | Workflow dispatch | Run exact-SHA baseline regression on logical environments |
| `internal-image-update.yml` | Merged PR + manual | Validate and promote candidate internal images |
| `release.yml` | Version tag `vX.Y.Z` | Publish PyPI package and Docker Hub release image |

The CPU checks are run manually through `ci-gate.yml` when needed. Operational workflows such as `submodule-sync.yml` may retain their own dispatch inputs.

## GPU Regression

Maintainers can request a baseline regression by commenting on a pull request:

```
/ok-to-test --env a|p|all [--model model1,model2] [--build-image a|h|p]
```

The public environment names are logical aliases. Protected variables provide
runner mappings and opaque configuration profile names; machine-specific values
are resolved by operator-managed hooks on each runner.
The default CI run currently uses the known-good `deepseek_v2_lite` baseline;
explicit models must also have a baseline. New commits invalidate previous results.
The self-hosted runner must provide `LOONGFORGE_REGRESSION_RUNNER`; direct execution
of PR-provided test scripts is rejected when running inside GitHub Actions.

Operator hook contracts:

- `LOONGFORGE_REGRESSION_RUNNER --source DIR --env a|p --sha SHA [--config-profile PROFILE] [--model LIST] [--candidate-revision REV]`
- `LOONGFORGE_IMAGE_BUILDER --source DIR --target a|h|p --sha SHA --pr NUMBER --tree-sha SHA`; stdout must contain only the candidate image reference
- `LOONGFORGE_IMAGE_PROMOTER --target a|h|p --pr NUMBER --head-sha SHA --merge-sha SHA --tree-sha SHA`

Image hooks read `CI_CONFIG_PATH_IMAGE` (or the wrapper's `CI_CONFIG_PATH`) from
the image runner. The config defines `IMAGE_REGISTRY`, `IMAGE_REPOSITORY_A/H/P`
and `IMAGE_DEFAULT_TAG_A/H/P`; registry credentials are supplied only through
the protected `internal-registry` environment. The builder labels each image
with the PR number, head SHA, tree SHA, target, and candidate revision before
push. The promoter validates those labels before updating the default tag.

Repository Variables are listed in `.github/ci-variables.example.json`, and the
image-to-runner contract is defined in `.github/image-mapping.yml`. Each image
runner provides its own `CI_CONFIG_PATH_IMAGE` environment variable; this is not
a Repository Variable because runner filesystem roots may differ. Create a
protected `internal-registry` Environment with required reviewer approval and
the `INTERNAL_REGISTRY_USERNAME` and `INTERNAL_REGISTRY_PASSWORD` secrets. The
operator-managed image config must follow the same schema on every image runner.

The complete image lifecycle can be exercised without GitHub Actions:

```bash
.github/scripts/self_runner/run_image_lifecycle.sh \
  --source "$PWD" \
  --config /path/to/image.env \
  --pr 151 \
  --target a \
  --promote
```

For a lightweight plumbing test, set `IMAGE_DOCKERFILE` to
`.github/scripts/self_runner/fixtures/Dockerfile`, `IMAGE_PUSH=false`, and
`IMAGE_LOCAL_ONLY=true`. Remove those three overrides to build and push the
production Dockerfile.

On the current A800 runner, the production preparation command is:

```bash
<runner-ci-root>/trigger_gpu_ci_local \
  --env a \
  --build-image a \
  --image-config <runner-ci-root>/image.env \
  --build-only \
  --pr 151
```

Omit `--build-only` only after the candidate push succeeds; that runs the DS
baseline against the returned candidate reference. Promotion is performed only
after merge by `internal-image-update.yml`.

Hooks must return nonzero on failure and must not print internal addresses, paths,
credentials, or physical accelerator details to the GitHub Actions log.

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
