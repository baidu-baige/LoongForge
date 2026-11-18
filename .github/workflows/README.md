# GitHub Actions Workflows

This directory contains the CI/CD workflows for LoongForge.

## Workflows

| Workflow | Trigger | Purpose |
|---|---|---|
| `static-checks.yml` | PR + workflow dispatch | Run and summarize all blocking CPU checks |
| `workflow-lint.yml` | Reusable workflow | Validate GitHub Actions expressions, YAML, and CI helper contracts |
| `pr-title.yml` | Reusable workflow | Validate PR title format (Conventional Commits preferred; legacy module format supported) |
| `license.yml` | Reusable workflow | Check SPDX Apache-2.0 header on newly added source files using language-specific comment styles |
| `doc-links.yml` | Reusable workflow | Check Markdown references to repository `examples/` and `configs/` paths |
| `secrets.yml` | Reusable workflow | Scan PR commits for leaked secrets via gitleaks |
| `lint.yml` | Reusable workflow | Run Ruff on changed Python files |
| `sensitive.yml` | Reusable workflow | Scan tracked files for internal identifiers via `.github/scripts/sensitive_scan.py` |
| `build.yml` | Reusable workflow | Build sdist + wheel and run Python 3.12 import smoke |
| `submodule-sync.yml` | repository dispatch / workflow dispatch + manual | Sync `third_party/Loong-Megatron` to its tracked branch and push the submodule pointer update |
| `auto-label.yml` | Issue/PR open/edit | Auto-label issues and PRs by keyword matching |
| `ok-to-test.yml` | Maintainer issue comment | Validate and dispatch `/ok-to-test` GPU regression |
| `gpu-regression.yml` | Workflow dispatch | Run exact-SHA baseline regression for one model suite |
| `gpu-invalidate.yml` | PR synchronize | Cancel stale suite GPU work after a new commit |
| `gpu-watchdog.yml` | GPU workflow completion | Finalize a check if cancellation bypasses normal cleanup |
| `release.yml` | Workflow dispatch | Build, regress, and publish the operator-selected release image |

Pull requests run `static-checks.yml` automatically. Its manual dispatch is diagnostic:
PR-only checks skip where appropriate and the final job is named
`manual-static-checks`, so it cannot satisfy the required PR `static-checks` check.
Operational workflows such as `submodule-sync.yml` may retain their own
dispatch inputs.

## GPU Regression

Maintainers can request a baseline regression by commenting on a pull request:

```
/ok-to-test --suite llm_vlm|embodied [--model model1,model2] [--build-image]
```

The suite selects both the test collection and its self-hosted runner:
`llm_vlm` runs on a and `embodied` runs on p. The `embodied` suite is enabled
by default; `llm_vlm` requires a registered runner and
`CI_ENABLE_LLM_VLM=true` in the workflow environment. With `--build-image`, that
same runner builds the PR source context with a trusted Dockerfile and the
runner-configured BuildKit APT, PyPI, and source mirrors, then immediately runs
regression against the local candidate image. The Dockerfile itself is
operator-managed; the PR contributes the source context. Without it, regression
uses the runner's configured default image. Explicit models must belong to the selected suite and have a
baseline. New commits invalidate previous results.
The PR `gpu-regression` check reports whether GPU work is queued, building a candidate
image, running regression, cancelled, passed, or failed. A new PR commit
cancels the older running or queued job for each suite, including an in-flight
candidate build or regression. Runner cleanup is targeted and best-effort:
temporary build contexts, regression containers, and candidate image tags are
removed where possible, while shared BuildKit caches are not globally pruned.
The watchdog observes cancelled or failed workflow runs and finalizes the
matching in-progress check by immutable PR head and suite when the normal
`finalize` job was itself cancelled.
The workflow points `LOONGFORGE_REGRESSION_RUNNER` at the trusted base-branch
checkout; direct execution of a PR-provided runner hook is rejected inside
GitHub Actions.

## Releases

The Release workflow is started by `workflow_dispatch`; the internal Portal is
the intended operator entry point. Pushing a Git tag never starts it. The
Formal Portal flow first creates a `v<version>` tag for a selected master
commit and dispatches that tag. Production publishing is accepted only from
that matching tag; the workflow verifies that the tag version matches
`pyproject.toml` and that the tagged commit is part of `master`.
Build or regression failure exits before any registry push. Debug requests use
a selected commit and a mandatory `_ci*` tag suffix; dry-run requests validate
inputs without executing stages. The
checkout includes tracked submodules and uses the
same trusted builder, automatically detected GPU target, runner-local mirrors,
proxy handling, BuildKit secrets, redacted logs, and image policy as a
candidate build.

Formal versions use strict SemVer tags: `vX.Y.Z` for stable releases and
`vX.Y.Z-rc.N` for release candidates. Choose the next version from the highest
existing upstream tag: a small fix or feature set is a patch bump, a larger
feature set is a minor bump, and a release candidate may be used when the
release needs staged validation. The version bump lives in its own release PR
that updates `project.version` in `pyproject.toml`; it is not mixed into a
feature or CI PR. GitHub Release notes are the changelog, so the repository
does not maintain a separate `CHANGELOG.md`.

When enabled, the release job creates one container, mounts both the LLM and
VLA artifact roots, and runs `deepseek_v2_lite` followed by `pi05_ddp` in that
container. The `RUN_RELEASE_REGRESSION` repository variable must be `true` or
`false`; set it to `true` when the GPU runner is available. When it is enabled,
the internal registry push happens only after both regressions pass. The
internal image tag is formed from the runner-local
`LOONGFORGE_RELEASE_IMAGE_TAG_PREFIX`, the validated version, and the
`_release` suffix. This keeps the tag tied to the actual Dockerfile base image.

Regression evidence includes numeric metrics and baselines from the tested
source checkout in `regression-metrics.json`. Training phases are kept separate,
and output directories include the run attempt so a retry cannot reuse stale
metrics. Evidence is collected on failed runs too and retained for 90 days;
the internal Portal renders actual-versus-baseline charts from these records.

After the internal push, the release runner invokes Docker over SSH on the
operator-configured publisher. That daemon pulls the exact internal image,
retags it as `<DOCKERHUB_IMAGE>:<version>`, and pushes it. Verification checks
the published manifest's image configuration against the pulled image and
its registry digest against the digest returned by push. Each stage reports
progress and elapsed time. The publisher does not run a GitHub runner.
Passwords are sent to Docker over stdin using a temporary Docker configuration.
Post-push verification uses Python 3 and the authenticated Registry API on the
publisher, independently of its Docker CLI version. It checks the tag's manifest
hash and image configuration, retrying transient HTTP or network failures up to
four times with bounded timeouts; authentication and content mismatches fail
immediately. Verification retries do not repeat the pull or push.
Registry login retries temporary network failures up to three times, waiting
two and four seconds between attempts; authentication failures stop immediately.
Reports distinguish `login_iregistry` from `login_dockerhub`. A redacted
`dockerhub-promotion.log` and the result report are archived as runner-local
evidence for the Portal; only the result report is uploaded to GitHub.
Only the newly created local destination tag and temporary credentials are
removed when promotion finishes; existing images and logins are preserved.
The workflow never moves Docker Hub's `latest` tag. A request can build and
regress without publishing, or publish when the
corresponding stage switches are enabled.

Debug runs retain their published internal registry tags. Only Docker Hub
debug tags are deleted automatically, using its tag-specific API; the release
workflow never deletes internal registry manifests. Formal release tags remain
in both registries. The 90-day evidence retention does not delete image tags.

When a formal request enables every stage and build, regression, internal
publication, and Docker Hub promotion all succeed, the workflow creates the
matching GitHub Release with generated notes. `rc` tags are marked as
pre-releases. The job is skipped for debug, dry-run, partial-stage, or failed
requests.

The four stage-switch fallbacks are required repository variables and must be
`true` or `false`: `BUILD_RELEASE_IMAGE`, `RUN_RELEASE_REGRESSION`,
`PUBLISH_TO_INTERNAL_REGISTRY`, and `PUBLISH_TO_DOCKERHUB`. Docker Hub
promotion is skipped automatically when internal registry publishing is
disabled, because there is no validated internal image to promote.

The optional `execution_slot` input must be `primary` or `secondary`; when it
is omitted the workflow uses `RELEASE_EXECUTION_SLOT`. Slots are resolved on
the Control Runner from runner-local configuration, so GitHub receives only
the stable public identifier.

The optional `llm_model` and `vla_model` inputs select the regression models.
When omitted they use `RELEASE_LLM_MODEL` and `RELEASE_VLA_MODEL`; the workflow
rejects names that do not exist in LoongForge's test definitions.

The optional `baseline_chip` input selects the baseline directory for both
regression suites; omitted inputs read `RELEASE_BASELINE_CHIP`. It is required
when regression is enabled. This selects comparison data, not GPU allocation.
LLM/VLM baselines live under `tests/llm_vlm/baseline/{default,optional}/<chip>/`;
embodied baselines live under `tests/embodied/baseline/<chip>/`. Missing selected
baselines fail regression without falling back to a different chip.
The P-chip DeepSeek V2 Lite baseline contains all 20 pretrain iterations from
debug run `38675600445`, source commit
`33295fc292f675a6c03634dd0b644290e192a302`. It preserves actual metrics,
including the first iteration's warmup time. No SFT baseline is implied.

The Release workflow does not publish to PyPI. If package publication is added
later, it should become an explicit stage switch rather than a Git-tag side
effect.

The image policy rejects `bcecmd`, common credential files, AK/SK and signed
authorization metadata, and internal endpoints or runner paths before a
candidate is run or a release is pushed. Its text scan excludes the copied
project source tree, which is independently covered by the repository
sensitive scan; prohibited executable and credential-file checks cover the
whole image.

Operator hook contracts:

- `LOONGFORGE_REGRESSION_RUNNER --source DIR --suite llm_vlm|embodied --sha SHA [--model LIST] [--candidate-revision REV]`
- `LOONGFORGE_IMAGE_BUILDER --source DIR --target a|p|auto --sha SHA --pr NUMBER --tree-sha SHA`; stdout must contain only the local candidate image reference
- `run_release_regression.sh --image IMAGE --source DIR --sha SHA --baseline-chip CHIP
  [--llm-model MODEL] [--vla-model MODEL]`; the release
  runner creates one container and runs the two release models serially. It
  also routes execution through the selected Execution Slot and synchronizes
  portable evidence back to the GitHub job.

The builder reads `CI_CONFIG_PATH_IMAGE` (or the wrapper's `CI_CONFIG_PATH`) from
the selected suite runner. It uses the Dockerfile's `BASE_IMAGE` build argument,
proxy settings without embedded credentials, and the configured BuildKit
secrets for APT, PyPI, and pinned source mirrors. Candidate images are tagged
locally with the PR number, head SHA, tree SHA, target, and candidate revision;
this workflow never pushes or promotes them.

The trusted wrapper normally passes `auto`. Before building, the selected
self-hosted runner is queried with `nvidia-smi`: compute capability 8.x maps to
the A-card/Ampere image and 10.x, 11.x, or 12.x maps to the P-card/Blackwell
image. Unknown or mixed GPU architectures fail closed, and an explicit `a` or
`p` target is rejected when it does not match the runner.

Repository Variables are configured in GitHub Settings. The workflow expects
`CI_RUNNER_A`, `CI_RUNNER_P`, `CI_REVIEW_RUNNER`, `CI_RELEASE_RUNNER`, and
`CLAUDE_REVIEW_MODEL`; release requests also require
`BUILD_RELEASE_IMAGE`, `RUN_RELEASE_REGRESSION`,
`PUBLISH_TO_INTERNAL_REGISTRY`, and `PUBLISH_TO_DOCKERHUB`.
`CLAUDE_CODE_EXECUTABLE` and `DOCKERHUB_IMAGE` are optional. Store
`INTERNAL_IMAGE_REPOSITORY` as a secret so GitHub masks the internal registry
reference in job logs. Every self-hosted
runner provides its own
`CI_CONFIG_PATH_IMAGE` environment variable; this is not a Repository Variable
because runner filesystem roots may differ.
The protected release runner must set
`LOONGFORGE_ALLOW_RELEASE_IMAGE_BUILD=true` in that config.

GitHub prints self-hosted runner metadata before the first workflow step.
Register the production runner with a non-sensitive name and run it on a host
or container whose hostname is also non-sensitive.
Before running a self-hosted release or GPU regression job, configure the
repository-level `RUNNER_PRIVATE_LOG_MASKS` secret in both the upstream and
fork repositories.
Store the exact private proxy URL and its hostname on separate lines, without
adding these values to source files or repository variables. Every self-hosted
release or GPU regression job references this secret in its job environment,
including debug cleanup, which does not use a deployment environment. The runner
registers
secret values and their individual lines before printing `Set up job`, so the
startup proxy endpoint is masked too. A workflow step or job-started hook runs
too late to protect these messages; the existing metadata-mask step still
protects values loaded by later steps.
Update the secret before changing a runner's proxy, and validate a small
self-hosted job's startup log before running a release. Release dry-run skips
self-hosted execution and cannot validate startup masking. Existing public
logs retain their original content and need separate cleanup.

Hooks must return nonzero on failure and must not print credentials, signed
source URLs, runner-local filesystem paths, or physical accelerator details to
the GitHub Actions log.

## PR Title Convention

Conventional Commits are preferred:

```
<type>(<scope>): <description>
<type>: <description>
<type>(<scope>)!: <description>
```

The scope accepts letters, numbers, `.`, `_`, `-`, and `/`. It is intentionally
not restricted to the module list so existing scopes such as `fastwam`, `dsa`,
and `pi05` remain valid.

The original module-prefixed format remains supported:

```
[<modules>] <type>: <description>
[BREAKING][<modules>] <type>: <description>
```

**Modules:** `llm, vlm, vla, diffusion, train, data, ops, ckpt, peft, docker, xpu, ci, docs, tests, scripts, release`

**Types:** `feat, fix, refactor, perf, docs, test, chore, ci`

**Examples:** `feat(ckpt): support Qwen3-Next checkpoint conversion` or
`[llm, ckpt] feat: support Qwen3-Next checkpoint conversion`

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

The GitHub App behind those secrets must be able to push to the configured target branch. This workflow is separate from the PR `Static Checks` and is not a required merge check.
