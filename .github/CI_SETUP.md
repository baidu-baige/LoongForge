# CI repository setup

Repository files define the workflows, but the following settings must be
created by an organization or repository administrator.

## Repository Variables

Configure the following Repository Variables in GitHub Settings. Runner values
are JSON arrays, not comma-separated strings. Set the custom labels to the
labels actually registered on the target runners; the `a`/`p` labels in
runner-local configuration are canonical aliases,
not assumptions made by the workflows. `llm_vlm` resolves to `CI_RUNNER_A` and
`embodied` resolves to `CI_RUNNER_P`; each suite runner builds its own local
candidate image when `--build-image` is requested.

- `CI_RUNNER_A`: JSON label array for the A-card regression runner.
- `CI_RUNNER_P`: JSON label array for the P-card regression runner.
- `CI_REVIEW_RUNNER`: JSON label array for the Claude review runner.
- `CI_RELEASE_RUNNER`: JSON label array for a GPU Docker runner with at least
  250 GiB free on Docker's storage filesystem. Its GPU target determines the
  release image architecture in the same way as a candidate build.
- `CI_ENABLE_LLM_VLM`: `true` only when the A-card suite is provisioned;
  otherwise `llm_vlm` dispatch is intentionally rejected.

Configure these Claude review repository variables:

- `CLAUDE_REVIEW_MODEL` is required and must be a model ID accepted by the
  configured `ANTHROPIC_BASE_URL`.
- `CLAUDE_CODE_EXECUTABLE` may point to a pinned Claude Code binary already
  installed on every selected review runner. When set, the action skips its
  network installation step. Leave it empty to use the action's automatic
  installation.

When a review fails, the workflow prints only SDK result metadata
(`api_error_status`, `terminal_reason`, turn count, and timing). The full
`claude-execution-output.json` contains model/tool content and remains in the
runner temp directory only; do not upload it or print it to public logs.

Each self-hosted runner must provide `CI_CONFIG_PATH_IMAGE` in its service
environment, pointing at that machine's private `image.env`. The service
environment and that file together must provide the values shown in
`.github/ci-config.example.env`, including the default image and isolated
mounts and BuildKit mirror paths. Do not put registry credentials or signed
source URLs in the repository; keep them in the operator-managed runner
configuration only. The candidate-image builder passes proxy settings without
embedded credentials and uses the configured APT, PyPI, and source-manifest
files as ephemeral BuildKit secrets.

The protected release runner must also provide `LOONGFORGE_RELEASE_CONFIG`
and `LOONGFORGE_RUNNER_MASK_SCRIPT`; `LOONGFORGE_RUNNER_MASK_VALUES` may point
to an additional newline-delimited private-literal list. The mask script path
points to an operator-controlled copy of `mask_runner_metadata.sh` outside the checkout.
It runs before checkout or any third-party Action and registers the runner
name, hostname, private config paths and values, proxy settings, and workspace
paths with GitHub's log masker. Release Docker command output passes through
the repository redactor as a second layer; raw Docker output must not be added
to this workflow.

GitHub emits self-hosted runner metadata before any workflow step. Register the
production runner with a non-sensitive name and use a host or container whose
hostname is also non-sensitive; otherwise `Runner name` and `Machine name` are
visible in public job logs before the mask script can run. If the runner proxy
line must also remain private, keep the executing workflow in a private runner
repository.

Each suite runner must also provide a working Docker Buildx plugin. The PR
checkout is the build context and the builder selects an operator-managed
trusted Dockerfile; configured BuildKit secrets are mounted for its
installation steps. An `IMAGE_DOCKERFILE` override must also be
operator-managed. Verify Buildx with `docker buildx version`;
installing the CLI plugin does not require restarting the Docker daemon.
The release job builds with the daemon's own BuildKit through
`DOCKER_BUILDKIT=1 docker build --network host`; this allows the isolated
release daemon to run with `bridge=none`.
The workflow deliberately does not create a
`docker-container` builder, because bootstrapping one pulls a builder image
from Docker Hub and an internal-only runner cannot reach it.

The builder forwards the runner's proxy settings into the image build, so the
runner's `NO_PROXY` must list every internal APT and PyPI mirror host. A mirror
that is only reachable directly fails with a gateway error when the build
sends it through the proxy.
The release runner uses this same builder and runner-local image config. Set
`LOONGFORGE_ALLOW_RELEASE_IMAGE_BUILD=true` only on the protected runner named
by `CI_RELEASE_RUNNER`; keep it false on general-purpose runners. Set
`LOONGFORGE_RELEASE_IMAGE_TAG_PREFIX` to a string that describes the actual
operator-managed `IMAGE_BASE_IMAGE`. The current protected base image is
Ubuntu 24.04, CUDA 13.3, PyTorch 2.13.0a0, and Python 3.12, so its prefix is
`ubuntu24.04-cu13.3-torch2.13.0a0-py312`.

Set the repository variable `RUN_RELEASE_REGRESSION=true` only when the GPU
runner is available for the two model regressions; set it to `false` while the
GPU runner is occupied.

The unified Release workflow also accepts per-request Stage Switch inputs:
`build_image`, `run_regression`, `publish_internal`, and `publish_dockerhub`.
When an input is omitted, the workflow reads the matching repository variable:
`BUILD_RELEASE_IMAGE`, `RUN_RELEASE_REGRESSION`,
`PUBLISH_TO_INTERNAL_REGISTRY`, or `PUBLISH_TO_DOCKERHUB`. The Release workflow
is started only by `workflow_dispatch`; pushing a Git tag never starts it.

Use strict SemVer for formal tags: `vX.Y.Z`, or `vX.Y.Z-rc.N` for a release
candidate. Prepare each formal release with a dedicated version-bump PR that
updates `project.version`; the Portal then creates the immutable matching tag
on the selected `master` commit. Small releases use a patch bump, larger
feature sets use a minor bump, and staged validation may use an `rc` tag.
GitHub Release notes are the changelog; do not add a root `CHANGELOG.md`.

Set `RELEASE_EXECUTION_SLOT` to `primary` or `secondary` as the fallback for
requests that omit `execution_slot`. The Control Runner resolves both slots
from an operator-managed, mode-`0600`
`LOONGFORGE_EXECUTION_SLOT_CONFIG` file. Public workflow inputs and GitHub
logs contain only the slot name; SSH aliases, Docker hosts, source roots, and
execution roots remain runner-local. The `primary` slot executes locally; the
`secondary` slot synchronizes source over SSH, executes regression on the
remote host, and returns only portable release evidence.

Set `RELEASE_LLM_MODEL` and `RELEASE_VLA_MODEL` to the fallback model names for
requests that omit `llm_model` or `vla_model`. The workflow validates both
selections against LoongForge's own test definitions before execution.

Set `RELEASE_BASELINE_CHIP` to the fallback baseline directory identifier for
requests that omit `baseline_chip`. Regression requires an explicit input or
this variable; there is no workflow default. The Portal sends its selected
chip on every regression request. Both suites receive the same `--chip`,
including when executed over SSH, and evidence collection reads that chip's baseline from
the tested checkout. Missing model or training-phase baselines fail regression;
the release never falls back to another chip or collects a new baseline.
Build-only requests do not require a chip.

`debug_tag_suffix` is an explicit per-request input. Debug Full Release uses
`_ci_test`; production requests leave it empty. The workflow does not infer a
suffix from runner-local configuration.

Published internal debug image tags are retained for registry-side or manual
management. Automatic debug cleanup removes only the Docker Hub tag; it never
deletes internal registry manifests. Image tags are separate from the 90-day
release evidence retention.

The release runner also needs `LOONGFORGE_DOCKERHUB_PROXY` and
`LOONGFORGE_DOCKERHUB_NO_PROXY` in its operator-managed release configuration.
They are used only by the Docker Hub cleanup step so temporary debug tags can be
removed through the publisher network path without exposing registry credentials
or internal proxy details in this repository.

Candidate builds require 250 GiB free by default on the filesystem containing
Docker's `DockerRootDir`. Set `LOONGFORGE_MIN_DOCKER_FREE_GB` to another integer
in the runner config, or set it explicitly to an empty value to disable the
check. `LOONGFORGE_MIN_FREE_GPU_MB` follows the same unset/default and
empty/disabled convention. GPU query errors fail closed and logs do not expose
device indices, counts, memory sizes, or runner paths.

Remove obsolete `INTERNAL_CI_*` variables after the old internal CI workflow
has been retired. They are not consumed by the current workflows and may
expose runner-local paths through the public repository configuration.

Create protected `iregistry-release` and `dockerhub-release`
Environments for the release workflow. They are approval gates; registry
credentials remain on the protected release runner and are not stored in
GitHub. Releases push the validated internal image first, then the same
release runner drives the publisher Docker daemon over SSH and pushes only
`<DOCKERHUB_IMAGE>:<version>`; the `latest` tag is never moved. A manual
`workflow_dispatch` can build, scan, regress, and publish when the
corresponding Stage Switches are enabled.
Production publishing requests that enable `publish_internal` or
`publish_dockerhub` without a debug suffix must still be dispatched from
`master`. Debug Full Release requests carry a non-production suffix and may be
dispatched from a non-production branch.

The Release workflow does not publish to PyPI. If package publication is added
later, it should become an explicit stage switch so it follows the same
operator-controlled request model as the image stages.

Both candidate and release images pass the trusted image policy before use or
push. It rejects `bcecmd`, common cloud credential files, persisted source
manifests, AK/SK or signed-authorization metadata, and internal endpoints or
runner paths. The filesystem text scan excludes the copied
`/workspace/LoongForge` source tree because that tree is already covered by the
repository sensitive scan and contains historical documentation fixtures. It
also excludes Docker's runtime-generated host, hostname, and resolver files;
file-name checks for prohibited executables and credential files still cover
the whole image.

The submodule sync workflow uses a GitHub App token so it can push through the
repository's protected branch rules. Configure `SUBMODULE_SYNC_APP_ID` and
`SUBMODULE_SYNC_APP_PRIVATE_KEY` as repository secrets, and install that App on
this repository with permission to write contents.

## Runner labels

- A-card regression runner: `self-hosted`, plus the registered A-card custom
  label (the example alias is `a`).
- P-card regression runner: `self-hosted`, plus the registered P-card custom
  label (the example alias is `p`).
- Candidate images are built on the same suite runner that performs regression.
  The trusted image wrapper detects the runner target from `nvidia-smi`: compute
  capability 8.x selects A/Ampere, while 10.x, 11.x, or 12.x selects
  P/Blackwell. Mixed or unknown GPU architectures fail closed.

## First activation

Enable the `ok-to-test`, `gpu-regression`, `gpu-invalidate`, and
`gpu-watchdog` workflows on the default branch, then verify a
maintainer-dispatched run on each labeled suite
runner. Confirm that the PR `gpu-regression` check progresses through queued, optional
candidate build, regression, and final status. Push a second commit while a GPU
run is active and confirm that the old check becomes cancelled and the runner's
targeted cleanup removes its regression container and candidate image where
possible. Cancellation cleanup is best-effort and must not globally prune the
shared BuildKit cache.

Configure branch protection to require the PR job named `static-checks`. A manual
dispatch ends in `manual-static-checks` and is intended only for diagnostics; it must
not satisfy the PR requirement.

Create the `baidu-baige/loongforge-maintainers` Team before activating the
`.github/CODEOWNERS` rule. The repository should use a `master-protection`
Ruleset requiring one non-author Code Owner approval, resolved conversations,
an up-to-date branch, and the `static-checks` check. Use a separate
`release-tag-protection` Ruleset for immutable `v*` tags.

Keep the default `GITHUB_TOKEN` permission read-only and require approval for
first-time fork workflows. Because this repository uses self-hosted runners,
only trusted dispatch workflows should reach those runners. The current
repository allows all third-party Actions and does not require SHA pinning;
enable mandatory pinning only after every `@vN` reference has been replaced by
an audited full commit SHA.

Configure these repository variables:

- `BUILD_RELEASE_IMAGE`, required: `true` builds a new release image; `false`
  reuses the image named by the request.
- `DOCKERHUB_IMAGE`, default `docker.io/loongforge/loongforge`.
- `RUN_RELEASE_REGRESSION`, required: `true` enables the GPU release
  regressions; `false` skips them.
- `PUBLISH_TO_INTERNAL_REGISTRY`, required: `true` pushes a validated
  release image to the internal registry; `false` skips that push.
- `PUBLISH_TO_DOCKERHUB`, required: `true` promotes the validated internal
  image to Docker Hub; `false` keeps it only in the internal registry. Docker
  Hub promotion also requires internal registry publishing to be enabled.

Configure `INTERNAL_IMAGE_REPOSITORY` as a repository secret, not a variable.
It contains the full internal image repository without a tag; the login host is
derived from it. Keeping it as a secret lets GitHub mask the reference if it is
accidentally printed in a job environment block.

The release workflow builds one image and creates one container on
`CI_RELEASE_RUNNER`. It runs `deepseek_v2_lite` and then `pi05_ddp` in that
same container when `RUN_RELEASE_REGRESSION=true`. While that repository
variable is false, image build and registry promotion continue without the GPU
regression step.

The release runner's operator-managed release directory contains
one non-secret configuration file and one credential file for each registry.

The non-secret configuration file contains an SSH Docker client destination
and its matching SSH alias. The credential files hold the push-capable
internal registry account and the Docker Hub login name plus Docker Hub PAT,
and the runner service environment points at both of them.
The public Docker Hub image name stays in a repository variable, while the
internal image repository stays in the secret described above.
All three files must be runner-local, mode `0600`, and excluded from version
control. The workflow uses a per-job Docker config directory and removes it
during cleanup.

The GitHub Release job runs only after a formal tag request enables every
stage and build, regression, internal publication, and Docker Hub promotion
succeed. It receives a job-local `contents: write` permission and creates a
GitHub Release with generated notes; `rc` tags are marked pre-release. No other
release job receives write access to repository contents.

The release runner needs Docker, read access to the internal registry, and
write access to Docker Hub through the SSH-selected daemon. The publishing
host needs at least 40 GiB free on the filesystem holding its
`DockerRootDir`, but it does not need a GitHub runner or a checkout.
