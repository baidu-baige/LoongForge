# CI repository setup

Repository files define the workflows, but the following settings must be
created by an organization or repository administrator.

## Repository Variables

Configure the following Repository Variables in GitHub Settings. Runner values
are JSON arrays, not comma-separated strings. Set the custom labels to the
labels actually registered on the target runners; the `a`/`p` labels in
runner-local configuration are canonical aliases,
not assumptions made by the workflows. `mcore` resolves to `CI_RUNNER_A` and
`torch` resolves to `CI_RUNNER_P`; each suite runner builds its own local
candidate image when `--build-image` is requested.

- `CI_RUNNER_A`: JSON label array for the A-card regression runner.
- `CI_RUNNER_P`: JSON label array for the P-card regression runner.
- `CI_REVIEW_RUNNER`: JSON label array for the Claude review runner.
- `CI_RELEASE_RUNNER`: JSON label array for a GPU Docker runner with at least
  250 GiB free on Docker's storage filesystem. Its GPU target determines the
  release image architecture in the same way as a candidate build.
- `CI_ENABLE_MCORE`: `true` only when the A-card suite is provisioned;
  otherwise `mcore` dispatch is intentionally rejected.

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

Each suite runner must also provide a working Docker Buildx plugin. The PR
checkout is the build context and the builder selects an operator-managed
trusted Dockerfile; configured BuildKit secrets are mounted for its
installation steps. An `IMAGE_DOCKERFILE` override must also be
operator-managed. Verify Buildx with `docker buildx version`;
installing the CLI plugin does not require restarting the Docker daemon.
The release runner uses this same builder and runner-local image config. Set
`LOONGFORGE_ALLOW_RELEASE_IMAGE_BUILD=true` only on the protected runner named
by `CI_RELEASE_RUNNER`; keep it false on general-purpose runners.

Candidate builds require 250 GiB free by default on the filesystem containing
Docker's `DockerRootDir`. Set `LOONGFORGE_MIN_DOCKER_FREE_GB` to another integer
in the runner config, or set it explicitly to an empty value to disable the
check. `LOONGFORGE_MIN_FREE_GPU_MB` follows the same unset/default and
empty/disabled convention. GPU query errors fail closed and logs do not expose
device indices, counts, memory sizes, or runner paths.

Remove obsolete `INTERNAL_CI_*` variables after the old internal CI workflow
has been retired. They are not consumed by the current workflows and may
expose runner-local paths through the public repository configuration.

Create protected `pypi-release` and `dockerhub-release` Environments for the
release workflow. Configure PyPI Trusted Publishing (OIDC) in the former, and
`DOCKERHUB_USERNAME`/`DOCKERHUB_TOKEN` secrets plus approval protection in the
latter. `DOCKERHUB_USERNAME` must identify a Docker Hub user or service account
with write access to the `loongforge` organization, and `DOCKERHUB_TOKEN` must
be a Personal Access Token rather than an account password. Releases push only
`docker.io/loongforge/loongforge:<version>` and never move Docker Hub's `latest`
tag. A manual `workflow_dispatch` builds and scans the package and image but
never logs in or pushes; it still uses the protected release environment and
may require its approval.

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

Docker Hub releases publish only the validated version tag. Do not configure
release automation that implicitly moves `latest`.

The repository currently has no post-merge workflow that promotes a validated
candidate image to the internal registry or updates the Runner default image.
Until that workflow is added, perform the promotion manually only after
checking the merged PR, required Checks, source tree SHA, and candidate image
revision; keep registry write credentials out of PR jobs.
