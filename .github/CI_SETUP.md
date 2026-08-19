# CI repository setup

Repository files define the workflows, but the following settings must be
created by an organization or repository administrator.

## Repository Variables

Create every variable from `.github/ci-variables.example.json`. Replace the
`P6K` label only if the registered P6K runner uses a different custom label.
The values of `CI_RUNNER_A`, `CI_RUNNER_P`, and `CI_RUNNER_IMAGE` are JSON
arrays, not comma-separated strings.

## Protected environments

Create these protected environments:

- `internal-registry`: required maintainer approval; secrets
  `INTERNAL_REGISTRY_USERNAME` and `INTERNAL_REGISTRY_PASSWORD`.
- `dockerhub-release`: required maintainer approval; secrets consumed by the
  release workflow.
- `pypi-release`: required maintainer approval and trusted publishing configuration.

Each image runner must provide `CI_CONFIG_PATH_IMAGE` in its local runner
environment, pointing at that machine's private `image.env`. Do not put
registry credentials in that file.

## Runner labels

- A800 regression runner: `self-hosted`, `A800`.
- P6K regression runner: `self-hosted`, `P6K`.
- Image build runner: the labels stored in `CI_RUNNER_IMAGE`; it may reuse the
  A800 runner because DeepEP compilation does not require a GPU.

## Rulesets

Configure `master-protection` for the default branch with:

- pull requests required;
- one approval from someone other than the PR author;
- stale approvals dismissed after a push;
- approval required for the latest push;
- CODEOWNERS review required;
- all review threads resolved;
- `ci-gate` as the only required status check;
- branch deletion and force pushes disabled;
- squash merge as the only enabled merge method.

Configure `release-tag-protection` for `refs/tags/v*` so only maintainers can
create release tags and tags cannot be updated or deleted.

## First activation

The dispatcher and scheduled workflows must exist on the default branch before
GitHub will accept `workflow_dispatch` or `issue_comment` events. Validate this
PR locally, merge it without enabling the required check, configure the
settings above, then enable `ci-gate` as required after its first successful
run on a follow-up PR.
