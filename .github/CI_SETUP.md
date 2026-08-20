# CI repository setup

Repository files define the workflows, but the following settings must be
created by an organization or repository administrator.

## Repository Variables

Create every variable from `.github/ci-variables.example.json`. Replace the
`P6K` label only if the registered P6K runner uses a different custom label.
The values of `CI_RUNNER_A`, `CI_RUNNER_P`, and `CI_RUNNER_IMAGE` are JSON
arrays, not comma-separated strings. `llm_vlm` resolves to `CI_RUNNER_A` and
`embodied` resolves to `CI_RUNNER_P`; each suite runner builds its own local
candidate image when `--build-image` is requested.

## Protected environment

The manual `internal-image-update.yml` workflow uses the `internal-registry`
environment. Require maintainer approval there and store its registry
credentials as environment secrets. These credentials are not used by the
per-PR local candidate build.

Each self-hosted runner must provide `CI_CONFIG_PATH_IMAGE` in its service
environment, pointing at that machine's private `image.env`. The service
environment and that file together must provide the values shown in
`.github/ci-config.example.env`, including the default image, isolated mounts,
and BuildKit mirror paths. Do not put registry credentials or signed source
URLs in the repository.

Each suite runner must also provide a working Docker Buildx plugin because the
PR Dockerfile uses BuildKit secrets for runner-local APT, PyPI, and source
mirrors. Verify it with `docker buildx version`; installing the CLI plugin does
not require restarting the Docker daemon.

## Runner labels

- A800 regression runner: `self-hosted`, `A800`.
- P6K regression runner: `self-hosted`, `P6K`.
- Candidate images are built on the same suite runner that performs regression.
- `CI_RUNNER_IMAGE` is used only for explicit, operator-dispatched promotion.

## First activation

Enable the `ok-to-test` and `gpu-regression` workflows on the default branch,
then verify a maintainer-dispatched run on each labeled suite runner.
