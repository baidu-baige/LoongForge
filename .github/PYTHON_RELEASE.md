# Python package releases

Use the **Python Package Release** workflow to build a wheel and source
distribution, check their metadata, and test installation in separate clean
environments. The `check` destination stops after creating the build artifact.
Hardware runtime setup and validation follow the installation guides for each
backend. The Docker image keeps its own release workflow.

## One-time setup

1. Create the `testpypi-release` and `pypi-release` GitHub environments. Restrict
   deployments to version tags and require a release maintainer's approval for
   `pypi-release`.
2. Configure a Trusted Publisher on each registry for the `loongforge`
   project. Use `baidu-baige` as owner, `LoongForge` as repository, and
   `python-release.yml` as workflow. Match each publisher to the corresponding
   GitHub environment name.

## Publish a version

1. Land the version bump in its own release PR. Create `v<version>` for the
   validated commit on `master`, with the version matching `project.version`.
2. Dispatch **Python Package Release** on the tag with `testpypi`. Confirm the
   registry artifact links back to this repository and can be installed in a
   clean environment. Use the selected release candidate for backend testing
   and record those CI runs in the release notes.
3. Dispatch on the stable tag with `pypi` and approve the deployment after
   reviewing the build and validation results. Install the exact published
   version in a clean environment outside the source tree, run `pip check`,
   `import loongforge`, and `LoongForge --help`, and verify the registry links.
