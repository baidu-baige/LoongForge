# Python package releases

Use the **Python Package Release** workflow to build a wheel and source
distribution, check their metadata, and test installation in separate clean
environments. The `check` destination stops after creating the build artifact.

The PyPI package is the installable Python library/CLI boundary. It includes
the model and data configuration defaults required by its documented entry
points. Docker remains the supported turnkey training environment because it
provides the pinned accelerator runtime, compiler stack, Megatron backend, and
hardware-specific operators. A base PyPI install does not install a CUDA or
Kunlun driver or claim that every GPU/XPU extra is a standalone training image.

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
3. Dispatch on the stable tag with `pypi`, provide the TestPyPI build run ID,
   and approve the deployment after reviewing that run and its validation
   results. PyPI must receive the same uploaded artifact from that run rather
   than rebuilding it. Install the exact published version in a clean
   environment outside the source tree, run `pip check`, `import loongforge`,
   execute `LoongForge --help`, resolve a packaged model configuration, and
   verify the registry links.
