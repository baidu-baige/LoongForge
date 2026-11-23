# Third-Party Notices

LoongForge is distributed under the [Apache License 2.0](LICENSE). Components derived
from or fetched from other projects retain the copyright, license, and notice
requirements of those projects.

## Loong-Megatron and Megatron-LM

The Loong-Megatron fork used as a training backend is a Git submodule at
`third_party/Loong-Megatron`. See the [submodule overview](third_party/README.md).

Megatron-LM source is distributed under the BSD 3-Clause License. The
submodule's own license and notice files at the pinned commit are authoritative
for the files it contains. Keep those files with the submodule when cloning,
building, or redistributing a checkout.

## Transformer Engine patches

`setup_env.py` applies [local patches](third_party/patches/README.md) to upstream
NVIDIA Transformer Engine sources. Keep the upstream project's copyright,
license, and notice files with the patched source tree.

## Other dependencies

Python packages, container base images, model implementations, and optional
runtime components are declared or referenced in `pyproject.toml`, Dockerfiles,
installation guides, and model-specific documentation. Refer to each
dependency's own license and notices.

When adding or adapting third-party code, preserve its original attribution and
license text and document the origin in the affected file as described in the
[file header guidelines](docs/source/HEADER_GUIDELINES.md).
