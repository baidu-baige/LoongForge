# Copyright 2026 The LoongForge Authors.
# SPDX-License-Identifier: Apache-2.0

"""Resolve configuration paths in source checkouts and built distributions."""

from pathlib import Path


def config_root() -> Path:
    """Return the packaged config root, falling back to the repository tree."""
    package_root = Path(__file__).resolve().parent / "_configs"
    if package_root.is_dir():
        return package_root

    source_root = Path(__file__).resolve().parent.parent / "configs"
    if source_root.is_dir():
        return source_root

    raise FileNotFoundError(
        "LoongForge configuration files were not found. "
        "Reinstall the package or pass an explicit --config-file."
    )


def resolve_config(path: str | Path) -> Path:
    """Resolve a path recorded relative to the repository's configs directory."""
    relative = Path(path)
    if relative.parts and relative.parts[0] == "configs":
        relative = Path(*relative.parts[1:])
    return config_root() / relative
