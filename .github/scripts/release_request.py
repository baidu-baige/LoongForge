#!/usr/bin/env python3
# Copyright 2026 The LoongForge Authors.
# SPDX-License-Identifier: Apache-2.0

"""Validate and normalize a unified Release Request."""

from __future__ import annotations

import argparse
import re
import sys
from dataclasses import asdict, dataclass


DEBUG_SUFFIX_RE = re.compile(r"^[A-Za-z0-9._-]*$")
MODEL_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]*$")
IMAGE_REF_RE = re.compile(
    r"^[A-Za-z0-9._-]+(?::[0-9]+)?(?:/[A-Za-z0-9._-]+)*:[A-Za-z0-9._-]+$"
)
DRY_RUN_REASON = "__dry_run__"


class ReleaseRequestError(ValueError):
    pass


@dataclass(frozen=True)
class ReleaseRequest:
    build_image: bool
    run_regression: bool
    publish_internal: bool
    publish_dockerhub: bool
    execution_slot: str = ""
    llm_model: str = ""
    vla_model: str = ""
    baseline_chip: str = ""
    debug_tag_suffix: str = ""
    existing_image_ref: str = ""
    skip_regression_reason: str = ""


def _parse_bool(name: str, value: str) -> bool:
    normalized = value.strip().lower()
    if normalized not in {"true", "false"}:
        raise ReleaseRequestError(f"{name} must be true or false")
    return normalized == "true"


def _validate_image_ref(name: str, value: str) -> None:
    if value and not IMAGE_REF_RE.fullmatch(value):
        raise ReleaseRequestError(f"{name} is not a valid image reference")


def parse_release_request(
    *,
    build_image: str,
    run_regression: str,
    publish_internal: str,
    publish_dockerhub: str,
    execution_slot: str,
    llm_model: str,
    vla_model: str,
    baseline_chip: str = "",
    debug_tag_suffix: str = "",
    existing_image_ref: str = "",
    skip_regression_reason: str = "",
) -> ReleaseRequest:
    request = ReleaseRequest(
        build_image=_parse_bool("build_image", build_image),
        run_regression=_parse_bool("run_regression", run_regression),
        publish_internal=_parse_bool("publish_internal", publish_internal),
        publish_dockerhub=_parse_bool("publish_dockerhub", publish_dockerhub),
        execution_slot=execution_slot.strip(),
        llm_model=llm_model.strip(),
        vla_model=vla_model.strip(),
        baseline_chip=baseline_chip.strip(),
        debug_tag_suffix=debug_tag_suffix.strip(),
        existing_image_ref=existing_image_ref.strip(),
        skip_regression_reason=skip_regression_reason.strip(),
    )
    validate_release_request(request)
    return request


def validate_release_request(request: ReleaseRequest) -> None:
    if request.execution_slot not in {"primary", "secondary"}:
        raise ReleaseRequestError("execution_slot must be primary or secondary")
    if not MODEL_RE.fullmatch(request.llm_model):
        raise ReleaseRequestError("llm_model is invalid")
    if not MODEL_RE.fullmatch(request.vla_model):
        raise ReleaseRequestError("vla_model is invalid")
    if request.baseline_chip and not MODEL_RE.fullmatch(request.baseline_chip):
        raise ReleaseRequestError("baseline_chip is invalid")
    if request.run_regression and not request.baseline_chip:
        raise ReleaseRequestError("run_regression requires baseline_chip")

    if not DEBUG_SUFFIX_RE.fullmatch(request.debug_tag_suffix):
        raise ReleaseRequestError("debug_tag_suffix contains invalid characters")

    _validate_image_ref("existing_image_ref", request.existing_image_ref)

    if request.skip_regression_reason == DRY_RUN_REASON:
        return

    if request.run_regression and not (request.build_image or request.existing_image_ref):
        raise ReleaseRequestError("run_regression requires build_image or existing_image_ref")

    if request.publish_internal and not (request.build_image or request.existing_image_ref):
        raise ReleaseRequestError("publish_internal requires build_image or existing_image_ref")

    if request.publish_dockerhub and not request.publish_internal:
        raise ReleaseRequestError("publish_dockerhub requires publish_internal")

    if (
        not request.build_image
        and not request.run_regression
        and not request.publish_internal
        and not request.publish_dockerhub
    ):
        raise ReleaseRequestError("at least one release stage must be enabled")

    publishing_without_regression = (
        request.publish_internal or request.publish_dockerhub
    ) and not request.run_regression
    if publishing_without_regression and not request.skip_regression_reason:
        raise ReleaseRequestError("publishing without regression requires skip_regression_reason")


def _add_argument(parser: argparse.ArgumentParser, name: str, *, required: bool) -> None:
    parser.add_argument(f"--{name.replace('_', '-')}", required=required)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    for name in (
        "build_image",
        "run_regression",
        "publish_internal",
        "publish_dockerhub",
        "execution_slot",
        "llm_model",
        "vla_model",
    ):
        _add_argument(parser, name, required=True)
    for name in (
        "baseline_chip",
        "debug_tag_suffix",
        "existing_image_ref",
        "skip_regression_reason",
    ):
        _add_argument(parser, name, required=False)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    try:
        request = parse_release_request(
            build_image=args.build_image,
            run_regression=args.run_regression,
            publish_internal=args.publish_internal,
            publish_dockerhub=args.publish_dockerhub,
            execution_slot=args.execution_slot,
            llm_model=args.llm_model,
            vla_model=args.vla_model,
            baseline_chip=args.baseline_chip or "",
            debug_tag_suffix=args.debug_tag_suffix or "",
            existing_image_ref=args.existing_image_ref or "",
            skip_regression_reason=args.skip_regression_reason or "",
        )
    except ReleaseRequestError as exc:
        print(str(exc), file=sys.stderr)
        return 2

    for key, value in asdict(request).items():
        print(f"{key}={str(value).lower() if isinstance(value, bool) else value}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
