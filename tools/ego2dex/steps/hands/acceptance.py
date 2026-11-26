# Copyright 2026 The LoongForge Authors.
# SPDX-License-Identifier: Apache-2.0
"""Versioned acceptance profiles shared by search, raster audit and export."""

from dataclasses import asdict, dataclass
import os


@dataclass(frozen=True)
class Acceptance:
    name: str
    position_m: float
    rotation_rad: float
    mean_work_region_occlusion: float
    max_work_region_occlusion: float
    mean_hand_proximal_occlusion: float
    max_arm_image_fraction: float = 0.60
    layout_m: float = 0.014

    @property
    def position_search_m(self):
        return 0.9 * self.position_m

    @property
    def rotation_search_rad(self):
        return 0.025 if self.name == "strict" else 0.9 * self.rotation_rad

    @property
    def position_finish_m(self):
        return self.position_m - 0.00002

    @property
    def rotation_finish_rad(self):
        return self.rotation_rad - 0.0015

    def report(self):
        return dict(
            asdict(self),
            version="hands_acceptance_20261116",
            native_joint_limits=True,
            native_speed_caps=True,
            coupling=True,
            rigid_mount=True,
            native_arm_collision_samples=0,
            source_frames_clock_camera_preserved=True,
            relative_original_angular_gate_preserved=True,
        )


PROFILES = {
    "strict": Acceptance("strict", 0.005, 0.15, 0.05, 0.25, 0.05),
    "mild": Acceptance("mild", 0.010, 0.25, 0.08, 0.35, 0.08),
    # B tier, user-approved 2026-11-24; opt-in only, never the default.
    "loose_b": Acceptance("loose_b", 0.020, 0.60, 0.08, 0.60, 0.08, 0.60, 0.025),
}


def current():
    name = os.environ.get("EGO2ROBOT_ACCEPTANCE_PROFILE", "mild")
    if name not in PROFILES:
        raise ValueError("Acceptance profile must be one of %s, got %r" % (sorted(PROFILES), name))
    return PROFILES[name]


def pixel_checks(measurements, profile=None):
    p = profile or current()
    return {
        key: float(measurements[key]) <= getattr(p, key)
        for key in (
            "mean_work_region_occlusion",
            "max_work_region_occlusion",
            "mean_hand_proximal_occlusion",
            "max_arm_image_fraction",
        )
    }


def reusable_record(record):
    """An older pass under an equal or stricter profile is valid, never the reverse."""
    previous = record.get("acceptance_profile", {}).get("name", "strict")
    order = ("strict", "mild", "loose_b")
    return previous in order and order.index(previous) <= order.index(current().name)


def gate(default, field, delta=0.0, scale=1.0):
    """Solver/search threshold: the historical literal under strict and mild (unchanged behaviour),
    the active profile value (times scale, plus delta) under loose_b."""
    a = current()
    if a.name != "loose_b":
        return default
    return getattr(a, field) * scale + delta
