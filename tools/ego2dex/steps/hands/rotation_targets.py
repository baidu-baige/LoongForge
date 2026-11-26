# Copyright 2026 The LoongForge Authors.
# SPDX-License-Identifier: Apache-2.0
"""Versioned source-rotation repair; raw Orientation outputs stay untouched."""

import hashlib
import json
import shutil
from pathlib import Path
import numpy as np
from scipy.spatial.transform import Rotation, Slerp


def prepare(cache, reference, destination, policy="stabilize"):
    cache = Path(cache).resolve()
    reference = Path(reference).resolve()
    manifest = json.loads((reference / "manifest.json").read_text())
    ep = Path(manifest["episode"]).name
    source = cache / "06_retarget" / (ep + "_joint_pose_seed.npz")
    seed = np.load(source)
    fields = {k: seed[k] for k in seed.files}
    ref = np.load(reference / "hand_reference.npz")
    times = (ref["timestamps_ns"] - ref["timestamps_ns"][0]) / 1e9
    raw = seed["R"]
    fixed = raw.copy()
    masks = seed["masks"].astype(bool)
    n = len(raw)
    hands = []
    if policy not in ["original", "stabilize"]:
        raise ValueError(policy)
    for side, s in enumerate(["left", "right"]):
        bad = np.zeros(n, bool)
        turns = Rotation.from_matrix(raw[:-1, side].transpose(0, 2, 1) @ raw[1:, side]).magnitude()
        jumps = (
            np.flatnonzero((turns > 0.25) & masks[:-1, side] & masks[1:, side])
            if policy == "stabilize"
            else np.array([], int)
        )
        for frame in jumps:
            bad[max(0, frame - 2) : min(n, frame + 4)] = True
        anchors = np.flatnonzero(~bad)
        if bad.any():
            if len(anchors) < 2:
                raise ValueError("Not enough stable orientation anchors; original source retained")
            fixed[bad, side] = Slerp(times[anchors], Rotation.from_matrix(raw[anchors, side]))(
                np.clip(times[bad], times[anchors[0]], times[anchors[-1]])
            ).as_matrix()
        delta = Rotation.from_matrix(raw[:, side].transpose(0, 2, 1) @ fixed[:, side]).magnitude()
        hands.append(
            dict(
                side=s,
                trigger_frames=jumps.tolist(),
                changed_frames=np.flatnonzero(delta > 1e-8).tolist(),
                max_change_rad=float(delta.max()),
            )
        )
    changed = sorted({i for h in hands for i in h["changed_frames"]})
    goal_hash = hashlib.sha256(seed["p"].tobytes() + fixed.tobytes() + masks.tobytes()).hexdigest()
    prepared = cache
    if changed:
        prepared = Path(destination) / (ep + "_" + goal_hash[:12]) / "output"
        retarget = prepared / "06_retarget"
        retarget.mkdir(exist_ok=True, parents=True)
        fields["R"] = fixed
        np.savez_compressed(retarget / source.name, **fields)
        shutil.copy2(source.with_name(ep + "_joint_pose.json"), retarget / (ep + "_joint_pose.json"))
        for name in ["03_mask", "04_inpaint", "05_depth"]:
            if (cache / name).exists() and not (prepared / name).exists():
                (prepared / name).symlink_to(cache / name, target_is_directory=True)
    target = prepared / "06_retarget" / source.name
    revision = dict(
        policy=policy,
        target_profile="source_rotation_stabilized_v1" if changed else "original",
        original_cache=str(cache),
        orientation_cache=str(prepared),
        reference=str(reference),
        raw_seed=str(source),
        seed=str(target),
        goal_sha256=goal_hash,
        changed_frames=changed,
        changed_frame_count=len(changed),
        hands=hands,
        positions_unchanged=True,
        source_clock_unchanged=True,
        source_camera_unchanged=True,
        original_seed_sha256=hashlib.sha256(source.read_bytes()).hexdigest(),
        target_seed_sha256=hashlib.sha256(target.read_bytes()).hexdigest(),
    )
    Path(destination).mkdir(exist_ok=True, parents=True)
    (Path(destination) / "source_rotation_revision.json").write_text(json.dumps(revision, indent=2))
    return revision
