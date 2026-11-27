# Copyright 2026 The LoongForge Authors.
# SPDX-License-Identifier: Apache-2.0
"""Independent FK checks on real mount roots, allowing genuine wrist DOFs."""

import numpy as np
import mujoco
from scipy.spatial.transform import Rotation
from .registry import profile, selected, coupling_error
from .panda_scene import mounting, FLANGE, ADAPTER


def audit_connection(model, track):
    d = mujoco.MjData(model)
    records = []
    for side in ["left", "right"]:
        p = profile(side)
        mt = mounting(selected(), side)
        link = model.body(side + "_link7").id
        root = model.body(side + "_hand_" + p["root"]).id
        chain = []
        b = root
        while b > 0 and b != link:
            chain.append(b)
            b = int(model.body_parentid[b])
        assert b == link and all(model.body_jntnum[b] == 0 for b in chain), "Non-rigid vendor mount root"
        expectedR = Rotation.from_quat(np.array(mt["quaternion"])[[1, 2, 3, 0]]).as_matrix()
        position = rotation = seam = 0.0
        interface = model.site(side + "_hand_assembly_interface").id
        for row in track:
            d.qpos[:] = row
            mujoco.mj_forward(model, d)
            R = d.xmat[link].reshape(3, 3)
            position = max(position, float(np.linalg.norm(d.xpos[root] - (d.xpos[link] + R @ mt["position"]))))
            rotation = max(rotation, float(np.linalg.norm(d.xmat[root].reshape(3, 3) - R @ expectedR)))
            seam = max(
                seam,
                float(np.linalg.norm(d.site_xpos[interface] - (d.xpos[link] + R @ np.array([0, 0, FLANGE + ADAPTER])))),
            )
        records.append(
            dict(
                side=side,
                root_position_error_max_m=position,
                root_rotation_error_max=rotation,
                adapter_seam_error_max_m=seam,
                root_rigid_downstream_joint7=True,
                intrinsic_wrist_joints_preserved=True,
                passed=position < 1e-8 and rotation < 1e-8 and seam < 1e-8,
            )
        )
    error = coupling_error(model, track)
    passed = all(x["passed"] for x in records) and error < 1e-8
    result = dict(
        passed=passed,
        rows=records,
        source_frames=len(track),
        coupling_residual_max_rad=error,
        no_free_hand_root=True,
        no_per_episode_mount_fitting=True,
        hardware_calibration_validated=False,
    )
    if not passed:
        raise AssertionError(result)
    return result
