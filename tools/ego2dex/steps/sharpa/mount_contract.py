# Copyright 2026 The LoongForge Authors.
# SPDX-License-Identifier: Apache-2.0
"""One model-defined flange-to-Sharpa transform, independent of an episode.

The Panda link7-to-flange distance is from the Panda MJCF. The two axial
couplers follow the pinned official replay model. Palm clocking follows the
Panda hand body's -45 degree rotation; the UR3e replay's 45/135 degree values
are expressed at a different robot flange and must not be copied to Panda.
These define a simulation assembly, not a calibrated real hardware adapter.
"""

import math
import numpy as np

LINK7_TO_FLANGE = 0.107
BLACK_HEIGHT = 0.01325
SILVER_HEIGHT = 0.0175
HAND_OFFSET = BLACK_HEIGHT + SILVER_HEIGHT
MOUNT_YAW_DEG = {"left": -45.0, "right": -45.0}
VERSION = "panda_sharpa_axial_rigid_v2"


def canonical_mounts():
    mounts = {}
    for side, yaw in MOUNT_YAW_DEG.items():
        angle = math.radians(yaw) / 2
        mounts[side] = dict(
            position_link7=[0.0, 0.0, LINK7_TO_FLANGE + HAND_OFFSET],
            quaternion_link7_wxyz=[math.cos(angle), 0.0, 0.0, math.sin(angle)],
            version=VERSION,
            policy="fixed_model_geometry; never fit mount to human wrist",
            hardware_calibration_validated=False,
        )
    return mounts


def validate_mounts(mounts):
    expected = canonical_mounts()
    for side in expected:
        actual = mounts[side]
        reference = expected[side]
        np.testing.assert_allclose(
            actual["position_link7"],
            reference["position_link7"],
            atol=1e-10,
            rtol=0,
            err_msg="Off-axis/per-episode hand mounts are unsupported; solve arm IK instead",
        )
        quat = np.asarray(actual["quaternion_link7_wxyz"], float)
        quat /= np.linalg.norm(quat)
        target = np.asarray(reference["quaternion_link7_wxyz"])
        if abs(float(quat @ target)) < 1 - 1e-10:
            raise ValueError("Palm clocking must follow the rigid adapter; solve arm IK instead")
    return mounts


def audit_connection(model, track):
    """Measure flange, both adapter seams and palm root on every source frame."""
    import mujoco

    data = mujoco.MjData(model)
    rows = []
    for side, contract in canonical_mounts().items():
        link = model.body(side + "_link7").id
        palm = model.body(side + "_hand_" + side + "_hand_C_MC").id
        path = []
        body = palm
        while body != link and body > 0:
            path.append(body)
            body = int(model.body_parentid[body])
        if body != link or any(model.body_jntnum[b] for b in path):
            raise AssertionError("Palm is not rigidly attached downstream of joint7")
        site = model.site(side + "_attachment_site").id
        black = model.geom(side + "_coupler_black").id
        silver = model.geom(side + "_coupler_silver").id
        angle = math.radians(MOUNT_YAW_DEG[side])
        Rz = np.array(
            [[math.cos(angle), -math.sin(angle), 0.0], [math.sin(angle), math.cos(angle), 0.0], [0.0, 0.0, 1.0]]
        )
        gap = rotation = seam = 0.0
        for q in track:
            data.qpos[:] = q
            mujoco.mj_forward(model, data)
            R = data.site_xmat[site].reshape(3, 3)
            origin = data.site_xpos[site]
            gap = max(gap, float(np.linalg.norm(data.xpos[palm] - (origin + R @ np.array([0, 0, HAND_OFFSET])))))
            rotation = max(rotation, float(np.linalg.norm(data.xmat[palm].reshape(3, 3) - R @ Rz)))
            rb = data.geom_xmat[black].reshape(3, 3)
            rs = data.geom_xmat[silver].reshape(3, 3)
            b0 = data.geom_xpos[black] - rb[:, 2] * model.geom_size[black, 1]
            b1 = data.geom_xpos[black] + rb[:, 2] * model.geom_size[black, 1]
            s0 = data.geom_xpos[silver] - rs[:, 2] * model.geom_size[silver, 1]
            s1 = data.geom_xpos[silver] + rs[:, 2] * model.geom_size[silver, 1]
            seam = max(
                seam,
                float(np.linalg.norm(b0 - origin)),
                float(np.linalg.norm(b1 - s0)),
                float(np.linalg.norm(s1 - data.xpos[palm])),
            )
        passed = gap < 1e-8 and rotation < 1e-8 and seam < 1e-8
        rows.append(
            dict(
                side=side,
                frames=len(track),
                palm_in_joint7_rigid_subtree=True,
                mount_position_error_max_m=gap,
                mount_rotation_matrix_error_max=rotation,
                connector_seam_error_max_m=seam,
                passed=passed,
                contract=contract,
            )
        )
    assert all(r["passed"] for r in rows), rows
    return dict(
        passed=True,
        mount_version=VERSION,
        source_frames=len(track),
        rows=rows,
        no_free_hand_root=True,
        no_per_episode_mount_fitting=True,
        contact_or_dynamics_validated=False,
    )
