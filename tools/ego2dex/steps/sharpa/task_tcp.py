# Copyright 2026 The LoongForge Authors.
# SPDX-License-Identifier: Apache-2.0
"""Sharpa's articulated counterpart of frozen Orientation Eq.1/Eq.2 TCP.

The point is the midpoint of thumb and virtual index(.7index+.3middle).
Its frame follows the same signed opening/approach construction as the source
pipeline. The attachment transform remains fixed while finger joints move.
"""

import mujoco
import numpy as np
from steps.retarget.continuity import _hat
from scipy.spatial.transform import Rotation


def tcp_ids(model, side):
    """Site ids (thumb,index,middle) and the C_MC root body id; resolve once per model."""
    return [model.site(f"{side}_hand_{side}_{digit}_tip").id for digit in ["thumb", "index", "middle"]], model.body(
        f"{side}_hand_{side}_hand_C_MC"
    ).id


def opening_and_jac(model, data, side, vadr, ids=None, jacobian=True):
    # ids/jacobian=False are speed-only options; values are bit-identical.
    sids = tcp_ids(model, side)[0] if ids is None else ids
    points = []
    jac = []
    jp = np.zeros((3, model.nv))
    for sid in sids:
        points.append(data.site_xpos[sid].copy())
        if jacobian:
            mujoco.mj_jacSite(model, data, jp, None, sid)
            jac.append(jp[:, vadr].copy())
    delta = points[0] - 0.7 * points[1] - 0.3 * points[2]
    width = float(np.linalg.norm(delta))
    return width, (delta / max(width, 1e-12)) @ (jac[0] - 0.7 * jac[1] - 0.3 * jac[2]) if jacobian else None


def pose_and_jac(model, data, side, vadr, ids=None, jacobian=True):
    # ids/jacobian=False are speed-only options; position/rotation are bit-identical.
    sids, root = tcp_ids(model, side) if ids is None else ids
    vadr = np.asarray(vadr, dtype=int)
    jp = np.zeros((3, model.nv))
    jr = np.zeros_like(jp)
    points = []
    jac = []
    for sid in sids:
        points.append(data.site_xpos[sid].copy())
        if jacobian:
            mujoco.mj_jacSite(model, data, jp, None, sid)
            jac.append(jp[:, vadr].copy())
    wrist = data.xpos[root]
    thumb, index, middle = points
    virtual = 0.7 * index + 0.3 * middle
    position = 0.5 * (thumb + virtual)
    sign = 1.0 if side == "right" else -1.0
    opening = sign * (thumb - virtual)
    width = float(np.linalg.norm(opening))
    if width < 1e-8:
        raise ValueError("Degenerate zero-width articulated TCP")
    z = opening / width
    approach = virtual - wrist
    a = np.cross(z, approach)
    length = float(np.linalg.norm(a))
    reference = None
    if length < 1e-6:
        reference = np.array([0.0, 0.0, 1.0]) if abs(z[2]) <= 0.99 else np.array([0.0, 1.0, 0.0])
        a = np.cross(z, reference)
        length = float(np.linalg.norm(a))
    y = a / max(length, 1e-12)
    x = np.cross(y, z)
    rotation = np.stack([y, z, x], axis=1)
    if not jacobian:
        return position, rotation, None, None
    mujoco.mj_jacBody(model, data, jp, jr, root)
    Jw = jp[:, vadr].copy()
    JT, JI, JM = jac
    JV = 0.7 * JI + 0.3 * JM
    Jp = 0.5 * (JT + JV)
    Jz = sign * (np.eye(3) - np.outer(z, z)) @ (JT - JV) / width
    Jd = JV - Jw
    Ja = -_hat(approach) @ Jz + _hat(z) @ Jd if reference is None else -_hat(reference) @ Jz
    Jy = (np.eye(3) - np.outer(y, y)) @ Ja / max(length, 1e-12)
    Jx = -_hat(z) @ Jy + _hat(y) @ Jz
    angular_jac = 0.5 * (_hat(y) @ Jy + _hat(z) @ Jz + _hat(x) @ Jx)
    return position, rotation, Jp, angular_jac


def track(model, q):
    data = mujoco.MjData(model)
    P = []
    R = []
    for row in q:
        data.qpos[:] = row
        mujoco.mj_forward(model, data)
        poses = [pose_and_jac(model, data, s, []) for s in ["left", "right"]]
        P.append([p[0] for p in poses])
        R.append([p[1] for p in poses])
    return np.asarray(P), np.asarray(R)


def angular_metrics(actual, target, masks):
    """Same common-TCP metrics consumed by frozen angular_eligible."""
    raw = (
        Rotation.from_matrix((target.transpose(0, 1, 3, 2) @ actual).reshape(-1, 3, 3))
        .magnitude()
        .reshape(len(actual), 2)
    )
    alternate = target @ np.diag([-1.0, -1.0, 1.0])
    other = (
        Rotation.from_matrix((alternate.transpose(0, 1, 3, 2) @ actual).reshape(-1, 3, 3))
        .magnitude()
        .reshape(len(actual), 2)
    )
    equivalent = np.minimum(raw, other)
    approach = np.arccos(np.clip(np.einsum("nsi,nsi->ns", target[:, :, :, 2], actual[:, :, :, 2]), -1, 1))
    opening = np.arccos(np.clip(abs(np.einsum("nsi,nsi->ns", target[:, :, :, 1], actual[:, :, :, 1])), 0, 1))

    def means(v):
        return [float(np.degrees(v[masks[:, s], s].mean())) if masks[:, s].any() else None for s in range(2)]

    return dict(
        raw_orientation_mean_deg=means(raw),
        symmetry_orientation_mean_deg=means(equivalent),
        approach_mean_deg=means(approach),
        opening_line_mean_deg=means(opening),
        symmetry_orientation_p95_deg=float(np.degrees(np.percentile(equivalent[masks], 95))),
    )
