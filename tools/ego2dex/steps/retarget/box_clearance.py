# Copyright 2026 The LoongForge Authors.
# SPDX-License-Identifier: Apache-2.0
"""Continuous separating-axis clearance and its derivative for oriented boxes."""

import numpy as np


def box_gap_gradient(p1, R1, half1, p2, R2, half2, jp1, jr1, jp2, jr2):
    a, b = R1.T, R2.T
    da = np.cross(jr1.T[None, :, :], a[:, None, :])
    db = np.cross(jr2.T[None, :, :], b[:, None, :])
    axes = np.concatenate([a, b, np.cross(a[:, None, :], b[None, :, :]).reshape(9, 3)])
    derivatives = np.concatenate(
        [
            da,
            db,
            (
                np.cross(da[:, None, :, :], b[None, :, None, :]) + np.cross(a[:, None, None, :], db[None, :, :, :])
            ).reshape(9, jp1.shape[1], 3),
        ]
    )
    lengths = np.linalg.norm(axes, axis=1)
    keep = lengths > 1e-7
    axes = axes[keep] / lengths[keep, None]
    derivatives = derivatives[keep]
    derivatives = (derivatives - axes[:, None, :] * np.einsum("ajc,ac->aj", derivatives, axes)[:, :, None]) / lengths[
        keep, None, None
    ]
    delta = p2 - p1
    center = axes @ delta
    center_jac = np.einsum("ajc,c->aj", derivatives, delta) + axes @ (jp2 - jp1)
    pa, pb = axes @ a.T, axes @ b.T
    pa_jac = np.einsum("ajc,kc->akj", derivatives, a) + np.einsum("ac,kjc->akj", axes, da)
    pb_jac = np.einsum("ajc,kc->akj", derivatives, b) + np.einsum("ac,kjc->akj", axes, db)
    gaps = abs(center) - abs(pa) @ half1 - abs(pb) @ half2
    jac = np.sign(center)[:, None] * center_jac
    jac -= np.einsum("ak,akj,k->aj", np.sign(pa), pa_jac, half1)
    jac -= np.einsum("ak,akj,k->aj", np.sign(pb), pb_jac, half2)
    selected = int(np.argmax(gaps))
    return float(gaps[selected]), jac[selected]
