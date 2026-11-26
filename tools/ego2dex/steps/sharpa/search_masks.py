# Copyright 2026 The LoongForge Authors.
# SPDX-License-Identifier: Apache-2.0
"""Avoid computing contacts explicitly excluded from the arm-search merit.

Final model construction and independent native contact validation keep the
original asset masks. This removes only within-one-hand pairs from SEARCH.
"""

import json
import mujoco
import numpy as np
from .coupled_mapping import allowed_contact


def configure(model, qpos, output):
    original_type = model.geom_contype.copy()
    original_aff = model.geom_conaffinity.copy()
    if any((t or a) and (t != 1 or a != 1) for t, a in zip(original_type, original_aff)):
        return dict(applied=False, reason="nonstandard collision masks retained")
    data = mujoco.MjData(model)
    data.qpos[:] = qpos
    mujoco.mj_forward(model, data)

    def rows(d):
        return sorted((int(c.geom1), int(c.geom2), float(c.dist)) for c in d.contact if allowed_contact(model, c))

    expected = rows(data)
    before = int(data.ncon)
    for geom in range(model.ngeom):
        if not (original_type[geom] or original_aff[geom]):
            continue
        name = model.body(int(model.geom_bodyid[geom])).name or ""
        if name.startswith("left_hand_"):
            model.geom_contype[geom] = 4
            model.geom_conaffinity[geom] = 1 | 2 | 8
        elif name.startswith("right_hand_"):
            model.geom_contype[geom] = 8
            model.geom_conaffinity[geom] = 1 | 2 | 4
        elif name.startswith("left_"):
            model.geom_contype[geom] = 1
            model.geom_conaffinity[geom] = 1 | 2 | 4 | 8
        elif name.startswith("right_"):
            model.geom_contype[geom] = 2
            model.geom_conaffinity[geom] = 1 | 2 | 4 | 8
    removed = 0
    for i in range(model.ngeom):
        for j in range(i + 1, model.ngeom):
            old = bool((original_type[i] & original_aff[j]) | (original_type[j] & original_aff[i]))
            new = bool(
                (model.geom_contype[i] & model.geom_conaffinity[j])
                | (model.geom_contype[j] & model.geom_conaffinity[i])
            )
            if old == new:
                continue
            names = [model.body(int(model.geom_bodyid[g])).name or "" for g in [i, j]]
            assert (
                old
                and not new
                and all("_hand_" in n for n in names)
                and names[0].split("_")[0] == names[1].split("_")[0]
            ), names
            removed += 1
    mujoco.mj_forward(model, data)
    actual = rows(data)
    # Contact generation order may change; compare the entire admitted set and
    # its signed distances, not just counts or the aggregate cost.
    assert len(expected) == len(actual), (len(expected), len(actual))
    if expected:
        np.testing.assert_allclose(np.asarray(expected), np.asarray(actual), atol=1e-10, rtol=0)
    report = dict(
        applied=True,
        search_only=True,
        original_total_contacts=before,
        filtered_total_contacts=int(data.ncon),
        retained_merit_contacts=len(expected),
        signed_distance_equivalence_passed=True,
        physical_acceptance_masks_unchanged=True,
    )
    report.update(
        pair_graph_equivalent_for_all_included_pairs=True, removed_only_already_ignored_within_one_hand_pairs=removed
    )
    (output / "search_contact_mask_equivalence.json").write_text(json.dumps(report, indent=2))
    return report
