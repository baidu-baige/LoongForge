# Copyright 2026 The LoongForge Authors.
# SPDX-License-Identifier: Apache-2.0
"""Native Sharpa model operations for the real registered-hand initializers.

Pass this module as ``backend=initializer_backend`` to
``steps.hands.global_initializer_v28/v29.initialize``. This module does not
choose a candidate, change an acceptance gate, or invoke a recovery solver.

The supported asset contract is deliberately narrow: 58 scalar coordinates,
58 independent, stateless, gear-one position actuators and no equalities or
tendons. Search models may additionally contain 12 unactuated static-base
coordinates. The 3 rad/s cap is a configured pipeline bound, not a vendor
velocity rating; the pinned Sharpa MJCF does not declare such a rating.
"""

from pathlib import Path
import copy
import xml.etree.ElementTree as ET

import mujoco
import numpy as np

from .panda_scene import build as _native_build, indices as _native_indices
from .mount_contract import canonical_mounts, audit_connection as _native_audit_connection


CONFIGURED_SPEED_CAP_RAD_S = 3.0
PHYSICAL_COORDINATES = 58
STATIC_JOINT_NAMES = tuple(
    f"{side}_static_{kind}{axis}" for side in ("left", "right") for kind in ("t", "r") for axis in "xyz"
)


def _require(condition, message):
    if not condition:
        raise ValueError("Sharpa initializer backend: " + message)


def _fmt(values):
    return " ".join(format(float(value), ".17g") for value in values)


def _native_addresses(model):
    addresses = []
    for side in ("left", "right"):
        ids = _native_indices(model, side)
        addresses.extend(int(a) for a in ids["arm_qadr"])
        addresses.extend(int(a) for a in ids["finger_qadr"])
    return np.asarray(addresses, dtype=np.int64)


def model_contract(model, *, allow_virtual=False):
    """Validate the actual transmissions; return qpos/actuator bookkeeping.

    Motor columns follow actuator order. Physical rows follow native qpos
    addresses, so a reordered actuator list is represented by a permutation
    rather than silently interpreted as qpos order.
    """
    expected = PHYSICAL_COORDINATES + (len(STATIC_JOINT_NAMES) if allow_virtual else 0)
    _require(model.nq == model.nv == model.njnt == expected, f"expected {expected} scalar qpos/dofs/joints")
    _require(model.nu == PHYSICAL_COORDINATES, "expected 58 independent actuators")
    _require(model.neq == 0, "equality or mimic constraints require a new explicit mapping")
    _require(model.ntendon == 0, "tendon transmission is outside this backend contract")
    physical = _native_addresses(model)
    _require(
        len(physical) == PHYSICAL_COORDINATES and len(set(physical.tolist())) == PHYSICAL_COORDINATES,
        "arm/finger indices must identify 58 distinct coordinates",
    )
    physical_set = set(physical.tolist())
    static = set()
    if allow_virtual:
        static = {int(model.joint(name).qposadr[0]) for name in STATIC_JOINT_NAMES}
    _require(
        physical_set.isdisjoint(static) and physical_set | static == set(range(expected)),
        "unknown, missing or duplicate physical/static coordinates",
    )
    for joint in range(model.njnt):
        _require(
            int(model.jnt_type[joint]) in (int(mujoco.mjtJoint.mjJNT_HINGE), int(mujoco.mjtJoint.mjJNT_SLIDE)),
            "only scalar hinge/slide joints are supported",
        )
        _require(
            int(model.jnt_qposadr[joint]) == int(model.jnt_dofadr[joint]) == joint,
            "initializer requires scalar joint id == qpos address == dof address",
        )
        _require(bool(model.jnt_limited[joint]), "all supported joints must declare finite limits")
        bounds = np.asarray(model.jnt_range[joint])
        _require(np.isfinite(bounds).all() and bounds[0] < bounds[1], "invalid joint range")
    joint_bounds = np.array(model.jnt_range, dtype=float, copy=True)
    command_bounds = np.column_stack((np.full(expected, -np.inf), np.full(expected, np.inf)))
    effective_bounds = joint_bounds.copy()
    command_limited = np.zeros(expected, dtype=bool)
    addresses = []
    actuator_names, motor_names, joint_ids = [], [], []
    matrix = np.zeros((expected, model.nu))
    for actuator in range(model.nu):
        _require(
            int(model.actuator_trntype[actuator]) == int(mujoco.mjtTrn.mjTRN_JOINT),
            "only direct joint transmission is supported",
        )
        joint = int(model.actuator_trnid[actuator, 0])
        address = int(model.jnt_qposadr[joint])
        _require(address in physical_set, "an actuator addresses a virtual or unknown coordinate")
        _require(
            np.array_equal(np.asarray(model.actuator_gear[actuator]), np.array([1.0, 0.0, 0.0, 0.0, 0.0, 0.0])),
            "actuator gear must be exactly one",
        )
        _require(
            int(model.actuator_dyntype[actuator]) == int(mujoco.mjtDyn.mjDYN_NONE),
            "stateful actuator dynamics require a different command mapping",
        )
        _require(
            int(model.actuator_gaintype[actuator]) == int(mujoco.mjtGain.mjGAIN_FIXED),
            "nonlinear actuator gain is unsupported",
        )
        _require(
            int(model.actuator_biastype[actuator]) == int(mujoco.mjtBias.mjBIAS_AFFINE),
            "actuator must be an ordinary position actuator",
        )
        kp = float(model.actuator_gainprm[actuator, 0])
        bias = np.asarray(model.actuator_biasprm[actuator, :3])
        _require(
            kp > 0 and bias[0] == 0 and bias[1] == -kp and bias[2] <= 0,
            "actuator parameters do not define a stateless position target",
        )
        if model.actuator_ctrllimited[actuator]:
            command = np.array(model.actuator_ctrlrange[actuator], dtype=float, copy=True)
            _require(np.isfinite(command).all() and command[0] < command[1], "invalid actuator command range")
            command_bounds[address] = command
            command_limited[address] = True
            effective_bounds[address, 0] = max(joint_bounds[address, 0], command[0])
            effective_bounds[address, 1] = min(joint_bounds[address, 1], command[1])
        addresses.append(address)
        matrix[address, actuator] = 1.0
        motor_names.append(model.joint(joint).name)
        actuator_names.append(model.actuator(actuator).name)
        joint_ids.append(joint)
    _require(
        len(set(addresses)) == PHYSICAL_COORDINATES and set(addresses) == physical_set,
        "every physical coordinate must have exactly one independent actuator",
    )
    _require(
        np.isfinite(effective_bounds).all() and np.all(effective_bounds[:, 1] - effective_bounds[:, 0] > 2e-8 + 2e-10),
        "joint/command intersection cannot contain the original core interior margin",
    )
    narrowed = []
    for actuator, address in enumerate(addresses):
        if not np.array_equal(joint_bounds[address], effective_bounds[address]):
            narrowed.append(
                dict(
                    joint=model.joint(joint_ids[actuator]).name,
                    actuator=actuator_names[actuator],
                    qpos_address=address,
                    native_joint_bounds=joint_bounds[address].tolist(),
                    command_bounds=command_bounds[address].tolist(),
                    effective_bounds=effective_bounds[address].tolist(),
                    lower_tightening=float(effective_bounds[address, 0] - joint_bounds[address, 0]),
                    upper_tightening=float(joint_bounds[address, 1] - effective_bounds[address, 1]),
                )
            )
    return dict(
        physical_addresses=physical,
        actuator_addresses=np.asarray(addresses, dtype=np.int64),
        physical_from_motor_matrix=matrix,
        motor_names=motor_names,
        actuator_names=actuator_names,
        actuator_joint_ids=np.asarray(joint_ids, dtype=np.int64),
        joint_bounds=joint_bounds,
        mapped_command_bounds=command_bounds,
        command_limited=command_limited,
        effective_bounds=effective_bounds,
        narrowed_coordinates=narrowed,
        configured_speed_cap_rad_s=CONFIGURED_SPEED_CAP_RAD_S,
        native_speed_limits_declared=False,
    )


def _contract(model):
    return model_contract(model, allow_virtual=model.nq != PHYSICAL_COORDINATES)


def prepare_model(original):
    """Return a private search clone with the exact joint/command intersection.

    The input model and native build/spec assets retain their declared ranges.
    No residual, search budget, geometry, transmission or acceptance threshold
    changes here. The original core simply reads these narrower search bounds
    through its existing ``model.jnt_range`` domain calculation.
    """
    contract = _contract(original)
    native_ranges = np.array(original.jnt_range, copy=True)
    native_commands = np.array(original.actuator_ctrlrange, copy=True)
    native_flags = int(original.opt.disableflags)
    prepared = copy.copy(original)
    _require(
        prepared is not original and not np.shares_memory(prepared.jnt_range, original.jnt_range),
        "MjModel copy must have independent joint-range storage",
    )
    physical = contract["physical_addresses"]
    prepared.jnt_range[physical] = contract["effective_bounds"][physical]
    _require(
        np.array_equal(prepared.jnt_range[physical], contract["effective_bounds"][physical]),
        "prepared search bounds differ from the exact intersection",
    )
    _require(
        np.array_equal(original.jnt_range, native_ranges)
        and np.array_equal(original.actuator_ctrlrange, native_commands)
        and int(original.opt.disableflags) == native_flags,
        "search preparation changed its input model",
    )
    _contract(prepared)
    return prepared


def build(base_poses=None, mounts=None, include_floor=True):
    spec = _native_build(base_poses=base_poses, mounts=mounts, include_floor=include_floor)
    model_contract(spec.compile())
    return spec


def indices(model, side):
    _require(side in ("left", "right"), "side must be left or right")
    _contract(model)
    return _native_indices(model, side)


def affine(model, side):
    """Physical finger q = 0 + I @ independent finger-position commands."""
    _require(side in ("left", "right"), "side must be left or right")
    _contract(model)
    ids = _native_indices(model, side)
    fingers = np.asarray(ids["finger_qadr"], dtype=np.int64)
    _require(len(fingers) == 22, "each Sharpa hand must have 22 independent coordinates")
    return fingers, np.zeros(22), np.eye(22)


def whole_affine(model, control=None):
    """Return an explicitly verified actuator-order motor permutation."""
    contract = _contract(model)
    control = np.sort(contract["physical_addresses"]) if control is None else np.asarray(control, dtype=np.int64)
    physical = set(contract["physical_addresses"].tolist())
    _require(
        control.ndim == 1 and len(control) == PHYSICAL_COORDINATES and set(control.tolist()) == physical,
        "whole_affine must contain every physical coordinate and no virtual coordinate",
    )
    matrix = contract["physical_from_motor_matrix"][control]
    return np.zeros(len(control)), matrix, list(contract["motor_names"])


def speed_limits(model, control=None):
    contract = _contract(model)
    values = np.full(model.nq, np.inf)
    values[contract["physical_addresses"]] = CONFIGURED_SPEED_CAP_RAD_S
    if control is None:
        return values
    control = np.asarray(control, dtype=np.int64)
    _require(
        control.ndim == 1 and set(control.tolist()) <= set(contract["physical_addresses"].tolist()),
        "speed caps apply to physical coordinates only",
    )
    return values[control]


def coupling_error(model, q):
    """No couplings are assumed: the no-equality independent model is checked."""
    _contract(model)
    q = np.asarray(q)
    _require(q.ndim == 2 and q.shape[1] == model.nq and np.isfinite(q).all(), "invalid trajectory")
    return 0.0


def project_seed(model, q, times, fixed_rows=None):
    """Nearest independently feasible trajectory, preserving feasible columns.

    With identity/permutation actuation the original Euclidean projection QP
    separates exactly by physical coordinate. Only violating columns enter
    the numerical QP; feasible finger columns remain bit-for-bit unchanged
    while arm speed/range violations are projected. This is not a relaxed
    feasibility test and adds no smoothing objective.
    """
    import clarabel
    from scipy import sparse

    contract = model_contract(model)
    q = np.asarray(q)
    _require(np.issubdtype(q.dtype, np.floating), "projection expects floating point joint coordinates")
    times = np.asarray(times, dtype=float)
    _require(q.ndim == 2 and q.shape[1] == model.nq and q.shape[0] == len(times), "trajectory/time shape mismatch")
    _require(
        len(times) >= 2 and np.isfinite(q).all() and np.isfinite(times).all(),
        "projection needs finite data and >=2 frames",
    )
    dt = np.diff(times)
    _require(np.all(dt > 0), "source clock must be strictly increasing")
    lower, upper = contract["effective_bounds"][:, 0], contract["effective_bounds"][:, 1]
    caps = dt[:, None] * CONFIGURED_SPEED_CAP_RAD_S - 1e-7
    _require(np.all(caps > 0), "source intervals are too short for the numerical speed margin")
    fixed = [] if fixed_rows is None else list(fixed_rows)
    for frame in fixed:
        _require(isinstance(frame, (int, np.integer)) and 0 <= frame < len(q), "invalid fixed frame")
        _require(
            np.all(q[frame] >= lower) and np.all(q[frame] <= upper),
            "fixed frame lies outside the exact joint/command intersection",
        )
    feasible = (
        np.all(q >= lower, axis=0) & np.all(q <= upper, axis=0) & np.all(np.abs(np.diff(q, axis=0)) <= caps, axis=0)
    )
    # The exact box projection is already the full nearest feasible point
    # whenever it satisfies every speed constraint. This avoids a numerical
    # QP perturbing a finger because its command bound differs by one ULP.
    result = np.clip(q, lower, upper)
    result[:, feasible] = q[:, feasible]
    box_speed_feasible = np.all(np.abs(np.diff(result, axis=0)) <= caps, axis=0)
    bad = np.flatnonzero(~box_speed_feasible)
    if not len(bad):
        _require(np.array_equal(result[:, feasible], q[:, feasible]), "box projection changed a feasible coordinate")
        return result
    n, count = len(q), len(bad)
    target = q[:, bad].ravel()
    difference = sparse.kron(
        sparse.diags([-np.ones(n - 1), np.ones(n - 1)], [0, 1], shape=(n - 1, n)), sparse.eye(count), format="csc"
    )
    low = np.tile(lower[bad], n)
    high = np.tile(upper[bad], n)
    for frame in fixed:
        low[frame * count : (frame + 1) * count] = q[frame, bad]
        high[frame * count : (frame + 1) * count] = q[frame, bad]
    eye = sparse.eye(n * count, format="csc")
    G = sparse.vstack([eye, -eye, difference, -difference], format="csc")
    h = np.r_[high, -low, np.repeat(caps[:, 0], count), np.repeat(caps[:, 0], count)]
    settings = clarabel.DefaultSettings()
    settings.verbose = False
    settings.max_iter = 500
    settings.tol_feas = 1e-9
    settings.tol_gap_abs = 1e-8
    solution = clarabel.DefaultSolver(eye, -target, G, h, [clarabel.NonnegativeConeT(len(h))], settings).solve()
    _require(str(solution.status) in ("Solved", "AlmostSolved"), f"feasible seed projection failed: {solution.status}")
    # Restore exact box bounds after the approximate numerical solve. Scalar
    # clipping to one fixed interval cannot increase adjacent differences.
    result[:, bad] = np.clip(np.asarray(solution.x).reshape(n, count), lower[bad], upper[bad])
    for frame in fixed:
        result[frame, bad] = q[frame, bad]
    _require(np.isfinite(result).all(), "projection returned non-finite coordinates")
    _require(np.all(result >= lower) and np.all(result <= upper), "projection violated effective joint/command bounds")
    _require(np.all(np.abs(np.diff(result, axis=0)) <= caps + 1e-7), "projection violated configured speed caps")
    _require(np.array_equal(result[:, feasible], q[:, feasible]), "projection changed a feasible coordinate")
    return result


def audit_connection(model, q):
    model_contract(model)
    return _native_audit_connection(model, q)


def standalone(spec):
    """Serialize native geometry with precise bases and canonical assembly."""
    from .hand_swap import _standalone_xml
    from .backend import upstream_root
    from steps import config

    model = spec.compile()
    model_contract(model)
    tree = _standalone_xml(spec, upstream_root(), Path(config.MENAGERIE_DIR) / "franka_emika_panda/assets")
    parents = {child: parent for parent in tree.iter() for child in parent}
    for side, mount in canonical_mounts().items():
        name = side + "_link0"
        body = tree.find(f'.//body[@name="{name}"]')
        bid = model.body(name).id
        _require(
            body is not None and parents[body].tag == "worldbody" and int(model.body_parentid[bid]) == 0,
            "serialized physical arm base must be a direct world body",
        )
        body.set("pos", _fmt(model.body_pos[bid]))
        body.set("quat", _fmt(model.body_quat[bid]))
        palm = tree.find(f'.//body[@name="{side}_hand_{side}_hand_C_MC"]')
        _require(palm is not None, "canonical Sharpa palm body is missing")
        palm_id = model.body(f"{side}_hand_{side}_hand_C_MC").id
        _require(
            parents[palm].tag == "body"
            and parents[palm].get("name") == side + "_link7"
            and int(model.body_parentid[palm_id]) == model.body(side + "_link7").id,
            "canonical palm transform must be expressed directly in link7",
        )
        palm.set("pos", _fmt(mount["position_link7"]))
        palm.set("quat", _fmt(mount["quaternion_link7_wxyz"]))
        for suffix in ("coupler_black", "coupler_silver"):
            geom = tree.find(f'.//geom[@name="{side}_{suffix}"]')
            _require(geom is not None, "canonical axial coupler is missing")
            gid = model.geom(f"{side}_{suffix}").id
            geom.set("pos", _fmt(model.geom_pos[gid]))
            geom.set("quat", _fmt(model.geom_quat[gid]))
        site = tree.find(f'.//site[@name="{side}_attachment_site"]')
        _require(site is not None, "Panda attachment site is missing")
        sid = model.site(side + "_attachment_site").id
        site.set("pos", _fmt(model.site_pos[sid]))
        site.set("quat", _fmt(model.site_quat[sid]))
    return tree


def virtual_model(bases, wide=False):
    """Same six-coordinate static-base wrapper used by the source algorithms."""
    tree = standalone(build(base_poses=bases, include_floor=False))
    for key in tree.findall("keyframe"):
        tree.remove(key)
    parents = {child: parent for parent in tree.iter() for child in parent}
    for side in ("left", "right"):
        body = tree.find(f'.//body[@name="{side}_link0"]')
        parent = parents[body]
        wrapper = ET.Element(
            "body", name=side + "_mount", pos=body.attrib.pop("pos", "0 0 0"), quat=body.attrib.pop("quat", "1 0 0 0")
        )
        ET.SubElement(wrapper, "inertial", mass="1", pos="0 0 0", diaginertia=".01 .01 .01")
        for kind, prefix, limit in [("slide", "t", 0.6 if wide else 0.3), ("hinge", "r", np.pi if wide else np.pi / 3)]:
            for axis, vector in zip("xyz", ["1 0 0", "0 1 0", "0 0 1"]):
                ET.SubElement(
                    wrapper,
                    "joint",
                    name=f"{side}_static_{prefix}{axis}",
                    type=kind,
                    axis=vector,
                    range=_fmt([-limit, limit]),
                    limited="true",
                )
        parent.remove(body)
        wrapper.append(body)
        parent.append(wrapper)
    model = mujoco.MjModel.from_xml_string(ET.tostring(tree, encoding="unicode"))
    model_contract(model, allow_virtual=True)
    forbidden = int(mujoco.mjtDisableBit.mjDSBL_CONTACT) | int(mujoco.mjtDisableBit.mjDSBL_NATIVECCD)
    _require(not (int(model.opt.disableflags) & forbidden), "search model must retain native contacts initially")
    return prepare_model(model)


def allowed_contact(model, contact):
    """Whether a native contact belongs in the v29 recovery penalty.

    The predicate matches the native Sharpa ``arm_contacts`` acceptance scope;
    internal fingers on the same hand remain a separately reported quantity.
    """
    from steps.retarget.base_search import _is_known_baseline_self_contact

    bodies = [model.body(int(model.geom_bodyid[geom])).name or "" for geom in (contact.geom1, contact.geom2)]
    if not all(name.startswith(("left_", "right_")) for name in bodies):
        return False
    if _is_known_baseline_self_contact(*bodies):
        return False
    if all("_hand_" in name for name in bodies) and bodies[0].split("_")[0] == bodies[1].split("_")[0]:
        return False
    return True
