# Copyright 2026 The LoongForge Authors.
# SPDX-License-Identifier: Apache-2.0

"""Mink and damped-least-squares inverse-kinematics solvers."""

import mink
import mujoco
import numpy as np


# ============ Mink 6-DOF IK (base frame) ============

MINK_POSITION_TOL = 1e-3
POSITION_FIRST_TOL = 5e-3
POSE_REFINEMENT_MAX_ITER = 200
MINK_ORIENTATION_TOL = 1e-3

# Legacy DLS search settings.  These are defined before the solver functions
# because the robust solver uses DLS_MAX_ITER as a default argument.
DLS_POS_FEAS_GATE = 0.75
DLS_POS_TOL = 5e-3
DLS_PREWARM_MAX_ITER = 200
DLS_KEYFRAMES = 8
DLS_MAX_ITER = 300
# The reference control loop integrates one warm-start solution frame by frame.
# A modest trust region and posture cost preserve that branch while still
# allowing a fallback solve when the previous branch cannot reach the target.
MINK_CONTINUITY_MAX_STEP = 0.35
MINK_CONTINUITY_COST = 0.2

def _ee_pos(model, data, ee_ref):
    kind, idx = ee_ref
    return data.site_xpos[idx].copy() if kind == "site" else data.xpos[idx].copy()


def _project_qpos_to_limits(model, q):
    """Project hinge/slide coordinates into their MuJoCo declared ranges."""
    q = np.asarray(q, dtype=float).copy()
    for jid in range(model.njnt):
        if not model.jnt_limited[jid]:
            continue
        if model.jnt_type[jid] not in (mujoco.mjtJoint.mjJNT_HINGE,
                                       mujoco.mjtJoint.mjJNT_SLIDE):
            continue
        qa = int(model.jnt_qposadr[jid])
        lo, hi = model.jnt_range[jid]
        q[qa] = np.clip(q[qa], lo, hi)
    return q


class _JointContinuityLimit:
    """Bound a Mink integration step to a reference joint configuration."""

    def __init__(self, model, arm_ids, max_step):
        self.indices = np.asarray(
            [int(model.jnt_dofadr[jid]) for jid in arm_ids], dtype=int)
        self.projection_matrix = np.eye(model.nv)[self.indices]
        self.reference = None
        self.max_step = float(max_step)

    def set_reference(self, q_ref, max_step=None):
        self.reference = None if q_ref is None else np.asarray(q_ref, dtype=float).copy()
        if max_step is not None:
            self.max_step = float(max_step)

    def compute_qp_inequalities(self, configuration, dt):
        del dt
        if self.reference is None or self.max_step <= 0.0:
            return mink.limits.Constraint()
        delta_ref = np.empty(configuration.nv)
        # mj_differentiatePos returns q_reference - q_current in tangent space.
        mujoco.mj_differentiatePos(
            m=configuration.model, qvel=delta_ref, dt=1.0,
            qpos1=configuration.q, qpos2=self.reference)
        upper = self.max_step + delta_ref[self.indices]
        lower = -self.max_step + delta_ref[self.indices]
        G = np.vstack([self.projection_matrix, -self.projection_matrix])
        h = np.hstack([upper, -lower])
        return mink.limits.Constraint(G=G, h=h)


class MinkIKContext:
    """Reusable Mink task stack for one single-arm MuJoCo model.

    Ego2Robot solves a differential IK QP at every step, then integrates the
    resulting tangent velocity.  The context owns the task and limit objects
    so candidate and trajectory solves only reset the configuration state.
    """

    def __init__(self, model, arm_ids, ee_ref, solver="daqp",
                 position_cost=10.0, orientation_cost=1.0,
                 continuity_max_step=MINK_CONTINUITY_MAX_STEP,
                 continuity_cost=MINK_CONTINUITY_COST):
        self.model = model
        self.solver = solver
        self.solvers = tuple(dict.fromkeys((solver, "quadprog")))
        self.configuration = mink.Configuration(model)
        self.position_cost = np.broadcast_to(
            np.asarray(position_cost, dtype=float), (3,)).copy()
        self.orientation_cost = np.broadcast_to(
            np.asarray(orientation_cost, dtype=float), (3,)).copy()
        self.ee_task = mink.FrameTask(
            frame_name=ee_ref[1],
            frame_type=ee_ref[0],
            position_cost=self.position_cost,
            orientation_cost=self.orientation_cost,
            lm_damping=0.1,
        )
        self.orientation_mask = self.orientation_cost > 0.0
        self.orientation_enabled = bool(np.any(self.orientation_mask))
        self.posture_task = mink.PostureTask(model, cost=1e-3)
        self.continuity_task = mink.PostureTask(model, cost=np.zeros(model.nv))
        continuity_costs = np.zeros(model.nv)
        for jid in arm_ids:
            continuity_costs[int(model.jnt_dofadr[jid])] = float(continuity_cost)
        self.continuity_costs = continuity_costs
        self.continuity_task.set_cost(continuity_costs)
        self.tasks = [self.ee_task, self.posture_task, self.continuity_task]
        velocity_limits = {
            model.joint(jid).name: np.array([1.0]) for jid in arm_ids
        }
        self.limits = [
            mink.ConfigurationLimit(model=model),
            mink.VelocityLimit(model, velocity_limits),
        ]
        self.continuity_limit = _JointContinuityLimit(
            model, arm_ids, continuity_max_step)

    def reset(self, q_init=None, posture_target=None):
        """Reset the IK configuration and posture target."""
        q = self.model.qpos0.copy()
        if q_init is not None:
            q[:] = np.asarray(q_init, dtype=float)
        # Some menagerie XMLs ship a qpos0 outside its declared range. Mink's
        # ConfigurationLimit correctly rejects that seed, so project it first.
        q = _project_qpos_to_limits(self.model, q)
        self.configuration.update(q)
        self.posture_task.set_target(
            q if posture_target is None else np.asarray(posture_target, dtype=float))

    def solve(self, target_pos, target_R, q_init=None, max_iterations=100,
              dt=0.05, tol_pos=8e-3, tol_rot=0.15, continuity_q=None,
              continuity_max_step=None, position_first=False):
        """Solve inverse kinematics for the requested end-effector pose."""
        target = mink.SE3.from_rotation_and_translation(
            mink.SO3.from_matrix(np.asarray(target_R, dtype=float)),
            np.asarray(target_pos, dtype=float),
        )
        self.ee_task.set_target(target)
        self.reset(q_init, posture_target=continuity_q)
        self.continuity_task.set_target(
            self.configuration.q if continuity_q is None else continuity_q)
        # A continuity posture is meaningful only relative to the previous
        # frame. Applying its cost against the freshly reset q_init also on
        # the first frame traps the solver in the position-only prewarm branch
        # and can leave a large wrist orientation residual.
        self.continuity_task.set_cost(
            self.continuity_costs if continuity_q is not None
            else np.zeros(self.model.nv))
        self.continuity_limit.set_reference(continuity_q, continuity_max_step)
        limits = self.limits + ([self.continuity_limit]
                                if continuity_q is not None else [])

        best = None
        for _ in range(max_iterations):
            error = self.ee_task.compute_error(self.configuration)
            err_pos = float(np.linalg.norm(error[:3]))
            # FrameTask always returns all three rotational residuals, even
            # when a morphology releases one axis by assigning it zero cost.
            # Success and candidate scoring must use only constrained axes.
            err_rot = float(np.linalg.norm(error[3:][self.orientation_mask]))
            # Select the best non-converged iterate using the same relative
            # task weights as the QP. This matters for underactuated arms:
            # unweighted radians otherwise dominate centimetre-scale position
            # residuals and return a visibly off-target TCP.
            orientation_objective = float(np.linalg.norm(
                error[3:] * self.orientation_cost))
            total = (err_pos + 1e-3 * orientation_objective
                     if position_first else
                     float(np.linalg.norm(error[:3] * self.position_cost)) +
                     orientation_objective)
            if best is None or total < best[0]:
                best = (total, err_pos, err_rot, self.configuration.q.copy())
            if err_pos < tol_pos and (position_first or err_rot < tol_rot):
                return self.configuration.q.copy(), True, err_pos, err_rot

            damping = 1e-4 if total > 0.01 else 1e-3
            velocity = None
            for backend in self.solvers:
                try:
                    velocity = mink.solve_ik(
                        self.configuration,
                        self.tasks,
                        dt,
                        backend,
                        limits=limits,
                        damping=damping,
                    )
                    break
                except Exception:
                    continue
            if velocity is None:
                break
            self.configuration.integrate_inplace(velocity, dt)

        if best is None:
            q = self.configuration.q.copy()
            return q, False, float("inf"), float("inf")
        _, err_pos, err_rot, q = best
        return q, False, err_pos, err_rot

    def measure_error(self, q, target_pos, target_R):
        """Measure this context's enabled task residuals at a configuration."""
        target = mink.SE3.from_rotation_and_translation(
            mink.SO3.from_matrix(np.asarray(target_R, dtype=float)),
            np.asarray(target_pos, dtype=float),
        )
        self.ee_task.set_target(target)
        self.configuration.update(_project_qpos_to_limits(self.model, q))
        error = self.ee_task.compute_error(self.configuration)
        err_pos = float(np.linalg.norm(error[:3]))
        err_rot = float(np.linalg.norm(error[3:][self.orientation_mask]))
        return err_pos, err_rot


def solve_arm_ik(model, arm_qadr, arm_vadr, arm_ids, ee_ref,
                 target_pos_base, target_R_base, q_init=None,
                 w_rot=0.3, max_iter=100, tol_pos=8e-3, tol_rot=0.15,
                 lmbda=1e-2, step=0.3, mink_context=None, continuity_q=None,
                 continuity_max_step=None):
    """Solve 6-DoF IK with the Ego2Robot Mink differential-QP method.

    The legacy DLS arguments remain accepted for call-site compatibility; the
    task weights, limits, damping schedule, QP solve, and integration step are
    controlled by ``MinkIKContext`` as in the Ego2Robot implementation.
    """
    del arm_qadr, arm_vadr, w_rot, lmbda, step
    context = mink_context or MinkIKContext(model, arm_ids, ee_ref)
    return context.solve(
        target_pos_base,
        target_R_base,
        q_init=q_init,
        max_iterations=max_iter,
        dt=0.05,
        tol_pos=MINK_POSITION_TOL,
        tol_rot=MINK_ORIENTATION_TOL,
        continuity_q=continuity_q,
        continuity_max_step=continuity_max_step,
    )


def solve_arm_ik_position_first(
        model, arm_qadr, arm_vadr, arm_ids, ee_ref,
        target_pos_base, target_R_base, q_init=None,
        max_iter=500, tol_pos=POSITION_FIRST_TOL, tol_rot=0.25,
        mink_context=None, continuity_q=None,
        continuity_max_step=MINK_CONTINUITY_MAX_STEP):
    """Track position continuously while retaining orientation as a soft task.

    This mirrors EgoDex's control loop: integrate from the previous frame and
    stop as soon as the pinch center reaches the target. Orientation remains
    in the QP objective and is reported as full-pose quality, but it cannot
    force an otherwise aligned frame into a distant IK branch.
    """
    del arm_qadr, arm_vadr
    context = mink_context or MinkIKContext(model, arm_ids, ee_ref)
    q, position_ok, err_pos, err_rot = context.solve(
        target_pos_base, target_R_base, q_init=q_init,
        max_iterations=max_iter, dt=0.01, tol_pos=tol_pos,
        tol_rot=tol_rot, continuity_q=continuity_q,
        continuity_max_step=continuity_max_step, position_first=True)
    pose_ok = bool(position_ok and err_rot < tol_rot)
    return q, pose_ok, err_pos, err_rot


def refine_arm_ik_pose(
        model, arm_qadr, arm_vadr, arm_ids, ee_ref,
        target_pos_base, target_R_base, position_result, previous_q=None,
        max_iter=POSE_REFINEMENT_MAX_ITER, tol_pos=POSITION_FIRST_TOL,
        tol_rot=0.25,
        mink_context=None, continuity_max_step=MINK_CONTINUITY_MAX_STEP):
    """Improve orientation without losing position or temporal continuity.

    The position-first pass is always the safe baseline. This second pass is
    accepted only when it remains inside the position tolerance, reduces the
    enabled-axis rotation residual, and stays on the previous joint branch.
    """
    del arm_vadr
    q_position, _, err_pos, err_rot = position_result
    context = mink_context or MinkIKContext(model, arm_ids, ee_ref)
    q_refined, _, refined_pos, refined_rot = context.solve(
        target_pos_base, target_R_base, q_init=q_position,
        max_iterations=max_iter, dt=0.01, tol_pos=tol_pos,
        tol_rot=tol_rot, continuity_q=previous_q,
        continuity_max_step=continuity_max_step, position_first=False)
    continuous = (
        previous_q is None or continuity_max_step <= 0.0 or
        np.max(np.abs(
            q_refined[arm_qadr] - previous_q[arm_qadr])) <=
        continuity_max_step + 1e-6)
    if (refined_pos < tol_pos and refined_rot + 1e-6 < err_rot and
            continuous):
        return (q_refined, bool(refined_rot < tol_rot),
                refined_pos, refined_rot, True)
    return q_position, bool(err_pos < tol_pos and err_rot < tol_rot), \
        err_pos, err_rot, False


def _ee_pose_and_jacobian(model, data, ee_ref, jacp, jacr):
    """Read an end-effector pose and its MuJoCo geometric Jacobian."""
    kind, idx = ee_ref
    if kind == "site":
        pos = data.site_xpos[idx].copy()
        mat = data.site_xmat[idx].reshape(3, 3)
        mujoco.mj_jacSite(model, data, jacp, jacr, idx)
        quat = np.zeros(4)
        mujoco.mju_mat2Quat(quat, mat.reshape(-1))
    else:
        pos = data.xpos[idx].copy()
        quat = data.xquat[idx].copy()
        # Keep the original Panda body-frame Jacobian call exactly. Site EE
        # morphologies use mj_jacSite above because the legacy code had no
        # representation for them.
        mujoco.mj_jac(model, data, jacp, jacr, pos, idx)
    return pos, quat


def solve_arm_ik_dls(model, arm_qadr, arm_vadr, arm_ids, ee_ref,
                     target_pos_base, target_R_base, q_init=None,
                     w_rot=0.05, max_iter=300, tol_pos=8e-3, tol_rot=0.15,
                     lmbda=1e-2, step=0.3, mink_context=None):
    """Solve IK with the legacy MuJoCo damped-least-squares method.

    This is intentionally separate from ``solve_arm_ik``: the latter is the
    default Mink differential-QP implementation, while DLS remains available
    as an explicit alternative.
    """
    del mink_context
    data = mujoco.MjData(model)
    if q_init is not None:
        data.qpos[:] = q_init

    target_quat = np.zeros(4)
    mujoco.mju_mat2Quat(target_quat, np.asarray(target_R_base).reshape(-1))
    jacp = np.zeros((3, model.nv))
    jacr = np.zeros((3, model.nv))
    best = (float("inf"), float("inf"), data.qpos.copy())
    err_pos = err_rot = float("inf")

    for _ in range(max_iter):
        mujoco.mj_forward(model, data)
        cur_pos, cur_quat = _ee_pose_and_jacobian(
            model, data, ee_ref, jacp, jacr)
        e_pos = np.asarray(target_pos_base, dtype=float) - cur_pos

        neg = np.zeros(4)
        quat_delta = np.zeros(4)
        rvec = np.zeros(3)
        mujoco.mju_negQuat(neg, cur_quat)
        mujoco.mju_mulQuat(quat_delta, target_quat, neg)
        mujoco.mju_quat2Vel(rvec, quat_delta, 1.0)
        err_pos = float(np.linalg.norm(e_pos))
        err_rot = float(np.linalg.norm(rvec))

        # Preserve the original DLS fallback: only remember configurations
        # that already satisfy position, then prefer the best orientation.
        if err_pos < tol_pos and err_rot < best[1]:
            best = (err_pos, err_rot, data.qpos.copy())
        if err_pos < tol_pos and err_rot < tol_rot:
            return data.qpos.copy(), True, err_pos, err_rot

        if w_rot > 0.0:
            J = np.vstack((jacp[:, arm_vadr], w_rot * jacr[:, arm_vadr]))
            error = np.concatenate((e_pos, w_rot * rvec))
            regularizer = lmbda * np.eye(6)
        else:
            J = jacp[:, arm_vadr]
            error = e_pos
            regularizer = lmbda * np.eye(3)
        dq_sub = J.T @ np.linalg.solve(
            J @ J.T + regularizer, error)
        for i, qa in enumerate(arm_qadr):
            new_q = data.qpos[qa] + step * dq_sub[i]
            jid = arm_ids[i]
            if model.jnt_limited[jid]:
                lo, hi = model.jnt_range[jid]
                new_q = np.clip(new_q, lo, hi)
            data.qpos[qa] = new_q

    if best[0] < float("inf"):
        return best[2], False, best[0], best[1]
    mujoco.mj_forward(model, data)
    return data.qpos.copy(), False, err_pos, err_rot


def _prewarm_dls(model, arm_qadr, arm_vadr, arm_ids, ee_ref,
                 target_pos_base):
    """Position-only DLS prewarm used by the legacy robust DLS solver."""
    data = mujoco.MjData(model)
    jacp = np.zeros((3, model.nv))
    jacr = np.zeros((3, model.nv))
    for _ in range(DLS_PREWARM_MAX_ITER):
        mujoco.mj_forward(model, data)
        pos, _ = _ee_pose_and_jacobian(model, data, ee_ref, jacp, jacr)
        error = np.asarray(target_pos_base, dtype=float) - pos
        if np.linalg.norm(error) < 3e-3:
            break
        J = jacp[:, arm_vadr]
        dq_sub = J.T @ np.linalg.solve(
            J @ J.T + 1e-2 * np.eye(3), error)
        for i, qa in enumerate(arm_qadr):
            new_q = data.qpos[qa] + 0.5 * dq_sub[i]
            jid = arm_ids[i]
            if model.jnt_limited[jid]:
                lo, hi = model.jnt_range[jid]
                new_q = np.clip(new_q, lo, hi)
            data.qpos[qa] = new_q
    return data.qpos.copy()


def solve_arm_ik_dls_robust(model, arm_qadr, arm_vadr, arm_ids, ee_ref,
                            target_pos_base, target_R_base, q_warm,
                            tol_pos=8e-3, tol_rot=0.25, n_random=2,
                            seed=0, jump_penalty=1.0,
                            max_iter=DLS_MAX_ITER,
                            continuity_max_step=0.0):
    """Legacy DLS warm/prewarm/random-restart solver used by dls search."""
    def score(q, err_pos):
        if q_warm is None:
            return err_pos
        return err_pos + jump_penalty * float(
            np.linalg.norm(q[arm_qadr] - q_warm[arm_qadr]))

    def within_continuity_step(q):
        if q_warm is None or continuity_max_step <= 0.0:
            return True
        return bool(np.max(np.abs(
            q[arm_qadr] - q_warm[arm_qadr])) <= continuity_max_step + 1e-6)

    candidates = [q_warm, None]
    best = None
    best_ok = None
    warm_record = None
    for init in candidates:
        q0 = init if init is not None else _prewarm_dls(
            model, arm_qadr, arm_vadr, arm_ids, ee_ref, target_pos_base)
        q, _, err_pos, err_rot = solve_arm_ik_dls(
            model, arm_qadr, arm_vadr, arm_ids, ee_ref,
            target_pos_base, target_R_base, q_init=q0,
            max_iter=max_iter, tol_pos=tol_pos)
        record = (q, err_pos < tol_pos and err_rot < tol_rot and
                  within_continuity_step(q),
                  err_pos, err_rot, score(q, err_pos))
        if init is q_warm and q_warm is not None:
            warm_record = record
        if init is q_warm and record[1]:
            return q, True, err_pos, err_rot
        if best is None or record[4] < best[4]:
            best = record
        if record[1] and (best_ok is None or record[4] < best_ok[4]):
            best_ok = record

    rng = np.random.default_rng(seed)
    for _ in range(n_random):
        # The original restart seed initialized the complete qpos vector to
        # zero, then randomized only arm joints.
        q0 = np.zeros(model.nq)
        for i, qa in enumerate(arm_qadr):
            lo, hi = model.jnt_range[arm_ids[i]]
            q0[qa] = rng.uniform(lo, hi)
        q, _, err_pos, err_rot = solve_arm_ik_dls(
            model, arm_qadr, arm_vadr, arm_ids, ee_ref,
            target_pos_base, target_R_base, q_init=q0,
            max_iter=max_iter, tol_pos=tol_pos)
        record = (q, err_pos < tol_pos and err_rot < tol_rot and
                  within_continuity_step(q),
                  err_pos, err_rot, score(q, err_pos))
        if record[4] < best[4]:
            best = record
        if record[1] and (best_ok is None or record[4] < best_ok[4]):
            best_ok = record

    # If no compliant candidate exists, keep the previous configuration and
    # let the episode-level failed-frame interpolation repair the target. A
    # distant random branch is much more damaging than one held frame.
    chosen = (best_ok if best_ok is not None else
              warm_record if warm_record is not None else best)
    return chosen[0], chosen[1], chosen[2], chosen[3]


def solve_arm_ik_position_only_dls(model, arm_qadr, arm_vadr, arm_ids,
                                   ee_ref, target_pos_base, q_init=None,
                                   max_iter=100, tol_pos=5e-3,
                                   lmbda=1e-2, step=0.5, mink_context=None):
    """Position-only DLS feasibility check used by the DLS base search."""
    del mink_context
    data = mujoco.MjData(model)
    if q_init is not None:
        data.qpos[:] = q_init
    jacp = np.zeros((3, model.nv))
    jacr = np.zeros((3, model.nv))
    kind, ee_idx = ee_ref
    target_pos_base = np.asarray(target_pos_base, dtype=float)
    current_pos = np.zeros(3)
    for _ in range(max_iter):
        mujoco.mj_forward(model, data)
        if kind == "site":
            current_pos = data.site_xpos[ee_idx].copy()
            mujoco.mj_jacSite(model, data, jacp, jacr, ee_idx)
        else:
            current_pos = data.xpos[ee_idx].copy()
            mujoco.mj_jac(model, data, jacp, jacr, current_pos, ee_idx)
        error = target_pos_base - current_pos
        err_pos = float(np.linalg.norm(error))
        if err_pos < tol_pos:
            return True, err_pos
        J = jacp[:, arm_vadr]
        dq_sub = J.T @ np.linalg.solve(
            J @ J.T + lmbda * np.eye(3), error)
        for i, qa in enumerate(arm_qadr):
            new_q = data.qpos[qa] + step * dq_sub[i]
            jid = arm_ids[i]
            if model.jnt_limited[jid]:
                lo, hi = model.jnt_range[jid]
                new_q = np.clip(new_q, lo, hi)
            data.qpos[qa] = new_q
    # Match the legacy function's stale post-loop state (the final update is
    # intentionally not forwarded before measuring the residual).
    return False, float(np.linalg.norm(target_pos_base - current_pos))


def solve_arm_ik_robust(model, arm_qadr, arm_vadr, arm_ids, ee_ref,
                        target_pos_base, target_R_base, q_warm,
                        tol_pos=8e-3, tol_rot=0.25, n_random=2, seed=0,
                        jump_penalty=1.0, mink_context=None,
                        mink_position_context=None,
                        continuity_max_step=MINK_CONTINUITY_MAX_STEP,
                        max_iter=100):
    """Mink 6-DOF IK with optional retries and continuity scoring.

    Motivation (measured over 374 right-hand frames): a plain warm-start chain meets
    the position target for only 31% of frames because it cannot escape after drifting
    into a poor IK branch. Prewarm and random restarts raise this to 100%, with a mean
    position error of 6.2 mm and mean rotation error of 16.2 degrees. The 0.25 rad
    (14 degree) orientation tolerance is a compromise: Panda's 7 DOF cannot strictly
    satisfy both position and orientation for these human-hand poses.

    **Temporal jump penalty**: the redundant 7-DOF Panda admits multiple solutions for
    one EEF target (elbow flip, wrist flip, or a large q1 change). Selecting only by
    err_pos often lets prewarm/random candidates choose a lower-error but entirely
    different configuration, making the arm flip while the base and end effector stay
    still. Candidate score = `err_pos + jump_penalty * norm(q_arm - q_warm_arm)` so
    temporal continuity contributes to selection. A target-compliant warm-start
    solution is accepted immediately; fallback branches are considered only
    when the previous branch cannot satisfy the target.

    Returns (q_full, ok, err_pos, err_rot)."""
    def score(q, ep_, er_):
        if q_warm is None:
            # Once position is within tolerance, prefer the orientation
            # compliant candidate. Position-only scoring can otherwise keep a
            # warm branch with a 2 rad wrist error over a valid restart.
            return ep_ + (0.02 * er_ if ep_ < tol_pos else er_)
        d = float(np.linalg.norm(q[arm_qadr] - q_warm[arm_qadr]))
        return ep_ + jump_penalty * d + (0.02 * er_ if ep_ < tol_pos else er_)

    def within_continuity_step(q):
        if q_warm is None or continuity_max_step <= 0.0:
            return True
        delta = np.abs(q[arm_qadr] - q_warm[arm_qadr])
        return bool(np.max(delta) <= continuity_max_step + 1e-6)

    cands = [q_warm, None]
    best = None       # (q, ok, ep_, er_, s): lowest score among all candidates.
    best_ok = None    # Lowest-scoring compliant candidate (preferred when available).
    warm_record = None
    for init in cands:
        q0 = init if init is not None else _prewarm(
            model, arm_qadr, arm_vadr, arm_ids, ee_ref, target_pos_base,
            mink_context=mink_position_context)
        # Only the warm-start candidate is hard-constrained to the previous
        # frame. Random restarts remain available for genuinely unreachable or
        # numerically difficult targets, then continuity scoring decides whether
        # their compliant solution should be used.
        continuity_q = q_warm if init is q_warm and q_warm is not None else None
        q, _, ep_, er_ = solve_arm_ik(model, arm_qadr, arm_vadr, arm_ids,
                                      ee_ref, target_pos_base, target_R_base,
                                      q_init=q0, tol_pos=tol_pos,
                                      tol_rot=tol_rot, mink_context=mink_context,
                                      max_iter=max_iter,
                                      continuity_q=continuity_q,
                                      continuity_max_step=continuity_max_step)
        s = score(q, ep_, er_)
        ok = (ep_ < tol_pos and er_ < tol_rot and
              within_continuity_step(q))
        if init is q_warm and q_warm is not None:
            warm_record = (q, ok, ep_, er_, s)
        # Match the reference's frame-by-frame control loop: once the previous
        # branch still reaches the target, keep it. A random/prewarm candidate
        # must not replace it for a marginally smaller residual.
        if init is q_warm and ok:
            return q, True, ep_, er_
        if best is None or s < best[4]:
            best = (q, ok, ep_, er_, s)
        if ok and (best_ok is None or s < best_ok[4]):
            best_ok = (q, ok, ep_, er_, s)
    rng = np.random.default_rng(seed)
    for _ in range(n_random):
        q0 = _project_qpos_to_limits(model, model.qpos0)
        for i, qa in enumerate(arm_qadr):
            lo, hi = model.jnt_range[arm_ids[i]]
            q0[qa] = rng.uniform(lo, hi)
        q, _, ep_, er_ = solve_arm_ik(model, arm_qadr, arm_vadr, arm_ids,
                                      ee_ref, target_pos_base, target_R_base,
                                      q_init=q0, tol_pos=tol_pos,
                                      tol_rot=tol_rot, mink_context=mink_context,
                                      max_iter=max_iter)
        s = score(q, ep_, er_)
        ok = (ep_ < tol_pos and er_ < tol_rot and
              within_continuity_step(q))
        if s < best[4]:
            best = (q, ok, ep_, er_, s)
        if ok and (best_ok is None or s < best_ok[4]):
            best_ok = (q, ok, ep_, er_, s)
    chosen = (best_ok if best_ok is not None else
              warm_record if warm_record is not None else best)
    return chosen[0], chosen[1], chosen[2], chosen[3]


def solve_arm_ik_position_priority(
        model, arm_qadr, arm_vadr, arm_ids, ee_ref,
        target_pos_base, target_R_base, q_warm,
        tol_pos=8e-3, tol_rot=0.25, branch_jump_threshold=0.6,
        mink_context=None, mink_position_context=None,
        mink_fallback_context=None,
        continuity_max_step=MINK_CONTINUITY_MAX_STEP):
    """Solve underactuated IK with a continuous position-first fallback.

    The regular pose solve remains preferred when it satisfies the complete
    morphology-specific task. If it fails, or reaches the target through a
    distant joint branch, retry with a position-priority task from the previous frame. This
    keeps the physical pinch center on the human hand without turning a soft
    orientation preference into a long run of failed frames that is later
    interpolated in joint space.

    Returns the normal IK tuple followed by ``used_fallback`` and
    ``avoided_branch_jump`` flags.
    """
    if (mink_context is None or mink_position_context is None or
            mink_fallback_context is None):
        raise ValueError("position-priority IK requires Mink task contexts")

    full = solve_arm_ik_robust(
        model, arm_qadr, arm_vadr, arm_ids, ee_ref,
        target_pos_base, target_R_base, q_warm,
        tol_pos=tol_pos, tol_rot=tol_rot,
        mink_context=mink_context,
        mink_position_context=mink_position_context,
        continuity_max_step=continuity_max_step)
    q_full, full_ok, _, _ = full
    full_jump = (0.0 if q_warm is None else float(np.max(np.abs(
        q_full[arm_qadr] - q_warm[arm_qadr]))))
    branch_jump = (branch_jump_threshold > 0.0 and
                   full_jump > branch_jump_threshold)
    if full_ok and not branch_jump:
        return (*full, False, False)

    q_pos, _, _, _ = mink_fallback_context.solve(
        target_pos_base,
        target_R_base,
        q_init=q_warm,
        max_iterations=200,
        dt=0.05,
        tol_pos=tol_pos,
        tol_rot=tol_rot,
        continuity_q=q_warm,
        continuity_max_step=continuity_max_step,
    )
    pos_jump = (0.0 if q_warm is None else float(np.max(np.abs(
        q_pos[arm_qadr] - q_warm[arm_qadr]))))
    fallback_improves_branch = not branch_jump or pos_jump < full_jump
    err_pos, err_rot = mink_context.measure_error(
        q_pos, target_pos_base, target_R_base)
    if err_pos < tol_pos and fallback_improves_branch:
        return q_pos, err_pos < tol_pos, err_pos, err_rot, True, branch_jump

    return (*full, False, False)


def solve_arm_ik_position_only(model, arm_qadr, arm_vadr, arm_ids, ee_ref,
                               target_pos_base, q_init=None,
                               max_iter=100, tol_pos=5e-3,
                               lmbda=1e-2, step=0.5,
                               mink_context=None):
    """Mink position-only QP used by Base Pose Search prefiltering."""
    del lmbda, step
    context = mink_context or MinkIKContext(
        model, arm_ids, ee_ref, orientation_cost=0.0)
    _, ok, err_pos, _ = context.solve(
        target_pos_base,
        np.eye(3),
        q_init=q_init,
        max_iterations=max_iter,
        dt=0.05,
        tol_pos=tol_pos,
        tol_rot=np.inf,
    )
    return ok, err_pos


def _prewarm(model, arm_qadr, arm_vadr, arm_ids, ee_ref, target_pos,
             mink_context=None):
    """Mink position-only prewarm used before the full pose solve."""
    del arm_qadr, arm_vadr
    context = mink_context or MinkIKContext(
        model, arm_ids, ee_ref, orientation_cost=0.0)
    q, _, _, _ = context.solve(
        target_pos,
        np.eye(3),
        q_init=model.qpos0,
        max_iterations=100,
        dt=0.05,
        tol_pos=3e-3,
        tol_rot=np.inf,
    )
    return q
