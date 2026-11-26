# Copyright 2026 The LoongForge Authors.
# SPDX-License-Identifier: Apache-2.0
"""Equivalent sparseGNQP without forming dense global-angular Hessian blocks."""

import numpy as np
from scipy import sparse
import clarabel


def bounded_step(
    jac, residual, x, lower, upper, differences, speed_step, caps, damping=1.0, coordinate_map=None, linear_guard=None
):
    mapping = sparse.eye(len(x), format="csc") if coordinate_map is None else sparse.csc_matrix(coordinate_map)
    J = sparse.csr_matrix(jac @ mapping)
    r = np.asarray(residual)
    J.eliminate_zeros()
    # Per-episode means/percentiles couple thousands of joints. Represent their
    # few residuals with auxiliary variables instead of materializing J.T@J.
    dense = np.diff(J.indptr) > 256
    global_J = sparse.csc_matrix(J[dense])
    local = sparse.csc_matrix(J[~dense])
    m = int(dense.sum())
    n = J.shape[1]
    H = sparse.csc_matrix(local.T @ local + damping * sparse.eye(n))
    g = np.asarray(local.T @ r[~dense]).ravel()
    units = 1 / np.sqrt(np.maximum(H.diagonal(), 1.0))
    S = sparse.diags(units)
    P = sparse.block_diag([S @ H @ S, sparse.eye(m)], format="csc")
    q = np.r_[units * g, r[dense]]
    A = sparse.vstack([mapping, differences @ mapping], format="csc")
    v = differences @ x
    L = np.r_[np.maximum(lower - x, -caps), -speed_step - v]
    U = np.r_[np.minimum(upper - x, caps), speed_step - v]
    if linear_guard is not None:
        G, GL, GU = linear_guard
        A = sparse.vstack([A, sparse.csc_matrix(G) @ mapping], format="csc")
        L = np.r_[L, GL]
        U = np.r_[U, GU]
    if np.any(L > U):
        raise ValueError("Empty step bounds")
    AA = sparse.hstack([A @ S, sparse.csc_matrix((A.shape[0], m))], format="csc")
    equal = np.isfinite(L) & np.isfinite(U) & (abs(L - U) < 1e-12)
    up = np.isfinite(U) & ~equal
    down = np.isfinite(L) & ~equal
    extra = sparse.hstack([-global_J @ S, sparse.eye(m)], format="csc")
    constraints = sparse.vstack([extra, AA[equal], AA[up], -AA[down]], format="csc")
    rhs = np.r_[np.zeros(m), U[equal], U[up], -L[down]]
    scale = 1 / np.maximum(np.asarray(abs(constraints).max(axis=1).toarray()).ravel(), 1e-9)
    scale = np.minimum(scale, 1e6)
    constraints = sparse.csc_matrix(sparse.diags(scale) @ constraints)
    rhs *= scale
    cones = [clarabel.ZeroConeT(m + int(equal.sum())), clarabel.NonnegativeConeT(int(up.sum() + down.sum()))]
    settings = clarabel.DefaultSettings()
    settings.verbose = False
    settings.max_iter = 150
    settings.tol_gap_abs = 1e-9
    settings.tol_gap_rel = 1e-9
    settings.tol_feas = 1e-9
    result = clarabel.DefaultSolver(sparse.triu(P, format="csc"), q, constraints, rhs, cones, settings).solve()
    report = dict(
        status=str(result.status),
        iterations=result.iterations,
        primal_residual=result.r_prim,
        dual_residual=result.r_dual,
        equivalent_global_residual_auxiliaries=m,
        hessian_nonzeros=P.nnz,
        solver="clarabel_lowrank",
    )
    # Search directions need not be QP-optimal. Clarabel occasionally reports
    # InsufficientProgress after already finding a useful near-KKT direction.
    # Keep that status visible and use it ONLY if explicit unscaled constraints
    # and the original quadratic descent test pass below. This is not a robot
    # acceptance gate; all nonlinear final acceptance remains independent.
    optimal = str(result.status) in ["Solved", "AlmostSolved"]
    near_kkt = str(result.status) == "InsufficientProgress" and result.r_prim <= 1e-7 and result.r_dual <= 1e-7
    if not (optimal or near_kkt):
        return None, report
    step = units * np.asarray(result.x[:n])
    actual = A @ step
    violation = max(float(np.max(L - actual)), float(np.max(actual - U)), 0.0)
    full_gradient = np.asarray(J.T @ r).ravel()
    report.update(constraint_violation=violation, predicted_linear_change=float(full_gradient @ step))
    if violation > 1e-7:
        report["rejected_reason"] = "linear_constraint_violation"
        return None, report
    change = float(full_gradient @ step + 0.5 * np.dot(J @ step, J @ step) + 0.5 * damping * np.dot(step, step))
    report.update(
        predicted_quadratic_change=change, qp_optimal_status=optimal, suboptimal_search_direction_only=not optimal
    )
    report["feasibility_restoration_direction"] = bool(linear_guard is not None and change >= -1e-10)
    if not np.isfinite(step).all() or not np.isfinite(change) or (change >= -1e-10 and linear_guard is None):
        report["rejected_reason"] = "not_a_finite_quadratic_descent_direction"
        return None, report
    return mapping @ step, report
