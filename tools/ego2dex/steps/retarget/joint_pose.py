# Copyright 2026 The LoongForge Authors.
# SPDX-License-Identifier: Apache-2.0
"""Bounded joint base/parallel-jaw symmetry search, validated over full tracks.

The 180-degree alternative is fixed for each arm over the entire episode.
Only the Panda's symmetric finger geometry is currently supported.
"""

from dataclasses import replace
from itertools import product
import numpy as np
import mujoco
from scipy.spatial.transform import Rotation
from .continuity import ContinuousProblem, ContinuityConfig, reconstruct_continuous_trajectory
from .posture import refine_posture, refine_recovery_stages
from .rendering import build_dual_model, mat_to_quat
from .targets import _quat_to_mat


def jaw_rotation(spec, side):
    quat = spec.tcp_rot_site_left if side == 0 else spec.tcp_rot_site_right
    return _quat_to_mat((quat or spec.tcp_rot_site) if spec.ee_site else spec.tcp_rot_body)


def symmetric_targets(p, R, spec, flips):
    if spec.name != "panda" or spec.gripper_mode != "symmetric":
        raise ValueError("Jaw symmetry has only been validated for Panda")
    offset = np.asarray(spec.tcp_pos_site if spec.ee_site else spec.tcp_pos_body)
    target_p, target_R = np.array(p, copy=True), np.array(R, copy=True)
    for side, flip in enumerate(flips):
        if not flip:
            continue
        tcp = jaw_rotation(spec, side)
        symmetry = tcp @ np.diag([-1.0, -1.0, 1.0]) @ tcp.T
        target_R[:, side] = R[:, side] @ symmetry
        # Keep the physical TCP fixed, even when the IK reference is offset.
        center = p[:, side] + np.einsum("nij,j->ni", R[:, side], offset)
        target_p[:, side] = center - np.einsum("nij,j->ni", target_R[:, side], offset)
    return target_p, target_R


def pose_metrics(problem, track, original_R, spec):
    raw = np.full((len(track), 2), np.nan)
    equivalent = raw.copy()
    approach = raw.copy()
    opening = raw.copy()
    for t, q in enumerate(track):
        problem.data.qpos[:] = q
        mujoco.mj_forward(problem.model, problem.data)
        for side, (kind, index) in enumerate(problem.refs):
            if not problem.masks[t, side]:
                continue
            R = (problem.data.site_xmat[index] if kind == "site" else problem.data.xmat[index]).reshape(3, 3)
            tcp = jaw_rotation(spec, side)
            target = original_R[t, side] @ tcp
            actual = R @ tcp
            raw[t, side] = Rotation.from_matrix(target.T @ actual).magnitude()
            alternative = target @ np.diag([-1.0, -1.0, 1.0])
            equivalent[t, side] = min(raw[t, side], Rotation.from_matrix(alternative.T @ actual).magnitude())
            approach[t, side] = np.arccos(np.clip(target[:, 2] @ actual[:, 2], -1, 1))
            opening[t, side] = np.arccos(np.clip(abs(target[:, 1] @ actual[:, 1]), 0, 1))

    def means(values):
        return [
            float(np.nanmean(values[:, s]) * 180 / np.pi) if np.isfinite(values[:, s]).any() else None for s in range(2)
        ]

    return dict(
        raw_orientation_mean_deg=means(raw),
        symmetry_orientation_mean_deg=means(equivalent),
        symmetry_orientation_p95_deg=float(np.nanpercentile(equivalent, 95) * 180 / np.pi),
        approach_mean_deg=means(approach),
        opening_line_mean_deg=means(opening),
    )


def angular_eligible(report, baseline):
    substantial = score(report) <= score(baseline) * 0.8
    for key in ("symmetry_orientation_mean_deg", "approach_mean_deg", "opening_line_mean_deg"):
        old = np.asarray(baseline[key], dtype=float)
        new = np.asarray(report[key], dtype=float)
        valid = np.isfinite(old)
        limit = np.maximum(old + 0.25, 5.0) if substantial else old + 0.25
        if np.any(~np.isfinite(new[valid])) or np.any(new[valid] > limit[valid]):
            return False
    return report["symmetry_orientation_p95_deg"] <= baseline["symmetry_orientation_p95_deg"] + 0.25


def eligible(report, baseline):
    if not report["accepted"] or not angular_eligible(report, baseline):
        return False
    # A stuck seed is not a temporal-quality baseline. Its measured angular
    # errors remain useful: recovery must not trade correct jaws for position.
    if not baseline["accepted"]:
        return True
    substantial = score(report) <= score(baseline) * 0.8
    # Changing a base/IK branch can require a longer, but smoother path.
    # This bounded tradeoff applies only to a >=20% pose improvement.
    velocity_ratio = 1.25 if substantial else 1.05
    if report["velocity_rms_rad_s"] > baseline["velocity_rms_rad_s"] * velocity_ratio + 1e-6:
        return False
    if report["acceleration_rms_rad_s2"] > baseline["acceleration_rms_rad_s2"] * 1.05 + 1e-6:
        return False
    if (
        "max_acceleration_rad_s2" in report
        and "max_acceleration_rad_s2" in baseline
        and report["max_acceleration_rad_s2"] > baseline["max_acceleration_rad_s2"] * 1.05 + 1e-6
    ):
        return False
    return True


def score(report):
    return float(np.nanmean(np.asarray(report["symmetry_orientation_mean_deg"], dtype=float)))


def joint_pose_search(
    model, initial, qadr, refs, p, R, masks, fps, spec, base_positions, base_quaternions, config=None, shortlist=4
):
    """Return a selected model, trajectory, targets, bases, and diagnostics."""
    cfg = config or ContinuityConfig()
    initial = np.asarray(initial)
    original = ContinuousProblem(model, qadr, refs, p, R, masks, fps, cfg)
    before, _, _ = original.measure(initial)
    before.update(pose_metrics(original, initial, R, spec))
    angular_guard = dict(targets=R, tcp=[jaw_rotation(spec, s) for s in range(2)], baseline=before)
    recovering = not before["accepted"]
    before.update(flips=[False, False], base_offset_m=0.0, base_yaw_deg=0.0)
    best = (
        dict(
            model=model,
            qpos=initial.copy(),
            p=p.copy(),
            R=R.copy(),
            positions=np.array(base_positions),
            quaternions=np.array(base_quaternions),
            report=before,
        )
        if not recovering
        else None
    )
    initial_repair = None
    if recovering and before["position_max_m"] <= 0.03:
        # Repair the existing branch before paying for alternate-base search.
        # All fractions must meet the same physical and angular guards.
        repaired_candidates = []
        _, diagnostics, _, _ = refine_recovery_stages(
            model,
            initial,
            qadr,
            refs,
            p,
            R,
            masks,
            fps,
            max_nfev=160,
            config=cfg,
            angular_guard=angular_guard,
            candidate_callback=lambda q, r: repaired_candidates.append((q, r)),
        )
        repaired_reports = []
        for q, report in repaired_candidates:
            report.update(pose_metrics(original, q, R, spec))
            report.update(
                flips=[False, False],
                base_offset_m=0.0,
                base_yaw_deg=0.0,
                recovery_method="initial_branch_whole_track",
                eligible=eligible(report, before),
            )
            repaired_reports.append(report)
            if report["eligible"] and (
                best is None or report["acceleration_rms_rad_s2"] < best["report"]["acceleration_rms_rad_s2"]
            ):
                best = dict(
                    model=model,
                    qpos=q,
                    p=p.copy(),
                    R=R.copy(),
                    positions=np.array(base_positions),
                    quaternions=np.array(base_quaternions),
                    report=report,
                )
        initial_repair = dict(optimizer=diagnostics, candidates=repaired_reports)
        if best is not None:
            return best, dict(
                before=before,
                after=best["report"],
                attempts=repaired_reports,
                initial_repair=initial_repair,
                recovery_refinements=[],
                recovered=True,
                initial_seed_accepted=False,
                search_exhausted=False,
                changed=True,
                screen=[],
                selection_policy="unchanged absolute physical and angular non-regression guards",
            )
    candidates = []
    attempts = []
    rejected = []
    n = len(initial)
    frames = np.unique(np.r_[np.linspace(0, n - 1, 7).astype(int), n // 2, n // 4, 3 * n // 4])
    direction = np.asarray(base_positions[1]) - base_positions[0]
    direction = direction / max(np.linalg.norm(direction), 1e-9)
    fovy = float(model.cam_fovy[mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_CAMERA, "ego")])
    bases = ((0.0, 0.0), (0.05, 0.0), (0.10, 0.0), (0.05, 20.0), (0.05, -20.0))
    rng = np.random.default_rng(20261028)
    for offset, yaw in bases:
        positions = np.array(base_positions) + offset * np.array([-direction, direction])
        quats = []
        for side, quat in enumerate(base_quaternions):
            turn = Rotation.from_rotvec([0, 0, np.deg2rad(yaw) * (1 if side == 0 else -1)]).as_matrix()
            quats.append(mat_to_quat(_quat_to_mat(quat) @ turn))
        trial = build_dual_model(positions[0], quats[0], positions[1], quats[1], fovy, spec)
        trial.geom_margin[:] = np.maximum(trial.geom_margin, cfg.clearance)
        for flips in product((False, True), repeat=2):
            tp, tR = symmetric_targets(p, R, spec, flips)
            problem = ContinuousProblem(
                trial, qadr, refs, tp, tR, masks, fps, replace(cfg, orientation_weight=1.0, max_nfev=45)
            )
            seeds = initial.copy()
            losses = []
            failed = 0
            for t in frames:
                problem.frame(t, initial[t])
                options = []
                for k in range(2):
                    seed = initial[t, qadr].copy()
                    if k:
                        seed = np.clip(seed + rng.normal(0, 0.65, len(qadr)), problem.lower, problem.upper)
                    q = problem.solve(seed, bounded=False)
                    residual, _ = problem.evaluate(q)
                    errors = [
                        np.linalg.norm(residual[s * 6 : s * 6 + 3]) / cfg.position_weight
                        for s in range(2)
                        if masks[t, s]
                    ]
                    loss = float(np.sum(residual[:12] ** 2) + np.sum(residual[12 + 2 * len(qadr) :] ** 2))
                    options.append((max(errors, default=0) > cfg.position_tolerance, loss, q))
                bad, loss, q = min(options, key=lambda v: (v[0], v[1]))
                seeds[t, qadr] = q
                losses.append(loss)
                failed += bad
            candidates.append(
                dict(
                    model=trial,
                    seeds=seeds,
                    p=tp,
                    R=tR,
                    positions=positions,
                    quaternions=np.array(quats),
                    flips=list(flips),
                    offset=offset,
                    yaw=yaw,
                    screen_failed=int(failed),
                    screen_loss=float(np.mean(losses)),
                )
            )
            print(
                f"joint-pose screen offset={offset} yaw={yaw} flips={flips} failed={failed} loss={np.mean(losses):.4f}",
                flush=True,
            )
    ordered = sorted(candidates, key=lambda c: (c["screen_failed"], c["screen_loss"]))
    # Include the best fixed-base symmetry alternative as an explicit control.
    chosen = ordered[:shortlist]
    fixed = next(c for c in ordered if c["offset"] == 0 and c["yaw"] == 0)
    if not any(c is fixed for c in chosen):
        chosen.append(fixed)
    if recovering:
        # A sparse screen ranks candidates, but cannot establish full-track
        # infeasibility. If the shortlist fails, evaluate the remaining bases.
        chosen += [c for c in ordered if not any(c is x for x in chosen)]
    for c in chosen:
        track, report, _, _ = reconstruct_continuous_trajectory(
            c["model"],
            c["seeds"],
            qadr,
            refs,
            c["p"],
            c["R"],
            masks,
            fps,
            replace(cfg, orientation_weight=1.0, max_seeds=3),
        )
        problem = ContinuousProblem(c["model"], qadr, refs, c["p"], c["R"], masks, fps, cfg)
        report.update(pose_metrics(problem, track, R, spec))
        report.update(flips=c["flips"], base_offset_m=c["offset"], base_yaw_deg=c["yaw"], screen_loss=c["screen_loss"])
        report["eligible"] = eligible(report, before)
        attempts.append(report)
        if recovering and not eligible(report, before):
            rejected.append((c, track.copy(), report))
        print(f"joint-pose candidate {report}", flush=True)
        # Try whole-track temporal repair on any feasible alternate branch.
        if report["accepted"]:
            refinement_candidates = []
            refined, refinement, _, _ = refine_posture(
                c["model"],
                track,
                qadr,
                refs,
                c["p"],
                c["R"],
                masks,
                fps,
                max_nfev=120,
                config=cfg,
                angular_guard=angular_guard,
                candidate_callback=lambda q, r: refinement_candidates.append((q, r)),
            )
            report["refinement_diagnostics"] = refinement
            for refined, r in refinement_candidates:
                r.update(pose_metrics(problem, refined, R, spec))
                r.update(
                    flips=c["flips"], base_offset_m=c["offset"], base_yaw_deg=c["yaw"], eligible=eligible(r, before)
                )
                attempts.append(r)
                if eligible(r, before) and (not eligible(report, before) or score(r) < score(report)):
                    track, report = refined, r
        if eligible(report, before) and (best is None or score(report) < score(best["report"]) - 0.05):
            best = dict(
                model=c["model"],
                qpos=track,
                p=c["p"],
                R=c["R"],
                positions=c["positions"],
                quaternions=c["quaternions"],
                report=report,
            )
        if recovering and best is not None:
            # First fully validated candidate establishes feasibility. Do not
            # compare acceleration/pose to an invalid trajectory or silently
            # spend the full search budget after recovery.
            break
    recovery_refinements = []
    if recovering and best is None:
        # Sparse/forward propagation can get trapped around a few difficult
        # frames. Optimize all frames together on the three closest tracks.
        # The original seed is also eligible for repair; alternate bases are
        # not assumed to be better than the original.
        rejected.append(
            (
                dict(
                    model=model,
                    p=p,
                    R=R,
                    positions=np.array(base_positions),
                    quaternions=np.array(base_quaternions),
                    flips=[False, False],
                    offset=0.0,
                    yaw=0.0,
                ),
                initial.copy(),
                before,
            )
        )
        ranked = sorted(
            rejected,
            key=lambda item: (
                not angular_eligible(item[2], before),
                item[2]["self_collision_frames"]
                + item[2]["cross_arm_collision_frames"]
                + item[2]["midpoint_collision_intervals"],
                item[2]["position_max_m"],
                item[2]["position_failed_frames"],
            ),
        )
        for c, track, failed in ranked[:3]:
            # This is a local repair budget, not a relaxed acceptance threshold.
            if failed["position_max_m"] > 0.03:
                continue
            print(
                f"joint-pose global recovery offset={c['offset']} yaw={c['yaw']} "
                f"flips={c['flips']} error_mm={failed['position_max_m'] * 1000:.3f}",
                flush=True,
            )
            refined, diagnostics, _, _ = refine_posture(
                c["model"],
                track,
                qadr,
                refs,
                c["p"],
                c["R"],
                masks,
                fps,
                max_nfev=160,
                config=cfg,
                allow_infeasible=True,
                angular_guard=angular_guard,
            )
            problem = ContinuousProblem(c["model"], qadr, refs, c["p"], c["R"], masks, fps, cfg)
            report, _, _ = problem.measure(refined)
            report.update(pose_metrics(problem, refined, R, spec))
            report.update(
                flips=c["flips"],
                base_offset_m=c["offset"],
                base_yaw_deg=c["yaw"],
                recovery_method="whole_track",
                eligible=eligible(report, before),
            )
            recovery_refinements.append(dict(before=failed, after=report, optimizer=diagnostics))
            print(f"joint-pose global recovery result {report}", flush=True)
            if eligible(report, before):
                best = dict(
                    model=c["model"],
                    qpos=refined,
                    p=c["p"],
                    R=c["R"],
                    positions=c["positions"],
                    quaternions=c["quaternions"],
                    report=report,
                )
                break
    # Search margins are not physical geometry; restore the original policy.
    if best is not None:
        best["model"].geom_margin[:] = model.geom_margin
    return best, dict(
        before=before,
        after=best["report"] if best else None,
        attempts=attempts,
        initial_repair=initial_repair,
        recovered=bool(recovering and best is not None),
        recovery_refinements=recovery_refinements,
        initial_seed_accepted=bool(before["accepted"]),
        search_exhausted=bool(best is None),
        changed=bool(best is not None and not np.array_equal(best["qpos"], initial)),
        screen=[{k: c[k] for k in ("flips", "offset", "yaw", "screen_failed", "screen_loss")} for c in candidates],
        symmetry_policy="one constant 180-degree TCP approach-axis rotation per arm; Panda only",
        selection_policy=dict(
            hard_velocity_rad_s=cfg.max_velocity,
            hard_position_tolerance_m=cfg.position_tolerance,
            acceleration_rms_max_ratio=1.05,
            acceleration_peak_max_ratio=1.05,
            substantial_pose_improvement_fraction=0.20,
            substantial_velocity_rms_max_ratio=1.25,
            other_velocity_rms_max_ratio=1.05,
            substantial_other_arm_angle_floor_deg=5.0,
        ),
    )
