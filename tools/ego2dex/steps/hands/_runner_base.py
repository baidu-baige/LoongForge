# Copyright 2026 The LoongForge Authors.
# SPDX-License-Identifier: Apache-2.0
"""Orientation-native end-effector stage, using only verified Orientation cache."""

import os
import hashlib
from .registry import selected
import json
from pathlib import Path
import shutil

import numpy as np

from .hand_swap import swap_grippers


def run_cached(args):
    os.environ["EGO2ROBOT_END_EFFECTOR"] = args.end_effector
    if args.robot_type != "panda":
        raise ValueError("Registered dexterous hands are currently supported only for Panda")
    if not args.orientation_cache or not args.hand_reference_dir:
        raise ValueError("Sharpa cache mode requires --orientation_cache and --hand_reference_dir")
    cache = Path(args.orientation_cache).resolve()
    reference = Path(args.hand_reference_dir).resolve()
    out = Path(args.output_dir).resolve()
    out.mkdir(parents=True, exist_ok=True)
    manifest = json.loads((reference / "manifest.json").read_text())
    ep = Path(manifest["episode"]).name
    accepted = cache / f"06_retarget/{ep}_ik.npz"
    source_path = accepted
    kind = "completed_orientation_output"
    run_status_path = cache.parent / "run_status.json"
    original_run = json.loads(run_status_path.read_text()) if run_status_path.exists() else {"status": "completed"}
    if not accepted.exists():
        source_path = cache / f"06_retarget/{ep}_joint_pose_seed.npz"
        if not source_path.exists():
            raise FileNotFoundError(source_path)
        kind = "unaccepted_parallel_gripper_seed_for_new_tool_optimization"
    verification = source_path.parent.parent.parent / "verification.json"
    # Verification belongs to the original case, outside its output stages.
    verification = Path(args.orientation_cache).resolve().parent / "verification.json"
    upstream_verified = json.loads(verification.read_text()).get("accepted", False) if verification.exists() else False
    status = {
        "stage": "registered_hand_render",
        "generation_completed": False,
        "original_orientation_run_status": original_run.get("status"),
        "original_arm_accepted": bool(kind == "completed_orientation_output" and upstream_verified),
        "arm_source_kind": kind,
        "arm_source": str(source_path),
    }
    (out / "status.json").write_text(json.dumps(status, indent=2))

    def stage(name):
        status["stage"] = name
        (out / "status.json").write_text(json.dumps(status, indent=2))

    report = swap_grippers(cache, reference, out, stage, depth_mode=args.hand_depth_mode, make_comparison=False)
    report.update(
        original_arm_accepted=status["original_arm_accepted"],
        original_orientation_run_status=original_run.get("status"),
        arm_source_kind=kind,
        arm_source=str(source_path),
        arm_source_sha256=hashlib.sha256(source_path.read_bytes()).hexdigest(),
        original_rejection_preserved=kind != "completed_orientation_output",
        result_only=True,
    )
    (out / "replacement_report.json").write_text(json.dumps(report, indent=2))
    status.update(
        stage="result_video_generated",
        generation_completed=True,
        result_video="robot_on_bg.mp4",
        rigid_tool_mapping_passed=report["rigid_arm_mapping"]["passed"],
        quality_passed=False,
    )
    (out / "status.json").write_text(json.dumps(status, indent=2))
    (out / "artifact_sha256.json").write_text(
        json.dumps(
            {
                str(p.relative_to(out)): hashlib.sha256(p.read_bytes()).hexdigest()
                for p in out.rglob("*")
                if p.is_file() and p.name not in ["artifact_sha256.json", "status.json"]
            },
            indent=2,
        )
    )
    return report


def replace_pipeline_outputs(args, root, original_retarget, replacement_fn=None):
    os.environ["EGO2ROBOT_END_EFFECTOR"] = args.end_effector
    """Post-process successful native Panda outputs, then feed the normal writer."""
    from steps.loader import load_episode, find_invalid_frames

    output = Path(root) / ("06_retarget_" + selected())
    output.mkdir(exist_ok=True)
    episodes = {p.name[: -len("_ik.npz")] for p in Path(original_retarget).glob("*_ik.npz")} | {
        p.name[: -len("_joint_pose_seed.npz")] for p in Path(original_retarget).glob("*_joint_pose_seed.npz")
    }
    for ep in sorted(episodes):
        ik = Path(original_retarget) / f"{ep}_ik.npz"
        if not ik.exists():
            ik = Path(original_retarget) / f"{ep}_joint_pose_seed.npz"
        if args.episodes and ep not in args.episodes:
            continue
        old = np.load(ik)
        data = load_episode(str(Path(args.input_dir) / ep))
        n = len(old["qpos"])
        assert data["total_frames"] == n
        reference = output / "_references" / ep
        reference.mkdir(parents=True, exist_ok=True)
        fields = {
            k: data[k]
            for k in [
                "left_obs_keypoints",
                "right_obs_keypoints",
                "left_obs_wrist_pose",
                "right_obs_wrist_pose",
                "obs_head_pose",
                "timestamps_ns",
            ]
        }
        # Resolution comes from the existing inpainted video, without resizing.
        import cv2

        cap = cv2.VideoCapture(str(Path(root) / f"04_inpaint/{ep}/bg.mp4"))
        ok, image = cap.read()
        cap.release()
        assert ok
        H, W = image.shape[:2]
        from steps.config import dataset_intrinsics_k

        K = old["camera_K"] if "camera_K" in old else dataset_intrinsics_k(data["attributes"], img_shape=(H, W))
        if K is None:
            raise ValueError("Original camera intrinsics required")
        np.savez_compressed(
            reference / "hand_reference.npz",
            **fields,
            valid=find_invalid_frames(data),
            intrinsics=K,
            source_frame_indices=np.arange(n),
            fps=args.target_fps,
        )
        (reference / "manifest.json").write_text(
            json.dumps(dict(episode=ep, frames=n, width=W, height=H, fps=args.target_fps), indent=2)
        )
        render = output / "_render" / ep
        report = (replacement_fn or swap_grippers)(
            root, reference, render, depth_mode=args.hand_depth_mode, make_comparison=False
        )
        new = np.load(render / "trajectory.npz")
        metadata = {k: old[k] for k in old.files}
        metadata.update(
            qpos=new["qpos"],
            state=new["state"],
            state_names=new["state_names"],
            robot_type=np.array("panda_" + selected()),
            original_orientation_qpos=old["qpos"],
            hand_quality_passed=np.array(False),
        )
        np.savez_compressed(output / f"{ep}_ik.npz", **metadata)
        shutil.copy2(render / "robot_on_bg.mp4", output / f"{ep}_robot_on_bg.mp4")
    return output
