# Copyright 2026 The LoongForge Authors.
# SPDX-License-Identifier: Apache-2.0

"""Scene reconstruction for ego2robot.

This package reconstructs the EgoVerse scene from an ego-centric video using
NVIDIA DVLT (Déjà View) and re-synthesizes the scene in the SLAM world frame
(metric), producing a ``scene_depth.npz`` that is interface-compatible with the
DA3 depth produced by :mod:`steps.depth_estimate`.

The DVLT checkpoint is feed-forward (no per-episode training): a few hundred
frames give camera poses, per-pixel depth, world points and confidence.  A
similarity transform (Umeyama) registers the DVLT reconstruction into the SLAM
metric frame, after which the world points are projected for every aligned
frame with the same effective camera as the retarget renderer
(``fx=fy, cx=w/2, cy=h/2``), so the resulting depth grid matches the MuJoCo
robot depth pixel-for-pixel.

Milestone 1: deterministic depth replacement (``backend="gauss"``) for the
retarget occlusion/compositing stage.  The raw DVLT output is cached as
``{ep}_dvlt.pt`` for the follow-up gsplat training milestone.
"""