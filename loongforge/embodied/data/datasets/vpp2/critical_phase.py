# Copyright 2026 The LoongForge Authors.
# SPDX-License-Identifier: Apache-2.0
# Adapted from MIT-licensed VPP2.
"""Offline event detection and row/stratum-preserving temporal sampling.

No simulator state, event labels, or future observations are policy inputs.
Index files are immutable artifacts bound to their source metadata and arrays.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from .video_event_dataset import VideoEventDataset

SCHEMA = "robodojo-critical-phase-portable-v1"
SALT = 0x43524954
SCORE_RECIPE = dict(
    mode="core_score",
    score="full_window_core_fraction",
    mixture=0.5,
    max_density_ratio=3.0,
    tail_policy="unchanged",
    protected_prefix=24,
)


def row_key(parquet_path, video_path, table_start, table_end, video_start, video_end, instruction):
    # These are metadata-relative asset names, not installation-specific paths.
    return json.dumps(
        [
            Path(parquet_path).as_posix(),
            Path(video_path).as_posix(),
            int(table_start),
            int(table_end),
            int(video_start),
            int(video_end),
            str(instruction),
        ],
        ensure_ascii=False,
        separators=(",", ":"),
    )


@dataclass(frozen=True)
class EventRules:
    min_amplitude: float = 0.10
    derivative_epsilon: float = 0.002
    merge_gap: int = 3
    plateau_frames: int = 3
    plateau_tolerance: float = 0.025
    before: int = 8
    after: int = 5
    fps: int = 25

    def __post_init__(self):
        if not (0 < self.min_amplitude <= 1 and 0 < self.derivative_epsilon < self.min_amplitude):
            raise ValueError("Invalid gripper amplitude/derivative thresholds")
        if self.plateau_frames < 2 or min(self.merge_gap, self.before, self.after) < 0 or self.fps <= 0:
            raise ValueError("Invalid event timing")


def detect_gripper_events(action, rules=EventRules()):
    """Detect sustained open/close commands, not physical contact or success.

    Returns all significant candidates, including rejected non-plateau changes.
    All intervals are half-open parquet row indices.
    """
    a = np.asarray(action, dtype=np.float64)
    if a.ndim != 2 or a.shape[1] != 16 or not np.isfinite(a).all():
        raise ValueError("Expected finite, unnormalized [T,16] actions")
    if len(a) < 3:
        return []
    events = []
    for arm, col in (("left", 7), ("right", 15)):
        g = a[:, col]
        if g.min() < -0.01 or g.max() > 1.01:
            raise ValueError("Gripper range is not normalized open fraction [0,1]")
        padded = np.pad(g, 1, mode="edge")
        filtered = np.median(np.lib.stride_tricks.sliding_window_view(padded, 3), axis=1)
        delta = np.diff(filtered)
        active = np.flatnonzero(np.abs(delta) >= rules.derivative_epsilon)
        groups = []
        for i in active:
            sign = int(np.sign(delta[i]))
            if groups and sign == groups[-1][2] and i - groups[-1][1] <= rules.merge_gap + 1:
                groups[-1][1] = int(i)
            else:
                groups.append([int(i), int(i), sign])
        for first, last, sign in groups:
            start, end = first, last + 2
            amplitude = float(filtered[end - 1] - filtered[start])
            if abs(amplitude) < rules.min_amplitude:
                continue
            pre = filtered[max(0, start - rules.plateau_frames + 1) : start + 1]
            post = filtered[end - 1 : min(len(g), end - 1 + rules.plateau_frames)]
            stable = (
                len(pre) == rules.plateau_frames
                and len(post) == rules.plateau_frames
                and np.ptp(pre) <= rules.plateau_tolerance
                and np.ptp(post) <= rules.plateau_tolerance
            )
            events.append(
                dict(
                    arm=arm,
                    start=start,
                    end=end,
                    amplitude=amplitude,
                    event_type="gripper_open_transition" if sign > 0 else "gripper_close_transition",
                    accepted=bool(stable),
                    reason="stable_platforms" if stable else "unstable_or_boundary_platform",
                )
            )
    return sorted(events, key=lambda e: (e["start"], e["arm"]))


def row_masks(events, start, end, rules=EventRules(), prefix=16, stride=1, action_offset=0):
    """Union events wholly belonging to this row; clip context, never import adjacent events."""
    length = int(end - start)
    if length <= 0 or prefix <= 0 or stride <= 0 or action_offset < 0:
        raise ValueError("Invalid row or action prefix")
    core = np.zeros(length, dtype=bool)
    context = np.zeros(length, dtype=bool)
    included, truncated = [], []
    for e in events:
        if not e["accepted"] or e["end"] <= start or e["start"] >= end:
            continue
        if e["start"] < start or e["end"] > end:
            truncated.append(e)
            continue
        lo, hi = e["start"] - start, e["end"] - start
        core[lo:hi] = True
        context[max(0, lo - rules.before) : min(length, hi + rules.after)] = True
        included.append(e)
    critical = np.zeros(length, dtype=bool)
    for offset in action_offset + np.arange(prefix) * stride:
        if offset < length:
            critical[: length - offset] |= context[offset:]
    return critical, core, context, included, truncated


def stratum_bounds(length, old_start, horizon=32, stride=1, action_offset=0):
    if not 0 <= old_start < length:
        raise ValueError("Start outside row")
    full_end = max(0, length - ((horizon - 1) * stride + action_offset))
    return (0, full_end) if old_start < full_end else (full_end, length)


def start_probabilities(critical, weight=3.0, horizon=32, stride=1, action_offset=0):
    """Analytical marginal with old uniform probability of each padding stratum."""
    c = np.asarray(critical, dtype=bool)
    if len(c) == 0 or not np.isfinite(weight) or weight < 1:
        raise ValueError("Invalid critical mask/weight")
    full_end = max(0, len(c) - ((horizon - 1) * stride + action_offset))
    p = np.zeros(len(c), dtype=np.float64)
    for lo, hi in ((0, full_end), (full_end, len(c))):
        if hi > lo:
            w = 1.0 + (weight - 1.0) * c[lo:hi]
            p[lo:hi] = ((hi - lo) / len(c)) * w / w.sum()
    return p


def token_exposure(probabilities, horizon=32, stride=1, action_offset=0, terminal=False):
    """Expected raw token draws and per-window-mean loss contributions; no padded fake events."""
    p = np.asarray(probabilities, dtype=np.float64)
    n = len(p)
    starts = np.arange(n)
    offsets = action_offset + np.arange(horizon) * stride
    valid_count = np.clip((n - 1 - starts - action_offset) // stride + 1, 0, horizon)
    denom = np.full(n, horizon) if terminal else np.maximum(1, valid_count)
    draws = np.zeros(n)
    loss = np.zeros(n)
    for off in offsets:
        count = max(0, n - off)
        if count:
            draws[off:] += p[:count]
            loss[off:] += p[:count] / denom[:count]
    terminal_mass = float(np.sum(p * (horizon - valid_count) / horizon)) if terminal else 0.0
    return draws, loss, terminal_mass


def full_window_scores(core, horizon=32, stride=1, action_offset=0):
    """Fraction of physical core tokens, only for starts with a full action window."""
    core = np.asarray(core, dtype=bool)
    if core.ndim != 1 or len(core) == 0 or horizon < 1 or stride < 1 or action_offset < 0:
        raise ValueError("Invalid core mask or temporal contract")
    count = max(0, len(core) - (horizon - 1) * stride - action_offset)
    if stride == 1:
        cumulative = np.concatenate(([0], np.cumsum(core, dtype=np.int64)))
        starts = np.arange(count) + action_offset
        return (cumulative[starts + horizon] - cumulative[starts]) / horizon
    scores = np.zeros(count, dtype=np.float64)
    for offset in action_offset + np.arange(horizon) * stride:
        scores += core[offset : offset + count]
    return scores / horizon


def score_density(scores, mixture=0.5, max_density_ratio=3.0):
    """Density relative to uniform; reduce mixture before normalization to enforce the cap.

    E_new[score] = mean + rho * variance / mean. Thus the expected core
    contribution cannot fall. The uniform floor is at least 1 - mixture.
    """
    scores = np.asarray(scores, dtype=np.float64)
    if (
        scores.ndim != 1
        or not np.isfinite(scores).all()
        or np.any(scores < 0)
        or not np.isfinite(mixture)
        or not 0 <= mixture <= 1
        or not np.isfinite(max_density_ratio)
        or max_density_ratio < 1
    ):
        raise ValueError("Invalid scores, mixture, or density cap")
    density = np.ones(len(scores), dtype=np.float64)
    if len(scores) == 0 or mixture == 0 or max_density_ratio == 1 or np.ptp(scores) == 0:
        return density, 0.0
    mean = float(scores.mean())
    delta = scores / mean - 1.0
    rho = min(float(mixture), (float(max_density_ratio) - 1.0) / float(delta.max()))
    return 1.0 + rho * delta, rho


def full_window_score_density(core, mixture=0.5, max_density_ratio=3.0, horizon=32, stride=1, action_offset=0):
    """Protect prefix24 exposure as well as full-window exposure.

    The default planner executes 24 actions. If full-window weighting would
    reduce core exposure in this prefix, preserve the row's uniform distribution.
    This does not change the runtime execution horizon or the training targets.
    """
    scores = full_window_scores(core, horizon, stride, action_offset)
    density, rho = score_density(scores, mixture, max_density_ratio)
    prefix = min(SCORE_RECIPE["protected_prefix"], horizon)
    if rho > 0 and prefix < horizon:
        early_scores = full_window_scores(core, prefix, stride, action_offset)[: len(scores)] * prefix / horizon
        delta = float((density - 1) @ early_scores / len(scores))
        if delta < -1e-12:
            return np.ones(len(scores)), 0.0
    return density, rho


def score_start_probabilities(core, mixture=0.5, max_density_ratio=3.0, horizon=32, stride=1, action_offset=0):
    """Reweight only full windows; every individual padded start retains its old mass."""
    density, _ = full_window_score_density(core, mixture, max_density_ratio, horizon, stride, action_offset)
    probabilities = np.full(len(core), 1.0 / len(core), dtype=np.float64)
    probabilities[: len(density)] *= density
    return probabilities


class CriticalPhaseIndex:
    def __init__(
        self,
        directory,
        *,
        metadata_paths,
        contract,
        weight=3.0,
        allow_unreviewed=False,
        mode="binary",
        mixture=0.5,
        max_density_ratio=3.0,
    ):
        self.directory = Path(directory).resolve()
        self.manifest = json.loads((self.directory / "manifest.json").read_text())
        m = self.manifest
        if m.get("schema") != SCHEMA or m.get("contract") != contract:
            raise ValueError("Critical index schema/temporal contract mismatch")
        if not np.isfinite(weight) or weight < 1:
            raise ValueError("critical_phase_weight must be finite and >= 1")
        if mode not in ("binary", "core_score"):
            raise ValueError("Unknown critical phase sampling mode")
        score_density([], mixture, max_density_ratio)  # Validate scalar settings even for empty rows.
        if mode == "core_score" and weight != 1:
            raise ValueError("core_score does not use critical_phase_weight; set it to 1")
        recipe = m.get("sampling")
        if mode == "core_score":
            expected = {
                **SCORE_RECIPE,
                "mixture": mixture or SCORE_RECIPE["mixture"],
                "max_density_ratio": max_density_ratio,
            }
            if recipe != expected:
                raise ValueError("Critical index sampling recipe mismatch")
        elif recipe is not None:
            raise ValueError("A score index cannot be used as a binary sampler")
        active = weight > 1 if mode == "binary" else mixture > 0 and max_density_ratio > 1
        if active and not allow_unreviewed and m.get("review_status") != "validated":
            raise ValueError("Critical phase index is unreviewed; audit/review before training")
        if not m.get("rows") or not (self.directory / "start_weights.npz").is_file():
            raise ValueError("Incomplete critical index")
        with np.load(self.directory / "start_weights.npz", allow_pickle=False) as arrays:
            self.flags = arrays["critical"].astype(bool)
            self.core = arrays["core"].astype(bool)
            self.offsets = arrays["offsets"].copy()
            keys = arrays["keys"].tolist()
        if len(set(keys)) != len(keys) or len(self.offsets) != len(keys) + 1:
            raise ValueError("Critical index duplicate keys or invalid offsets")
        if self.offsets[0] != 0 or self.offsets[-1] != len(self.flags) or np.any(np.diff(self.offsets) <= 0):
            raise ValueError("Critical index offset bounds mismatch")
        if self.core.shape != self.flags.shape:
            raise ValueError("Critical index core mask shape mismatch")
        self.lookup = {key: i for i, key in enumerate(keys)}
        self.weight = float(weight)
        self.mode = mode
        self.mixture = float(mixture)
        self.max_density_ratio = float(max_density_ratio)
        self.active = active
        self.contract = contract
        self._cdf = {}

    def bind(self, keys, lengths):
        if len(keys) != len(lengths):
            raise ValueError("Critical index row key/length count mismatch")
        result = []
        for key, length in zip(keys, lengths):
            if key not in self.lookup:
                raise ValueError(f"Critical index missing row: {key}")
            i = self.lookup[key]
            if self.offsets[i + 1] - self.offsets[i] != length:
                raise ValueError("Critical index row length mismatch")
            result.append(i)
        return result

    def probabilities(self, row_index):
        """Exact per-row marginal for the configured runtime sampler and audit."""
        a, b = self.offsets[row_index : row_index + 2]
        c = self.contract
        temporal = dict(horizon=c["action_horizon"], stride=c["stride"], action_offset=c["action_offset"])
        if self.mode == "core_score":
            return score_start_probabilities(self.core[a:b], self.mixture, self.max_density_ratio, **temporal)
        return start_probabilities(self.flags[a:b], self.weight, **temporal)

    def choose(self, row_index, old_start, *, seed, epoch, sample_index):
        if not self.active:
            return int(old_start)
        begin, end = self.offsets[row_index : row_index + 2]
        length = int(end - begin)
        c = self.contract
        lo, hi = stratum_bounds(length, old_start, c["action_horizon"], c["stride"], c["action_offset"])
        if self.mode == "core_score":
            full_end = max(0, length - (c["action_horizon"] - 1) * c["stride"] - c["action_offset"])
            if old_start >= full_end:
                return int(old_start)
            key = (row_index, 0)
            if key not in self._cdf:
                density, rho = full_window_score_density(
                    self.core[begin:end],
                    self.mixture,
                    self.max_density_ratio,
                    c["action_horizon"],
                    c["stride"],
                    c["action_offset"],
                )
                self._cdf[key] = np.cumsum(density) if rho > 0 else None
            cdf = self._cdf[key]
            if cdf is None:
                return int(old_start)
            rng = np.random.default_rng(np.random.SeedSequence([int(seed), int(epoch), int(sample_index), SALT]))
            return int(np.searchsorted(cdf, rng.random() * cdf[-1], side="right"))
        flags = self.flags[begin + lo : begin + hi]
        if not flags.any() or flags.all():
            return int(old_start)
        key = (row_index, lo)
        if key not in self._cdf:
            self._cdf[key] = np.cumsum(1.0 + (self.weight - 1.0) * flags)
        cdf = self._cdf[key]
        rng = np.random.default_rng(np.random.SeedSequence([int(seed), int(epoch), int(sample_index), SALT]))
        return lo + int(np.searchsorted(cdf, rng.random() * cdf[-1], side="right"))


class CriticalPhaseVideoEventDataset(VideoEventDataset):
    def __init__(
        self,
        *,
        critical_phase_index=None,
        critical_phase_weight=1.0,
        critical_phase_allow_unreviewed=False,
        critical_phase_prefix=16,
        critical_phase_mode="binary",
        critical_phase_mixture=0.5,
        critical_phase_max_density_ratio=3.0,
        **kwargs,
    ):
        self._critical_phase = None
        super().__init__(**kwargs)
        # runtime's heldout_action evaluator clones data.train and changes its
        # metadata/is_training_set. Do not bind a training index to that clone.
        if not self.is_training_set:
            return
        if critical_phase_index is None:
            if critical_phase_weight != 1.0 or critical_phase_mode != "binary":
                raise ValueError("Critical phase sampling requires critical_phase_index")
            return
        if (
            self.segment_sampling_mode != "step_uniform"
            or self.skip_padding_as_possible
            or kwargs.get("val_set_proportion", 0.05) > 1e-6
            or len(self.video_path_columns) != 1
            or self.video_hdf5_path_column
        ):
            raise ValueError(
                "Critical phase requires step_uniform, explicit split, single video, and no padding retries"
            )
        if "fps" not in self.metadata or not self.metadata.fps.eq(25).all():
            raise ValueError("Critical phase requires verified 25 Hz metadata")
        if not 1 <= int(critical_phase_prefix) <= self.action_horizon:
            raise ValueError("critical_phase_prefix must be within action horizon")
        contract = dict(
            action_horizon=self.action_horizon,
            stride=self.global_sample_stride,
            action_offset=self.action_time_offset,
            prefix=int(critical_phase_prefix),
            fps=25,
        )
        self._critical_phase = CriticalPhaseIndex(
            critical_phase_index,
            metadata_paths=self.metadata_paths,
            contract=contract,
            weight=critical_phase_weight,
            allow_unreviewed=critical_phase_allow_unreviewed,
            mode=critical_phase_mode,
            mixture=critical_phase_mixture,
            max_density_ratio=critical_phase_max_density_ratio,
        )
        keys = [
            row_key(
                row[self.parquet_path_column],
                row[self.video_path_columns[0]],
                self.table_start_steps[i],
                self.table_start_steps[i] + self.trajectory_lengths[i],
                self.video_start_steps[i],
                self.video_start_steps[i] + self.trajectory_lengths[i],
                self._get_instruction(row),
            )
            for i, (_, row) in enumerate(self.metadata.iterrows())
        ]
        self._critical_phase_rows = self._critical_phase.bind(keys, self.trajectory_lengths)

    def _resolve_step_index_uniform(self, idx):
        return super()._resolve_step_index(idx)

    def _resolve_step_index(self, idx):
        row, start, absolute = self._resolve_step_index_uniform(idx)
        if self._critical_phase is None:
            return row, start, absolute
        logical_idx = int(idx) if idx >= 0 else int(idx) + self.total_steps
        start = self._critical_phase.choose(
            self._critical_phase_rows[row],
            start,
            seed=self.seed,
            epoch=self.sampling_epoch,
            sample_index=logical_idx,
        )
        return row, start, int(self.trajectory_start_steps[row] + start)

    def __getitem__(self, idx):
        if self._critical_phase is None:
            return super().__getitem__(idx)
        try:
            return self._get(idx)
        except Exception as error:
            raise RuntimeError(f"Critical phase sample {idx} failed; no random fallback") from error
