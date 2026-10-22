# Copyright 2026 The LoongForge Authors.
# SPDX-License-Identifier: Apache-2.0
#
# Modified from lerobot (https://github.com/huggingface/lerobot).
# Copyright 2025 Physical Intelligence and The HuggingFace Inc. team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Stats aggregation utilities for LeRobot v2.1 datasets.

Ported from lerobot/datasets/compute_stats.py (lerobot v0.3.3 / dataset format v2.1).
Algorithm: Chan's parallel algorithm for combining per-episode mean/std/count.

Functions:
    aggregate_feature_stats: Aggregate stats for a single feature across episodes.
    aggregate_stats: Aggregate stats for all features across episodes.
"""

import numpy as np


def aggregate_feature_stats(stats_ft_list: list[dict[str, dict]]) -> dict[str, dict[str, np.ndarray]]:
    """Aggregate stats for a single feature across multiple episodes.

    Uses Chan's parallel algorithm to combine per-episode mean/std/count into
    global stats without needing access to raw data.

    Args:
        stats_ft_list: List of per-episode stat dicts, each containing:
            - "mean"  (np.ndarray)
            - "std"   (np.ndarray)
            - "count" (np.ndarray, shape (1,))
            - "min"   (np.ndarray)
            - "max"   (np.ndarray)

    Returns:
        Aggregated stat dict with the same keys, values as np.ndarray.
    """
    means = np.stack([s["mean"] for s in stats_ft_list])
    variances = np.stack([s["std"] ** 2 for s in stats_ft_list])
    counts = np.stack([s["count"] for s in stats_ft_list])
    total_count = counts.sum(axis=0)

    # Expand counts dims to broadcast against means/variances
    c = counts
    while c.ndim < means.ndim:
        c = np.expand_dims(c, axis=-1)

    # Weighted mean: total_mean = sum(count_i * mean_i) / total_count
    total_mean = (means * c).sum(axis=0) / total_count

    # Parallel variance: total_var = sum((var_i + delta_i^2) * count_i) / total_count
    delta_means = means - total_mean
    total_variance = ((variances + delta_means ** 2) * c).sum(axis=0) / total_count

    return {
        "min": np.min(np.stack([s["min"] for s in stats_ft_list]), axis=0),
        "max": np.max(np.stack([s["max"] for s in stats_ft_list]), axis=0),
        "mean": total_mean,
        "std": np.sqrt(total_variance),
        "count": total_count,
    }


def aggregate_stats(stats_list: list[dict[str, dict]]) -> dict[str, dict[str, np.ndarray]]:
    """Aggregate stats for all features across multiple episodes.

    Takes the union of all feature keys and aggregates each independently.

    Args:
        stats_list: List of per-episode stats dicts. Each dict maps feature key
            (e.g. "observation.state") to a stat dict with min/max/mean/std/count.

    Returns:
        Aggregated stats dict with same structure, values as np.ndarray.
    """
    data_keys = {key for stats in stats_list for key in stats}
    aggregated_stats = {}
    for key in data_keys:
        stats_with_key = [stats[key] for stats in stats_list if key in stats]
        aggregated_stats[key] = aggregate_feature_stats(stats_with_key)
    return aggregated_stats
