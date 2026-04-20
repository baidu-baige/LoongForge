# Copyright 2026 The LoongForge Authors.
# SPDX-License-Identifier: Apache-2.0

"""plot loss diff"""

import os
from typing import List, Union, Optional

import numpy as np

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


def _to_float_list(values: List[Union[float, str]]) -> List[float]:
	return [float(v) for v in values]


def _align_lists(baseline_list: List[float], current_list: List[float]):
	n = min(len(baseline_list), len(current_list))
	return baseline_list[:n], current_list[:n]


def _resolve_output_dir(output_dir: Optional[str], category: str = "default", model_name: Optional[str] = None) -> str:
	if output_dir:
		return output_dir

	try:
		import yaml
		current_dir = os.path.dirname(os.path.abspath(__file__))
		common_yaml = os.path.abspath(os.path.join(current_dir, "..", "configs", "common.yaml"))
		if os.path.exists(common_yaml):
			with open(common_yaml, "r") as f:
				cfg = yaml.safe_load(f) or {}
			pfs_path = cfg.get("pfs_path")
			if pfs_path:
				parts = [pfs_path, "E2E", "diff", category]
				if model_name:
					parts.append(model_name)
				return os.path.join(*parts)
	except Exception:
		pass

	parts = [os.getcwd(), "E2E", "diff", category]
	if model_name:
		parts.append(model_name)
	return os.path.join(*parts)


def save_loss_diff_plots(
	baseline_list: List[Union[float, str]],
	current_list: List[Union[float, str]],
	model_name: str,
	training_type: str,
	output_dir: Optional[str] = None,
	category: str = "default",
):
	output_dir = _resolve_output_dir(output_dir, category=category, model_name=model_name)
	os.makedirs(output_dir, exist_ok=True)

	baseline = _to_float_list(baseline_list)
	current = _to_float_list(current_list)
	baseline, current = _align_lists(baseline, current)

	if len(baseline) == 0:
		return

	x = np.arange(1, len(baseline) + 1)
	baseline_arr = np.array(baseline, dtype=np.float64)
	current_arr = np.array(current, dtype=np.float64)

	abs_err = np.abs(current_arr - baseline_arr)
	with np.errstate(divide="ignore", invalid="ignore"):
		rel_err = np.where(baseline_arr == 0, 0.0, abs_err / np.abs(baseline_arr))

	base_name = f"{model_name}_{training_type}"

	# 1) Loss comparison
	plt.figure(figsize=(10, 4), dpi=180)
	plt.plot(x, baseline_arr, label="baseline", linewidth=2)
	plt.plot(x, current_arr, label="current", linewidth=2)
	plt.title(f"Loss compare: {model_name} ({training_type})")
	plt.xlabel("Iteration")
	plt.ylabel("Loss")
	plt.grid(True, alpha=0.3)
	plt.legend()
	plt.tight_layout()
	plt.savefig(os.path.join(output_dir, f"{base_name}_loss_compare.png"), dpi=180)
	plt.close()

	# 2) Absolute error
	plt.figure(figsize=(10, 4), dpi=180)
	plt.plot(x, abs_err, label="|current - baseline|", color="#d62728", linewidth=2)
	plt.title(f"Loss absolute error: {model_name} ({training_type})")
	plt.xlabel("Iteration")
	plt.ylabel("Absolute Error")
	plt.grid(True, alpha=0.3)
	plt.legend()
	plt.tight_layout()
	plt.savefig(os.path.join(output_dir, f"{base_name}_loss_abs_error.png"), dpi=180)
	plt.close()

	# 3) Relative error
	plt.figure(figsize=(10, 4), dpi=180)
	plt.plot(x, rel_err, label="|current - baseline| / |baseline|", color="#ff7f0e", linewidth=2)
	plt.title(f"Loss relative error: {model_name} ({training_type})")
	plt.xlabel("Iteration")
	plt.ylabel("Relative Error")
	plt.grid(True, alpha=0.3)
	plt.legend()
	plt.tight_layout()
	plt.savefig(os.path.join(output_dir, f"{base_name}_loss_rel_error.png"), dpi=180)
	plt.close()


def save_metric_compare_plot(
	baseline_list: List[Union[float, str]],
	current_list: List[Union[float, str]],
	model_name: str,
	training_type: str,
	metric_name: str,
	y_label: Optional[str] = None,
	output_dir: Optional[str] = None,
	category: str = "default",
):
	output_dir = _resolve_output_dir(output_dir, category=category, model_name=model_name)
	os.makedirs(output_dir, exist_ok=True)

	baseline = _to_float_list(baseline_list)
	current = _to_float_list(current_list)
	baseline, current = _align_lists(baseline, current)

	if len(baseline) == 0:
		return

	x = np.arange(1, len(baseline) + 1)
	baseline_arr = np.array(baseline, dtype=np.float64)
	current_arr = np.array(current, dtype=np.float64)

	avg_baseline = float(np.mean(baseline_arr))
	avg_current = float(np.mean(current_arr))

	base_name = f"{model_name}_{training_type}"
	label_name = y_label or metric_name

	plt.figure(figsize=(10, 4), dpi=180)
	plt.plot(x, baseline_arr, label="baseline", linewidth=2)
	plt.plot(x, current_arr, label="current", linewidth=2)
	plt.axhline(avg_baseline, color="#1f77b4", linestyle="--", linewidth=1)
	plt.axhline(avg_current, color="#ff7f0e", linestyle="--", linewidth=1)

	plt.title(f"{metric_name} compare: {model_name} ({training_type})")
	plt.xlabel("Iteration")
	plt.ylabel(label_name)
	plt.grid(True, alpha=0.3)
	plt.legend()

	plt.text(
		0.01,
		0.98,
		f"avg baseline: {avg_baseline:.6g}\navg current: {avg_current:.6g}",
		transform=plt.gca().transAxes,
		va="top",
		ha="left",
		fontsize=9,
		bbox=dict(facecolor="white", alpha=0.6, edgecolor="none"),
	)

	plt.tight_layout()
	plt.savefig(os.path.join(output_dir, f"{base_name}_{metric_name}_compare.png"), dpi=180)
	plt.close()
