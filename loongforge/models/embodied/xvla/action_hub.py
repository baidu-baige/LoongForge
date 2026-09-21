# Copyright 2026 The LoongForge Authors.
# SPDX-License-Identifier: Apache-2.0
#
# Modified from X-VLA (https://github.com/2toinf/X-VLA).
# Copyright 2025 2toINF. All rights reserved.
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
"""Action space registry and implementations for X-VLA embodied model."""

from __future__ import annotations
from typing import Iterable, Tuple, Dict, Type
import torch
import torch.nn as nn

# =============================================================================
# Registry
# =============================================================================
ACTION_REGISTRY: Dict[str, Type["BaseActionSpace"]] = {}


def register_action(name: str):
    """Decorator for registering a new action space."""
    def _wrap(cls):
        """
        Inner decorator that adds the class to ACTION_REGISTRY under the given name.

        Raises KeyError if the name is already registered.
        Sets cls.name to the lowercased key for introspection.
        """
        key = name.lower()
        if key in ACTION_REGISTRY:
            raise KeyError(f"ActionSpace '{key}' already registered -> {ACTION_REGISTRY[key]}")
        ACTION_REGISTRY[key] = cls
        cls.name = key
        return cls
    return _wrap


def build_action_space(name: str, **kwargs) -> "BaseActionSpace":
    """Instantiate a registered action space by name."""
    key = name.lower()
    if key not in ACTION_REGISTRY:
        raise KeyError(f"Unknown action space '{name}'. Available: {list(ACTION_REGISTRY.keys())}")
    return ACTION_REGISTRY[key](**kwargs)


# =============================================================================
# Base class
# =============================================================================
class BaseActionSpace(nn.Module):
    """
    Abstract base class for all action-space definitions.

    Each subclass defines:
      - `dim_action`: dimension of the action vector.
      - `gripper_idx`: indices of gripper channels.
      - `compute_loss(pred, target)`: supervised loss for this space.
      - `preprocess(proprio, action, mode)`: pre-step modifications.
      - `postprocess(action)`: post-step corrections (e.g. apply sigmoid).
    """

    name: str = "base"
    dim_action: int = 0
    gripper_idx: Tuple[int, ...] = ()

    @property
    def dim_proprio(self) -> int:
        """Proprioception dimension expected by this action space."""
        return self.dim_action

    def __init__(self):
        """Initialize the BaseActionSpace module (no learnable parameters at this level)."""
        super().__init__()

    # ---------------------------------------------------------------------
    def compute_loss(self, pred: torch.Tensor, target: torch.Tensor) -> Dict[str, torch.Tensor]:
        """Compute supervised loss for this action space. Must be implemented by subclasses."""
        raise NotImplementedError

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> Dict[str, torch.Tensor]:
        """Alias for compute_loss."""
        return self.compute_loss(pred, target)

    # ---------------------------------------------------------------------
    # Space-level hooks
    # ---------------------------------------------------------------------
    def preprocess(
        self,
        proprio: torch.Tensor,
        action: torch.Tensor,
        mode: str = "train",
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Default: return unchanged."""
        return proprio, action

    def postprocess(self, action: torch.Tensor) -> torch.Tensor:
        """Default: return unchanged."""
        return action


# =============================================================================
# Utilities
# =============================================================================
def _ensure_indices_valid(D: int, idx: Iterable[int], name: str) -> None:
    """
    Validate that all indices in idx are within [0, D).

    Raises IndexError with a descriptive message if any index is out of range.
    """
    bad = [i for i in idx if i < 0 or i >= D]
    if bad:
        raise IndexError(f"{name} contains out-of-range indices {bad} for action dim D={D}")


# =============================================================================
# Implementations
# =============================================================================
@register_action("ee6d")
class EE6DActionSpace(BaseActionSpace):
    """End-effector layout with xyz, 6D rotation, and gripper channels."""

    dim_action = 20
    gripper_idx = (9, 19)
    GRIPPER_SCALE = 1.0
    XYZ_SCALE = 500.0
    ROT_SCALE = 10.0

    # Contiguous ranges — expressed as (start, stop) so ``compute_loss`` can
    # use plain slicing (much faster than tuple advanced-indexing; the latter
    # emits ``index_put_`` with accumulate on the backward, which forces
    # ``torch.compile`` to skip CUDA graphs).
    POS_IDX_1 = (0, 3)     # was (0, 1, 2)
    POS_IDX_2 = (10, 13)   # was (10, 11, 12)
    ROT_IDX_1 = (3, 9)     # was (3, 4, 5, 6, 7, 8)
    ROT_IDX_2 = (13, 19)   # was (13, 14, 15, 16, 17, 18)

    def __init__(self):
        """Initialize EE6DActionSpace with MSE loss (position/rotation) and BCE loss (gripper)."""
        super().__init__()
        self.mse = nn.MSELoss()
        self.bce = nn.BCEWithLogitsLoss()
        # Pre-bake ``gripper_idx`` as a long tensor buffer so that advanced-
        # indexing scatter in :meth:`preprocess` (``t[..., idx] = 0``) does
        # not trigger a per-step Python-tuple -> CPU tensor -> H2D copy.
        # NSys attributed ~150 ms/step to that hidden H2D. Register as a
        # non-persistent buffer so ``.to(device)`` migrates it once.
        self.register_buffer(
            "_gripper_idx_t",
            torch.as_tensor(self.gripper_idx, dtype=torch.long),
            persistent=False,
        )

    def compute_loss(self, pred, target, valid_mask=None):
        assert pred.shape == target.shape, "pred/target shapes must match"
        B, T, D = pred.shape
        _ensure_indices_valid(D, self.gripper_idx, "gripper_idx")

        # ``valid_mask`` [B]: True for real samples, False for padded rows
        # inserted by the collator's static-shape padding (CUDA-graph path).
        # We compute per-element squared errors / BCE terms manually and
        # reduce with the mask so padded rows contribute exactly zero and
        # the denominator matches the number of *real* samples.
        # When ``valid_mask`` is None (default path) this reduces to the
        # original ``nn.MSELoss(mean)`` / ``nn.BCEWithLogitsLoss(mean)``.
        if valid_mask is None:
            # Fast path: original semantics.
            g0, g1 = self.gripper_idx
            gripper_loss = (
                self.bce(pred[..., g0:g0 + 1], target[..., g0:g0 + 1])
                + self.bce(pred[..., g1:g1 + 1], target[..., g1:g1 + 1])
            ) / 2 * self.GRIPPER_SCALE

            p1s, p1e = self.POS_IDX_1
            p2s, p2e = self.POS_IDX_2
            pos_loss = (
                self.mse(pred[..., p1s:p1e], target[..., p1s:p1e])
                + self.mse(pred[..., p2s:p2e], target[..., p2s:p2e])
            ) * self.XYZ_SCALE

            r1s, r1e = self.ROT_IDX_1
            r2s, r2e = self.ROT_IDX_2
            rot_loss = (
                self.mse(pred[..., r1s:r1e], target[..., r1s:r1e])
                + self.mse(pred[..., r2s:r2e], target[..., r2s:r2e])
            ) * self.ROT_SCALE

            return {
                "position_loss": pos_loss,
                "rotate6D_loss": rot_loss,
                "gripper_loss": gripper_loss,
            }

        # Masked reduction path: preserve the original scalar-loss magnitude
        # by dividing by (n_valid * T * width) where width is the channel
        # count each sub-loss originally averaged over.
        vmask = valid_mask.to(pred.dtype).view(B, 1, 1)  # broadcast over T, D
        n_valid = valid_mask.sum().clamp(min=1).to(pred.dtype)
        inv_v = 1.0 / n_valid  # scalar

        def _masked_mse(p, t, width):
            # ``nn.MSELoss(mean)`` divides by numel = B*T*width; the masked
            # analog divides by n_valid*T*width so padded rows drop out.
            sq = (p - t).pow(2) * vmask
            return sq.sum() / (T * width) * inv_v

        # Gripper BCE: implement mean-BCE with mask, same expression as
        # F.binary_cross_entropy_with_logits(reduction='none') then averaged.
        def _masked_bce(logits, target_):
            # numerically-stable BCE-with-logits, elementwise:
            #   loss = max(l,0) - l*t + log(1 + exp(-|l|))
            l = logits
            per = l.clamp(min=0) - l * target_ + torch.log1p(torch.exp(-l.abs()))
            per = per * vmask  # zero out padded rows
            # Original ``BCEWithLogitsLoss(mean)`` averages over all
            # elements = B*T*1; masked analog averages over n_valid*T*1.
            return per.sum() / T * inv_v

        g0, g1 = self.gripper_idx
        gripper_loss = (
            _masked_bce(pred[..., g0:g0 + 1], target[..., g0:g0 + 1])
            + _masked_bce(pred[..., g1:g1 + 1], target[..., g1:g1 + 1])
        ) / 2 * self.GRIPPER_SCALE

        p1s, p1e = self.POS_IDX_1
        p2s, p2e = self.POS_IDX_2
        pos_loss = (
            _masked_mse(pred[..., p1s:p1e], target[..., p1s:p1e], p1e - p1s)
            + _masked_mse(pred[..., p2s:p2e], target[..., p2s:p2e], p2e - p2s)
        ) * self.XYZ_SCALE

        r1s, r1e = self.ROT_IDX_1
        r2s, r2e = self.ROT_IDX_2
        rot_loss = (
            _masked_mse(pred[..., r1s:r1e], target[..., r1s:r1e], r1e - r1s)
            + _masked_mse(pred[..., r2s:r2e], target[..., r2s:r2e], r2e - r2s)
        ) * self.ROT_SCALE

        return {
            "position_loss": pos_loss,
            "rotate6D_loss": rot_loss,
            "gripper_loss": gripper_loss,
        }

    def preprocess(self, proprio, action, mode="train"):
        """Zero-out gripper channels in proprio/action."""
        proprio_m = proprio.clone()
        action_m = action.clone()
        proprio_m[..., self._gripper_idx_t] = 0.0
        action_m[..., self._gripper_idx_t] = 0.0
        return proprio_m, action_m

    def postprocess(self, action: torch.Tensor) -> torch.Tensor:
        """Apply sigmoid to gripper logits."""
        if action.size(-1) > max(self.gripper_idx):
            action[..., self.gripper_idx] = torch.sigmoid(action[..., self.gripper_idx])
        return action


@register_action("joint")
class JointActionSpace(BaseActionSpace):
    """Joint-space layout with joints + gripper only."""

    dim_action = 14
    gripper_idx = (6, 13)
    GRIPPER_SCALE = 0.1
    JOINTS_SCALE = 1.0

    def __init__(self):
        """Initialize JointActionSpace with MSE (joints) and BCE (gripper) losses."""
        super().__init__()
        self.mse = nn.MSELoss()
        self.bce = nn.BCEWithLogitsLoss()

    def compute_loss(self, pred, target):
        """
        Compute joint-space loss: joint MSE (scaled) + gripper BCE (scaled).

        Returns a dict with 'joints_loss' and 'gripper_loss'.
        """
        assert pred.shape == target.shape
        B, T, D = pred.shape
        _ensure_indices_valid(D, self.gripper_idx, "gripper_idx")

        g_losses = [self.bce(pred[:, :, gi], target[:, :, gi]) for gi in self.gripper_idx]
        gripper_loss = sum(g_losses) / len(self.gripper_idx) * self.GRIPPER_SCALE

        joints_idx = tuple(i for i in range(D) if i not in set(self.gripper_idx))
        joints_loss = self.mse(pred[:, :, joints_idx], target[:, :, joints_idx]) * self.JOINTS_SCALE

        return {
            "joints_loss": joints_loss,
            "gripper_loss": gripper_loss,
        }

    def preprocess(self, proprio, action, mode="train"):
        """Zero-out gripper channels in proprio/action."""
        proprio_m = proprio.clone()
        action_m = action.clone()
        proprio_m[..., self.gripper_idx] = 0.0
        action_m[..., self.gripper_idx] = 0.0
        return proprio_m, action_m

    def postprocess(self, action: torch.Tensor) -> torch.Tensor:
        """Apply sigmoid to gripper logits."""
        if action.size(-1) > max(self.gripper_idx):
            action[..., self.gripper_idx] = torch.sigmoid(action[..., self.gripper_idx])
        return action


@register_action("agibot_ee6d")
class AGIBOTEE6DActionSpace(BaseActionSpace):
    """AGI-bot variant of EE6DActionSpace using MSE for all components."""

    dim_action = 20
    gripper_idx = (9, 19)
    GRIPPER_SCALE = 10.0
    XYZ_SCALE = 500.0
    ROT_SCALE = 10.0
    POS_IDX_1 = (0, 1, 2)
    POS_IDX_2 = (10, 11, 12)
    ROT_IDX_1 = (3, 4, 5, 6, 7, 8)
    ROT_IDX_2 = (13, 14, 15, 16, 17, 18)

    def __init__(self):
        """Initialize AGIBOTEE6DActionSpace with MSE loss for all components."""
        super().__init__()
        self.mse = nn.MSELoss()

    def compute_loss(self, pred, target):
        """
        Compute AGIBOT EE6D loss using MSE for all components (no BCE for gripper).

        Returns a dict with 'position_loss', 'rotate6D_loss', 'gripper_loss'.
        """
        assert pred.shape == target.shape
        B, T, D = pred.shape
        _ensure_indices_valid(D, self.gripper_idx, "gripper_idx")

        gripper_loss = self.mse(pred[:, :, self.gripper_idx], target[:, :, self.gripper_idx]) * self.GRIPPER_SCALE
        pos_loss = (
            self.mse(pred[:, :, self.POS_IDX_1], target[:, :, self.POS_IDX_1]) +
            self.mse(pred[:, :, self.POS_IDX_2], target[:, :, self.POS_IDX_2])
        ) * self.XYZ_SCALE
        rot_loss = (
            self.mse(pred[:, :, self.ROT_IDX_1], target[:, :, self.ROT_IDX_1]) +
            self.mse(pred[:, :, self.ROT_IDX_2], target[:, :, self.ROT_IDX_2])
        ) * self.ROT_SCALE

        return {
            "position_loss": pos_loss,
            "rotate6D_loss": rot_loss,
            "gripper_loss": gripper_loss,
        }

    def preprocess(self, proprio, action, mode="train"):
        """No preprocessing applied in AGIBOT variant."""
        return proprio, action

    def postprocess(self, action: torch.Tensor) -> torch.Tensor:
        """AGIBOT does not postprocess."""
        return action





@register_action("auto")
class AutoActionSpace(BaseActionSpace):
    """
    Auto-detecting action space that adapts to any action dimension.

    - Model outputs max_dim for compatibility with pretrained models
    - Loss is computed only on the first real_dim dimensions
    - Postprocess trims output back to real_dim

    Args:
        real_dim: The actual action dimension from the dataset/policy feature
        max_dim: The model's output dimension for pretrained VLA compatibility
    """

    JOINTS_SCALE = 100.0

    def __init__(self, real_dim: int, max_dim: int = 20):
        """
        Initialize AutoActionSpace.

        Args:
            real_dim: Actual action dimension from the dataset/policy.
            max_dim: Model's output dimension for pretrained VLA compatibility (padding target).
        """
        super().__init__()
        self.real_dim = real_dim
        self.dim_action = max_dim  # Model-facing dimension
        self.mse = nn.MSELoss()

    def _pad_to_model_dim(self, x: torch.Tensor) -> torch.Tensor:
        """Pad real_dim → max_dim (zeros for the dummy channels)."""
        if x is None:
            return None
        if x.size(-1) == self.dim_action:
            return x
        if x.size(-1) != self.real_dim:
            # If dimension doesn't match either, pad/trim to real_dim first
            if x.size(-1) < self.real_dim:
                pad_shape = list(x.shape[:-1]) + [self.real_dim - x.size(-1)]
                pad = x.new_zeros(pad_shape)
                x = torch.cat([x, pad], dim=-1)
            else:
                x = x[..., : self.real_dim]

        pad_shape = list(x.shape[:-1]) + [self.dim_action - self.real_dim]
        pad = x.new_zeros(pad_shape)
        return torch.cat([x, pad], dim=-1)

    def _trim_to_real_dim(self, x: torch.Tensor) -> torch.Tensor:
        """Trim model output max_dim → real_dim."""
        return x[..., : self.real_dim]

    def compute_loss(self, pred: torch.Tensor, target: torch.Tensor) -> dict[str, torch.Tensor]:
        """
        Compute loss only on the first real_dim dimensions.

        pred:   [B, T, max_dim] from the model
        target: [B, T, real_dim] or [B, T, max_dim]

        Loss = MSE(pred[:,:,:real_dim], target[:,:,:real_dim])
        """
        pred = self._pad_to_model_dim(pred)
        target = self._pad_to_model_dim(target)
        assert pred.shape == target.shape, f"Shape mismatch: pred {pred.shape} vs target {target.shape}"

        # only compute loss on the real dimensions
        joints_loss = (
            self.mse(
                pred[:, :, : self.real_dim],
                target[:, :, : self.real_dim],
            )
            * self.JOINTS_SCALE
        )

        return {"joints_loss": joints_loss}

    def preprocess(self, proprio: torch.Tensor, action: torch.Tensor, mode: str = "train"):
        """
        Pad action from real_dim to max_dim for the model.
        """
        return proprio, self._pad_to_model_dim(action)

    def postprocess(self, action: torch.Tensor) -> torch.Tensor:
        """
        Trim model output from max_dim to real_dim for real robot control.
        """
        return self._trim_to_real_dim(action)



# =============================================================================
# Exports
# =============================================================================
__all__ = [
    "BaseActionSpace",
    "build_action_space",
    "register_action",
    "EE6DActionSpace",
    "JointActionSpace",
    "AGIBOTEE6DActionSpace",
    "AutoActionSpace",
    "ACTION_REGISTRY",
]
