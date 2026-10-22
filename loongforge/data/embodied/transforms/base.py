# Copyright 2026 The LoongForge Authors.
# SPDX-License-Identifier: Apache-2.0

"""
Base Transform classes.

Design ModalityTransform / ComposedModalityTransform.
"""

from abc import ABC, abstractmethod
from typing import Any, Dict, List


class BaseTransform(ABC):
    """
    Data transform base class.

    Each transform operates on a data dict, transforming values of specified keys.
    """

    def __init__(self, apply_to: List[str], training: bool = True):
        """
        Args:
            apply_to: List of keys to transform
            training: Whether in training mode (affects random operations like augmentation)
        """
        self.apply_to = apply_to
        self.training = training

    def __call__(self, data: Dict[str, Any]) -> Dict[str, Any]:
        return self.apply(data)

    @abstractmethod
    def apply(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Forward transform."""
        ...

    def unapply(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Inverse transform (for denormalization during inference). Default: no-op."""
        return data

    def train(self):
        """Set training mode."""
        self.training = True

    def eval(self):
        """Set eval mode."""
        self.training = False


class ComposedTransform(BaseTransform):
    """
    Compose multiple transforms, executed in order.

    Usage:
        transform = ComposedTransform([
            ImageTransform(apply_to=["image"], size=(224, 224)),
            ActionTransform(apply_to=["action"], ...),
        ])
        data = transform(raw_data)
    """

    def __init__(self, transforms: List[BaseTransform]):
        super().__init__(apply_to=[], training=True)
        self.transforms = transforms

    def apply(self, data: Dict[str, Any]) -> Dict[str, Any]:
        for transform in self.transforms:
            data = transform(data)
        return data

    def unapply(self, data: Dict[str, Any]) -> Dict[str, Any]:
        for transform in reversed(self.transforms):
            data = transform.unapply(data)
        return data

    def train(self):
        for t in self.transforms:
            t.train()

    def eval(self):
        for t in self.transforms:
            t.eval()
