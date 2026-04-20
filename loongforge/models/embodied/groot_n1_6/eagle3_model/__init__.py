"""Eagle3 model package for Gr00tN1d6.

This subpackage mirrors the organization in LeRobot:
- configuration in dedicated module
- model loading helpers in dedicated module
- backbone wrapper module used by Gr00tN1d6
"""

from .configuration_eagle3_vl import Eagle3VLConfig
from .eagle_backbone import EagleBackbone

__all__ = ["Eagle3VLConfig", "EagleBackbone"]
