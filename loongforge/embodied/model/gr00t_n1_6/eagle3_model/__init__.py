# Copyright 2026 The LoongForge Authors.
# SPDX-License-Identifier: Apache-2.0
#
# Modified from NVIDIA Eagle3 under the Apache-2.0 License.
#
# Copyright 2024 NVIDIA. All rights reserved.
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

"""Eagle3 model package for Gr00tN1d6.

This subpackage mirrors the organization in LeRobot:
- configuration in dedicated module
- model loading helpers in dedicated module
- backbone wrapper module used by Gr00tN1d6
"""

from .configuration_eagle3_vl import Eagle3VLConfig
from .eagle_backbone import EagleBackbone

__all__ = ["Eagle3VLConfig", "EagleBackbone"]
