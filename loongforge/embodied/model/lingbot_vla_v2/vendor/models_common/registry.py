# Copyright 2025 Bytedance Ltd. and/or its affiliates
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


# Adapted from https://github.com/sgl-project/sglang/blob/main/python/sglang/srt/models/registry.py

from functools import lru_cache


class _ModelRegistry:
    def __init__(self):
        from ..lingbot_vla.modeling_lingbot_vla_v2 import LingbotVlaV2Policy

        self.model_arch_name_to_cls = {"LingbotVlaV2Policy": LingbotVlaV2Policy}

    @property
    def supported_models(self):
        return self.model_arch_name_to_cls.keys()

    def get_model_cls_from_model_arch(self, model_arch: str):
        return self.model_arch_name_to_cls[model_arch]


@lru_cache
def get_registry():
    return _ModelRegistry()
