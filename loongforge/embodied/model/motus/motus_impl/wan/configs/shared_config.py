# Copyright 2026 The LoongForge Authors.
# SPDX-License-Identifier: Apache-2.0
#
# Modified from https://github.com/thu-ml/Motus under the Apache-2.0 License.
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

"""Shared default config values common to all Wan model variants."""
import torch
from easydict import EasyDict

#------------------------ Wan shared config ------------------------#
wan_shared_cfg = EasyDict()

# t5
wan_shared_cfg.t5_model = 'umt5_xxl'
wan_shared_cfg.t5_dtype = torch.bfloat16
wan_shared_cfg.text_len = 512

# transformer
wan_shared_cfg.param_dtype = torch.bfloat16

# inference
wan_shared_cfg.num_train_timesteps = 1000
wan_shared_cfg.sample_fps = 16
wan_shared_cfg.sample_neg_prompt = \
    ('色调艳丽，过曝，静态，细节模糊不清，'
     '字幕，风格，作品，画作，'
     '画面，静止，整体发灰，最差质量，低质量，JPEG压缩残留，丑陋的，'
     '残缺的，多余的手指，画得不好的手部，画得不好的脸部，'
     '畸形的，毁容的，形态畸形的肢体，手指融合，静止不动的画面，'
     '杂乱的背景，三条腿，背景人很多，倒着走')
wan_shared_cfg.frame_num = 81
