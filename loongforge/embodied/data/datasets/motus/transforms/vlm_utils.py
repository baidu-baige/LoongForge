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

# VLM Utilities
# Simple functions for VLM data preprocessing

"""VLM data preprocessing helpers.

Thin wrappers that build chat-style messages and run the Qwen VLM processor to turn
a text instruction plus image into model-ready VLM inputs.
"""

from qwen_vl_utils import process_vision_info
from typing import List, Dict, Any, Tuple
import logging

logger = logging.getLogger(__name__)


def preprocess_vlm_messages(text_instruction: str, image_pil, processor):
    """
    Complete VLM preprocessing - create messages, process vision, and get final inputs.
    
    Args:
        text_instruction: Robot task instruction
        image_pil: PIL Image object
        processor: VLM processor (AutoProcessor)
        
    Returns:
        VLM inputs ready for model forward
    """
    # Create VLM messages format
    messages = [
        {
            "role": "user",
            "content": [
                {"type": "image", "image": image_pil},
                {"type": "text", "text": text_instruction}
            ]
        }
    ]
    
    # Apply chat template
    text = processor.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )
    
    # Process vision info
    image_inputs, video_inputs = process_vision_info(messages)
    
    # Get final processor inputs
    inputs = processor(
        text=[text],
        images=image_inputs,
        videos=video_inputs,
        padding=True,
        return_tensors="pt",
    )
    
    return inputs
