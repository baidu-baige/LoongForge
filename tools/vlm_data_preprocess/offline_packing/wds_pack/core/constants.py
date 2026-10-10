# Copyright 2026 The LoongForge Authors.
# SPDX-License-Identifier: Apache-2.0

"""Shared constants for offline data packing."""

# TODO TEMPLATES needs to be expanded with more templates
TEMPLATES = {
    "packed_captioning": "<|vision_start|><|image_pad|><|vision_end|>{{ captions[0].content }}<|im_end|>",
    "packed_vqa": {
        "qwenvl": """
                    {% set image_count = namespace(value=0) %}
                    {% set video_count = namespace(value=0) %}

                    {% for message in messages %}
                        {% if loop.first and message['from'] != 'system' %}
                    <|im_start|>system
                    You are a helpful assistant.<|im_end|>
                        {% endif %}
                    <|im_start|>{{ message['from'] }}
                    {{ message['value'] | replace('<image>', '<|vision_start|><|image_pad|><|vision_end|>') }}<|im_end|>
                    {% endfor %}

                    {% if add_generation_prompt %}
                    <|im_start|>assistant
                    {% endif %}
        """
    },
    "packed_multi_mix_qa": {
        "qwenvl": """
                    {% set image_count = namespace(value=0) %}
                    {% set video_count = namespace(value=0) %}

                    {% for message in messages %}
                        {% if loop.first and message['from'] != 'system' %}
                    <|im_start|>system
                    You are a helpful assistant.<|im_end|>
                        {% endif %}
                    <|im_start|>{{ message['from'] }}
                    {{ message['value'] | replace('<image>', '<|vision_start|><|image_pad|><|vision_end|>') }}<|im_end|>
                    {% endfor %}

                    {% if add_generation_prompt %}
                    <|im_start|>assistant
                    {% endif %}
        """
    },
}

VALID_MEDIA_EXT = {"image": [".jpg", ".jpeg", ".png"], "video": [".mp4", ".avi"]}

# Supported media types, additional modalities may be added in the future.
# mix indicates heterogeneous sample modalities (pure-text / text-image / text-video).
VALID_MEDIA_TYPE = ["text", "image", "video", "mix"]
