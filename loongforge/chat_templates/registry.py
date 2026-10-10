# Copyright 2026 The LoongForge Authors.
# SPDX-License-Identifier: Apache-2.0
#
# Modified from LLaMA-Factory (https://github.com/hiyouga/LLaMA-Factory).
# Copyright 2024 the LlamaFactory team.
#
# Licensed under the Apache License, Version 2.0 (the License);
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an AS IS BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.


"""Chat template registry: name -> template, with all built-in registrations."""

import importlib.resources as resources
from typing import Any, Dict, List, Optional, Sequence, Type

from .base import (
    ChatTemplate,
    EmptyFormatter,
    Formatter,
    Llama2Template,
    StringFormatter,
)
from .hf import HFChatTemplate
from .plugins.mm_plugin import MMPlugin, Qwen2VLPlugin, Qwen3VLPlugin
from .plugins.kimi_plugin import KimiPlugin
from .plugins.minicpm_v_4_6_plugin import MiniCPMV46Plugin


MAPPING_NAME_TO_TEMPLATE: Dict[str, ChatTemplate] = {}


def _register_chat_template(
    name: str,
    cls: Type[ChatTemplate] = ChatTemplate,
    format_user: Optional[Formatter] = None,
    format_assistant: Optional[Formatter] = None,
    format_system: Optional[Formatter] = None,
    format_separator: Optional[Formatter] = None,
    format_prefix: Optional[Formatter] = None,
    default_system: str = "",
    stop_words: Sequence[str] = [],
    efficient_eos: bool = False,
    replace_eos: bool = False,
    mm_plugin: Optional[MMPlugin] = None,
    chat_template: Optional[str] = None,
    chat_template_kwargs: Optional[Dict[str, Any]] = None,
) -> None:
    """
    Registers a chat template.

    To add the following chat template:
    ```
    [HUMAN]:
    user prompt here
    [AI]:
    model response here

    [HUMAN]:
    user prompt here
    [AI]:
    model response here
    ```

    The corresponding code should be:
    ```
    _register_chat_template(
        name="custom",
        format_user=StringFormatter(slots=["[HUMAN]:\n{{content}}\n[AI]:\n"]),
        format_separator=EmptyFormatter(slots=["\n\n"]),
        efficient_eos=True,
    )
    ```
    """
    if name in MAPPING_NAME_TO_TEMPLATE:
        raise ValueError(f"Cannot register duplicate template with name {name}.")

    template = cls(
        format_user=format_user,
        format_assistant=format_assistant,
        format_system=format_system,
        format_separator=format_separator,
        format_prefix=format_prefix,
        default_system=default_system,
        stop_words=stop_words,
        efficient_eos=efficient_eos,
        replace_eos=replace_eos,
        mm_plugin=mm_plugin,
    )
    if chat_template is not None:
        if not isinstance(template, HFChatTemplate):
            raise ValueError("chat_template can only be set for HFChatTemplate.")
        template.chat_template = chat_template
    if chat_template_kwargs is not None:
        if not isinstance(template, HFChatTemplate):
            raise ValueError("chat_template_kwargs can only be set for HFChatTemplate.")
        template.chat_template_kwargs = dict(chat_template_kwargs)

    MAPPING_NAME_TO_TEMPLATE[name] = template


def get_support_templates() -> List[str]:
    """
    Returns a list of supported chat templates.
    """
    return list(MAPPING_NAME_TO_TEMPLATE.keys())


def _read_builtin_chat_template(filename: str) -> str:
    """Read a packaged Jinja chat template."""
    return (
        resources.files(__package__)
        .joinpath("jinja", filename)
        .read_text(encoding="utf-8")
    )


_register_chat_template(
    name="kimi-k2.5-hf",
    cls=HFChatTemplate,
    chat_template=_read_builtin_chat_template("kimi_k2_5_training.jinja"),
    mm_plugin=KimiPlugin(
        image_token="<|media_content|>",
        video_token="<|media_content|>",
        merge_kernel_size=(2, 2),
        temporal_merge_kernel_size=4,
    ),
)


_register_chat_template(
    name="kimi-k2.6-hf",
    cls=HFChatTemplate,
    chat_template=_read_builtin_chat_template("kimi_k2_5_training.jinja"),
    mm_plugin=KimiPlugin(
        image_token="<|media_content|>",
        video_token="<|media_content|>",
        merge_kernel_size=(2, 2),
        temporal_merge_kernel_size=4,
    ),
)


_register_chat_template(
    name="kimi-k2.7-code-hf",
    cls=HFChatTemplate,
    chat_template=_read_builtin_chat_template("kimi_k2_7_code_training.jinja"),
    mm_plugin=KimiPlugin(
        image_token="<|media_content|>",
        video_token="<|media_content|>",
        merge_kernel_size=(2, 2),
        temporal_merge_kernel_size=4,
    ),
)

# XTML generation blocks define the assistant loss mask.
_register_chat_template(
    name="kimi-k3-hf",
    cls=HFChatTemplate,
    chat_template=_read_builtin_chat_template("kimi_k3_hf_training.jinja"),
    stop_words=["<|end_of_msg|>"],
    mm_plugin=KimiPlugin(
        image_token="<|media_content|>",
        video_token="<|media_content|>",
        merge_kernel_size=(2, 2),
        temporal_merge_kernel_size=4,
        include_image_size=True,
    ),
)


_register_chat_template(
    name="qwen1.5-hf",
    cls=HFChatTemplate,
    chat_template=_read_builtin_chat_template("qwen_chat_hf_training.jinja"),
)

_register_chat_template(
    name="qwen2-hf",
    cls=HFChatTemplate,
    chat_template=_read_builtin_chat_template("qwen_chat_hf_training.jinja"),
)

_register_chat_template(
    name="qwen2.5-hf",
    cls=HFChatTemplate,
    chat_template=_read_builtin_chat_template("qwen2_5_hf_training.jinja"),
)

_register_chat_template(
    name="qwen3-hf",
    cls=HFChatTemplate,
    chat_template=_read_builtin_chat_template("qwen3_hf_training.jinja"),
)

_register_chat_template(
    name="qwen3-coder-hf",
    cls=HFChatTemplate,
    chat_template=_read_builtin_chat_template("qwen3_coder_hf_training.jinja"),
)

_register_chat_template(
    name="qwen3-next-hf",
    cls=HFChatTemplate,
    chat_template=_read_builtin_chat_template("qwen3_next_hf_training.jinja"),
)

_register_chat_template(
    name="qwen3.5-hf",
    cls=HFChatTemplate,
    chat_template=_read_builtin_chat_template("qwen3_5_think_hf_training.jinja"),
    mm_plugin=Qwen3VLPlugin(image_token="<|image_pad|>", video_token="<|video_pad|>"),
)

_register_chat_template(
    name="qwen3.5-think-hf",
    cls=HFChatTemplate,
    chat_template=_read_builtin_chat_template("qwen3_5_think_hf_training.jinja"),
    mm_plugin=Qwen3VLPlugin(image_token="<|image_pad|>", video_token="<|video_pad|>"),
)

_register_chat_template(
    name="qwen3.5-nothink-hf",
    cls=HFChatTemplate,
    chat_template=_read_builtin_chat_template("qwen3_5_nothink_hf_training.jinja"),
    mm_plugin=Qwen3VLPlugin(image_token="<|image_pad|>", video_token="<|video_pad|>"),
)

_register_chat_template(
    name="qwen3.8-hf",
    cls=HFChatTemplate,
    chat_template=_read_builtin_chat_template("qwen3_8_hf_training.jinja"),
    chat_template_kwargs={
        "enable_thinking": False,
        "reasoning_effort": "low",
        "preserve_thinking": True,
    },
    mm_plugin=Qwen3VLPlugin(image_token="<|image_pad|>", video_token="<|video_pad|>"),
)

_register_chat_template(
    name="qwen3.6-hf",
    cls=HFChatTemplate,
    chat_template=_read_builtin_chat_template("qwen3_6_hf_training.jinja"),
    mm_plugin=Qwen3VLPlugin(image_token="<|image_pad|>", video_token="<|video_pad|>"),
)

_register_chat_template(
    name="qwen2.5-vl-hf",
    cls=HFChatTemplate,
    chat_template=_read_builtin_chat_template("qwen2_5_vl_hf_training.jinja"),
    mm_plugin=Qwen2VLPlugin(image_token="<|image_pad|>", video_token="<|video_pad|>"),
)

_register_chat_template(
    name="qwen3-vl-hf",
    cls=HFChatTemplate,
    chat_template=_read_builtin_chat_template("qwen3_vl_hf_training.jinja"),
    mm_plugin=Qwen3VLPlugin(image_token="<|image_pad|>", video_token="<|video_pad|>"),
)

_register_chat_template(
    name="llava-onevision-hf",
    cls=HFChatTemplate,
    chat_template=_read_builtin_chat_template("qwen_chat_hf_training.jinja"),
)

_register_chat_template(
    name="minicpm-v-4.6-hf",
    cls=HFChatTemplate,
    chat_template=_read_builtin_chat_template("minicpm_v_4_6_hf_training.jinja"),
    mm_plugin=MiniCPMV46Plugin(image_token="<|image_pad|>"),
)

_register_chat_template(
    name="deepseek-v2-hf",
    cls=HFChatTemplate,
    chat_template=_read_builtin_chat_template("deepseek_v2_hf_training.jinja"),
)

_register_chat_template(
    name="deepseek-v3-hf",
    cls=HFChatTemplate,
    chat_template=_read_builtin_chat_template("deepseek_v3_hf_training.jinja"),
)

_register_chat_template(
    name="internlm2.5-hf",
    cls=HFChatTemplate,
    chat_template=_read_builtin_chat_template("internlm2_5_hf_training.jinja"),
)

_register_chat_template(
    name="mimo-v2-hf",
    cls=HFChatTemplate,
    chat_template=_read_builtin_chat_template("mimo_v2_hf_training.jinja"),
)


_register_chat_template(
    name="empty",
    efficient_eos=True,
)


_register_chat_template(
    name="default",
    format_user=StringFormatter(slots=["Human: {{content}}\nAssistant:"]),
    format_system=StringFormatter(slots=["{{content}}\n"]),
    format_separator=EmptyFormatter(slots=["\n"]),
)


_register_chat_template(
    name="alpaca",
    format_user=StringFormatter(
        slots=["### Instruction:\n{{content}}\n\n### Response:\n"]
    ),
    format_separator=EmptyFormatter(slots=["\n\n"]),
    default_system=(
        "Below is an instruction that describes a task. "
        "Write a response that appropriately completes the request.\n\n"
    ),
)


_register_chat_template(
    name="baichuan",
    format_user=StringFormatter(
        slots=[{"token": "<reserved_102>"}, "{{content}}", {"token": "<reserved_103>"}]
    ),
    efficient_eos=True,
)


_register_chat_template(
    name="baichuan2",
    format_user=StringFormatter(slots=["<reserved_106>{{content}}<reserved_107>"]),
    efficient_eos=True,
)


_register_chat_template(
    name="llama2",
    cls=Llama2Template,
    format_user=StringFormatter(slots=[{"bos_token"}, "[INST] {{content}} [/INST]"]),
    format_system=StringFormatter(slots=["<<SYS>>\n{{content}}\n<</SYS>>\n\n"]),
)


_register_chat_template(
    name="llama2_zh",
    cls=Llama2Template,
    format_user=StringFormatter(slots=[{"bos_token"}, "[INST] {{content}} [/INST]"]),
    format_system=StringFormatter(slots=["<<SYS>>\n{{content}}\n<</SYS>>\n\n"]),
    default_system="You are a helpful assistant. 你是一个乐于助人的助手。",
)


_register_chat_template(
    name="llama3",
    format_user=StringFormatter(
        slots=[
            (
                "<|start_header_id|>user<|end_header_id|>\n\n{{content}}<|eot_id|>"
                "<|start_header_id|>assistant<|end_header_id|>\n\n"
            )
        ]
    ),
    format_system=StringFormatter(
        slots=["<|start_header_id|>system<|end_header_id|>\n\n{{content}}<|eot_id|>"]
    ),
    format_prefix=EmptyFormatter(slots=[{"bos_token"}]),
    stop_words=["<|eot_id|>"],
    replace_eos=True,
)

_register_chat_template(
    name="llama3.1",
    format_user=StringFormatter(
        slots=[
            (
                "<|start_header_id|>user<|end_header_id|>\n\n{{content}}<|eot_id|>"
                "<|start_header_id|>assistant<|end_header_id|>\n\n"
            )
        ]
    ),
    format_system=StringFormatter(
        slots=["<|start_header_id|>system<|end_header_id|>\n\n{{content}}<|eot_id|>"]
    ),
    format_prefix=EmptyFormatter(slots=[{"bos_token"}]),
    stop_words=["<|eot_id|>"],
    replace_eos=True,
)


_register_chat_template(
    name="mistral",
    format_user=StringFormatter(slots=["[INST] {{content}} [/INST]"]),
    format_prefix=EmptyFormatter(slots=[{"bos_token"}]),
)


_register_chat_template(
    name="qwen",
    format_user=StringFormatter(
        slots=["<|im_start|>user\n{{content}}<|im_end|>\n<|im_start|>assistant\n"]
    ),
    format_system=StringFormatter(
        slots=["<|im_start|>system\n{{content}}<|im_end|>\n"]
    ),
    format_separator=EmptyFormatter(slots=["\n"]),
    default_system="You are a helpful assistant.",
    stop_words=["<|im_end|>"],
    replace_eos=True,
)

_register_chat_template(
    name="qwen2-vl",
    format_user=StringFormatter(
        slots=["<|im_start|>user\n{{content}}<|im_end|>\n<|im_start|>assistant\n"]
    ),
    format_system=StringFormatter(
        slots=["<|im_start|>system\n{{content}}<|im_end|>\n"]
    ),
    format_separator=EmptyFormatter(slots=["\n"]),
    default_system="You are a helpful assistant.",
    stop_words=["<|im_end|>"],
    replace_eos=True,
    mm_plugin=Qwen2VLPlugin(image_token="<|image_pad|>", video_token="<|video_pad|>"),
)

_register_chat_template(
    name="qwen3-vl",
    format_user=StringFormatter(slots=["<|im_start|>user\n{{content}}<|im_end|>\n<|im_start|>assistant\n"]),
    format_system=StringFormatter(slots=["<|im_start|>system\n{{content}}<|im_end|>\n"]),
    format_separator=EmptyFormatter(slots=["\n"]),
    default_system="You are a helpful assistant.",
    stop_words=["<|im_end|>"],
    replace_eos=True,
    mm_plugin=Qwen3VLPlugin(image_token="<|image_pad|>", video_token="<|video_pad|>"),
)

_register_chat_template(
    name="deepseek",
    format_user=StringFormatter(slots=["User: {{content}}\n\nAssistant:"]),
    format_system=StringFormatter(slots=["{{content}}\n\n"]),
    format_prefix=EmptyFormatter(slots=[{"bos_token"}]),
)

_register_chat_template(
    name="deepseek3",
    format_user=StringFormatter(slots=["<｜User｜>{{content}}<｜Assistant｜>"]),
    format_prefix=EmptyFormatter(slots=[{"bos_token"}]),
)

_register_chat_template(
    name="deepseek4",
    format_user=StringFormatter(slots=["<｜User｜>{{content}}<｜Assistant｜><think>"]),
    format_system=StringFormatter(slots=["{{content}}"]),
    format_prefix=EmptyFormatter(slots=[{"bos_token"}]),
)

_register_chat_template(
    name="deepseek3.1-nothink",
    format_user=StringFormatter(slots=["<｜User｜>{{content}}<｜Assistant｜></think>"]),
    format_prefix=EmptyFormatter(slots=[{"bos_token"}]),
)

_register_chat_template(
    name="minimax-m2",
    format_user=StringFormatter(slots=["]~b]user\n{{content}}[e~[\n]~b]ai\n"]),
    format_assistant=StringFormatter(slots=["{{content}}[e~[\n"]),
    format_system=StringFormatter(slots=["]~!b[]~b]system\n{{content}}[e~[\n"]),
    stop_words=["[e~["],
)

_register_chat_template(
    name="no-template",
    format_user=StringFormatter(slots=["{{content}}"]),
    format_prefix=EmptyFormatter(slots=[{"bos_token"}]),
)

_register_chat_template(
    name="mimo",
    format_user=StringFormatter(slots=["<|im_start|>user\n{{content}}<|im_end|>\n<|im_start|>assistant\n"]),
    format_system=StringFormatter(slots=["<|im_start|>system\n{{content}}<|im_end|>\n"]),
    format_separator=EmptyFormatter(slots=["\n"]),
    default_system="You are a helpful assistant.",
    stop_words=["<|im_end|>"],
    replace_eos=True,
)

# Kimi K2.5 chat template
# Format: <|im_user|>user<|im_middle|>{content}<|im_end|><|im_assistant|>assistant\
#   <|im_middle|><think></think>{response}<|im_end|>
_register_chat_template(
    name="kimi-k2.5",
    format_user=StringFormatter(
        slots=["<|im_user|>user<|im_middle|>{{content}}<|im_end|><|im_assistant|>assistant<|im_middle|><think></think>"]
    ),
    format_system=StringFormatter(
        slots=["<|im_system|>system<|im_middle|>{{content}}<|im_end|>"]
    ),
    format_assistant=StringFormatter(
        slots=["{{content}}<|im_end|>"]
    ),
    format_separator=EmptyFormatter(slots=[""]),
    stop_words=["<|im_end|>"],
    replace_eos=True,
    mm_plugin=KimiPlugin(
        image_token="<|media_content|>",
        video_token="<|media_content|>",
        merge_kernel_size=(2, 2),
        temporal_merge_kernel_size=4,
    ),
)

# Kimi K2.5 with thinking (reasoning) enabled
_register_chat_template(
    name="kimi-k2.5-think",
    format_user=StringFormatter(
        slots=["<|im_user|>user<|im_middle|>{{content}}<|im_end|><|im_assistant|>assistant<|im_middle|><think>"]
    ),
    format_system=StringFormatter(
        slots=["<|im_system|>system<|im_middle|>{{content}}<|im_end|>"]
    ),
    format_assistant=StringFormatter(
        slots=["{{content}}<|im_end|>"]
    ),
    format_separator=EmptyFormatter(slots=[""]),
    stop_words=["<|im_end|>"],
    replace_eos=True,
    mm_plugin=KimiPlugin(
        image_token="<|media_content|>",
        video_token="<|media_content|>",
        merge_kernel_size=(2, 2),
        temporal_merge_kernel_size=4,
    ),
)

# Kimi K2.6 uses the same chat format and shared Kimi multimodal plugin as K2.5.
_register_chat_template(
    name="kimi-k2.6",
    format_user=StringFormatter(
        slots=["<|im_user|>user<|im_middle|>{{content}}<|im_end|><|im_assistant|>assistant<|im_middle|><think></think>"]
    ),
    format_system=StringFormatter(
        slots=["<|im_system|>system<|im_middle|>{{content}}<|im_end|>"]
    ),
    format_assistant=StringFormatter(
        slots=["{{content}}<|im_end|>"]
    ),
    format_separator=EmptyFormatter(slots=[""]),
    stop_words=["<|im_end|>"],
    replace_eos=True,
    mm_plugin=KimiPlugin(
        image_token="<|media_content|>",
        video_token="<|media_content|>",
        merge_kernel_size=(2, 2),
        temporal_merge_kernel_size=4,
    ),
)

_register_chat_template(
    name="kimi-k2.6-think",
    format_user=StringFormatter(
        slots=["<|im_user|>user<|im_middle|>{{content}}<|im_end|><|im_assistant|>assistant<|im_middle|><think>"]
    ),
    format_system=StringFormatter(
        slots=["<|im_system|>system<|im_middle|>{{content}}<|im_end|>"]
    ),
    format_assistant=StringFormatter(
        slots=["{{content}}<|im_end|>"]
    ),
    format_separator=EmptyFormatter(slots=[""]),
    stop_words=["<|im_end|>"],
    replace_eos=True,
    mm_plugin=KimiPlugin(
        image_token="<|media_content|>",
        video_token="<|media_content|>",
        merge_kernel_size=(2, 2),
        temporal_merge_kernel_size=4,
    ),
)

_register_chat_template(
    name="glm5",
    format_user=StringFormatter(slots=["<|user|>{{content}}<|assistant|>"]),
    format_assistant=StringFormatter(slots=["{{content}}"]),
    format_system=StringFormatter(slots=["<|system|>{{content}}"]),
    format_prefix=EmptyFormatter(slots=["[gMASK]<sop>"]),
    stop_words=["<|user|>", "<|observation|>"],
    efficient_eos=True,
)

# GLM-5.2 changes the prompt format relative to GLM-5: a `Reasoning Effort` system line
# (default Max, `high` selectable via --chat-template-kwargs), an empty `<think></think>`
# pair on non-thinking turns where 5.1 emitted a bare `</think>`, and a multi-modal
# refusal reminder. Registered as an HFChatTemplate so OpenAI-style SFT gets assistant-only
# loss masks from the template's generation blocks.
_register_chat_template(
    name="glm5.2-hf",
    cls=HFChatTemplate,
    chat_template=_read_builtin_chat_template("glm5_2_hf_training.jinja"),
    stop_words=["<|user|>", "<|observation|>"],
    mm_plugin=KimiPlugin(
        image_token="<|image|>",
        video_token=None,
        merge_kernel_size=(2, 2),
        image_prefix="<|begin_of_image|>",
        image_suffix="<|end_of_image|>",
    ),
)

# GLM-5.3-Flash prompt format: Reasoning Effort system line (max/high/low, default max,
# selectable via --chat-template-kwargs), a `clear_thinking` switch, tool_reference
# responses, and real image/video/audio token emission inside the template.
# Registered as an HFChatTemplate so OpenAI-style SFT gets assistant-only loss masks
# from the template's generation blocks. Video SFT additionally needs plugin support
# for the <|begin_of_video|>/<|end_of_video|> wrapper.
_register_chat_template(
    name="glm5.3-hf",
    cls=HFChatTemplate,
    chat_template=_read_builtin_chat_template("glm5_3_hf_training.jinja"),
    stop_words=["<|user|>", "<|assistant|>"],
    mm_plugin=KimiPlugin(
        image_token="<|image|>",
        video_token=None,
        merge_kernel_size=(2, 2),
        image_prefix="<|begin_of_image|>",
        image_suffix="<|end_of_image|>",
    ),
)
