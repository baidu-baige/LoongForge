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


"""Chat templates"""

import importlib.resources as resources
import logging
import re
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import (
    TYPE_CHECKING,
    Type,
    Dict,
    Any,
    List,
    Optional,
    Sequence,
    Set,
    Tuple,
    Union,
)

from loongforge.utils.constants import DataRoles
from .mm_plugin import MMPlugin, Qwen2VLPlugin, Qwen3VLPlugin
from .kimi_k25_plugin import KimiK25Plugin


logger = logging.getLogger(__name__)


if TYPE_CHECKING:
    from loongforge.tokenizer import AutoTokenizerFromHF


SlotsType = Sequence[Union[str, Set[str], Dict[str, str]]]


@dataclass
class Formatter(ABC):
    """Base class of all formatters."""

    slots: SlotsType = field(default_factory=list)

    @abstractmethod
    def apply(self, **kwargs) -> SlotsType:
        """Apply the formatter to the given arguments"""
        raise NotImplementedError


@dataclass
class EmptyFormatter(Formatter):
    """An empty formatter that does nothing"""

    def __post_init__(self):
        has_placeholder = False
        for slot in filter(lambda s: isinstance(s, str), self.slots):
            if re.search(r"\{\{[a-zA-Z_][a-zA-Z0-9_]*\}\}", slot):
                has_placeholder = True

        if has_placeholder:
            raise ValueError("Empty formatter should not contain any placeholder.")

    def apply(self, **kwargs) -> SlotsType:
        """Apply the formatter to the given arguments"""
        return self.slots


@dataclass
class StringFormatter(Formatter):
    """String formatter"""

    def __post_init__(self):
        has_placeholder = False
        for slot in filter(lambda s: isinstance(s, str), self.slots):
            if re.search(r"\{\{[a-zA-Z_][a-zA-Z0-9_]*\}\}", slot):
                has_placeholder = True

        if not has_placeholder:
            raise ValueError("A placeholder is required in the string formatter.")

    def apply(self, **kwargs) -> SlotsType:
        """Apply the formatter to the given arguments"""
        elements = []
        for slot in self.slots:
            if isinstance(slot, str):
                for name, value in kwargs.items():
                    if not isinstance(value, str):
                        raise RuntimeError("Expected a string, got {}".format(value))

                    slot = slot.replace("{{" + name + "}}", value, 1)
                elements.append(slot)
            elif isinstance(slot, (dict, set)):
                elements.append(slot)
            else:
                raise RuntimeError(
                    "Input must be string, set[str] or dict[str, str], got {}".format(
                        type(slot)
                    )
                )

        return elements


@dataclass
class ChatTemplate:
    """ChatTemplate class."""

    format_user: Optional[Formatter] = None
    format_assistant: Optional[Formatter] = None
    format_system: Optional[Formatter] = None
    format_separator: Optional[Formatter] = None
    format_prefix: Optional[Formatter] = None
    default_system: str = ""
    stop_words: List[str] = field(default_factory=list)
    efficient_eos: bool = False
    replace_eos: bool = False
    mm_plugin: Optional[MMPlugin] = None

    def __post_init__(self):
        if self.format_user is None:
            self.format_user = StringFormatter(slots=["{{content}}"])

        # if efficient_eos=true, we will not add eos_token among the multiple turns,
        # and it will be added in the end of the last response.
        eos_slots = [] if self.efficient_eos else [{"eos_token"}]
        if self.format_assistant is None:
            self.format_assistant = StringFormatter(slots=["{{content}}"] + eos_slots)

        if self.format_system is None:
            self.format_system = StringFormatter(slots=["{{content}}"])

        if self.format_separator is None:
            self.format_separator = EmptyFormatter()

        if self.format_prefix is None:
            self.format_prefix = EmptyFormatter()

    def encode_multiturn(
        self,
        tokenizer: "AutoTokenizerFromHF",
        messages: Sequence[Dict[str, str]],
        system: Optional[str] = None,
    ) -> List[Tuple[List[int], List[int]]]:
        """
        Returns multiple pairs of token ids representing prompts and responses respectively.
        """
        encoded_messages = self._encode(tokenizer, messages, system)
        return [
            (encoded_messages[i], encoded_messages[i + 1])
            for i in range(0, len(encoded_messages), 2)
        ]

    def encode_oneturn(
        self,
        tokenizer: "AutoTokenizerFromHF",
        messages: Sequence[Dict[str, str]],
        system: Optional[str] = None,
    ) -> Tuple[List[int], List[int]]:
        """
        Returns a single pair of token ids representing prompt and response respectively.
        """
        encoded_messages = self._encode(tokenizer, messages, system)
        prompt_ids = []
        for encoded_ids in encoded_messages[:-1]:
            prompt_ids += encoded_ids

        answer_ids = encoded_messages[-1]
        return prompt_ids, answer_ids

    def encode_openai(
        self,
        tokenizer: "AutoTokenizerFromHF",
        messages: Sequence[Dict[str, Any]],
        tools: Optional[Sequence[Dict[str, Any]]] = None,
        train_on_prompt: bool = False,
        history_mask_loss: bool = False,
        ignore_index: int = -100,
        max_length: Optional[int] = None,
    ) -> Tuple[List[int], List[int], List[int], int]:
        """Encode OpenAI-style messages. Only HFChatTemplate supports this."""
        raise NotImplementedError(
            "OpenAI-style SFT data requires an HFChatTemplate registered for "
            "the target model, usually selected by a model-specific `*-hf` "
            "chat template name."
        )

    def _encode(
        self,
        tokenizer: "AutoTokenizerFromHF",
        messages: Sequence[Dict[str, str]],
        system: Optional[str],
    ) -> List[List[int]]:
        """
        Encodes formatted inputs to pairs of token ids.
        Turn 0: prefix + system + query     resp
        Turn t: sep + query                 resp
        """
        system = system or self.default_system
        encoded_messages = []
        for i, message in enumerate(messages):
            elements = []

            if i == 0:
                elements += self.format_prefix.apply()
                if system:
                    elements += self.format_system.apply(content=system)

            elif i > 0 and i % 2 == 0:
                elements += self.format_separator.apply()

            if message["role"] == DataRoles.USER:
                elements += self.format_user.apply(
                    content=message["content"], idx=str(i // 2)
                )
            elif message["role"] == DataRoles.ASSISTANT:
                elements += self.format_assistant.apply(content=message["content"])
            else:
                raise NotImplementedError("Unexpected role: {}".format(message["role"]))

            encoded_messages.append(self._convert_elements_to_ids(tokenizer, elements))

        return encoded_messages

    def _convert_elements_to_ids(
        self,
        tokenizer: "AutoTokenizerFromHF",
        elements: "SlotsType",
    ) -> List[int]:
        """
        Converts elements to token ids.
        """
        token_ids = []
        for elem in elements:
            if isinstance(elem, str):
                if len(elem) != 0:
                    token_ids += tokenizer.tokenize(elem, add_special_tokens=False)

            elif isinstance(elem, dict):
                token_ids += [tokenizer.convert_tokens_to_ids(elem.get("token"))]

            elif isinstance(elem, set):
                if "bos_token" in elem and tokenizer.bos is not None:
                    token_ids += [tokenizer.bos]

                elif "eos_token" in elem and tokenizer.eos is not None:
                    token_ids += [tokenizer.eos]

            else:
                raise ValueError(
                    "Input must be string, set[str] or dict[str, str], got {}".format(
                        type(elem)
                    )
                )

        return token_ids

    @classmethod
    def from_name(cls, name: str) -> "ChatTemplate":
        """build template."""
        return MAPPING_NAME_TO_TEMPLATE.get(name, None)


@dataclass
class HFChatTemplate(ChatTemplate):
    """Chat template backed by HuggingFace tokenizer.apply_chat_template."""

    chat_template: Optional[str] = None
    chat_template_kwargs: Dict[str, Any] = field(default_factory=dict)

    def _encode(
        self,
        tokenizer: "AutoTokenizerFromHF",
        messages: Sequence[Dict[str, str]],
        system: Optional[str],
    ) -> List[List[int]]:
        """Reject legacy prompt/response encoding for HF Jinja templates."""
        raise NotImplementedError(
            "HFChatTemplate only supports OpenAI-style `messages` data through "
            "encode_openai(). Use an OpenAI chat-completions dataset format "
            "with a registered model-specific `*-hf` chat template."
        )

    @staticmethod
    def _require_generation_template(chat_template: Optional[str]) -> str:
        """Return a template that contains HF generation blocks, or fail fast."""
        if chat_template is None:
            raise ValueError("HFChatTemplate does not provide a chat_template.")
        has_start = re.search(r"{%-?\s*generation\s*-?%}", chat_template)
        has_end = re.search(r"{%-?\s*endgeneration\s*-?%}", chat_template)
        if has_start and has_end:
            return chat_template
        raise ValueError(
            "HF chat_template must contain paired `{% generation %}` / "
            "`{% endgeneration %}` blocks for OpenAI-style assistant loss masks. "
            "Use a registered model-specific `*-hf` training template."
        )

    @staticmethod
    def _as_list(value) -> List[int]:
        """Normalize tokenizer outputs to a flat Python list of token ids."""
        if hasattr(value, "tolist"):
            value = value.tolist()
        if value and isinstance(value[0], list):
            return list(value[0])
        return list(value)

    def _tokenize(
        self,
        tokenizer: "AutoTokenizerFromHF",
        messages: Sequence[Dict[str, Any]],
        tools: Optional[Sequence[Dict[str, Any]]] = None,
        add_generation_prompt: bool = False,
    ) -> List[int]:
        """Tokenize chat messages with the tokenizer's HF chat template path."""
        hf_tokenizer = tokenizer.hf_tokenizer()
        kwargs = dict(self.chat_template_kwargs)
        if tools:
            kwargs["tools"] = tools
        kwargs.update(
            {
                "tokenize": True,
                "return_dict": False,
                "add_generation_prompt": add_generation_prompt,
            }
        )

        rendered = hf_tokenizer.apply_chat_template(
            list(messages),
            chat_template=self.chat_template,
            **kwargs,
        )
        return self._as_list(rendered)

    @staticmethod
    def _encode_text(hf_tokenizer, text: str) -> List[int]:
        """Encode already-rendered chat text without adding special tokens."""
        if not text:
            return []
        return list(hf_tokenizer.encode(text, add_special_tokens=False))

    @staticmethod
    def _prepare_tools_for_render(
        hf_tokenizer,
        tools: Optional[Sequence[Dict[str, Any]]],
        kwargs: Dict[str, Any],
    ) -> Tuple[Optional[List[Dict[str, Any]]], Dict[str, Any]]:
        """Prepare tools for direct Jinja rendering outside apply_chat_template."""
        if not tools:
            return None, kwargs

        tools = list(tools)
        apply_chat_template = hf_tokenizer.apply_chat_template
        if hasattr(apply_chat_template, "__func__"):
            apply_chat_template = apply_chat_template.__func__
        apply_globals = getattr(apply_chat_template, "__globals__", {})
        deep_sort_dict = apply_globals.get("deep_sort_dict")
        encode_tools_to_typescript_style = apply_globals.get(
            "encode_tools_to_typescript_style"
        )

        if deep_sort_dict is not None:
            tools = deep_sort_dict(tools)

        if (
            "tools_ts_str" not in kwargs
            and encode_tools_to_typescript_style is not None
        ):
            try:
                kwargs["tools_ts_str"] = encode_tools_to_typescript_style(tools)
            except Exception as exc:
                logger.warning(
                    "Failed to render tools_ts_str with HF tokenizer helper; "
                    "falling back to raw tools for chat template rendering: %s",
                    exc,
                )

        return tools, kwargs

    def _render_with_generation_indices(
        self,
        tokenizer: "AutoTokenizerFromHF",
        messages: Sequence[Dict[str, Any]],
        tools: Optional[Sequence[Dict[str, Any]]] = None,
    ) -> Tuple[str, List[Tuple[int, int]]]:
        """Render chat text and return assistant generation character ranges."""
        from transformers.utils.chat_template_utils import render_jinja_template

        hf_tokenizer = tokenizer.hf_tokenizer()
        chat_template = self._require_generation_template(self.chat_template)

        kwargs = dict(self.chat_template_kwargs)
        documents = kwargs.get("documents")
        tools, kwargs = self._prepare_tools_for_render(hf_tokenizer, tools, kwargs)
        render_kwargs = {**hf_tokenizer.special_tokens_map, **kwargs}
        render_kwargs.update(
            {
                "conversations": [list(messages)],
                "tools": tools,
                "documents": documents,
                "chat_template": chat_template,
                "return_assistant_tokens_mask": True,
                "continue_final_message": False,
                "add_generation_prompt": False,
            }
        )
        rendered_chats, generation_indices = render_jinja_template(**render_kwargs)
        return rendered_chats[0], generation_indices[0]

    @staticmethod
    def _offsets_from_tokenizer(
        hf_tokenizer,
        text: str,
        input_ids: List[int],
    ) -> Optional[List[Tuple[int, int]]]:
        """Return token character offsets using fast offsets or decode offsets."""
        try:
            encoded = hf_tokenizer(
                text,
                add_special_tokens=False,
                return_offsets_mapping=True,
            )
            encoded_ids = encoded.get("input_ids")
            offsets = encoded.get("offset_mapping")
            if encoded_ids == input_ids and offsets is not None:
                return [(int(start), int(end)) for start, end in offsets]
        except (NotImplementedError, TypeError, ValueError):
            pass

        model = getattr(hf_tokenizer, "model", None)
        if hasattr(model, "decode_with_offsets"):
            decoded_text, offsets = model.decode_with_offsets(input_ids)
            if decoded_text == text and len(offsets) == len(input_ids):
                starts = [int(offset) for offset in offsets]
                ends = starts[1:] + [len(text)]
                return list(zip(starts, ends))

        return None

    @staticmethod
    def _span_mask_from_offsets(
        offsets: List[Tuple[int, int]],
        generation_ranges: List[Tuple[int, int]],
    ) -> List[int]:
        """Convert generation character ranges into a per-token assistant mask."""
        mask: List[int] = []
        range_index = 0

        for token_start, token_end in offsets:
            while (
                range_index < len(generation_ranges)
                and generation_ranges[range_index][1] <= token_start
            ):
                range_index += 1

            if range_index >= len(generation_ranges):
                mask.append(0)
                continue

            range_start, range_end = generation_ranges[range_index]
            overlaps = token_start < range_end and token_end > range_start
            if not overlaps:
                mask.append(0)
                continue

            if token_start < range_start or token_end > range_end:
                raise ValueError(
                    "HuggingFace generation boundary splits a token. "
                    "Move `{% generation %}` boundaries to tokenizer boundaries."
                )
            mask.append(1)

        return mask

    def _chunk_tokenize_mask(
        self,
        hf_tokenizer,
        text: str,
        input_ids: List[int],
        generation_ranges: List[Tuple[int, int]],
    ) -> Optional[List[int]]:
        """Fallback mask builder that tokenizes generation/non-generation chunks."""
        boundaries = sorted(
            {0, len(text)}
            | {boundary for span in generation_ranges for boundary in span}
        )
        chunk_ids: List[int] = []
        chunk_mask: List[int] = []

        for start, end in zip(boundaries, boundaries[1:]):
            chunk = text[start:end]
            token_ids = self._encode_text(hf_tokenizer, chunk)
            in_generation = any(
                range_start <= start and end <= range_end
                for range_start, range_end in generation_ranges
            )
            chunk_ids.extend(token_ids)
            chunk_mask.extend([1 if in_generation else 0] * len(token_ids))

        if chunk_ids == input_ids:
            return chunk_mask
        return None

    def _assistant_mask_from_generation_ranges(
        self,
        hf_tokenizer,
        text: str,
        input_ids: List[int],
        generation_ranges: List[Tuple[int, int]],
    ) -> List[int]:
        """Align assistant generation character ranges to input token positions."""
        offsets = self._offsets_from_tokenizer(hf_tokenizer, text, input_ids)
        if offsets is not None:
            return self._span_mask_from_offsets(offsets, generation_ranges)

        chunk_mask = self._chunk_tokenize_mask(
            hf_tokenizer=hf_tokenizer,
            text=text,
            input_ids=input_ids,
            generation_ranges=generation_ranges,
        )
        if chunk_mask is not None:
            return chunk_mask

        raise ValueError(
            "Unable to align HuggingFace generation ranges to token positions. "
            "Use a fast tokenizer with offsets, a tokenizer exposing decode_with_offsets, "
            "or place `{% generation %}` boundaries exactly on tokenizer boundaries."
        )

    def _tokenize_with_generation_indices(
        self,
        tokenizer: "AutoTokenizerFromHF",
        messages: Sequence[Dict[str, Any]],
        tools: Optional[Sequence[Dict[str, Any]]] = None,
    ) -> Tuple[List[int], List[int]]:
        """Render OpenAI chat messages and build assistant-token masks."""
        hf_tokenizer = tokenizer.hf_tokenizer()
        rendered, generation_ranges = self._render_with_generation_indices(
            tokenizer=tokenizer,
            messages=messages,
            tools=tools,
        )
        input_ids = self._encode_text(hf_tokenizer, rendered)
        assistant_masks = self._assistant_mask_from_generation_ranges(
            hf_tokenizer=hf_tokenizer,
            text=rendered,
            input_ids=input_ids,
            generation_ranges=generation_ranges,
        )
        return input_ids, assistant_masks

    @staticmethod
    def _mask_to_final_span(mask: List[int]) -> List[int]:
        """Keep only the final contiguous trainable assistant span."""
        last_start = None
        last_end = None
        index = 0
        while index < len(mask):
            if mask[index]:
                start = index
                while index < len(mask) and mask[index]:
                    index += 1
                last_start, last_end = start, index
            else:
                index += 1

        final_mask = [0] * len(mask)
        if last_start is not None:
            final_mask[last_start:last_end] = [1] * (last_end - last_start)
        return final_mask

    @staticmethod
    def _truncate_to_assistant_boundary(
        input_ids: List[int],
        assistant_masks: List[int],
        max_length: Optional[int],
    ) -> Tuple[List[int], List[int]]:
        """Keep a prefix whose final token is inside assistant generation."""
        if len(input_ids) != len(assistant_masks):
            raise ValueError(
                "assistant mask length must match input_ids length, got "
                f"{len(assistant_masks)} vs {len(input_ids)}"
            )
        if max_length is None or len(input_ids) <= max_length:
            return input_ids, assistant_masks
        if max_length <= 0:
            return [], []

        end = max_length
        if assistant_masks[end - 1]:
            return input_ids[:end], assistant_masks[:end]

        # If the nominal boundary is in source/user/tool text, roll back to the
        # previous assistant token so the kept sample ends at a trainable answer.
        boundary = end - 1
        while boundary >= 0 and not assistant_masks[boundary]:
            boundary -= 1

        if boundary < 0:
            return [], []

        end = boundary + 1
        return input_ids[:end], assistant_masks[:end]

    @classmethod
    def _build_labels_from_assistant_mask(
        cls,
        input_ids: List[int],
        assistant_masks: List[int],
        ignore_index: int,
        history_mask_loss: bool,
    ) -> Tuple[List[int], List[int]]:
        """Build labels and loss mask from the assistant-token mask."""
        if len(input_ids) != len(assistant_masks):
            raise ValueError(
                "assistant mask length must match input_ids length, got "
                f"{len(assistant_masks)} vs {len(input_ids)}"
            )

        loss_mask = [1 if mask else 0 for mask in assistant_masks]
        if history_mask_loss:
            loss_mask = cls._mask_to_final_span(loss_mask)
        labels = [
            token_id if mask else ignore_index
            for token_id, mask in zip(input_ids, loss_mask)
        ]
        return labels, loss_mask

    def encode_openai(
        self,
        tokenizer: "AutoTokenizerFromHF",
        messages: Sequence[Dict[str, Any]],
        tools: Optional[Sequence[Dict[str, Any]]] = None,
        train_on_prompt: bool = False,
        history_mask_loss: bool = False,
        ignore_index: int = -100,
        max_length: Optional[int] = None,
    ) -> Tuple[List[int], List[int], List[int], int]:
        """Encode OpenAI-style chat data into input ids, labels, and loss mask."""
        input_ids, assistant_masks = self._tokenize_with_generation_indices(
            tokenizer=tokenizer,
            messages=messages,
            tools=tools,
        )
        ori_total_len = len(input_ids)
        input_ids, assistant_masks = self._truncate_to_assistant_boundary(
            input_ids=input_ids,
            assistant_masks=assistant_masks,
            max_length=max_length,
        )

        if train_on_prompt:
            labels = list(input_ids)
            return input_ids, labels, [1] * len(input_ids), ori_total_len

        labels, loss_mask = self._build_labels_from_assistant_mask(
            input_ids=input_ids,
            assistant_masks=assistant_masks,
            ignore_index=ignore_index,
            history_mask_loss=history_mask_loss,
        )
        return input_ids, labels, loss_mask, ori_total_len


@dataclass
class Llama2Template(ChatTemplate):
    """LLaMA-2 Template"""

    def _encode(
        self,
        tokenizer: "AutoTokenizerFromHF",
        messages: Sequence[Dict[str, str]],
        system: str,
    ) -> List[List[int]]:
        """
        Encodes formatted inputs to pairs of token ids.
        Turn 0: prefix + system + query    resp
        Turn t: sep + query                resp
        """
        system = system or self.default_system
        encoded_messages = []
        for i, message in enumerate(messages):
            elements = []

            system_text = ""

            if i == 0:
                elements += self.format_prefix.apply()
                if system:
                    system_text = self.format_system.apply(content=system)[0]

            if i > 0 and i % 2 == 0:
                elements += self.format_separator.apply()

            if message["role"] == DataRoles.USER:
                elements += self.format_user.apply(
                    content=system_text + message["content"]
                )
            elif message["role"] == DataRoles.ASSISTANT:
                elements += self.format_assistant.apply(content=message["content"])
            else:
                raise NotImplementedError("Unexpected role: {}".format(message["role"]))

            encoded_messages.append(self._convert_elements_to_ids(tokenizer, elements))

        return encoded_messages


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


_register_chat_template(
    name="kimi-k2.5-hf",
    cls=HFChatTemplate,
    chat_template=(
        resources.files("loongforge.data.chat_templates")
        .joinpath("kimi_k2_5_training.jinja")
        .read_text(encoding="utf-8")
    ),
    mm_plugin=KimiK25Plugin(
        image_token="<|media_content|>",
        video_token="<|media_content|>",
        merge_kernel_size=(2, 2),
        temporal_merge_kernel_size=4,
    ),
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
    mm_plugin=KimiK25Plugin(
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
    mm_plugin=KimiK25Plugin(
        image_token="<|media_content|>",
        video_token="<|media_content|>",
        merge_kernel_size=(2, 2),
        temporal_merge_kernel_size=4,
    ),
)

# Kimi K2.6 uses the same chat format and media plugin as Kimi K2.5.
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
    mm_plugin=KimiK25Plugin(
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
    mm_plugin=KimiK25Plugin(
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
