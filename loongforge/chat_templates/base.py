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


"""Slot-based chat templates (prompt/response encoding) and their formatters."""

import re
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import (
    TYPE_CHECKING,
    Any,
    Dict,
    List,
    Optional,
    Sequence,
    Set,
    Tuple,
    Union,
)

from loongforge.constants import IGNORE_INDEX

from .plugins.mm_plugin import MMPlugin

if TYPE_CHECKING:
    from loongforge.engines.mcore.tokenizer import AutoTokenizerFromHF


SlotsType = Sequence[Union[str, Set[str], Dict[str, str]]]


class DataRoles(object):
    """data roles"""

    USER = "user"
    ASSISTANT = "assistant"
    OBSERVATION = "observation"
    FUNCTION = "function"
    SYSTEM = "system"


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
        ignore_index: int = IGNORE_INDEX,
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
        from .registry import MAPPING_NAME_TO_TEMPLATE

        return MAPPING_NAME_TO_TEMPLATE.get(name, None)


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
