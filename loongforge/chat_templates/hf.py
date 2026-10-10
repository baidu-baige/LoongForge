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


"""Chat template backed by HuggingFace jinja templates (OpenAI-style messages)."""

import json
import logging
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Any, Dict, List, Optional, Sequence, Tuple

from loongforge.constants import IGNORE_INDEX

from .base import ChatTemplate

if TYPE_CHECKING:
    from loongforge.engines.mcore.tokenizer import AutoTokenizerFromHF

logger = logging.getLogger(__name__)


def load_chat_template_kwargs(raw_kwargs: Optional[str]) -> Dict[str, Any]:
    """Load HF chat template kwargs from a JSON object string or JSON file path."""
    if raw_kwargs is None:
        return {}

    kwargs_text = raw_kwargs
    if not kwargs_text.lstrip().startswith("{"):
        kwargs_path = Path(kwargs_text)
        if kwargs_path.is_file():
            kwargs_text = kwargs_path.read_text(encoding="utf-8")

    chat_template_kwargs = json.loads(kwargs_text)
    if not isinstance(chat_template_kwargs, dict):
        raise ValueError("chat_template_kwargs must be a JSON object")
    return chat_template_kwargs


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
        hf_tokenizer = tokenizer.tokenizer
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

        hf_tokenizer = tokenizer.tokenizer
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
        hf_tokenizer = tokenizer.tokenizer
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
        ignore_index: int = IGNORE_INDEX,
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
