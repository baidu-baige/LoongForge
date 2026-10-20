import html
import os
import re
import urllib.parse as ul
from typing import Any

import ftfy
import torch
from bs4 import BeautifulSoup
from transformers import CLIPTextModel, CLIPTextModelWithProjection, CLIPTokenizer, T5EncoderModel, T5Tokenizer


class PromptTransform:
    """End-to-end prompt tokenizer+encoder pipeline for CLIP/T5 variants.

    Args:
        mode (str): One of {'clip_text', 'clip_text_proj', 'clip_text_and_proj', 't5'}
            controlling which tokenizer(s) and text encoder(s) are used.
        model_path (str): Root path or HF id containing the tokenizer/encoder folders.
        device (str | None): Device string such as 'cuda:0' or 'cpu'. If None, use defaults.
        dtype (str | torch.dtype | None): Mixed precision dtype. String values are
            resolved via ``getattr(torch, dtype)``. Defaults to float16 on CUDA, otherwise float32.
    """

    def __init__(self, mode: str, model_path: str, device: str | None = None, dtype: str | torch.dtype | None = None):
        if dtype is None:
            if device is not None and 'cuda' in device:
                dtype = torch.float16
            else:
                dtype = torch.float32
        if isinstance(dtype, str):
            dtype = getattr(torch, dtype)
        self.mode = mode
        self.model_path = model_path
        self.device = device
        self.dtype = dtype
        self.tokenizers = None
        self.text_encoders = None

    def load_model(self) -> None:
        """Lazily initialize tokenizer(s) and encoder(s).

        Notes:
            This function is idempotent and safe to call multiple times.
        """
        if self.text_encoders is None:
            # Lazily initialize tokenizers and text encoders on first call
            self.tokenizers = load_tokenizers(self.mode, self.model_path)
            self.text_encoders = load_text_encoders(self.mode, self.model_path, self.device, self.dtype)

    @torch.no_grad()
    def __call__(
        self, prompt: str | list[str], clean: bool = False, max_length: int | None = None, with_attention_mask: bool = False
    ) -> tuple[Any, Any | None]:
        """Tokenize and encode text prompt(s) into embeddings.

        Args:
            prompt (str | list[str]): Single prompt or list of prompts.
            clean (bool): Whether to normalize/sanitize the prompt using ``clean_prompt``.
            max_length (int | None): Optional maximum token length for padding/truncation.
            with_attention_mask (bool): If True, also return attention mask(s).

        Returns:
            tuple: (prompt_embeds, prompt_masks_or_none)
                - prompt_embeds: torch.Tensor or tuple of tensors depending on mode
                - prompt_masks_or_none: attention mask tensor(s) or None
        """
        self.load_model()
        if clean:
            # Normalize and sanitize prompt text before tokenization
            prompt = clean_prompt(prompt)
            prompt = clean_prompt(prompt)
        # Tokenize into ids and attention masks (optionally capped by max_length)
        prompt_ids, prompt_masks = forward_tokenizers(self.tokenizers, prompt, max_length)
        if not with_attention_mask:
            prompt_masks = None
        # Encode tokens via chosen text encoder(s) into embeddings
        prompt_embeds = forward_text_encoders(self.text_encoders, self.mode, prompt_ids, prompt_masks)
        if with_attention_mask and len(prompt_masks) == 1:
            # Unwrap singleton list for convenience
            prompt_masks = prompt_masks[0]
        return prompt_embeds, prompt_masks


class PromptTokenizerTransform:
    """Only run tokenization and return token ids and masks.

    Args:
        mode (str): See :class:`PromptTransform` for valid values.
        model_path (str): Root path or HF id to load tokenizer(s).
    """

    def __init__(self, mode: str, model_path: str):
        self.mode = mode
        self.model_path = model_path
        self.tokenizers = None

    def load_model(self) -> None:
        """Lazily initialize tokenizer(s)."""
        if self.tokenizers is None:
            # Lazily initialize tokenizers on first call
            self.tokenizers = load_tokenizers(self.mode, self.model_path)

    def __call__(self, prompt: str | list[str], clean: bool = False, max_length: int | None = None) -> tuple[Any, Any]:
        """Tokenize prompts into ids and masks.

        Args:
            prompt (str | list[str]): Single prompt or list of prompts.
            clean (bool): Whether to normalize/sanitize the prompt using ``clean_prompt``.
            max_length (int | None): Optional maximum token length for padding/truncation.

        Returns:
            tuple | (Tensor, Tensor): If a single tokenizer is used, returns a pair
            (input_ids, attention_mask). Otherwise returns lists for each tokenizer.
        """
        self.load_model()
        if clean:
            # Optional prompt normalization
            prompt = clean_prompt(prompt)
            prompt = clean_prompt(prompt)
        # Batch tokenize with padding/truncation
        prompt_ids, prompt_masks = forward_tokenizers(self.tokenizers, prompt, max_length)
        if len(prompt_ids) == 1:
            return prompt_ids[0], prompt_masks[0]
        else:
            return prompt_ids, prompt_masks


class PromptEncoderTransform:
    """Only run text encoders on input token ids to get embeddings.

    Args:
        mode (str): See :class:`PromptTransform` for valid values.
        model_path (str): Root path or HF id to load text encoder(s).
        device (str | None): Device string such as 'cuda:0' or 'cpu'. If None, use defaults.
        dtype (str | torch.dtype | None): Mixed precision dtype; string resolved via ``getattr(torch, ...)``.
    """

    def __init__(self, mode: str, model_path: str, device: str | None = None, dtype: str | torch.dtype | None = None):
        if dtype is None:
            if device is not None and 'cuda' in device:
                dtype = torch.float16
            else:
                dtype = torch.float32
        if isinstance(dtype, str):
            dtype = getattr(torch, dtype)
        self.mode = mode
        self.model_path = model_path
        self.device = device
        self.dtype = dtype
        self.text_encoders = None

    def load_model(self) -> None:
        """Lazily initialize encoder(s)."""
        if self.text_encoders is None:
            # Lazily initialize encoders on first call
            self.text_encoders = load_text_encoders(self.mode, self.model_path, self.device, self.dtype)

    @torch.no_grad()
    def __call__(self, prompt_ids: Any, prompt_masks: Any | None = None) -> Any:
        """Encode token ids (and optional masks) into embeddings.

        Args:
            prompt_ids (Any): Token id tensor(s). For multi-encoder mode, pass a list aligned with encoders.
            prompt_masks (Any | None): Attention mask tensor(s) or None.

        Returns:
            torch.Tensor | tuple: Embedding tensor(s) depending on the selected mode.
        """
        self.load_model()
        # Forward through encoders; masks may be None depending on caller
        prompt_embeds = forward_text_encoders(self.text_encoders, self.mode, prompt_ids, prompt_masks)
        return prompt_embeds


def load_tokenizers(mode: str, model_path: str) -> list[Any]:
    """Load one or two tokenizers based on the selected mode and filesystem
    layout.

    Args:
        mode (str): Determines tokenizer type: 'clip' in mode uses CLIPTokenizer, 't5' uses T5Tokenizer.
        model_path (str): Directory that may contain 'tokenizer' and 'tokenizer_2' subfolders.

    Returns:
        list: List of tokenizer instances (length 1 or 2).
    """
    tokenizer_model_paths = []
    for tokenizer_name in ['tokenizer', 'tokenizer_2']:
        tokenizer_model_path = os.path.join(model_path, tokenizer_name)
        if os.path.exists(tokenizer_model_path):
            tokenizer_model_paths.append(tokenizer_model_path)
    if len(tokenizer_model_paths) == 0:
        # Single-tokenizer layout: point to the root model path
        tokenizer_model_paths = [model_path]
    if 'clip' in mode:
        tokenizers = [CLIPTokenizer.from_pretrained(_) for _ in tokenizer_model_paths]
    elif 't5' in mode:
        tokenizers = [T5Tokenizer.from_pretrained(_) for _ in tokenizer_model_paths]
    else:
        assert False
    return tokenizers


def load_text_encoders(mode: str, model_path: str, device: str | None, dtype: torch.dtype) -> list[Any]:
    """Load text encoder module(s) given a mode and model layout.

    Args:
        mode (str): 'clip_text', 'clip_text_proj', 'clip_text_and_proj', or 't5'.
        model_path (str): Directory containing 'text_encoder' and optionally 'text_encoder_2'.
        device (str | None): Device to place the model(s) on.
        dtype (torch.dtype): Parameter dtype for the model(s).

    Returns:
        list: List of initialized, eval-mode encoders with gradients disabled.
    """
    text_encoder_model_paths = []
    for text_encoder_name in ['text_encoder', 'text_encoder_2']:
        text_encoder_model_path = os.path.join(model_path, text_encoder_name)
        if os.path.exists(text_encoder_model_path):
            text_encoder_model_paths.append(text_encoder_model_path)
    if len(text_encoder_model_paths) == 0:
        # Fallback to single-encoder layout
        text_encoder_model_paths = [model_path]
    if mode == 'clip_text':
        text_encoders = [CLIPTextModel.from_pretrained(text_encoder_model_paths[0], torch_dtype=dtype)]
    elif mode == 'clip_text_proj':
        text_encoders = [CLIPTextModelWithProjection.from_pretrained(text_encoder_model_paths[0], torch_dtype=dtype)]
    elif mode == 'clip_text_and_proj':
        text_encoders = [
            CLIPTextModel.from_pretrained(text_encoder_model_paths[0], torch_dtype=dtype),
            CLIPTextModelWithProjection.from_pretrained(text_encoder_model_paths[1], torch_dtype=dtype),
        ]
    elif mode == 't5':
        text_encoders = [T5EncoderModel.from_pretrained(text_encoder_model_paths[0], torch_dtype=dtype)]
    else:
        assert False
    for text_encoder in text_encoders:
        text_encoder.requires_grad_(False)
        if device is not None:
            # Place modules onto the requested device and dtype
            text_encoder.to(device)
    return text_encoders


def forward_tokenizers(
    tokenizers: list[Any], prompt: str | list[str], max_length: int | None = None
) -> tuple[list[torch.Tensor], list[torch.Tensor]]:
    """Run tokenizers on prompts with padding/truncation.

    Args:
        tokenizers: List of tokenizer instances.
        prompt (str | list[str]): Single prompt or list of prompts.
        max_length (int | None): If provided, pad/truncate to this length; else use model default.

    Returns:
        tuple[list[Tensor], list[Tensor]]: Lists of input_ids and attention_masks (one per tokenizer).
    """
    prompt_ids = []
    prompt_masks = []
    for tokenizer in tokenizers:
        # Use provided max_length or fall back to the tokenizer default
        max_length_i = max_length or tokenizer.model_max_length
        text_inputs = tokenizer(
            prompt,
            padding='max_length',
            max_length=max_length_i,
            truncation=True,
            return_tensors='pt',
        )
        input_ids = text_inputs.input_ids
        attention_mask = text_inputs.attention_mask
        prompt_ids.append(input_ids)
        prompt_masks.append(attention_mask)
    return prompt_ids, prompt_masks


def forward_text_encoders(
    text_encoders: list[Any], mode: str, prompt_ids: Any, prompt_masks: Any | None = None
) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
    """Forward text encoder(s) and return embeddings depending on mode.

    Args:
        text_encoders: List of encoders (length 1 or 2).
        mode (str): Encoding mode controlling outputs and concatenation.
        prompt_ids (Any): Token id tensor(s) aligned with encoders.
        prompt_masks (Any | None): Mask tensor(s) aligned with encoders or None.

    Returns:
        torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
            - For single-encoder modes: hidden states or pooled feature.
            - For 'clip_text_and_proj': (concat penultimate hidden states, pooled feature from final encoder).
    """
    device = text_encoders[0].device
    if mode in ('clip_text', 'clip_text_proj', 't5'):
        # Single-encoder flow: move ids/masks to the encoder device
        if isinstance(prompt_ids, list):
            prompt_ids = prompt_ids[0].to(device)
        else:
            prompt_ids = prompt_ids.to(device)
        if prompt_masks is not None:
            if isinstance(prompt_masks, list):
                prompt_masks = prompt_masks[0].to(device)
            else:
                prompt_masks = prompt_masks.to(device)
        with torch.no_grad():
            # Return hidden states or pooled outputs depending on mode
            prompt_embeds = text_encoders[0](prompt_ids, attention_mask=prompt_masks)
        if mode in ('clip_text', 't5'):
            prompt_embeds = prompt_embeds[0]
        else:
            prompt_embeds = prompt_embeds[0].unsqueeze(1)

    elif mode == 'clip_text_and_proj':
        # Two-encoder flow: concatenate penultimate hidden states, keep pooled output from final encoder
        prompt_embeds_list = []
        for i, text_encoder in enumerate(text_encoders):
            prompt_ids_i = prompt_ids[i].to(device)
            prompt_masks_i = prompt_masks[i].to(device) if prompt_masks is not None else None
            with torch.no_grad():
                prompt_embeds = text_encoder(prompt_ids_i, attention_mask=prompt_masks_i, output_hidden_states=True)
            # We are only ALWAYS interested in the pooled output of the final text encoder
            pooled_prompt_embeds = prompt_embeds[0]
            prompt_embeds = prompt_embeds.hidden_states[-2]
            prompt_embeds_list.append(prompt_embeds)
        prompt_embeds = torch.concat(prompt_embeds_list, dim=-1)
        prompt_embeds = (prompt_embeds, pooled_prompt_embeds)

    else:
        assert False
    return prompt_embeds


def truncate_prompt(prompt_embeds: list[torch.Tensor], prompt_masks: list[torch.Tensor]) -> torch.Tensor | list[torch.Tensor]:
    """Trim embeddings to the valid token length according to masks.

    Args:
        prompt_embeds: List of [L, D] embedding tensors.
        prompt_masks: List of [L] attention masks (1 for keep, 0 for pad).

    Returns:
        torch.Tensor | list[torch.Tensor]: Trimmed embedding(s). Returns a single
        tensor if the input list length is 1, else a list.
    """
    assert len(prompt_embeds) == len(prompt_masks)
    new_prompt_embeds = []
    for i in range(len(prompt_embeds)):
        # Determine valid token length via attention mask sum
        keep_index = prompt_masks[i].sum().item()
        prompt_embed = prompt_embeds[i][:keep_index]
        new_prompt_embeds.append(prompt_embed)
    return new_prompt_embeds[0] if len(new_prompt_embeds) == 1 else new_prompt_embeds


def pad_prompt(prompt_embeds: torch.Tensor, max_length: int, prompt_masks: torch.Tensor | None = None) -> tuple[torch.Tensor, torch.Tensor]:
    """Pad embeddings (and masks) to a target max length.

    Args:
        prompt_embeds: [L, D] embedding tensor.
        max_length (int): Target sequence length to pad to.
        prompt_masks: Optional [L] mask tensor; if None, defaults to all-ones for existing tokens.

    Returns:
        tuple[torch.Tensor, torch.Tensor]: (padded_embeds, padded_masks) with length ``max_length``.
    """
    cur_length = prompt_embeds.shape[0]
    assert cur_length <= max_length
    if prompt_masks is None:
        # If caller did not provide mask, default to all-ones for current tokens
        prompt_masks = torch.ones((cur_length,), device=prompt_embeds.device, dtype=torch.int64)
    if cur_length == max_length:
        return prompt_embeds, prompt_masks
    new_shape = list(prompt_embeds.shape)
    new_shape[0] = max_length
    # Allocate zero-padded tensors and copy current tokens/mask into prefix
    new_prompt_embeds = torch.zeros(new_shape, device=prompt_embeds.device, dtype=prompt_embeds.dtype)
    new_prompt_masks = torch.zeros((max_length,), device=prompt_masks.device, dtype=prompt_masks.dtype)
    new_prompt_embeds[:cur_length] = prompt_embeds
    new_prompt_masks[:cur_length] = prompt_masks
    return new_prompt_embeds, new_prompt_masks


def clean_prompt(prompt: str) -> str:
    """Normalize/sanitize a raw prompt string for more stable tokenization.

    Steps include URL decoding, lowercasing, stripping HTML, removing mentions,
    trimming excessive punctuation, and several heuristic cleanups.

    Args:
        prompt (str): Raw user-provided prompt.

    Returns:
        str: Cleaned prompt string.
    """
    bad_punct_regex = re.compile(r'[' + '#®•©™&@·º½¾¿¡§~' + r'\)' + r'\(' + r'\]' + r'\[' + r'\}' + r'\{' + r'\|' + '\\' + r'\/' + r'\*' + r']{1,}')
    caption = str(prompt)
    # URL-decoding, lowercasing, and common replacements
    caption = ul.unquote_plus(caption)
    caption = caption.strip().lower()
    caption = re.sub('<person>', 'person', caption)
    # Strip URLs and hostnames
    caption = re.sub(
        r'\b((?:https?:(?:\/{1,3}|[a-zA-Z0-9%])|[a-zA-Z0-9.\-]+[.](?:com|co|ru|net|org|edu|gov|it)[\w/-]*\b\/?(?!@)))',  # noqa
        '',
        caption,
    )  # regex for urls
    caption = re.sub(
        r'\b((?:www:(?:\/{1,3}|[a-zA-Z0-9%])|[a-zA-Z0-9.\-]+[.](?:com|co|ru|net|org|edu|gov|it)[\w/-]*\b\/?(?!@)))',  # noqa
        '',
        caption,
    )  # regex for urls
    # Remove HTML markup and mentions
    caption = BeautifulSoup(caption, features='html.parser').text

    # @<nickname>
    caption = re.sub(r'@[\w\d]+\b', '', caption)

    # 31C0—31EF CJK Strokes
    # 31F0—31FF Katakana Phonetic Extensions
    # 3200—32FF Enclosed CJK Letters and Months
    # 3300—33FF CJK Compatibility
    # 3400—4DBF CJK Unified Ideographs Extension A
    # 4DC0—4DFF Yijing Hexagram Symbols
    # 4E00—9FFF CJK Unified Ideographs
    caption = re.sub(r'[\u31c0-\u31ef]+', '', caption)
    caption = re.sub(r'[\u31f0-\u31ff]+', '', caption)
    caption = re.sub(r'[\u3200-\u32ff]+', '', caption)
    caption = re.sub(r'[\u3300-\u33ff]+', '', caption)
    caption = re.sub(r'[\u3400-\u4dbf]+', '', caption)
    caption = re.sub(r'[\u4dc0-\u4dff]+', '', caption)
    caption = re.sub(r'[\u4e00-\u9fff]+', '', caption)
    #######################################################

    # все виды тире / all types of dash --> "-"
    caption = re.sub(
        r'[\u002D\u058A\u05BE\u1400\u1806\u2010-\u2015\u2E17\u2E1A\u2E3A\u2E3B\u2E40\u301C\u3030\u30A0\uFE31\uFE32\uFE58\uFE63\uFF0D]+',  # noqa
        '-',
        caption,
    )

    # кавычки к одному стандарту
    caption = re.sub(r'[`´«»“”¨]', '"', caption)
    caption = re.sub(r'[‘’]', "'", caption)

    # &quot;
    caption = re.sub(r'&quot;?', '', caption)
    # &amp
    caption = re.sub(r'&amp', '', caption)

    # ip adresses:
    caption = re.sub(r'\d{1,3}\.\d{1,3}\.\d{1,3}\.\d{1,3}', ' ', caption)

    # article ids:
    caption = re.sub(r'\d:\d\d\s+$', '', caption)

    # \n
    caption = re.sub(r'\\n', ' ', caption)

    # "#123"
    caption = re.sub(r'#\d{1,3}\b', '', caption)
    # "#12345.."
    caption = re.sub(r'#\d{5,}\b', '', caption)
    # "123456.."
    caption = re.sub(r'\b\d{6,}\b', '', caption)
    # filenames:
    caption = re.sub(r'[\S]+\.(?:png|jpg|jpeg|bmp|webp|eps|pdf|apk|mp4)', '', caption)

    #
    caption = re.sub(r"[\"\']{2,}", r'"', caption)  # """AUSVERKAUFT"""
    caption = re.sub(r'[\.]{2,}', r' ', caption)  # """AUSVERKAUFT"""

    caption = re.sub(bad_punct_regex, r' ', caption)  # ***AUSVERKAUFT***, #AUSVERKAUFT
    caption = re.sub(r'\s+\.\s+', r' ', caption)  # " . "

    # this-is-my-cute-cat / this_is_my_cute_cat
    regex2 = re.compile(r'(?:\-|\_)')
    if len(re.findall(regex2, caption)) > 3:
        caption = re.sub(regex2, ' ', caption)

    # Unicode fixes, unescape HTML entities, normalize punctuation and noise
    caption = ftfy.fix_text(caption)
    caption = html.unescape(html.unescape(caption))

    caption = re.sub(r'\b[a-zA-Z]{1,3}\d{3,15}\b', '', caption)  # jc6640
    caption = re.sub(r'\b[a-zA-Z]+\d+[a-zA-Z]+\b', '', caption)  # jc6640vc
    caption = re.sub(r'\b\d+[a-zA-Z]+\d+\b', '', caption)  # 6640vc231

    caption = re.sub(r'(worldwide\s+)?(free\s+)?shipping', '', caption)
    caption = re.sub(r'(free\s)?download(\sfree)?', '', caption)
    caption = re.sub(r'\bclick\b\s(?:for|on)\s\w+', '', caption)
    caption = re.sub(r'\b(?:png|jpg|jpeg|bmp|webp|eps|pdf|apk|mp4)(\simage[s]?)?', '', caption)
    caption = re.sub(r'\bpage\s+\d+\b', '', caption)

    caption = re.sub(r'\b\d*[a-zA-Z]+\d+[a-zA-Z]+\d+[a-zA-Z\d]*\b', r' ', caption)  # j2d1a2a...

    caption = re.sub(r'\b\d+\.?\d*[xх×]\d+\.?\d*\b', '', caption)

    caption = re.sub(r'\b\s+\:\s+', r': ', caption)
    caption = re.sub(r'(\D[,\./])\b', r'\1 ', caption)
    caption = re.sub(r'\s+', ' ', caption)

    caption.strip()

    caption = re.sub(r"^[\"\']([\w\W]+)[\"\']$", r'\1', caption)
    caption = re.sub(r"^[\'\_,\-\:;]", r'', caption)
    caption = re.sub(r"[\'\_,\-\:\-\+]$", r'', caption)
    caption = re.sub(r'^\.\S+$', '', caption)

    return caption.strip()
