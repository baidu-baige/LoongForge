"""AIAK Global Variables"""

from typing import TYPE_CHECKING, Optional

from megatron.training import get_args as _get_args
from megatron.training.global_vars import _ensure_var_is_initialized, _ensure_var_is_not_initialized

from aiak_training_omni.tokenizer import build_tokenizer
from aiak_training_omni.data import ChatTemplate

from .constants import TrainingPhase


if TYPE_CHECKING:
    from megatron.core.datasets.megatron_tokenizer import MegatronTokenizer


_GLOBAL_CHAT_TEMPLATE: Optional["ChatTemplate"] = None
_GLOBAL_AIAK_TOKENIZER: Optional["MegatronTokenizer"] = None


def set_aiak_extra_global_vars(args, build_tokenizer=True) -> None:
    """Set AIAK extra global variables"""
    assert args is not None
    if build_tokenizer:
        _ = _build_chat_template(args)
        _ = _build_tokenizer(args)

    # set AIAK-ACCELERATOR extra global vars
    try:
        from aiak_accelerator.multiacc_engine import multiacc_set_accelerator_extra_global_vars
    except ImportError:
        multiacc_set_accelerator_extra_global_vars = None

    if multiacc_set_accelerator_extra_global_vars is not None:
        multiacc_set_accelerator_extra_global_vars(args, build_tokenizer)


def  _build_chat_template(args) -> Optional["ChatTemplate"]:
    """Build the chat template."""
    if args.training_phase == TrainingPhase.SFT and args.chat_template is not None:
        global _GLOBAL_CHAT_TEMPLATE
        _ensure_var_is_not_initialized(_GLOBAL_CHAT_TEMPLATE, 'aiak-chat-template')
        _GLOBAL_CHAT_TEMPLATE = ChatTemplate.from_name(args.chat_template)
        assert _GLOBAL_CHAT_TEMPLATE is not None, f"chat_template {args.chat_template} not supported."
        return _GLOBAL_CHAT_TEMPLATE

    return None


def _build_tokenizer(args) -> Optional["MegatronTokenizer"]:
    """Initialize tokenizer."""
    global _GLOBAL_AIAK_TOKENIZER
    _ensure_var_is_not_initialized(_GLOBAL_AIAK_TOKENIZER, 'aiak-tokenizer')
    _GLOBAL_AIAK_TOKENIZER = build_tokenizer(args, chat_template=_GLOBAL_CHAT_TEMPLATE)
    return _GLOBAL_AIAK_TOKENIZER


def get_tokenizer() -> Optional["MegatronTokenizer"]:
    """Return tokenizer."""
    _ensure_var_is_initialized(_GLOBAL_AIAK_TOKENIZER, 'aiak-tokenizer')
    return _GLOBAL_AIAK_TOKENIZER


def get_chat_template() -> Optional["ChatTemplate"]:
    """Return chat template."""
    _ensure_var_is_initialized(_GLOBAL_CHAT_TEMPLATE, 'aiak-chat-template')
    return _GLOBAL_CHAT_TEMPLATE


def get_args():
    """Return args for Megatron now."""
    return _get_args()
