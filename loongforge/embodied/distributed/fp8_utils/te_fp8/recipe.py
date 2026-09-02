# Copyright 2026 The LoongForge Authors.
# SPDX-License-Identifier: Apache-2.0

"""TransformerEngine scaling recipes and FP8 execution contexts.

Linear replacement only changes module implementations; FP8 GEMMs become active
when the complete model forward runs inside ``te.fp8_autocast``. Ordinary
``torch.autocast`` may remain active at the same time for BF16/FP16 operations
outside TE modules.
"""

from __future__ import annotations

import logging
from contextlib import contextmanager, nullcontext

logger = logging.getLogger(__name__)

# Stable CLI spelling -> TransformerEngine class. The indirection keeps training
# arguments independent of TE class names while still reporting when an older TE
# build does not provide a requested recipe.
_RECIPE_CLASSES = {
    "blockwise": "Float8BlockScaling",
    "current": "Float8CurrentScaling",
    "delayed": "DelayedScaling",
}

FP8_RECIPE_CHOICES = tuple(_RECIPE_CLASSES)


def _import_te_recipe():
    """Import the TE recipe module lazily and provide a unit-test seam."""
    from transformer_engine.common import recipe as te_recipe

    return te_recipe


def _resolve_fp8_format(te_recipe, raw_format: str | None):
    """Map the CLI spelling to TransformerEngine's ``Format`` enum."""
    if raw_format is None:
        return None
    format_name = raw_format.upper()
    fp8_format = getattr(te_recipe.Format, format_name, None)
    if fp8_format is None:
        raise ValueError(
            f"Unsupported --fp8-te-format {raw_format!r} in this "
            "TransformerEngine build; expected e4m3 or hybrid."
        )
    return fp8_format


def _recipe_kwargs(recipe_name: str, recipe_args, te_recipe) -> dict:
    """Translate training arguments into kwargs supported by one TE recipe."""
    if recipe_args is None:
        return {}

    kwargs = {}
    fp8_format = _resolve_fp8_format(
        te_recipe,
        getattr(recipe_args, "fp8_te_format", None),
    )
    if fp8_format is not None:
        kwargs["fp8_format"] = fp8_format

    if recipe_name == "delayed":
        kwargs.update(
            margin=recipe_args.fp8_te_margin,
            amax_history_len=recipe_args.fp8_te_amax_history_len,
            amax_compute_algo=recipe_args.fp8_te_amax_compute_algo,
            reduce_amax=recipe_args.fp8_te_reduce_amax,
        )
    elif recipe_name == "current":
        kwargs["use_power_2_scales"] = (
            recipe_args.fp8_te_current_use_power_2_scales
        )
    elif recipe_name == "blockwise":
        kwargs["use_f32_scales"] = recipe_args.fp8_te_block_use_f32_scales
    return kwargs


def build_fp8_recipe(recipe_name: str, recipe_args=None):
    """Instantiate the scaling policy selected by ``--fp8-recipe``.

    Importing TE lazily keeps non-FP8 training usable in environments where the
    TransformerEngine PyTorch extension is not installed. ``recipe_args`` is
    normally the complete ``TrainingArgs`` object; only fields supported by the
    selected recipe are forwarded.
    """
    try:
        class_name = _RECIPE_CLASSES[recipe_name]
    except KeyError:
        raise ValueError(
            f"Unknown --fp8-recipe {recipe_name!r}; expected one of "
            + ", ".join(FP8_RECIPE_CHOICES)
        ) from None

    te_recipe = _import_te_recipe()

    recipe_cls = getattr(te_recipe, class_name, None)
    if recipe_cls is None:
        raise RuntimeError(
            f"--fp8-recipe={recipe_name} needs {class_name}, which this "
            "TransformerEngine build does not provide."
        )
    return recipe_cls(**_recipe_kwargs(recipe_name, recipe_args, te_recipe))


@contextmanager
def fp8_autocast_ctx(recipe_name: str, fp8_group=None, recipe_args=None):
    """Run the enclosed forward under ``te.fp8_autocast``.

    TE uses this context to quantize eligible inputs and weights, choose scales,
    execute FP8 GEMMs, and maintain scaling metadata. Parameters themselves keep
    the dtype used to construct ``te.Linear``.

    ``fp8_group`` is the process group amax/scale state is reduced over. Left as
    None it defaults to the whole world, which is what the embodied stack wants:
    ``DistributedContext`` keeps a single flat group and data parallelism spans
    every rank.
    """
    from transformer_engine.pytorch import fp8_autocast

    recipe = build_fp8_recipe(recipe_name, recipe_args=recipe_args)
    with fp8_autocast(enabled=True, fp8_recipe=recipe, fp8_group=fp8_group):
        yield


def resolve_fp8_autocast_ctx(training_args, fp8_group=None):
    """Return a uniform trainer context: TE FP8 when enabled, otherwise no-op."""
    if not getattr(training_args, "fp8", False):
        return nullcontext()
    return fp8_autocast_ctx(
        training_args.fp8_te_recipe,
        fp8_group=fp8_group,
        recipe_args=training_args,
    )


def te_checkpoint_fn():
    """Return TransformerEngine's activation-checkpoint function.

    Needed because TE's FP8 path saves a different set of tensors during the
    original forward than during recompute, which makes PyTorch's non-reentrant
    checkpoint raise ``CheckpointError``. TE's implementation also preserves
    the FP8 scaling state needed to make recompute consistent with the original
    forward.
    """
    from transformer_engine.pytorch import checkpoint

    return checkpoint
