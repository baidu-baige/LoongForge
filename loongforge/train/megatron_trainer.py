# Copyright 2026 The LoongForge Authors.
# SPDX-License-Identifier: Apache-2.0

"""Megatron trainer and MCore 0.20 runtime compatibility."""

import os
import sys
from copy import copy
from functools import partial

from loongforge.train.parser import parse_args_from_config
from loongforge.utils import constants
from loongforge.utils.global_vars import (
    get_model_config,
    set_loongforge_extra_global_vars,
)

from megatron.training.argument_utils import pretrain_cfg_container_from_args
from megatron.training.callbacks import CallbackManager
from megatron.training.global_vars import get_args, set_global_variables
from megatron.training.training import pretrain


def _is_hf_checkpoint(load_path):
    """Return whether a path contains a supported HuggingFace checkpoint."""
    if load_path is None:
        return False
    names = (
        "model.safetensors.index.json",
        "model.safetensors",
        "pytorch_model.bin.index.json",
        "pytorch_model.bin",
    )
    return any(os.path.exists(os.path.join(load_path, name)) for name in names)


def _load_hf_weights(context, *, args, model_config):
    repo_root = os.path.dirname(
        os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    )
    for path in (repo_root, os.path.join(repo_root, "tools")):
        # Keep this checkout ahead of other repo roots that may define ``tools``.
        while path in sys.path:
            sys.path.remove(path)
        sys.path.insert(0, path)
    from tools.mcore_bridge.checkpoint.hf_checkpoint_loader import load_hf_checkpoint_online

    layout = getattr(model_config, "pipeline_model_parallel_layout", None)
    if layout is None:
        foundation = getattr(model_config, "foundation", None)
        layout = getattr(foundation, "pipeline_model_parallel_layout", None)
    if layout is not None and getattr(layout, "layout", None) is not None:
        args.convert_pp_layout = layout
    load_hf_checkpoint_online(
        context.model, context.optimizer, context.scheduler, args, model_config=model_config
    )


def _restore_variable_seq_lengths(model_config):
    """Restore LoongForge's variable-length batch contract after validation."""
    runtime_args = get_args()
    runtime_args.variable_seq_lengths = True
    for target in (
        model_config,
        getattr(model_config, "foundation", None),
        getattr(model_config, "image_encoder", None),
    ):
        if target is not None and hasattr(target, "variable_seq_lengths"):
            target.variable_seq_lengths = True


def _sft_uses_variable_seq_lengths(train_args):
    """Mirror LoongForge SFT validation for the MCore argument boundary."""
    if getattr(train_args, "use_fixed_seq_lengths", False):
        return False
    if getattr(train_args, "tp_comm_overlap", False):
        return False
    if getattr(train_args, "num_experts", 0) and getattr(
        train_args, "moe_token_dispatcher_type", None
    ) in ("allgather", "alltoall_seq"):
        return False
    return True


def _normalize_sft_args(train_args):
    """Map LoongForge SFT flags onto MCore's flags."""
    if getattr(train_args, "training_phase", None) != constants.TrainingPhase.SFT:
        return
    train_args.sft = True
    train_args.sft_chunkpipe_mode = bool(
        getattr(train_args, "enable_chunkpipe", False)
    )
    train_args.sft_loss_log_mode = "microbatch"
    train_args.variable_seq_lengths = _sft_uses_variable_seq_lengths(train_args)


def _sync_chunkpipe_model_config(model_config, enabled):
    """Keep MCore and LoongForge model views aligned."""
    for target in (
        model_config,
        getattr(model_config, "foundation", None),
        getattr(model_config, "image_encoder", None),
    ):
        if target is not None and hasattr(target, "sft_chunkpipe_mode"):
            target.sft_chunkpipe_mode = enabled


def _validate_chunkpipe_entrypoint(train_args, model_config, *, allow_chunkpipe):
    """Fail fast unless this is a supported chunkpipe entrypoint."""
    if not getattr(train_args, "enable_chunkpipe", False):
        return
    if not allow_chunkpipe:
        raise NotImplementedError(
            "ChunkPipe is supported only by the LLM SFT trainer."
        )
    if getattr(train_args, "training_phase", None) != constants.TrainingPhase.SFT:
        raise NotImplementedError("ChunkPipe is supported only for SFT.")
    if _is_vlm_config(model_config):
        raise NotImplementedError("ChunkPipe is supported only for LLM SFT.")
    if getattr(train_args, "rerun_mode", None) != "disabled":
        raise NotImplementedError(
            "SFT ChunkPipe requires --rerun-mode disabled."
        )
    if getattr(train_args, "eval_iters", 0):
        raise NotImplementedError("SFT ChunkPipe requires eval_iters=0.")
    if getattr(train_args, "calculate_per_token_loss", False):
        raise NotImplementedError(
            "SFT ChunkPipe supports only the per-sample loss path."
        )
    if getattr(train_args, "tensor_model_parallel_size", 1) != 1:
        raise NotImplementedError(
            "SFT ChunkPipe requires tensor_model_parallel_size=1."
        )
    if getattr(train_args, "num_virtual_stages_per_pipeline_rank", 1) not in (None, 1):
        raise NotImplementedError(
            "SFT ChunkPipe requires virtual pipeline size 1."
        )


def _validate_linear_ce_entrypoint(train_args):
    """Reject unsupported linear-CE + parameter-gather overlap."""
    if not (
        getattr(train_args, "cross_entropy_loss_fusion", False)
        and getattr(train_args, "cross_entropy_fusion_impl", "native") == "linear"
    ):
        return
    if getattr(train_args, "overlap_param_gather", False):
        raise NotImplementedError(
            "cross_entropy_fusion_impl='linear' requires overlap_param_gather=False."
        )


def _is_vlm_config(model_config):
    return getattr(model_config, "image_encoder", None) is not None


def _is_llava_ov_config(model_config):
    return (
        getattr(model_config, "model_type", None) == "llava_ov_1_5"
        or getattr(getattr(model_config, "image_encoder", None), "model_type", None)
        == "rice_vit"
    )


def _is_mimo_or_omni_config(model_config):
    return getattr(model_config, "model_type", None) in ("mimo", "omni")


def _requires_variable_seq_lengths(train_args, model_config):
    """Preserve the legacy variable-length batch contract."""
    training_phase = getattr(train_args, "training_phase", None)
    if training_phase == constants.TrainingPhase.PRETRAIN:
        if _is_llava_ov_config(model_config):
            return True
        if _is_mimo_or_omni_config(model_config):
            return False
        return _is_vlm_config(model_config)
    if training_phase == constants.TrainingPhase.SFT:
        return _sft_uses_variable_seq_lengths(train_args)
    return False


def _requires_legacy_scalar_grad_norm(train_args, model_config):
    """Select the clip root used by the corresponding legacy trainer."""
    training_phase = getattr(train_args, "training_phase", None)
    if training_phase == constants.TrainingPhase.PRETRAIN:
        if _is_mimo_or_omni_config(model_config):
            return False
        return _is_vlm_config(model_config) or _is_llava_ov_config(model_config)
    if training_phase == constants.TrainingPhase.SFT:
        return True
    return False


def _select_sft_clip_root(train_args, packed_seq_params):
    """Packed VLM SFT needs tensor root; non-packed SFT keeps scalar root."""
    if getattr(train_args, "training_phase", None) != constants.TrainingPhase.SFT:
        return
    if not _is_vlm_config(get_model_config()):
        return
    from megatron.core.optimizer.clip_grads import set_legacy_scalar_grad_norm

    set_legacy_scalar_grad_norm(packed_seq_params is None)


def pretrain_with_mcore(
    train_args,
    train_valid_test_dataset_provider,
    model_provider,
    model_type,
    forward_step_func,
    process_non_loss_data_func=None,
    get_embedding_ranks=None,
    get_position_embedding_ranks=None,
    allow_chunkpipe=False,
):
    """Run LoongForge's provider through MCore 0.20 pretrain."""
    _normalize_sft_args(train_args)
    parse_args_from_config(train_args)
    model_config = get_model_config()
    _validate_linear_ce_entrypoint(train_args)
    _validate_chunkpipe_entrypoint(
        train_args, model_config, allow_chunkpipe=allow_chunkpipe
    )

    from megatron.core.optimizer.clip_grads import set_legacy_scalar_grad_norm

    set_legacy_scalar_grad_norm(
        _requires_legacy_scalar_grad_norm(train_args, model_config)
    )
    if _requires_variable_seq_lengths(train_args, model_config):
        train_args.variable_seq_lengths = True
        for target in (model_config, getattr(model_config, "foundation", None)):
            if target is not None and hasattr(target, "variable_seq_lengths"):
                target.variable_seq_lengths = True
    if getattr(train_args, "training_phase", None) == constants.TrainingPhase.SFT:
        _sync_chunkpipe_model_config(
            model_config, getattr(train_args, "enable_chunkpipe", False)
        )
    if train_args.enable_ema or getattr(model_config, "peft_config", None) is not None:
        raise NotImplementedError("This MCore runtime adapter does not yet support EMA or PEFT")

    callbacks = CallbackManager()
    if getattr(model_config, "image_encoder", None) is not None:
        from loongforge.train.initialize import initialize_mimo_vision_parallel_state

        callbacks.register(
            "on_setup_start",
            lambda _context: initialize_mimo_vision_parallel_state(
                model_config.image_encoder
            ),
        )
    if _requires_variable_seq_lengths(train_args, model_config):
        callbacks.register(
            "on_data_init_start",
            lambda _context: _restore_variable_seq_lengths(model_config),
        )

    runtime_args = train_args
    if _is_hf_checkpoint(train_args.load):
        if train_args.use_megatron_fsdp or train_args.moe_use_upcycling:
            raise NotImplementedError("Online HF loading does not support Megatron FSDP or upcycling")
        if train_args.pretrained_checkpoint is not None:
            raise ValueError("Specify one source when loading HF weights through --load")
        runtime_args = copy(train_args)
        runtime_args.load = None
        callbacks.register(
            "on_data_init_start",
            partial(_load_hf_weights, args=train_args, model_config=model_config),
        )

    if getattr(runtime_args, "training_phase", None) == constants.TrainingPhase.SFT:
        runtime_args.sft = True
        runtime_args.sft_chunkpipe_mode = bool(
            getattr(runtime_args, "enable_chunkpipe", False)
        )
        runtime_args.sft_loss_log_mode = "microbatch"
        runtime_args.variable_seq_lengths = _sft_uses_variable_seq_lengths(
            runtime_args
        )

    runtime_args.model_type = model_type
    cfg = pretrain_cfg_container_from_args(runtime_args, None)
    set_global_variables(runtime_args, cfg, build_tokenizer=False)
    set_loongforge_extra_global_vars(runtime_args, build_tokenizer=True)
    if getattr(model_config, "image_encoder", None) is not None:
        from loongforge.models.encoder.internvl_vision_models.internvl_config import (
            resolve_image_token_id,
        )

        resolve_image_token_id(model_config.image_encoder)
    return pretrain(
        cfg,
        train_valid_test_dataset_provider,
        model_type,
        forward_step_func,
        model_provider=model_provider,
        process_non_loss_data_func=process_non_loss_data_func,
        get_embedding_ranks=get_embedding_ranks,
        get_position_embedding_ranks=get_position_embedding_ranks,
        callbacks=callbacks,
    )


class MegatronTrainer(object):
    """Trainer that selects the MCore or legacy runtime explicitly."""

    def __init__(
        self,
        train_args,
        train_valid_test_dataset_provider,
        model_provider,
        model_type,
        forward_step_func,
        process_non_loss_data_func=None,
        get_embedding_ranks=None,
        get_position_embedding_ranks=None,
        pretrain_func=None,
    ):
        self.train_args = train_args
        self.train_valid_test_datasets_provider = train_valid_test_dataset_provider
        self.model_provider = model_provider
        self.model_type = model_type
        self.forward_step_func = forward_step_func
        self.process_non_loss_data_func = process_non_loss_data_func
        self.get_embedding_ranks = get_embedding_ranks
        self.get_position_embedding_ranks = get_position_embedding_ranks
        self.pretrain_func = pretrain_func

    def train(self):
        """Start training."""
        if self.pretrain_func is None:
            raise RuntimeError(
                "MegatronTrainer requires an explicit pretrain_func; "
                "select pretrain_with_mcore or the diffusion legacy runtime."
            )
        self.train_valid_test_datasets_provider.is_distributed = True
        self.pretrain_func(
            train_args=self.train_args,
            train_valid_test_dataset_provider=self.train_valid_test_datasets_provider,
            model_provider=self.model_provider,
            model_type=self.model_type,
            forward_step_func=self.forward_step_func,
            process_non_loss_data_func=self.process_non_loss_data_func,
            get_embedding_ranks=self.get_embedding_ranks,
            get_position_embedding_ranks=self.get_position_embedding_ranks,
        )
