# Copyright 2026 The LoongForge Authors.
# SPDX-License-Identifier: Apache-2.0

"""ERNIE 5.5 model for the native LoongForge training path."""

import torch
from megatron.core.process_groups_config import ProcessGroupCollection

from loongforge.models.foundation.base import BaseGPTModel
from .ernie_config import Ernie55Config
from .ernie_ops import Ernie55Multimax, Ernie55Norm


def _multimax_output(module, inputs, output):
    logits, bias = output
    return Ernie55Multimax.apply(logits, module.multimax_ranges, module.multimax_ts), bias


def _drop_mtp_aliases(module, state, prefix, metadata):
    for alias in module.mtp_aliases:
        state.pop(prefix + alias, None)


def _prepare_state_dict(module, state, prefix, metadata, strict, missing, unexpected, errors):
    if module.mtp_aliases and metadata.get("assign_to_params_buffers", False):
        raise ValueError("Shared MTP parameters require load_state_dict(assign=False).")
    # BF16 HF checkpoints have no Transformer Engine quantization state.
    for name, value in module.state_dict().items():
        if name.endswith("._extra_state") and (
            value is None or isinstance(value, torch.Tensor) and value.numel() == 0
        ):
            state.setdefault(prefix + name, value)
    for alias, source in module.mtp_aliases.items():
        if prefix + source in state:
            if prefix + alias in state and not torch.equal(state[prefix + alias], state[prefix + source]):
                errors.append(f"Shared MTP values differ: {alias} and {source}.")
            state[prefix + alias] = state[prefix + source]


class Ernie55Model(BaseGPTModel):
    config_class = Ernie55Config

    def __init__(
        self,
        config,
        pre_process=True,
        post_process=True,
        parallel_output=True,
        vp_stage=None,
        pg_collection: ProcessGroupCollection = None,
        **kwargs,
    ):
        from .ernie_layer_spec import get_ernie55_layer_spec

        layer_spec, mtp_spec = get_ernie55_layer_spec(config, vp_stage=vp_stage)
        if mtp_spec is not None:
            for spec in mtp_spec.layer_specs:
                spec.submodules.enorm = Ernie55Norm
                spec.submodules.hnorm = Ernie55Norm
                spec.submodules.layer_norm = Ernie55Norm
        super().__init__(
            config=config,
            transformer_layer_spec=layer_spec,
            mtp_block_spec=mtp_spec,
            vocab_size=config.padded_vocab_size,
            max_sequence_length=config.max_position_embeddings,
            pre_process=pre_process,
            post_process=post_process,
            parallel_output=parallel_output,
            share_embeddings_and_output_weights=False,
            position_embedding_type="none",
            vp_stage=vp_stage,
            pg_collection=pg_collection,
            **kwargs,
        )
        # Parameter order sets the NCCL gradient-buffer layout.
        if hasattr(self.decoder, "head_hyper_connection"):
            self.decoder._modules = {
                "head_hyper_connection": self.decoder.head_hyper_connection,
                **self.decoder._modules,
            }
            for parameter in self.decoder.head_hyper_connection.parameters():
                parameter.keep_in_fp32 = True
        for layer in self.decoder.layers:
            modules = {}
            for name, module in layer._modules.items():
                if name in ("self_attention_hyper_connection", "mlp_hyper_connection"):
                    continue
                modules[name] = module
                if name in ("self_attention", "mlp"):
                    modules[name + "_hyper_connection"] = getattr(layer, name + "_hyper_connection")
            layer._modules = modules
        if self.post_process:
            for name in ("multimax_ranges", "multimax_ts"):
                self.output_layer.register_parameter(
                    name,
                    torch.nn.Parameter(torch.zeros(4, dtype=config.params_dtype)),
                )
            self.output_layer.register_forward_hook(_multimax_output)
        self.mtp_aliases = {}
        if self.mtp_process:
            source_prefix = f"decoder.layers.{len(self.decoder.layers) - 1}."
            source_params = dict(self.decoder.layers[-1].named_parameters())
            for index, layer in enumerate(self.mtp.layers):
                for name in ("hc_head_fn", "hc_head_base", "hc_head_scale"):
                    getattr(layer, name).keep_in_fp32 = True
                body = layer.transformer_layer
                target_params = dict(body.named_parameters())
                if source_params.keys() != target_params.keys():
                    raise ValueError("The MTP body must have the last decoder layer's parameters.")
                for name, parameter in source_params.items():
                    if parameter.shape != target_params[name].shape:
                        raise ValueError(f"MTP parameter shape differs: {name}.")
                    owner, _, leaf = name.rpartition(".")
                    body.get_submodule(owner)._parameters[leaf] = parameter
                    self.mtp_aliases[f"mtp.layers.{index}.transformer_layer.{name}"] = source_prefix + name
            self.register_state_dict_post_hook(_drop_mtp_aliases)
        self.register_load_state_dict_pre_hook(_prepare_state_dict)

    def _apply(self, fn, recurse=True):
        def keep_fp32(tensor):
            converted = fn(tensor)
            if getattr(tensor, "keep_in_fp32", False):
                return tensor.to(device=converted.device, dtype=torch.float32)
            return converted

        result = super()._apply(keep_fp32, recurse=recurse)
        for alias, source in self.mtp_aliases.items():
            owner, _, leaf = alias.rpartition(".")
            self.get_submodule(owner)._parameters[leaf] = self.get_parameter(source)
        return result

    def sharded_state_dict(self, prefix="", sharded_offsets=(), metadata=None):
        state = super().sharded_state_dict(prefix, sharded_offsets, metadata)
        for alias in self.mtp_aliases:
            state.pop(prefix + alias, None)
        return state

    def forward(self, input_ids, position_ids, attention_mask, decoder_input=None, **kwargs):
        if decoder_input is None and self.pre_process:
            decoder_input = self.embedding(input_ids=input_ids, position_ids=position_ids).float()
        return super().forward(
            input_ids,
            position_ids,
            attention_mask,
            decoder_input=decoder_input,
            **kwargs,
        )
