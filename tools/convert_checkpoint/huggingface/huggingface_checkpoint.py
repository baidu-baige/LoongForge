# Copyright 2026 The LoongForge Authors.
# SPDX-License-Identifier: Apache-2.0

"""Load, save, and convert HuggingFace checkpoints within the common conversion pipeline."""

import os
import torch
import json
import re
import logging

logging.basicConfig(level=logging.INFO)

import concurrent.futures
from convert_checkpoint.common.abstact_checkpoint import AbstractCheckpoint
from convert_checkpoint.arguments import parse_args
from convert_checkpoint.common.common_checkpoint import VISION_MAP, VISION_WORD_EMBEDDINGS, CommonCheckpoint

from convert_checkpoint.utils.utils import (
    get_done_keys,
    touch_file
)

from convert_checkpoint.common.common_checkpoint import (
    TRANSFORMER, MTP_TRANSFORMER, MTP_LAYER_PREFIX, LAYER_PREFIX, MOE_EXPERT, MOE_SHARED_EXPERT, LAYER_IS_DICT_FOR_EXPERT,
    FIRST_LAYER_NAMES, BASE_NAMES, MOE_EXPERT_PROJS, LAST_LAYER_NAMES, MTP_NAMES, MTP_WORD_EMBEDDING,
    MOE_EXPERT_H_TO_4H, MOE_EXPERT_4H_TO_H, MOE_SHARED_EXPERT_H_TO_4H, MOE_SHARED_EXPERT_4H_TO_H,
    MTP_MOE_EXPERT_H_TO_4H, MTP_MOE_EXPERT_4H_TO_H, MTP_MOE_SHARED_EXPERT_H_TO_4H, MTP_MOE_SHARED_EXPERT_4H_TO_H
)

from convert_checkpoint.common.common_checkpoint import CommonCheckpoint
from convert_checkpoint.huggingface.huggingface_base import HuggingfaceBase, is_dsv4_hybrid_config
from convert_checkpoint.huggingface.huggingface_moe import HuggingfaceMoe
from convert_checkpoint.huggingface.compressed_tensors_dequant import (
    DTYPE_MAP as HF_DEQUANT_DTYPE_MAP,
    dequantize_state_dict,
    get_packed_weight_keys,
)
from convert_checkpoint.huggingface.compressed_tensors_quant import (
    pack_state_dict_from_official_config,
)
from convert_checkpoint.huggingface.mxfp4_dequant import (
    dequantize_mxfp4_state_dict,
    progress_print,
)


def _hf_dequantize_int4_enabled(args):
    return bool(args is not None and getattr(args, "hf_dequantize_int4", False))


def _hf_dequantize_mxfp4_enabled(args):
    return bool(args is not None and getattr(args, "hf_dequantize_mxfp4", False))


def _packed_to_weight_key(key):
    packed_suffix = ".weight_packed"
    if key.endswith(packed_suffix):
        return f"{key[:-len(packed_suffix)]}.weight"
    return None


def _add_dequant_weight_key(weight_map, dequant_weight_keys, weight_key, args=None):
    if not _hf_dequantize_int4_enabled(args) or dequant_weight_keys is None:
        return
    if weight_key in weight_map:
        return
    packed_key = weight_key[: -len(".weight")] + ".weight_packed" if weight_key.endswith(".weight") else f"{weight_key}.weight_packed"
    if packed_key in weight_map:
        dequant_weight_keys.add(weight_key)


def _add_hf_weight_file(weight_map, filenames, weight_key, args=None, dequant_weight_keys=None):
    if weight_key in weight_map:
        filenames.add(weight_map[weight_key])
    if not _hf_dequantize_int4_enabled(args):
        return
    _add_dequant_weight_key(weight_map, dequant_weight_keys, weight_key, args=args)
    for packed_key in get_packed_weight_keys(weight_key):
        if packed_key in weight_map:
            filenames.add(weight_map[packed_key])


def get_hf_checkpoint_names(c_config, weight_map, layer_ids, expert_ids=None, mtp_num_layers=0, args=None,
                            return_dequant_weight_keys=False):
    name_map = c_config.get("name_map")["huggingface"]
    cargs = c_config.get_args("common")
    hargs = c_config.get_args("huggingface")
    ori_num_layers = cargs["num_layers"]
    num_layers = ori_num_layers + mtp_num_layers
    mtp_transformer = name_map.get(MTP_TRANSFORMER, None)
    mtp_layer_id = hargs.get("mtp_layer_id", None)
    is_dsv4_hybrid = is_dsv4_hybrid_config(c_config)

    filenames_in_the_layer = set()
    dequant_weight_keys = set()

    if 0 in layer_ids or num_layers - 1 in layer_ids:
        for c_name in FIRST_LAYER_NAMES:
            if c_name in name_map:
                if name_map[c_name] is None:
                    continue
                hf_name, is_direct_name, _, _, _, _, _ = HuggingfaceBase.get_hf_name_and_args(name_map[c_name])
                for ext in ["", ".weight"]:
                    name = hf_name + ext
                    _add_hf_weight_file(
                        weight_map, filenames_in_the_layer, name, args=args, dequant_weight_keys=dequant_weight_keys
                    )

    if args is not None and args.enable_full_hetero_dp:
        c_name = VISION_WORD_EMBEDDINGS
        if c_name in name_map and name_map[c_name] is not None:
            hf_name, _, _, _, _, _, _ = HuggingfaceBase.get_hf_name_and_args(name_map[c_name])
            name = hf_name + ".weight"
            _add_hf_weight_file(
                weight_map, filenames_in_the_layer, name, args=args, dequant_weight_keys=dequant_weight_keys
            )

    if 0 in layer_ids:
        for c_name in name_map.keys():
            if c_name.startswith(VISION_MAP):
                hf_name, _, _, _, no_layer_id, _, _ = HuggingfaceBase.get_hf_name_and_args(name_map[c_name])
                for ext in ["", ".weight", ".bias"]:
                    name = hf_name + ext
                    if ext == ".bias":
                        if name in weight_map:
                            filenames_in_the_layer.add(weight_map[name])
                    else:
                        _add_hf_weight_file(
                            weight_map, filenames_in_the_layer, name, args=args,
                            dequant_weight_keys=dequant_weight_keys
                        )



    if (num_layers - 1) in layer_ids:
        for c_name in LAST_LAYER_NAMES:
            if c_name in name_map:
                if name_map[c_name] is None:
                    continue
                hf_name, is_direct_name, _, _, _, _, _ = HuggingfaceBase.get_hf_name_and_args(name_map[c_name])
                for ext in ["", ".weight"]:
                    name = hf_name + ext
                    _add_hf_weight_file(
                        weight_map, filenames_in_the_layer, name, args=args,
                        dequant_weight_keys=dequant_weight_keys
                    )
        if mtp_num_layers > 0:
            for c_name in MTP_NAMES:
                if c_name in name_map:
                    if name_map[c_name] is None:
                        continue
                    hf_name, _, _, _, no_layer_id, _, _ = HuggingfaceBase.get_hf_name_and_args(name_map[c_name])
                    if not no_layer_id:
                        continue
                    for ext in ["", ".weight"]:
                        name = hf_name + ext
                        _add_hf_weight_file(
                            weight_map, filenames_in_the_layer, name, args=args,
                            dequant_weight_keys=dequant_weight_keys
                        )
                        if is_dsv4_hybrid:
                            # Also try without "layers." sub-prefix for V4-style MTP keys
                            # (e.g. "mtp.layers.0.emb.tok_emb" -> "mtp.0.emb.tok_emb")
                            alt_name = name.replace(".layers.", ".", 1) if ".layers." in name else None
                            if alt_name and alt_name in weight_map:
                                filenames_in_the_layer.add(weight_map[alt_name])

    ori_transformer = name_map[TRANSFORMER]
    layer_prefix = name_map[LAYER_PREFIX]
    mtp_layer_prefix = name_map.get(MTP_LAYER_PREFIX, None)
    if expert_ids is not None:
        moe_expert = name_map[MOE_EXPERT]
    for layer_id in layer_ids:
        transformer = ori_transformer
        cur_layer_id = layer_id
        is_mtp_layer = layer_id >= ori_num_layers and mtp_num_layers > 0 and mtp_transformer is not None
        if is_mtp_layer:
            transformer = mtp_transformer
            if mtp_layer_id is not None:
                cur_layer_id = mtp_layer_id
        cur_layer_prefix = mtp_layer_prefix if is_mtp_layer and mtp_layer_prefix is not None else layer_prefix
        name_prefix = ".".join([p for p in [transformer, cur_layer_prefix, str(cur_layer_id)] if p]) + "."
        alt_prefixes = set()
        alt_prefixes.add(name_prefix)
        if is_dsv4_hybrid and is_mtp_layer:
            mtp_relative_id = layer_id - ori_num_layers
            # For DSV4 MTP layers, also try alternative prefixes that match original HF naming
            # (e.g. "mtp.0." in weight_map but "mtp.layers.0." after preprocessing)
            alt_prefixes.add(f"{mtp_transformer}.{mtp_relative_id}.")
            if cur_layer_prefix:
                alt_prefixes.add(f"{mtp_transformer}.{cur_layer_prefix}.{mtp_relative_id}.")
        for key, value in weight_map.items():
            matched = any(key.startswith(pfx) for pfx in alt_prefixes)
            if not matched:
                continue
            if expert_ids is None:
                include_key = True
            else:
                is_expert_key = any(key.startswith(f"{pfx}{moe_expert}.") for pfx in alt_prefixes)
                include_key = not is_expert_key
                if not include_key:
                    for expert_id in expert_ids:
                        if any(key.startswith(f"{pfx}{moe_expert}.{expert_id}.") for pfx in alt_prefixes):
                            include_key = True
                            break
            if not include_key:
                continue

            filenames_in_the_layer.add(value)
            dequant_weight_key = _packed_to_weight_key(key) if _hf_dequantize_int4_enabled(args) else None
            if dequant_weight_key is not None and dequant_weight_key not in weight_map:
                dequant_weight_keys.add(dequant_weight_key)

    checkpoint_names = list(filenames_in_the_layer)
    if return_dequant_weight_keys:
        return checkpoint_names, dequant_weight_keys
    return checkpoint_names

def merge_transformers_sharded_states(path, checkpoint_names, load_safe=False, max_workers=1, hf_checkpoint_device="cpu"):
    """
    Merge sharded checkpoints from transformers into a single checkpoint.

    Args:
        path (str): the path to the sharded checkpoints
        checkpoint_names (list): the names of the checkpoints to merge
    """
    if load_safe:
        from safetensors.torch import load_file
    import time
    state_dict = {}
    current_chunks = [None] * len(checkpoint_names)
    n_total = len(checkpoint_names)
    done = []  # list.append is atomic under the GIL — safe as a thread counter
    def load_files(checkpoint_path, i):
        t0 = time.perf_counter()
        if load_safe:
            current_chunks[i] = load_file(checkpoint_path, device=hf_checkpoint_device)
        else:
            current_chunks[i] = torch.load(checkpoint_path, map_location=hf_checkpoint_device, weights_only=False)
        done.append(1)
        progress_print(
            f"[hf-load] {len(done)}/{n_total} {os.path.basename(checkpoint_path)} "
            f"({time.perf_counter() - t0:.1f}s)"
        )
    if max_workers is None:
        for i in range(len(checkpoint_names)):
            load_files(os.path.join(path, checkpoint_names[i]), i)
    else:
        futures = []
        with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
            for i in range(len(checkpoint_names)):
                futures.append(executor.submit(load_files, os.path.join(path, checkpoint_names[i]), i))
        concurrent.futures.wait(futures)
        for future in futures:
            try:
                result = future.result()
            except Exception as e:
                logging.info(f"An error occurred: {e}")
                raise e
    for i in range(len(checkpoint_names)):
        state_dict.update(current_chunks[i])
    return state_dict


class HuggingFaceCheckpoint(AbstractCheckpoint):
    """
       HuggingFaceCheckpoint
    """

    def __init__(self, c_config, args):
        super().__init__(c_config)
        self.args = args
        self.margs = self.c_config.get_args("mcore")
        self.cargs = self.c_config.get_args("common")
        self.h_base = HuggingfaceBase(c_config, args)
        self.h_moe = HuggingfaceMoe(c_config, args)
        self.state_dict = {}

    @staticmethod
    def check_done_files(save_path, layer_dict, expert_dict=None):
        done_dir = os.path.join(save_path, "dones")
        p = list(layer_dict.keys())[0]
        ep_ids = expert_dict.keys() if expert_dict is not None else None
        if os.path.exists(done_dir):
            done_keys = get_done_keys(done_dir, p, ep_ids)
            if ep_ids is None:
                if (p, None) in done_keys:
                    logging.info(f"> p: {p} already converted. pass...")
                    return True
            else:
                all_done = True
                for ep_id in ep_ids:
                    if not (p, ep_id) in done_keys:
                        all_done = False
                if all_done:
                    logging.info(f"> p: {p}, ep_id: {ep_ids} already converted. pass...")
                    return True
        else:
            os.makedirs(done_dir, exist_ok=True)
        return False


    def convert_from_common(self, c_ckpt, layer_dict, expert_dict=None, save_path=None, save_file=True):
        """
        Convert HuggingFace checkpoint to common checkpoint.
        """

        logging.info("==================== Common -> HuggingFace ====================")

        cargs = self.c_config.get_args("common")
        hargs = self.c_config.get_args("huggingface")
        num_layers = cargs["num_layers"]
        mtp_layer_id = hargs.get("mtp_layer_id", None)
        name_map = self.c_config.get("name_map")["huggingface"]
        mtp_transformer = name_map.get(MTP_TRANSFORMER, None)
        mtp_layer_prefix = name_map.get(MTP_LAYER_PREFIX, None)

        p = list(layer_dict.keys())[0]
        layer_ids = layer_dict[p]
        ep_ids = list(expert_dict.keys()) if expert_dict is not None else None

        if save_file and self.check_done_files(save_path, layer_dict, expert_dict=expert_dict):
            return

        if 0 in layer_ids:
            for c_name in FIRST_LAYER_NAMES:
                self.h_base.common_to_hf(c_name, c_ckpt, self.state_dict)
            for c_name in name_map.keys():
                if c_name.startswith(VISION_MAP):
                    self.h_base.common_to_hf(c_name, c_ckpt, self.state_dict)

        for layer_id in layer_ids:
            hf_layer_id = mtp_layer_id + (layer_id - num_layers) if (layer_id >= num_layers and mtp_layer_id is not None) else layer_id
            transformer = mtp_transformer if layer_id >= num_layers else None
            layer_prefix = mtp_layer_prefix if layer_id >= num_layers else None
            for c_name in BASE_NAMES:
                self.h_base.common_to_hf(c_name, c_ckpt, self.state_dict, layer_id=layer_id,
                                         hf_layer_id=hf_layer_id, transformer=transformer, layer_prefix=layer_prefix)
            # ====moe shared_expert
            for c_name in MOE_EXPERT_PROJS:
                if c_name == MOE_EXPERT_H_TO_4H:
                    spec_name = MTP_MOE_SHARED_EXPERT_H_TO_4H if layer_id >= num_layers else MOE_SHARED_EXPERT_H_TO_4H
                else:
                    spec_name = MTP_MOE_SHARED_EXPERT_4H_TO_H if layer_id >= num_layers else MOE_SHARED_EXPERT_4H_TO_H
                self.h_base.common_to_hf(c_name, c_ckpt, self.state_dict, layer_id=layer_id,
                                         hf_layer_id=hf_layer_id, expert_name=MOE_SHARED_EXPERT,
                                         transformer=transformer, layer_prefix=layer_prefix, spec_name=spec_name)

            # EXPERT
            if expert_dict is not None:
                for ep_id, expert_ids in expert_dict.items():
                    for expert_id in expert_ids:
                        for c_name in MOE_EXPERT_PROJS:
                            if c_name == MOE_EXPERT_H_TO_4H:
                                spec_name = MTP_MOE_EXPERT_H_TO_4H if layer_id >= num_layers else None
                            else:
                                spec_name = MTP_MOE_EXPERT_4H_TO_H if layer_id >= num_layers else None
                            self.h_moe.common_e_to_hf(MOE_EXPERT, c_name, c_ckpt, self.state_dict, layer_id=layer_id,
                                                      hf_layer_id=hf_layer_id, expert_id=expert_id,
                                                      transformer=transformer, layer_prefix=layer_prefix, spec_name=spec_name)
            self.merge_dict_tensor(self.state_dict)
            # MTP
            if layer_id >= num_layers:
                for c_name in MTP_NAMES:
                    self.h_base.common_to_hf(c_name, c_ckpt, self.state_dict, layer_id=layer_id,
                                             hf_layer_id=hf_layer_id, transformer=transformer, layer_prefix=layer_prefix)

        if num_layers - 1 in layer_ids:
            for c_name in LAST_LAYER_NAMES:
                self.h_base.common_to_hf(c_name, c_ckpt, self.state_dict, layer_prefix=layer_prefix)

        if save_file:
            self.save_ckpt_file(save_path, p, ep_ids, self.state_dict)
        else:
            return self.state_dict

    def save_ckpt_file(self, save_path, p, ep_ids, state_dict):
        done_dir = os.path.join(save_path, "dones")
        if ep_ids is None or len(ep_ids) == 0:
            file_tag = p
            if self.args.sub_file_tag is not None:
                file_tag = self.args.sub_file_tag * 1000 + p
            self.save(f"{save_path}/sub_checkpoint/{file_tag}", state_dict, None)
            touch_file(done_dir=done_dir, p=p, sub_file_tag=self.args.sub_file_tag)
            logging.info(f"touch file: {done_dir=}, {p=}")
        else:
            file_tag = p * 1000 + ep_ids[0]
            if self.args.sub_file_tag is not None:
                file_tag = self.args.sub_file_tag * 1000000 + file_tag
            self.save(f"{save_path}/sub_checkpoint/{file_tag}", state_dict, None)
            for ep_id in ep_ids:
                touch_file(done_dir=done_dir, p=p, ep_id=ep_id, sub_file_tag=self.args.sub_file_tag)
                logging.info(f"touch file: {done_dir=}, {p=}, {ep_id=}")


    def convert_to_common(self, layer_dict, expert_dict=None):
        """
        Convert HuggingFace checkpoint to common checkpoint.
        """

        logging.info("==================== HuggingFace -> Common ====================")

        cargs = self.c_config.get_args("common")
        hargs = self.c_config.get_args("huggingface")
        c_ckpt = CommonCheckpoint(self.c_config)
        num_layers = cargs["num_layers"]
        mtp_layer_id = hargs.get("mtp_layer_id", None)
        name_map = self.c_config.get("name_map")["huggingface"]
        mtp_transformer = name_map.get(MTP_TRANSFORMER, None)
        mtp_layer_prefix = name_map.get(MTP_LAYER_PREFIX, None)

        p = list(layer_dict.keys())[0]
        layer_ids = layer_dict[p]

        if 0 in layer_ids:
            for c_name in FIRST_LAYER_NAMES:
                self.h_base.hf_to_common(c_name, c_ckpt, self.state_dict)
            for c_name in name_map.keys():
                if c_name.startswith(VISION_MAP):
                    self.h_base.hf_to_common(c_name, c_ckpt, self.state_dict)
        elif self.args.enable_full_hetero_dp:
            self.h_base.hf_to_common(VISION_WORD_EMBEDDINGS, c_ckpt, self.state_dict)

        for layer_id in layer_ids:
            hf_layer_id = mtp_layer_id if (layer_id >= num_layers and mtp_layer_id is not None) else layer_id
            transformer = mtp_transformer if layer_id >= num_layers else None
            layer_prefix = mtp_layer_prefix if layer_id >= num_layers else None
            for c_name in BASE_NAMES:
                self.h_base.hf_to_common(c_name, c_ckpt, self.state_dict, layer_id=layer_id,
                                         hf_layer_id=hf_layer_id, transformer=transformer, layer_prefix=layer_prefix)
            # ====moe shared_expert
            for c_name in MOE_EXPERT_PROJS:
                if c_name == MOE_EXPERT_H_TO_4H:
                    spec_name = MTP_MOE_SHARED_EXPERT_H_TO_4H if layer_id >= num_layers else MOE_SHARED_EXPERT_H_TO_4H
                else:
                    spec_name = MTP_MOE_SHARED_EXPERT_4H_TO_H if layer_id >= num_layers else MOE_SHARED_EXPERT_4H_TO_H
                self.h_base.hf_to_common(c_name, c_ckpt, self.state_dict, layer_id=layer_id, hf_layer_id=hf_layer_id,
                                         transformer=transformer, expert_name=MOE_SHARED_EXPERT, layer_prefix=layer_prefix, spec_name=spec_name)

            # EXPERT
            if expert_dict is not None:
                for ep_id, expert_ids in expert_dict.items():
                    for expert_id in expert_ids:
                        for c_name in MOE_EXPERT_PROJS:
                            if c_name == MOE_EXPERT_H_TO_4H:
                                spec_name = MTP_MOE_EXPERT_H_TO_4H if layer_id >= num_layers else None
                            else:
                                spec_name = MTP_MOE_EXPERT_4H_TO_H if layer_id >= num_layers else None
                            self.h_moe.hf_e_to_common(MOE_EXPERT, c_name, c_ckpt, self.state_dict,
                                                      layer_id=layer_id, hf_layer_id=hf_layer_id,
                                                      transformer=transformer, expert_id=expert_id, layer_prefix=layer_prefix, spec_name=spec_name)

            # MTP
            if layer_id >= num_layers:
                for c_name in MTP_NAMES:
                    self.h_base.hf_to_common(c_name, c_ckpt, self.state_dict, layer_id=layer_id,
                                             hf_layer_id=hf_layer_id, transformer=transformer, layer_prefix=layer_prefix)
                

        if num_layers - 1 in layer_ids:
            for c_name in LAST_LAYER_NAMES:
                self.h_base.hf_to_common(c_name, c_ckpt, self.state_dict)

        return c_ckpt

    def dequantize_compressed_tensors_if_needed(self, load_path, target_weight_keys=None):
        if not _hf_dequantize_int4_enabled(self.args):
            return

        dtype_name = getattr(self.args, "hf_dequantize_dtype", "bfloat16")
        try:
            output_dtype = HF_DEQUANT_DTYPE_MAP[dtype_name.lower()]
        except KeyError as exc:
            raise ValueError(f"Unsupported --hf-dequantize-dtype: {dtype_name}") from exc

        converted = dequantize_state_dict(
            self.state_dict,
            load_path,
            output_dtype=output_dtype,
            config_file=getattr(self.args, "hf_quant_config_file", None),
            target_weight_keys=target_weight_keys,
        )
        logging.info(
            "On-the-fly dequantized %d compressed-tensors packed INT4 weight(s) to %s.",
            converted,
            dtype_name,
        )

    def _drop_unowned_expert_tensors(self, c_config, expert_ids):
        """Drop routed-expert tensors this rank does not own.

        ``expert_ids`` narrows which shard FILES are read, but shards are read
        whole and mix all experts, so after the merge the state_dict still
        holds every expert of the loaded layers. Under expert parallelism a
        rank only consumes its own ``expert_ids`` (the rest are re-read by
        their owning ranks), so drop the others right away: host memory and
        MXFP4 dequant then scale with 1/EP instead of the full expert count
        (DSV4-Flash: 8 ranks x full PP-stage experts dequantized to BF16
        OOM-killed the node at ~1.7TB).
        """
        if expert_ids is None or c_config is None or not self.state_dict:
            return
        name_map = c_config.get("name_map")["huggingface"]
        moe_expert = name_map.get(MOE_EXPERT)
        if not moe_expert:
            return
        keep = {str(e) for e in expert_ids}
        pattern = re.compile(rf"(?:^|\.){re.escape(moe_expert)}\.([0-9]+)\.")
        dropped = 0
        for key in list(self.state_dict.keys()):
            m = pattern.search(key)
            if m and m.group(1) not in keep:
                del self.state_dict[key]
                dropped += 1
        if dropped:
            progress_print(
                f"[hf-load] EP filter: dropped {dropped} expert tensor(s) not owned "
                f"by this rank, kept experts {min(map(int, keep))}..{max(map(int, keep))}"
            )

    def dequantize_mxfp4_if_needed(self, target_weight_keys=None):
        """On-the-fly MXFP4 (E2M1 + E8M0) packed FP4 -> floating-point materialization."""
        if not _hf_dequantize_mxfp4_enabled(self.args):
            return

        dtype_name = getattr(self.args, "hf_dequantize_dtype", "bfloat16")
        try:
            output_dtype = HF_DEQUANT_DTYPE_MAP[dtype_name.lower()]
        except KeyError as exc:
            raise ValueError(f"Unsupported --hf-dequantize-dtype: {dtype_name}") from exc

        converted = dequantize_mxfp4_state_dict(
            self.state_dict,
            output_dtype=output_dtype,
            target_weight_keys=target_weight_keys,
        )
        progress_print(
            f"[mxfp4-dequant] done: materialized {converted} packed FP4 weight(s) to {dtype_name}."
        )

    def load(self, load_path, load_safe=False, c_config=None, layer_ids=[], expert_ids=None, mtp_num_layers=0):
        """ load ckpt """
        dequant_weight_keys = None
        if load_safe:
            from safetensors.torch import load_file
            sub_dirs = [x for x in os.listdir(load_path) if x.endswith("safetensors")]
            if not os.path.exists(os.path.join(load_path, "model.safetensors.index.json")):
                checkpoint_name = "model.safetensors"
                self.state_dict = load_file(os.path.join(load_path, checkpoint_name), device=self.args.hf_checkpoint_device)
                logging.info(f"Load HuggingFace from: {os.path.join(load_path, checkpoint_name)}")
            else:
                meta_path = f"{load_path}/model.safetensors.index.json"
                with open(meta_path, 'r') as f:
                    file_content = json.load(f)
                weight_map = file_content["weight_map"]
                checkpoint_names, dequant_weight_keys = get_hf_checkpoint_names(
                    c_config, weight_map, layer_ids, expert_ids=expert_ids, mtp_num_layers=mtp_num_layers,
                    args=self.args, return_dequant_weight_keys=True)
                logging.info(
                    "Selected %d HuggingFace shard(s), %d packed INT4 target weight(s).",
                    len(checkpoint_names),
                    0 if dequant_weight_keys is None else len(dequant_weight_keys),
                )
                self.state_dict = merge_transformers_sharded_states(
                    load_path, checkpoint_names, load_safe=True, max_workers=self.args.max_workers, hf_checkpoint_device=self.args.hf_checkpoint_device)
                logging.info(f"merge_transformers_sharded_states: {load_path}")
        else:
            sub_dirs = [x for x in os.listdir(load_path) if x.startswith("pytorch_model")]
            if len(sub_dirs) == 1:
                checkpoint_name = "pytorch_model.bin"
                self.state_dict = torch.load(os.path.join(load_path, checkpoint_name),
                                             map_location=self.args.hf_checkpoint_device, weights_only=False)
                logging.info(f"Load HuggingFace from: {os.path.join(load_path, checkpoint_name)}")
            else:
                meta_path = f"{load_path}/pytorch_model.bin.index.json"
                with open(meta_path, 'r') as f:
                    file_content = json.load(f)
                weight_map = file_content["weight_map"]
                checkpoint_names, dequant_weight_keys = get_hf_checkpoint_names(
                    c_config, weight_map, layer_ids, expert_ids=expert_ids, mtp_num_layers=mtp_num_layers,
                    args=self.args, return_dequant_weight_keys=True)
                logging.info(
                    "Selected %d HuggingFace shard(s), %d packed INT4 target weight(s).",
                    len(checkpoint_names),
                    0 if dequant_weight_keys is None else len(dequant_weight_keys),
                )
                self.state_dict = merge_transformers_sharded_states(
                    load_path, checkpoint_names, max_workers=self.args.max_workers, hf_checkpoint_device=self.args.hf_checkpoint_device)
                logging.info(f"merge_transformers_sharded_states: {load_path}")


        # EP filter must run before the dequant hooks so per-rank memory and
        # dequant work scale with 1/EP.
        self._drop_unowned_expert_tensors(c_config, expert_ids)

        self.dequantize_compressed_tensors_if_needed(load_path, target_weight_keys=dequant_weight_keys)
        # MXFP4 dequant must run before the DSV4 `.scale` -> `.weight_scale_inv`
        # rename below: it matches routed-expert scale companions by the raw
        # `.scale` suffix and pops them once consumed. If the rename ran first,
        # the pairing key would already be gone and the packed MXFP4 weights
        # would reach mcore conversion still as int8 with a stale scale.
        self.dequantize_mxfp4_if_needed()

        # DeepSeek-V4 key preprocessing: add model. prefix, rename FP8 scales,
        # restructure MTP keys, split hyper-connection scale into alpha_pre/alpha_post/alpha_res
        if (is_dsv4_hybrid_config(c_config) and self.state_dict
                and any(k.startswith('layers.') or k.startswith('mtp.') or k == 'embed.weight' for k in self.state_dict.keys())):
            new_sd = {}
            for key, value in self.state_dict.items():
                new_key = key
                # Rename FP8 scale keys: *.scale -> *.weight_scale_inv
                if new_key.endswith('.scale'):
                    new_key = new_key[:-len('.scale')] + '.weight_scale_inv'
                # Restructure MTP keys: mtp.{j}.* -> mtp.layers.{j}.*
                if new_key.startswith('mtp.'):
                    m = re.match(r'mtp\.([0-9]+)\.(.*)', new_key)
                    if m:
                        j = m.group(1)
                        rest = m.group(2)
                        new_key = f"mtp.layers.{j}.{rest}"
                # Split per-layer hc_*_scale (shape [3]) into alpha_pre/alpha_post/alpha_res
                if new_key.endswith('.hc_attn_scale'):
                    prefix = new_key.rsplit('.', 1)[0]
                    new_sd[f"{prefix}.hc_attn_alpha_pre"] = value[0:1]
                    new_sd[f"{prefix}.hc_attn_alpha_post"] = value[1:2]
                    new_sd[f"{prefix}.hc_attn_alpha_res"] = value[2:3]
                    continue
                elif new_key.endswith('.hc_ffn_scale'):
                    prefix = new_key.rsplit('.', 1)[0]
                    new_sd[f"{prefix}.hc_ffn_alpha_pre"] = value[0:1]
                    new_sd[f"{prefix}.hc_ffn_alpha_post"] = value[1:2]
                    new_sd[f"{prefix}.hc_ffn_alpha_res"] = value[2:3]
                    continue
                # hc_head_scale (shape [1]) kept as-is for mcore HyperHead
                # mtp hc_head_scale kept as-is for mcore MTP HyperHead
                new_sd[new_key] = value
            # MTP e_proj and h_proj kept separate for mcore (upstream uses them independently)
            self.state_dict = new_sd

    def print_memory_usage(self, desc):
        import psutil
        process = psutil.Process(os.getpid())
        mem = process.memory_info().rss / 1024**2  # Convert to MB
        logging.info(f"{desc} memory usage: {mem:.2f} MB")
   
    def save(self, save_path, state_dict, h_config=None, save_optim=False):
        """ save ckpt """
        from huggingface_hub import split_torch_state_dict_into_shards
        from transformers.modeling_utils import SAFE_WEIGHTS_INDEX_NAME
        from safetensors.torch import save_file
        os.makedirs(save_path, exist_ok=True)

        if getattr(self.args, "hf_pack_quantized_from_config", False):
            config_file = getattr(self.args, "hf_official_config_file", None) \
                or getattr(self.args, "hf_quant_config_file", None)
            if config_file is None:
                raise ValueError(
                    "--hf-pack-quantized-from-config requires --hf-official-config-file "
                    "or --hf-quant-config-file"
                )
            pack_state_dict_from_official_config(
                state_dict,
                config_file=config_file,
                target_regex=getattr(self.args, "hf_pack_quantized_target_regex", None),
            )

        state_dict_split = split_torch_state_dict_into_shards(state_dict)
        self.print_memory_usage(f"before save {save_path}")
        has_safetensor_file = False

        def save_hf_shard(tensors, shard_file):
            shard = {}
            for tensor in tensors:
                shard[tensor] = state_dict[tensor].contiguous()
                del state_dict[tensor]
            shard_path = os.path.join(save_path, shard_file)
            save_file(shard, shard_path, metadata={"format": "pt"})
            logging.info(f"Saving HuggingFace shard to: {shard_path}")

        if self.args.max_workers > 1:
            futures = []
            with concurrent.futures.ThreadPoolExecutor(max_workers=self.args.max_workers) as executor:
                for shard_file, tensors in state_dict_split.filename_to_tensors.items():
                    has_safetensor_file = True
                    futures.append(executor.submit(save_hf_shard, tensors=tensors, shard_file=shard_file))
            concurrent.futures.wait(futures)
            for future in futures:
                try:
                    result = future.result()
                except Exception as e:
                    logging.info(f"An error occurred: {e}")
                    raise e
        else:
            for shard_file, tensors in state_dict_split.filename_to_tensors.items():
                has_safetensor_file = True
                save_hf_shard(tensors, shard_file)
        self.print_memory_usage(f"after save {save_path}")

        if state_dict_split.is_sharded:
            index = {
                "metadata": state_dict_split.metadata,
                "weight_map": state_dict_split.tensor_to_filename,
            }
            save_index_file = SAFE_WEIGHTS_INDEX_NAME
            save_index_file = os.path.join(save_path, save_index_file)
            with open(save_index_file, "w", encoding="utf-8") as f:
                content = json.dumps(index, indent=2, sort_keys=True) + "\n"
                f.write(content)
        elif has_safetensor_file:
            for key in state_dict_split.tensor_to_filename.keys():
                if state_dict_split.tensor_to_filename[key] == "model.safetensors":
                    state_dict_split.tensor_to_filename[key] = "model-00001-of-00001.safetensors"
            index = {
                "metadata": state_dict_split.metadata,
                "weight_map": state_dict_split.tensor_to_filename,
            }
            save_index_file = SAFE_WEIGHTS_INDEX_NAME
            save_index_file = os.path.join(save_path, save_index_file)
            with open(save_index_file, "w", encoding="utf-8") as f:
                content = json.dumps(index, indent=2, sort_keys=True) + "\n"
                f.write(content)
            os.rename(os.path.join(save_path, 'model.safetensors'), \
                      os.path.join(save_path, 'model-00001-of-00001.safetensors'))

        if h_config is not None:
            h_config.save(save_path)

    def merge_dict_tensor(self, state_dict):
        for key, value in state_dict.items():
            if isinstance(value, dict) and LAYER_IS_DICT_FOR_EXPERT in value and value[LAYER_IS_DICT_FOR_EXPERT]:
                value.pop(LAYER_IS_DICT_FOR_EXPERT)
                sorted_items = sorted(value.items())
                tensors = [tensor for _, tensor in sorted_items]
                state_dict[key] = torch.stack(tensors, dim=0)

    @staticmethod
    def save_vlm_checkpoint(hf_ckpt, hf_vision_ckpt, c_vision_patch_config, c_ckpt, c_vision_ckpt, save_path, layer_dict, expert_dict=None):
        if hf_ckpt.check_done_files(save_path, layer_dict, expert_dict=expert_dict):
            return
        vision_num_layers = c_vision_patch_config.get_args("common")["num_layers"]
        vision_layer_dict = {}
        vision_layer_dict[0] = list(range(vision_num_layers)) 
        state_dict = hf_ckpt.convert_from_common(c_ckpt, layer_dict, expert_dict=expert_dict, save_path=save_path, save_file=False)
        vision_ckpt = hf_vision_ckpt.convert_from_common(c_vision_ckpt, vision_layer_dict, save_file=False)
        state_dict.update(vision_ckpt)
        # save checkpoint file
        ep_ids = list(expert_dict.keys()) if expert_dict is not None else None
        hf_ckpt.save_ckpt_file(save_path, 0, ep_ids, state_dict)
