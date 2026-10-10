# Copyright 2026 The LoongForge Authors.
# SPDX-License-Identifier: Apache-2.0
#
# Modified from FastWAM (https://github.com/yuantianyuan01/FastWAM).
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""State-dict conversion helpers for FastWAM Wan components."""

EXTRA_STATE_SUFFIX = "._extra_state"


def drop_extra_state(state_dict):
    """Return ``state_dict`` without Transformer Engine ``_extra_state`` entries.

    Transformer Engine modules add a ``<prefix>._extra_state`` key holding FP8 recipe
    bookkeeping (an empty ``uint8`` tensor under bf16, so dropping it is lossless).
    Their learnable parameter is still named ``weight``, so this key is the only
    schema difference between the TE and Wan RMSNorm implementations.

    Apply this on both sides of a checkpoint round trip: on save so the file is
    implementation-agnostic, and on load so an existing TE-saved file carries no
    keys the Wan implementation cannot place. A freshly built TE module already
    holds valid bookkeeping, so leaving it untouched is the correct outcome.

    Filtering the key set up front is preferred over a
    ``register_load_state_dict_post_hook`` that edits ``incompatible_keys``: the
    hook would also mask a genuinely absent key at every call site it is installed
    on, whereas here ``strict`` keeps its full meaning for real weights.

    This works because every load site that can see a cross-implementation file
    passes ``strict=False`` (``mot.fastwam.load_checkpoint``, ``wan.core``,
    ``wan.loader``). Under ``strict=True`` a TE module would still report its own
    ``_extra_state`` as missing, so seed such a load from the module's own
    ``state_dict()`` — this is what ``ActionDiT.from_pretrained`` does, together
    with ``ActionDiT.backbone_key_set`` filtering the same suffix out of the keys
    it expects the payload to provide.

    Args:
        state_dict: Mapping of parameter name to tensor.

    Returns:
        A new dict with every ``._extra_state`` key removed.
    """
    return {key: value for key, value in state_dict.items() if not key.endswith(EXTRA_STATE_SUFFIX)}


def wan_video_vae_state_dict_converter(state_dict):
    """Convert Wan VAE checkpoint keys to the local module prefix layout."""
    converted = {}
    if "model_state" in state_dict:
        state_dict = state_dict["model_state"]
    for name, value in state_dict.items():
        converted[f"model.{name}"] = value
    return converted


def wan_video_dit_from_diffusers(state_dict):
    """Convert Diffusers Wan video DiT keys to the local DiT key layout."""
    rename_dict = {
        "blocks.0.attn1.norm_k.weight": "blocks.0.self_attn.norm_k.weight",
        "blocks.0.attn1.norm_q.weight": "blocks.0.self_attn.norm_q.weight",
        "blocks.0.attn1.to_k.bias": "blocks.0.self_attn.k.bias",
        "blocks.0.attn1.to_k.weight": "blocks.0.self_attn.k.weight",
        "blocks.0.attn1.to_out.0.bias": "blocks.0.self_attn.o.bias",
        "blocks.0.attn1.to_out.0.weight": "blocks.0.self_attn.o.weight",
        "blocks.0.attn1.to_q.bias": "blocks.0.self_attn.q.bias",
        "blocks.0.attn1.to_q.weight": "blocks.0.self_attn.q.weight",
        "blocks.0.attn1.to_v.bias": "blocks.0.self_attn.v.bias",
        "blocks.0.attn1.to_v.weight": "blocks.0.self_attn.v.weight",
        "blocks.0.attn2.norm_k.weight": "blocks.0.cross_attn.norm_k.weight",
        "blocks.0.attn2.norm_q.weight": "blocks.0.cross_attn.norm_q.weight",
        "blocks.0.attn2.to_k.bias": "blocks.0.cross_attn.k.bias",
        "blocks.0.attn2.to_k.weight": "blocks.0.cross_attn.k.weight",
        "blocks.0.attn2.to_out.0.bias": "blocks.0.cross_attn.o.bias",
        "blocks.0.attn2.to_out.0.weight": "blocks.0.cross_attn.o.weight",
        "blocks.0.attn2.to_q.bias": "blocks.0.cross_attn.q.bias",
        "blocks.0.attn2.to_q.weight": "blocks.0.cross_attn.q.weight",
        "blocks.0.attn2.to_v.bias": "blocks.0.cross_attn.v.bias",
        "blocks.0.attn2.to_v.weight": "blocks.0.cross_attn.v.weight",
        "blocks.0.attn2.add_k_proj.bias": "blocks.0.cross_attn.k_img.bias",
        "blocks.0.attn2.add_k_proj.weight": "blocks.0.cross_attn.k_img.weight",
        "blocks.0.attn2.add_v_proj.bias": "blocks.0.cross_attn.v_img.bias",
        "blocks.0.attn2.add_v_proj.weight": "blocks.0.cross_attn.v_img.weight",
        "blocks.0.attn2.norm_added_k.weight": "blocks.0.cross_attn.norm_k_img.weight",
        "blocks.0.ffn.net.0.proj.bias": "blocks.0.ffn.0.bias",
        "blocks.0.ffn.net.0.proj.weight": "blocks.0.ffn.0.weight",
        "blocks.0.ffn.net.2.bias": "blocks.0.ffn.2.bias",
        "blocks.0.ffn.net.2.weight": "blocks.0.ffn.2.weight",
        "blocks.0.norm2.bias": "blocks.0.norm3.bias",
        "blocks.0.norm2.weight": "blocks.0.norm3.weight",
        "blocks.0.scale_shift_table": "blocks.0.modulation",
        "condition_embedder.text_embedder.linear_1.bias": "text_embedding.0.bias",
        "condition_embedder.text_embedder.linear_1.weight": "text_embedding.0.weight",
        "condition_embedder.text_embedder.linear_2.bias": "text_embedding.2.bias",
        "condition_embedder.text_embedder.linear_2.weight": "text_embedding.2.weight",
        "condition_embedder.time_embedder.linear_1.bias": "time_embedding.0.bias",
        "condition_embedder.time_embedder.linear_1.weight": "time_embedding.0.weight",
        "condition_embedder.time_embedder.linear_2.bias": "time_embedding.2.bias",
        "condition_embedder.time_embedder.linear_2.weight": "time_embedding.2.weight",
        "condition_embedder.time_proj.bias": "time_projection.1.bias",
        "condition_embedder.time_proj.weight": "time_projection.1.weight",
        "condition_embedder.image_embedder.ff.net.0.proj.bias": "img_emb.proj.1.bias",
        "condition_embedder.image_embedder.ff.net.0.proj.weight": "img_emb.proj.1.weight",
        "condition_embedder.image_embedder.ff.net.2.bias": "img_emb.proj.3.bias",
        "condition_embedder.image_embedder.ff.net.2.weight": "img_emb.proj.3.weight",
        "condition_embedder.image_embedder.norm1.bias": "img_emb.proj.0.bias",
        "condition_embedder.image_embedder.norm1.weight": "img_emb.proj.0.weight",
        "condition_embedder.image_embedder.norm2.bias": "img_emb.proj.4.bias",
        "condition_embedder.image_embedder.norm2.weight": "img_emb.proj.4.weight",
        "patch_embedding.bias": "patch_embedding.bias",
        "patch_embedding.weight": "patch_embedding.weight",
        "scale_shift_table": "head.modulation",
        "proj_out.bias": "head.head.bias",
        "proj_out.weight": "head.head.weight",
    }
    converted = {}
    for name, value in state_dict.items():
        if name in rename_dict:
            converted[rename_dict[name]] = value
        else:
            probe_name = ".".join(name.split(".")[:1] + ["0"] + name.split(".")[2:])
            if probe_name in rename_dict:
                mapped = rename_dict[probe_name]
                mapped = ".".join(mapped.split(".")[:1] + [name.split(".")[1]] + mapped.split(".")[2:])
                converted[mapped] = value
    return converted


def wan_video_dit_state_dict_converter(state_dict):
    """Drop unsupported Wan video DiT keys and remove the optional model prefix."""
    converted = {}
    skip_prefixes = ["pose_patch_embedding", "face_adapter", "face_encoder", "motion_encoder"]
    for name, value in state_dict.items():
        if name.startswith("vace"):
            continue
        if name.split(".")[0] in skip_prefixes:
            continue
        key = name[6:] if name.startswith("model.") else name
        converted[key] = value
    return converted
