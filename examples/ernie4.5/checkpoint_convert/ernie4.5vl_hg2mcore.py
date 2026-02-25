# pylint: skip-file
import os
import re
import argparse
import torch
from pathlib import Path
from dataclasses import dataclass
from typing import Callable
from einops import rearrange
from safetensors.torch import load_file


def forward_func(input_ops, hf_state_dict, expected_shape=None):
    input_names = [op.src_name for op in input_ops]

    if len(input_names) == 1:
        name = input_names[0]
        if "@" in name:
            print(f"1hg to 2mcore: {input_names}")
            # 1 hg tensor to split into multi mcore tensor
            part = name.split("@")[1]
            name = name.split("@")[0]
            result = hf_state_dict[name].split([1, 1], dim=0)[int(part)]
        else:
            result = hf_state_dict[name]
        if expected_shape is not None and result.shape != expected_shape:
            if len(result.shape) == len(expected_shape):
                assert len(result.shape) == 2 and len(expected_shape) == 2
                result = result.transpose(1, 0)
                print(f"forward_func: transpose {input_names}")
            else:
                result = result.reshape(expected_shape)
                print(f"forward_func: reshape {input_names}")

        return result

    # attention q,k,v -> combined: [q1, k1, v1, q2, k2, v2, ..., qn, k n, vn]
    elif len(input_names) == 3:
        q_name, k_name, v_name = input_names
        q = hf_state_dict[q_name]
        k = hf_state_dict[k_name]
        v = hf_state_dict[v_name]

        # A:(4,5,128,2560), B:(4,1,128,2560), C:(4,1,128,2560)
        # combined:(4,5,128,7680)
        q = q.reshape(4, 5, 128, 2560)
        k = k.reshape(4, 1, 128, 2560)
        v = v.reshape(4, 1, 128, 2560)
        mcore_qkv = torch.cat([q, k, v], dim=1).reshape(3584, 2560)

        print("mcore_qkv.shape", mcore_qkv.shape)
        return mcore_qkv
    else:
        print(f"forward_func: unsupported input names {input_names}")
        raise ValueError(f"forward_func: unsupported input names {input_names}")


@dataclass
class TransformOp:
    src_name: str
    dst_name: str
    name_type: str = "hg2mcore"
    forward_fn: Callable[[torch.Tensor], torch.Tensor] = forward_func  # 正向转换函数
    backward_fn: Callable[[torch.Tensor], torch.Tensor] = forward_func  # 反向转换函数

    def swap(self) -> None:
        """swap src_name and dst_name"""
        self.src_name, self.dst_name = self.dst_name, self.src_name
        self.name_type = "mcore2hg" if self.name_type == "hg2mcore" else "hg2mcore"


class WeightsModel:
    def __init__(self):
        self.common_mapping: list[TransformOp] = []
        self.loop_mapping: list[TransformOp] = []

    def add_common_weight(
        self, src, dst, forward_fn=forward_func, backward_fn=forward_func
    ):
        self.common_mapping.append(TransformOp(src, dst, forward_fn, backward_fn))

    def add_loop_weight(
        self, src, dst, forward_fn=forward_func, backward_fn=forward_func
    ):
        self.loop_mapping.append(TransformOp(src, dst, forward_fn, backward_fn))


def ernie_model_weights(args):

    # vision
    vision = WeightsModel()
    vision.add_common_weight(
        "vision_model.patch_embed.proj.weight",
        "encoder_model.image_encoder.patch_embed.proj.weight",
    )
    vision.add_common_weight(
        "vision_model.ln.bias", "encoder_model.image_encoder.ln.bias"
    )
    vision.add_common_weight(
        "vision_model.ln.weight", "encoder_model.image_encoder.ln.weight"
    )
    for i in range(args.num_vit_layers):
        # vision.add_loop_weight(f"vision_model.blocks.{i}.attn.proj.weight",f"encoder_model.image_encoder.blocks.{i}.attn.proj.weight")
        suffix = [
            ".attn.proj.bias",
            ".attn.proj.weight",
            ".attn.qkv.bias",
            ".attn.qkv.weight",
            ".mlp.fc1.bias",
            ".mlp.fc1.weight",
            ".mlp.fc2.bias",
            ".mlp.fc2.weight",
            ".norm1.bias",
            ".norm1.weight",
            ".norm2.bias",
            ".norm2.weight",
        ]
        for s in suffix:
            vision.add_loop_weight(
                f"vision_model.blocks.{i}{s}",
                f"encoder_model.image_encoder.blocks.{i}{s}",
            )

    suffix = [
        ".mlp.bias",
        ".mlp.weight",
        ".spatial_linear.0.bias",
        ".spatial_linear.0.weight",
        ".spatial_linear.2.bias",
        ".spatial_linear.2.weight",
        ".spatial_linear.3.bias",
        ".spatial_linear.3.weight",
        ".temporal_linear.0.bias",
        ".temporal_linear.0.weight",
        ".temporal_linear.2.bias",
        ".temporal_linear.2.weight",
        ".temporal_linear.3.bias",
        ".temporal_linear.3.weight",
    ]
    for s in suffix:
        vision.add_common_weight(
            f"model.resampler_model{s}", f"encoder_model.image_encoder.resampler{s}"
        )

    # adapter
    adapter = WeightsModel()
    adapter.add_common_weight(
        "model.resampler_model.mlp.weight", "encoder_model.image_projector.mlp.weight"
    )
    adapter.add_common_weight(
        "model.resampler_model.mlp.bias", "encoder_model.image_projector.mlp.bias"
    )
    adapter.add_common_weight(
        "model.resampler_model.after_norm.weight",
        "encoder_model.image_projector.after_norm.weight",
    )

    # language
    language = WeightsModel()
    language.add_common_weight(
        "model.embed_tokens.weight", "encoder_model.text_encoder.word_embeddings.weight"
    )
    language.add_common_weight(
        "model.norm.weight", "foundation_model.decoder.final_layernorm.weight"
    )
    language.add_common_weight(
        "model.embed_tokens.weight", "foundation_model.output_layer.weight"
    )
    language.add_common_weight(
        "model.embed_tokens.weight", "foundation_model.embedding.word_embeddings.weight"
    )

    # foundation_model.decoder.final_layernorm.weight

    for i in range(args.num_lm_layers):
        # self_attention: (input layernorm + attention + layernorm)
        language.add_loop_weight(
            f"model.layers.{i}.input_layernorm.weight",
            f"foundation_model.decoder.layers.{i}.input_layernorm.weight",
        )
        language.add_loop_weight(
            f"model.layers.{i}.self_attn.q_proj.weight",
            f"foundation_model.decoder.layers.{i}.self_attention.linear_qkv.weight",
        )
        language.add_loop_weight(
            f"model.layers.{i}.self_attn.k_proj.weight",
            f"foundation_model.decoder.layers.{i}.self_attention.linear_qkv.weight",
        )
        language.add_loop_weight(
            f"model.layers.{i}.self_attn.v_proj.weight",
            f"foundation_model.decoder.layers.{i}.self_attention.linear_qkv.weight",
        )
        language.add_loop_weight(
            f"model.layers.{i}.self_attn.o_proj.weight",
            f"foundation_model.decoder.layers.{i}.self_attention.linear_proj.weight",
        )
        language.add_loop_weight(
            f"model.layers.{i}.post_attention_layernorm.weight",
            f"foundation_model.decoder.layers.{i}.pre_mlp_layernorm.weight",
        )

        # moe
        if i == 0:
            # layer_0 mlp weights
            language.add_loop_weight(
                f"model.layers.0.mlp.gate_proj.weight",
                f"foundation_model.decoder.layers.0.mlp.linear_fc1.weight",
            )
            language.add_loop_weight(
                f"model.layers.0.mlp.up_proj.weight",
                f"foundation_model.decoder.layers.0.mlp.linear_fc1_1.weight",
            )
            language.add_loop_weight(
                f"model.layers.0.mlp.down_proj.weight",
                f"foundation_model.decoder.layers.0.mlp.linear_fc2.weight",
            )

        else:
            # gate
            language.add_loop_weight(
                f"model.layers.{i}.mlp.gate.weight",
                f"foundation_model.decoder.layers.{i}.mlp.text_moe_layer.router.weight",
            )
            language.add_loop_weight(
                f"model.layers.{i}.mlp.gate.weight_1",
                f"foundation_model.decoder.layers.{i}.mlp.vision_moe_layer.router.weight",
            )
            language.add_loop_weight(
                f"model.layers.{i}.mlp.moe_statics.e_score_correction_bias@0",
                f"foundation_model.decoder.layers.{i}.mlp.text_moe_layer.router.expert_bias",
            )
            language.add_loop_weight(
                f"model.layers.{i}.mlp.moe_statics.e_score_correction_bias@1",
                f"foundation_model.decoder.layers.{i}.mlp.vision_moe_layer.router.expert_bias",
            )

            # shared_experts(split 2->1)
            language.add_loop_weight(
                f"model.layers.{i}.mlp.shared_experts.gate_proj.weight",
                f"foundation_model.decoder.layers.{i}.mlp.shared_experts.linear_fc1.weight",
            )
            # language.add_loop_weight(f"model.layers.{i}.mlp.shared_experts.gate_proj.weight",f"foundation_model.decoder.layers.{i}.mlp.shared_experts.linear_fc1.weight")
            language.add_loop_weight(
                f"model.layers.{i}.mlp.shared_experts.up_proj.weight",
                f"foundation_model.decoder.layers.{i}.mlp.shared_experts.linear_fc1_1.weight",
            )
            # language.add_loop_weight(f"model.layers.{i}.mlp.shared_experts.up_proj.weight",f"foundation_model.decoder.layers.{i}.mlp.shared_experts.linear_fc1_1.weight")
            language.add_loop_weight(
                f"model.layers.{i}.mlp.shared_experts.down_proj.weight",
                f"foundation_model.decoder.layers.{i}.mlp.shared_experts.linear_fc2.weight",
            )
            # language.add_loop_weight(f"model.layers.{i}.mlp.shared_experts.down_proj.weight",f"foundation_model.decoder.layers.{i}.mlp.shared_experts.linear_fc2.weight")

            # text_experts
            for j in range(args.num_experts):
                language.add_loop_weight(
                    f"model.layers.{i}.mlp.experts.{j}.gate_proj.weight",
                    f"foundation_model.decoder.layers.{i}.mlp.text_moe_layer.experts.local_experts.{j}.linear_fc1.weight",
                )
                language.add_loop_weight(
                    f"model.layers.{i}.mlp.experts.{j}.up_proj.weight",
                    f"foundation_model.decoder.layers.{i}.mlp.text_moe_layer.experts.local_experts.{j}.linear_fc1_1.weight",
                )
                language.add_loop_weight(
                    f"model.layers.{i}.mlp.experts.{j}.down_proj.weight",
                    f"foundation_model.decoder.layers.{i}.mlp.text_moe_layer.experts.local_experts.{j}.linear_fc2.weight",
                )

            # vision_experts
            for j in range(args.num_experts):
                language.add_loop_weight(
                    f"model.layers.{i}.mlp.experts.{j+args.num_experts}.gate_proj.weight",
                    f"foundation_model.decoder.layers.{i}.mlp.vision_moe_layer.experts.local_experts.{j}.linear_fc1.weight",
                )
                language.add_loop_weight(
                    f"model.layers.{i}.mlp.experts.{j+args.num_experts}.up_proj.weight",
                    f"foundation_model.decoder.layers.{i}.mlp.vision_moe_layer.experts.local_experts.{j}.linear_fc1_1.weight",
                )
                language.add_loop_weight(
                    f"model.layers.{i}.mlp.experts.{j+args.num_experts}.down_proj.weight",
                    f"foundation_model.decoder.layers.{i}.mlp.vision_moe_layer.experts.local_experts.{j}.linear_fc2.weight",
                )

    return (vision, adapter, language)


class CheckpointConverter:
    """A class to convert checkpoints."""

    def __init__(self):
        self.args = self._parse_arguments()
        self._validate_paths()

    def _parse_arguments(self):
        """Parse command line arguments."""
        parser = argparse.ArgumentParser(description="Process some checkpoints.")
        parser.add_argument(
            "--model_name",
            type=str,
            choices=["ernie4.5vl"],
            default="ernie4.5vl",
            help=f"Select model version. Options: [ernie4.5vl]",
        )
        parser.add_argument(
            "--convert_mode",
            type=str,
            choices=["hg2mcore", "mcore2hg"],
            required=True,
            help="convert mode",
        )
        parser.add_argument(
            "--load_mcore_path",
            type=str,
            required=True,
            help="Path to load checkpoints from",
        )
        parser.add_argument(
            "--save_mcore_path",
            type=str,
            required=True,
            help="Path to save checkpoints to",
        )
        parser.add_argument(
            "--load_hg_path",
            type=str,
            required=True,
            help="Path to load HuggingFace checkpoints",
        )
        parser.add_argument(
            "--save_hg_path",
            type=str,
            required=True,
            help="Path to save HuggingFace checkpoints",
        )
        parser.add_argument(
            "--tp", type=int, required=True, help="Tensor parallel size"
        )
        parser.add_argument(
            "--pp", type=int, required=True, help="Pipeline parallel size"
        )
        parser.add_argument(
            "--num_vit_layers", type=int, required=True, help="Number of vit layers"
        )
        parser.add_argument(
            "--num_lm_layers", type=int, required=True, help="Number of lm layers"
        )
        parser.add_argument(
            "--num_experts", type=int, default=64, help="Number of experts"
        )

        return parser.parse_args()

    def _validate_paths(self):
        """Validate input/output paths."""
        assert self.args.load_mcore_path != self.args.save_mcore_path
        assert self.args.load_hg_path != self.args.save_hg_path
        print(f"convert weight mode: {self.args.convert_mode}")
        print(f"model_name: {self.args.model_name}")
        print(f"load_mcore_path: {self.args.load_mcore_path}")
        print(f"save_mcore_path: {self.args.save_mcore_path}")
        print(f"load_hg_path: {self.args.load_hg_path}")
        print(f"save_hg_path: {self.args.save_hg_path}")
        print(f"tp: {self.args.tp}")
        print(f"pp: {self.args.pp}")
        print(f"num_vit_layers: {self.args.num_vit_layers}")
        print(f"num_lm_layers: {self.args.num_lm_layers}")
        print(f"num_experts: {self.args.num_experts}")

    def load_mcore_checkpoints(self):
        """Load parallel checkpoints from disk."""
        state_dict = []
        for p in range(self.args.pp):
            state_dict.append([])
            for t in range(self.args.tp):
                state_dict[p].append({})
                sub_dir = (
                    f"mp_rank_{t:02d}"
                    if self.args.pp == 1
                    else f"mp_rank_{t:02d}_{p:03d}"
                )
                ckpt_path = os.path.join(
                    self.args.load_mcore_path, sub_dir, "model_optim_rng.pt"
                )
                state_dict[p][t] = torch.load(
                    ckpt_path, map_location="cpu", weights_only=False
                )
        return state_dict

    def save_checkpoints(self, state_dict):
        """Save parallel checkpoints to disk."""
        save_mcore_path = Path(self.args.save_mcore_path)
        release_path = save_mcore_path / "release"

        for p in range(self.args.pp):
            for t in range(self.args.tp):
                sub_dir = (
                    f"mp_rank_{t:02d}"
                    if self.args.pp == 1
                    else f"mp_rank_{t:02d}_{p:03d}"
                )
                sub_dir_path = release_path / sub_dir
                sub_dir_path.mkdir(parents=True, exist_ok=True)
                ckpt_path = sub_dir_path / "model_optim_rng.pt"
                torch.save(state_dict[p][t], ckpt_path)

        # Create checkpoint marker file
        with open(f"{save_mcore_path}/latest_checkpointed_iteration.txt", "w") as f:
            f.write("release")

    def load_huggingface_checkpoints(self):
        """Load and merge all .safetensors files in the given directory."""
        state_dict = {}
        if not os.path.isdir(self.args.load_hg_path):
            raise FileNotFoundError(
                f"HuggingFace checkpoint directory not found: {self.args.load_hg_path}"
            )
        # Find all .safetensors files
        checkpoint_files = [
            f for f in os.listdir(self.args.load_hg_path) if f.endswith(".safetensors")
        ]
        print(f"Found {len(checkpoint_files)} checkpoint files")
        # Load and merge all checkpoints
        for checkpoint_file in checkpoint_files:
            checkpoint_path = os.path.join(self.args.load_hg_path, checkpoint_file)
            print(f"Loading checkpoint: {checkpoint_path}")
            current_chunk = load_file(checkpoint_path)
            state_dict.update(current_chunk)

        return state_dict

    def convert_hg2mcore_checkpoints(self):
        """Main conversion workflow."""
        # Load source checkpoints
        hf_state_dict = self.load_huggingface_checkpoints()
        mcore_ckpt = self.load_mcore_checkpoints()

        ######## print for debug info ##########
        for key in hf_state_dict.keys():
            if "model.layers." in key:
                layer_num = int(key.split(".")[2])  # 得到layer_num
                if layer_num < 2:
                    if ".experts" in key:
                        if int(key.split(".")[5]) < 6:
                            print(key, hf_state_dict[key].shape)
                            continue
                        else:
                            continue
                else:
                    continue
            if "vision_model.blocks." in key:
                layer_num = int(key.split(".")[2])
                if layer_num < 2:
                    print(key, hf_state_dict[key].shape)
                    continue
                else:
                    continue

            print(key, hf_state_dict[key].shape)
        # vision loop: 2, language loop:2 + moe:4
        for key in mcore_ckpt[0][0]["model"].keys():
            if "_extra_state" in key:
                continue
            print(f"mcore_ckpt: {key}, {mcore_ckpt[0][0]['model'][key].shape}")
        ######## print for debug info ##########

        (vision, adapter, language) = ernie_model_weights(self.args)

        new_dict = []
        for model in (vision, adapter, language):
            for op in model.common_mapping:
                new_dict.append(op)
            for op in model.loop_mapping:
                # assert(op.name_type == "mcore2hg")
                new_dict.append(op)

        def get_matching_ops(ops, name):
            ops = [op for op in ops if op.dst_name == name]
            assert len(ops) >= 1, f"can not find weight: {name}"

            def sort_key(op):
                s = op.src_name
                parts = s.split(".")
                proj_type = parts[4]
                # define q/k/v priority
                proj_priority = {
                    "q_proj": 0,
                    "k_proj": 1,
                    "v_proj": 2,
                }.get(
                    proj_type, 3
                )  # if not q/k/v，the put it in the end
                return proj_priority  # sort by proj_priority

            if len(ops) > 1:
                ops = sorted(ops, key=sort_key)
            return ops

        # Update mcore checkpoints with converted weights
        for p in range(self.args.pp):
            for t in range(self.args.tp):
                print(f"Processing [pp{p}, tp{t}]....")
                for k, v in mcore_ckpt[p][t]["model"].items():
                    if k.endswith("extra_state"):
                        continue
                    ops = get_matching_ops(new_dict, k)
                    converted_mcore = op.forward_fn(ops, hf_state_dict, v.shape)

                    src_name = [op.src_name for op in ops]

                    assert (
                        mcore_ckpt[p][t]["model"][k].shape == converted_mcore.shape
                    ), f"{mcore_ckpt[p][t]["model"][k].shape} vs {converted_mcore.shape} not equal, src:{ops[0].src_name}, des: {k}"
                    hg_weight = hf_state_dict[ops[0].src_name.split("@")[0]]
                    print(
                        f"converted: {src_name} -> {ops[0].dst_name} [shape]: hg:{hg_weight.shape} -> mcore_ori:{v.shape}, mcore_converted:{converted_mcore.shape}"
                    )
                    mcore_ckpt[p][t]["model"][k] = converted_mcore


        # Save converted checkpoints
        self.save_checkpoints(mcore_ckpt)
        print(
            f"Conversion successful! Checkpoints saved to: {self.args.save_mcore_path}"
        )


if __name__ == "__main__":
    converter = CheckpointConverter()
    converter.convert_hg2mcore_checkpoints()
