# pylint: skip-file
import os
import torch
import argparse
from safetensors.torch import load_file
from einops import rearrange
from huggingface_hub import split_torch_state_dict_into_shards
from safetensors.torch import load_file, save_file
from transformers.modeling_utils import SAFE_WEIGHTS_INDEX_NAME, SAFE_WEIGHTS_NAME
from pathlib import Path
import json

parser = argparse.ArgumentParser(description="Process some checkpoints.")
parser.add_argument(
    "--model_name", type=str, required=True, help="Supported model name: [wan2_2_i2v]"
)
parser.add_argument(
    "--load_path",
    type=str,
    required=True,
    help="Path to load megatron checkpoints from",
)
parser.add_argument(
    "--save_path", type=str, required=True, help="Path to save hg checkpoints to"
)
parser.add_argument(
    "--checkpoint_path",
    type=str,
    required=True,
    help="Path to the Hugging original checkpoints",
)
parser.add_argument(
    "--num_checkpoints",
    type=int,
    required=True,
    help="Number of Hugging Face checkpoints",
)
parser.add_argument("--tp", type=int, required=True, help="Tensor parallel size")
parser.add_argument("--pp", type=int, required=True, help="Pipeline parallel size")
parser.add_argument("--num_layers", type=int, required=True, help="Number of layers")

args = parser.parse_args()

print(f"model_name: {args.model_name}")
print(f"load_path: {args.load_path}")
print(f"save_path: {args.save_path}")
print(f"checkpoint_path: {args.checkpoint_path}")
print(f"num_checkpoints: {args.num_checkpoints}")
print(f"tp: {args.tp}")
print(f"pp: {args.pp}")
print(f"num_layers: {args.num_layers}")
assert args.load_path != args.save_path
assert args.checkpoint_path != args.save_path
tp = args.tp
pp = args.pp
num_layers = args.num_layers
num_checkpoints = args.num_checkpoints
checkpoint_path = args.checkpoint_path
load_path = args.load_path
save_path = args.save_path
model_name = args.model_name

def find_parent_of_mp_rank(base_dir: str):
    """find parent path of mp_rank"""
    base_path = Path(base_dir)
    for subdir in base_path.rglob('*'):
        if subdir.is_dir() and 'mp_rank' in subdir.name:
            return subdir.parent
    return None

def load(load_path):
    load_path = str(find_parent_of_mp_rank(load_path))
    state_dict = []
    for p in range(pp):
        state_dict.append([])
        for t in range(tp):
            state_dict[p].append({})
            sub_dir_name = f"mp_rank_{t:02d}" if pp == 1 else f"mp_rank_{t:02d}_{p:03d}"
            checkpoint_path = os.path.join(
                load_path, sub_dir_name, "model_optim_rng.pt"
            )
            state_dict[p][t] = torch.load(
                checkpoint_path, map_location="cpu", weights_only=False
            )
    return state_dict


def save_huggingface_checkpoint(state_dict, save_path):
    """save ckpt"""
    os.makedirs(save_path, exist_ok=True)
    checkpoint_path = os.path.join(save_path, "model.safetensors")

    state_dict_split = split_torch_state_dict_into_shards(state_dict)
    for shard_file, tensors in state_dict_split.filename_to_tensors.items():
        shard = {}
        for tensor in tensors:
            shard[tensor] = state_dict[tensor].contiguous()
            del state_dict[tensor]
        shard_path = os.path.join(save_path, shard_file)
        save_file(shard, shard_path, metadata={"format": "pt"})
        print(f"Saving HuggingFace shard to: {shard_path}")

    if state_dict_split.is_sharded:
        index = {
            "metadata": state_dict_split.metadata,
            "weight_map": state_dict_split.tensor_to_filename,
        }
        save_index_file = os.path.join(save_path, SAFE_WEIGHTS_INDEX_NAME)
        with open(save_index_file, "w", encoding="utf-8") as f:
            content = json.dumps(index, indent=2, sort_keys=True) + "\n"
            f.write(content)


def load_huggingface_chekckpoints(path, num_checkpoints):
    """
    Merge sharded checkpoints from transformers into a single checkpoint.

    Args:
        path (str): the path to the sharded checkpoints
        num_checkpoints (int): the number of checkpoints to merge
    """
    state_dict = {}
    for i in range(1, num_checkpoints + 1):
        checkpoint_path = os.path.join(
            path,
            f"diffusion_pytorch_model-{i:05d}-of-{num_checkpoints:05d}.safetensors",
        )
        current_chunk = load_file(checkpoint_path)
        state_dict.update(current_chunk)
    return state_dict


state_dict = load_huggingface_chekckpoints(args.checkpoint_path, args.num_checkpoints)
# """Convert megatron format to HuggingFace state_dict"""

# Model first part
base_first_part_list = [
    "patch_embedding.weight",
    "patch_embedding.bias",
    "text_embedding.0.weight",
    "text_embedding.0.bias",
    "text_embedding.2.weight",
    "text_embedding.2.bias",
    "time_embedding.0.weight",
    "time_embedding.0.bias",
    "time_embedding.2.weight",
    "time_embedding.2.bias",
    "time_projection.1.weight",
    "time_projection.1.bias",
]
extra_first_part_dict = {
    "wan2_2_i2v": []
}
first_part_list = base_first_part_list + extra_first_part_dict.get(model_name, [])
# All weight correspondence for second part
second_part_dict = {
    "decoder.layers.0.ffn.0.weight": "blocks.0.ffn.0.weight",
    "decoder.layers.0.ffn.0.bias": "blocks.0.ffn.0.bias",
    "decoder.layers.0.ffn.2.weight": "blocks.0.ffn.2.weight",
    "decoder.layers.0.ffn.2.bias": "blocks.0.ffn.2.bias",
    "decoder.layers.0.norm3.weight": "blocks.0.norm3.weight",
    "decoder.layers.0.norm3.bias": "blocks.0.norm3.bias",
    "decoder.layers.0.modulation": "blocks.0.modulation",  # Need to transpose
    "decoder.layers.0.self_attn.linear_proj.weight": "blocks.0.self_attn.o.weight",
    "decoder.layers.0.self_attn.linear_proj.bias": "blocks.0.self_attn.o.bias",
    "decoder.layers.0.self_attn.linear_qkv.weight": [
        "blocks.0.self_attn.q.weight",
        "blocks.0.self_attn.k.weight",
        "blocks.0.self_attn.v.weight",
    ],
    "decoder.layers.0.self_attn.linear_qkv.bias": [
        "blocks.0.self_attn.q.bias",
        "blocks.0.self_attn.k.bias",
        "blocks.0.self_attn.v.bias",
    ],
    "decoder.layers.0.self_attn.q_layernorm.weight": "blocks.0.self_attn.norm_q.weight",
    "decoder.layers.0.self_attn.k_layernorm.weight": "blocks.0.self_attn.norm_k.weight",
    "decoder.layers.0.cross_attn.linear_proj.weight": "blocks.0.cross_attn.o.weight",
    "decoder.layers.0.cross_attn.linear_proj.bias": "blocks.0.cross_attn.o.bias",
    "decoder.layers.0.cross_attn.linear_q.weight": "blocks.0.cross_attn.q.weight",
    "decoder.layers.0.cross_attn.linear_q.bias": "blocks.0.cross_attn.q.bias",
    "decoder.layers.0.cross_attn.linear_kv.weight": [
        "blocks.0.cross_attn.k.weight",
        "blocks.0.cross_attn.v.weight",
    ],
    "decoder.layers.0.cross_attn.linear_kv.bias": [
        "blocks.0.cross_attn.k.bias",
        "blocks.0.cross_attn.v.bias",
    ],
    "decoder.layers.0.cross_attn.q_layernorm.weight": "blocks.0.cross_attn.norm_q.weight",
    "decoder.layers.0.cross_attn.k_layernorm.weight": "blocks.0.cross_attn.norm_k.weight",
}
# Parts that do not need transpose inside
inside_blk_replace_dict = {
    "blocks.0.ffn.0.weight": "decoder.layers.0.ffn.0.weight",
    "blocks.0.ffn.0.bias": "decoder.layers.0.ffn.0.bias",
    "blocks.0.ffn.2.weight": "decoder.layers.0.ffn.2.weight",
    "blocks.0.ffn.2.bias": "decoder.layers.0.ffn.2.bias",
    "blocks.0.norm3.weight": "decoder.layers.0.norm3.weight",
    "blocks.0.norm3.bias": "decoder.layers.0.norm3.bias",
    "blocks.0.self_attn.norm_q.weight": "decoder.layers.0.self_attn.q_layernorm.weight",
    "blocks.0.self_attn.norm_k.weight": "decoder.layers.0.self_attn.k_layernorm.weight",
    "blocks.0.cross_attn.norm_q.weight": "decoder.layers.0.cross_attn.q_layernorm.weight",
    "blocks.0.cross_attn.norm_k.weight": "decoder.layers.0.cross_attn.k_layernorm.weight",
}
# Model last part
third_part_dict = {
    "head.modulation",
    "head.head.weight",
    "head.head.bias",
}

new_state_dict = {}

cp_pp = load(args.load_path)

for i in range(num_layers):
    layers_in_each_stage = num_layers // pp
    pp_idx = i // layers_in_each_stage
    shift = i % layers_in_each_stage
    print(f"layer_idx: {i}, pp_idx: {pp_idx}, shift: {shift}")

    ## self_attention qkv split
    mcore_dict = cp_pp[pp_idx][0]["model"]
    src_qkv_weight = mcore_dict[
        "decoder.layers." + str(shift) + ".self_attn.linear_qkv.weight"
    ]
    trans_qkv = rearrange(
        src_qkv_weight, "(N R D) H -> (R N D) H", R=3, N=40, D=128, H=5120
    )
    q_weight, k_weight, v_weight = torch.split(trans_qkv, 5120, dim=0)
    new_state_dict["blocks." + str(i) + ".self_attn.q.weight"] = q_weight
    new_state_dict["blocks." + str(i) + ".self_attn.k.weight"] = k_weight
    new_state_dict["blocks." + str(i) + ".self_attn.v.weight"] = v_weight

    src_qkv_bias = mcore_dict[
        "decoder.layers." + str(shift) + ".self_attn.linear_qkv.bias"
    ]
    trans_qkv = rearrange(src_qkv_bias, "(N R D H) -> (R N D H)", R=3, N=40, D=128, H=1)
    q_bias, k_bias, v_bias = torch.split(trans_qkv, 5120, dim=0)
    new_state_dict["blocks." + str(i) + ".self_attn.q.bias"] = q_bias
    new_state_dict["blocks." + str(i) + ".self_attn.k.bias"] = k_bias
    new_state_dict["blocks." + str(i) + ".self_attn.v.bias"] = v_bias

    # Convert to o
    linear_proj_weight = mcore_dict[
        "decoder.layers." + str(shift) + ".self_attn.linear_proj.weight"
    ]
    linear_proj_bias = mcore_dict[
        "decoder.layers." + str(shift) + ".self_attn.linear_proj.bias"
    ]
    linear_proj_weight = rearrange(
        linear_proj_weight, "(N R D) H -> (R N D) H", R=1, N=40, D=128, H=5120
    )
    linear_proj_bias = rearrange(
        linear_proj_bias, "(N R D H) -> (R N D H)", R=1, N=40, D=128, H=1
    )
    new_state_dict["blocks." + str(i) + ".self_attn.o.weight"] = linear_proj_weight
    new_state_dict["blocks." + str(i) + ".self_attn.o.bias"] = linear_proj_bias

    ## cross_attention q transpose
    cross_q_weight = mcore_dict[
        "decoder.layers." + str(shift) + ".cross_attn.linear_q.weight"
    ]
    cross_q_weight = rearrange(
        cross_q_weight, "(N R D) H -> (R N D) H", R=1, N=40, D=128, H=5120
    )
    cross_q_bias = mcore_dict[
        "decoder.layers." + str(shift) + ".cross_attn.linear_q.bias"
    ]
    cross_q_bias = rearrange(
        cross_q_bias, "(N R D H) -> (R N D H)", R=1, N=40, D=128, H=1
    )
    new_state_dict["blocks." + str(i) + ".cross_attn.q.weight"] = cross_q_weight
    new_state_dict["blocks." + str(i) + ".cross_attn.q.bias"] = cross_q_bias

    # cross_attention kv split
    kv_weight = mcore_dict[
        "decoder.layers." + str(shift) + ".cross_attn.linear_kv.weight"
    ]
    kv_weight = rearrange(kv_weight, "(N R D) H -> (R N D) H", R=2, N=40, D=128, H=5120)
    k_weight, v_weight = torch.split(kv_weight, 5120, dim=0)

    kv_bias = mcore_dict["decoder.layers." + str(shift) + ".cross_attn.linear_kv.bias"]
    kv_bias = rearrange(kv_bias, "(N R D H) -> (R N D H)", R=2, N=40, D=128, H=1)
    k_bias, v_bias = torch.split(kv_bias, 5120, dim=0)
    new_state_dict["blocks." + str(i) + ".cross_attn.k.weight"] = k_weight
    new_state_dict["blocks." + str(i) + ".cross_attn.k.bias"] = k_bias
    new_state_dict["blocks." + str(i) + ".cross_attn.v.weight"] = v_weight
    new_state_dict["blocks." + str(i) + ".cross_attn.v.bias"] = v_bias

    # cross_attention o
    cross_o_weight = mcore_dict[
        "decoder.layers." + str(shift) + ".cross_attn.linear_proj.weight"
    ]
    cross_o_weight = rearrange(
        cross_o_weight, "(N R D) H ->(R N D) H", R=1, N=40, D=128, H=5120
    )
    new_state_dict["blocks." + str(i) + ".cross_attn.o.weight"] = cross_o_weight

    cross_o_bias = mcore_dict[
        "decoder.layers." + str(shift) + ".cross_attn.linear_proj.bias"
    ]
    cross_o_bias = rearrange(
        cross_o_bias, "(N R D H) ->(R N D H)", R=1, N=40, D=128, H=1
    )
    new_state_dict["blocks." + str(i) + ".cross_attn.o.bias"] = cross_o_bias

    # 1, 6, 5120 -> 6, 1, 5120 # modulation transpose
    modulation = mcore_dict["decoder.layers." + str(shift) + ".modulation"]
    modulation = rearrange(modulation, "D M L -> M D L")
    new_state_dict["blocks." + str(i) + ".modulation"] = modulation

    ## General replacement
    for key, value in inside_blk_replace_dict.items():
        key = key.replace("blocks.0", "blocks." + str(i))
        value = value.replace("decoder.layers.0", "decoder.layers." + str(shift))
        new_state_dict[key] = mcore_dict[value]


# new_state_dict = {}
for key in first_part_list:
    new_state_dict[key] = cp_pp[0][0]["model"][key]

for key in third_part_dict:
    new_state_dict[key] = cp_pp[pp - 1][0]["model"][key]

save_huggingface_checkpoint(new_state_dict, args.save_path)
print(f"convert success! checkpoint path: {save_path}")
