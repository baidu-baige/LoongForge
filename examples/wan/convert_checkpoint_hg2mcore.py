# pylint: skip-file
import os
import torch
import argparse
from safetensors.torch import load_file
from einops import rearrange
from pathlib import Path
import re


parser = argparse.ArgumentParser(description="Process some checkpoints.")
parser.add_argument(
    "--model_name", type=str, required=True, help="Supported model name: [wan2_2_i2v]"
)
parser.add_argument(
    "--load_path", type=str, required=True, help="Path to load checkpoints from"
)
parser.add_argument(
    "--save_path", type=str, required=True, help="Path to save checkpoints to"
)
parser.add_argument(
    "--checkpoint_path",
    type=str,
    required=True,
    help="Path to the Hugging Face checkpoints",
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


def load(load_path):
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


def save(state_dict, save_path):
    for p in range(pp):
        for t in range(tp):
            sub_dir_name = f"mp_rank_{t:02d}" if pp == 1 else f"mp_rank_{t:02d}_{p:03d}"
            sub_dir = save_path / sub_dir_name
            sub_dir.mkdir(parents=True, exist_ok=True)
            checkpoint_path = sub_dir / "model_optim_rng.pt"
            torch.save(state_dict[p][t], checkpoint_path)


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
# """Convert HuggingFace state_dict to megatron format state_dict."""

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


for i in range(num_layers):
    print("layer: ", i)
    # self_attention qkv merge
    q_weight = state_dict["blocks." + str(i) + ".self_attn.q.weight"]
    q_bias = state_dict["blocks." + str(i) + ".self_attn.q.bias"]
    k_weight = state_dict["blocks." + str(i) + ".self_attn.k.weight"]
    k_bias = state_dict["blocks." + str(i) + ".self_attn.k.bias"]
    v_weight = state_dict["blocks." + str(i) + ".self_attn.v.weight"]
    v_bias = state_dict["blocks." + str(i) + ".self_attn.v.bias"]

    # Convert to huggingface linear_qkv
    concat_qkv_weight = torch.concat([q_weight, k_weight, v_weight], dim=0)
    concat_qkv_weight = rearrange(
        concat_qkv_weight, "(R N D) H -> (N R D) H", R=3, N=40, D=128, H=5120
    )
    concat_qkv_bias = torch.concat([q_bias, k_bias, v_bias], dim=0)
    concat_qkv_bias = rearrange(
        concat_qkv_bias, "(R N D H) -> (N R D H)", R=3, N=40, D=128, H=1
    )

    new_state_dict["decoder.layers." + str(i) + ".self_attn.linear_qkv.weight"] = (
        concat_qkv_weight
    )
    new_state_dict["decoder.layers." + str(i) + ".self_attn.linear_qkv.bias"] = (
        concat_qkv_bias
    )
    # Convert o
    o_weight = state_dict["blocks." + str(i) + ".self_attn.o.weight"]
    o_weight = rearrange(o_weight, "(R N D) H -> (N R D) H", R=1, N=40, D=128, H=5120)
    o_bias = state_dict["blocks." + str(i) + ".self_attn.o.bias"]
    o_bias = rearrange(o_bias, "(R N D H) -> (N R D H)", R=1, N=40, D=128, H=1)

    new_state_dict["decoder.layers." + str(i) + ".self_attn.linear_proj.bias"] = o_bias
    new_state_dict["decoder.layers." + str(i) + ".self_attn.linear_proj.weight"] = (
        o_weight
    )

    # cross_attention q transpose
    cross_q_w = state_dict["blocks." + str(i) + ".cross_attn.q.weight"]
    cross_q_w = rearrange(cross_q_w, "(R N D) H -> (N R D) H", R=1, N=40, D=128, H=5120)
    cross_q_b = state_dict["blocks." + str(i) + ".cross_attn.q.bias"]
    cross_q_b = rearrange(cross_q_b, "(R N D H) -> (N R D H)", R=1, N=40, D=128, H=1)
    new_state_dict["decoder.layers." + str(i) + ".cross_attn.linear_q.weight"] = (
        cross_q_w
    )
    new_state_dict["decoder.layers." + str(i) + ".cross_attn.linear_q.bias"] = cross_q_b

    # cross_attention kv merge
    cross_attn_k_weight = state_dict["blocks." + str(i) + ".cross_attn.k.weight"]
    cross_attn_k_bias = state_dict["blocks." + str(i) + ".cross_attn.k.bias"]
    cross_attn_v_weight = state_dict["blocks." + str(i) + ".cross_attn.v.weight"]
    cross_attn_v_bias = state_dict["blocks." + str(i) + ".cross_attn.v.bias"]
    concat_kv_weight = torch.concat([cross_attn_k_weight, cross_attn_v_weight], dim=0)
    concat_kv_weight = rearrange(
        concat_kv_weight, "(R N D) H -> (N R D) H", R=2, N=40, D=128, H=5120
    )
    concat_kv_bias = torch.concat([cross_attn_k_bias, cross_attn_v_bias], dim=0)
    concat_kv_bias = rearrange(
        concat_kv_bias, "(R N D H) -> (N R D H)", R=2, N=40, D=128, H=1
    )
    new_state_dict["decoder.layers." + str(i) + ".cross_attn.linear_kv.weight"] = (
        concat_kv_weight
    )
    new_state_dict["decoder.layers." + str(i) + ".cross_attn.linear_kv.bias"] = (
        concat_kv_bias
    )

    # cross_attention o transpose
    cross_o_weight = state_dict["blocks." + str(i) + ".cross_attn.o.weight"]
    cross_o_weight = rearrange(
        cross_o_weight, "(R N D) H -> (N R D) H", R=1, N=40, D=128, H=5120
    )
    cross_o_bias = state_dict["blocks." + str(i) + ".cross_attn.o.bias"]
    cross_o_bias = rearrange(
        cross_o_bias, "(R N D H) -> (N R D H)", R=1, N=40, D=128, H=1
    )
    new_state_dict["decoder.layers." + str(i) + ".cross_attn.linear_proj.weight"] = (
        cross_o_weight
    )
    new_state_dict["decoder.layers." + str(i) + ".cross_attn.linear_proj.bias"] = (
        cross_o_bias
    )
    # 1, 6, 5120 -> 6, 1, 5120 # modulation transpose
    modulation = state_dict["blocks." + str(i) + ".modulation"]
    new_state_dict["decoder.layers." + str(i) + ".modulation"] = rearrange(
        modulation, "D M L -> M D L"
    )
    ## General replacement
    for key, value in inside_blk_replace_dict.items():
        key = key.replace("blocks.0", "blocks." + str(i))
        value = value.replace("decoder.layers.0", "decoder.layers." + str(i))
        new_state_dict[value] = state_dict[key]


# new_state_dict = {}
for key in first_part_list:
    new_state_dict[key] = state_dict[key]

for key in third_part_dict:
    new_state_dict[key] = state_dict[key]

mcore_dict = new_state_dict
cp_pp = load(args.load_path)

# Set into tp_pp dict
used = set()
for p in range(pp):
    for t in range(tp):
        print(f" ----------------- [pp{p}, tp{t}] ----------------")
        for k, v in cp_pp[p][t]["model"].items():
            # print("k: ", k)
            # if "t_embedder" in k:
            #     print("t_embedder skip: {k}",k)
            #     continue
            if k.endswith("extra_state"):
                continue
            if v.shape == mcore_dict[k].shape:
                if "decoder.layers." in k:
                    match = re.search(r"decoder\.layers\.(\d+)", k)
                    if match:
                        # number = int(match.group(1))
                        old_number = int(match.group(1))
                        new_number = int(old_number + num_layers / pp * p)
                        real_k = k.replace(
                            f"decoder.layers.{old_number}.",
                            f"decoder.layers.{new_number}.",
                        )
                        cp_pp[p][t]["model"][k] = mcore_dict[real_k].clone()
                        used.add(real_k)
                    else:
                        raise ValueError
                else:
                    cp_pp[p][t]["model"][k] = mcore_dict[k].clone()
                    used.add(k)

                print("【-】", k, v.shape)

            else:
                assert False, f"{k} {v.shape} {mcore_dict[k].shape}"

for k in mcore_dict.keys():
    assert k in used, k

save_path = Path(args.save_path)
release_path = save_path / "release"
save(cp_pp, release_path)

with open(f"{save_path}/latest_checkpointed_iteration.txt", "w") as f:
    f.write("release")


print(f"convert success! checkpoint path: {save_path}")
