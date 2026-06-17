# pylint: skip-file
import os
import torch
import argparse
import tempfile
from einops import rearrange
from huggingface_hub import split_torch_state_dict_into_shards
from safetensors.torch import save_file
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
parser.add_argument("--num_layers", type=int, required=True, help="Number of layers")

args = parser.parse_args()

print(f"model_name: {args.model_name}")
print(f"load_path: {args.load_path}")
print(f"save_path: {args.save_path}")
print(f"num_layers: {args.num_layers}")
assert args.load_path != args.save_path
num_layers = args.num_layers
load_path = args.load_path
save_path = args.save_path
model_name = args.model_name


def load_dcp(load_path):
    """Load DCP (fsdp_dtensor) checkpoint and return the model state dict."""
    from torch.distributed.checkpoint.format_utils import dcp_to_torch_save

    iter_file = os.path.join(load_path, "latest_checkpointed_iteration.txt")
    with open(iter_file) as f:
        iteration = f.read().strip()
    dcp_dir = os.path.join(load_path, f"iter_{int(iteration):07d}")
    print(f"Loading DCP checkpoint from: {dcp_dir}")

    with tempfile.NamedTemporaryFile(suffix=".pt", delete=False) as tmp:
        tmp_path = tmp.name
    try:
        dcp_to_torch_save(dcp_dir, tmp_path)
        raw = torch.load(tmp_path, map_location="cpu", weights_only=False)
    finally:
        os.unlink(tmp_path)

    # Strip "module." prefix added by _save_dcp in hg2mcore
    model_dict = {}
    for k, v in raw["model"].items():
        new_key = k.removeprefix("module.")
        model_dict[new_key] = v

    return model_dict


def save_huggingface_checkpoint(state_dict, save_path):
    """save ckpt"""
    os.makedirs(save_path, exist_ok=True)

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
    "wan2_1_i2v": [
        "img_emb.proj.0.weight",
        "img_emb.proj.0.bias",
        "img_emb.proj.1.weight",
        "img_emb.proj.1.bias",
        "img_emb.proj.3.weight",
        "img_emb.proj.3.bias",
        "img_emb.proj.4.weight",
        "img_emb.proj.4.bias",
    ],
    "wan2_2_i2v": []
}
first_part_list = base_first_part_list + extra_first_part_dict.get(model_name, [])

# Parts that do not need transpose inside
inside_blk_replace_dict = {
    "blocks.0.ffn.0.weight": "decoder.layers.0.ffn.0.weight",
    "blocks.0.ffn.0.bias": "decoder.layers.0.ffn.0.bias",
    "blocks.0.ffn.2.weight": "decoder.layers.0.ffn.2.weight",
    "blocks.0.ffn.2.bias": "decoder.layers.0.ffn.2.bias",
    "blocks.0.norm3.weight": "decoder.layers.0.norm3.weight",
    "blocks.0.norm3.bias": "decoder.layers.0.norm3.bias",
    "blocks.0.self_attn.norm_q.weight": "decoder.layers.0.self_attention.q_layernorm.weight",
    "blocks.0.self_attn.norm_k.weight": "decoder.layers.0.self_attention.k_layernorm.weight",
    "blocks.0.cross_attn.norm_q.weight": "decoder.layers.0.cross_attn.q_layernorm.weight",
    "blocks.0.cross_attn.norm_k.weight": "decoder.layers.0.cross_attn.k_layernorm.weight",
}
wan2_1_inside_blk_replace_dict = {
    "blocks.0.cross_attn.norm_k_img.weight": "decoder.layers.0.cross_attn.k_img_layernorm.weight",
}
if model_name == "wan2_1_i2v":
    inside_blk_replace_dict.update(wan2_1_inside_blk_replace_dict)

# Model last part
third_part_dict = {
    "head.modulation",
    "head.head.weight",
    "head.head.bias",
}

mcore_dict = load_dcp(args.load_path)

new_state_dict = {}

for i in range(num_layers):
    print(f"layer_idx: {i}")

    ## self_attention qkv split
    src_qkv_weight = mcore_dict[
        "decoder.layers." + str(i) + ".self_attention.linear_qkv.weight"
    ]
    trans_qkv = rearrange(
        src_qkv_weight, "(N R D) H -> (R N D) H", R=3, N=40, D=128, H=5120
    )
    q_weight, k_weight, v_weight = torch.split(trans_qkv, 5120, dim=0)
    new_state_dict["blocks." + str(i) + ".self_attn.q.weight"] = q_weight
    new_state_dict["blocks." + str(i) + ".self_attn.k.weight"] = k_weight
    new_state_dict["blocks." + str(i) + ".self_attn.v.weight"] = v_weight

    src_qkv_bias = mcore_dict[
        "decoder.layers." + str(i) + ".self_attention.linear_qkv.bias"
    ]
    trans_qkv = rearrange(src_qkv_bias, "(N R D H) -> (R N D H)", R=3, N=40, D=128, H=1)
    q_bias, k_bias, v_bias = torch.split(trans_qkv, 5120, dim=0)
    new_state_dict["blocks." + str(i) + ".self_attn.q.bias"] = q_bias
    new_state_dict["blocks." + str(i) + ".self_attn.k.bias"] = k_bias
    new_state_dict["blocks." + str(i) + ".self_attn.v.bias"] = v_bias

    # Convert to o
    linear_proj_weight = mcore_dict[
        "decoder.layers." + str(i) + ".self_attention.linear_proj.weight"
    ]
    linear_proj_bias = mcore_dict[
        "decoder.layers." + str(i) + ".self_attention.linear_proj.bias"
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
        "decoder.layers." + str(i) + ".cross_attn.linear_q.weight"
    ]
    cross_q_weight = rearrange(
        cross_q_weight, "(N R D) H -> (R N D) H", R=1, N=40, D=128, H=5120
    )
    cross_q_bias = mcore_dict[
        "decoder.layers." + str(i) + ".cross_attn.linear_q.bias"
    ]
    cross_q_bias = rearrange(
        cross_q_bias, "(N R D H) -> (R N D H)", R=1, N=40, D=128, H=1
    )
    new_state_dict["blocks." + str(i) + ".cross_attn.q.weight"] = cross_q_weight
    new_state_dict["blocks." + str(i) + ".cross_attn.q.bias"] = cross_q_bias

    # cross_attention kv split
    kv_weight = mcore_dict[
        "decoder.layers." + str(i) + ".cross_attn.linear_kv.weight"
    ]
    kv_weight = rearrange(kv_weight, "(N R D) H -> (R N D) H", R=2, N=40, D=128, H=5120)
    k_weight, v_weight = torch.split(kv_weight, 5120, dim=0)

    kv_bias = mcore_dict["decoder.layers." + str(i) + ".cross_attn.linear_kv.bias"]
    kv_bias = rearrange(kv_bias, "(N R D H) -> (R N D H)", R=2, N=40, D=128, H=1)
    k_bias, v_bias = torch.split(kv_bias, 5120, dim=0)
    new_state_dict["blocks." + str(i) + ".cross_attn.k.weight"] = k_weight
    new_state_dict["blocks." + str(i) + ".cross_attn.k.bias"] = k_bias
    new_state_dict["blocks." + str(i) + ".cross_attn.v.weight"] = v_weight
    new_state_dict["blocks." + str(i) + ".cross_attn.v.bias"] = v_bias

    if model_name == "wan2_1_i2v":
        cross_k_img_weight = mcore_dict[
            "decoder.layers." + str(i) + ".cross_attn.linear_k_img.weight"
        ]
        cross_k_img_weight = rearrange(
            cross_k_img_weight, "(N R D) H -> (R N D) H", R=1, N=40, D=128, H=5120
        )
        cross_k_img_bias = mcore_dict[
            "decoder.layers." + str(i) + ".cross_attn.linear_k_img.bias"
        ]
        cross_k_img_bias = rearrange(
            cross_k_img_bias, "(N R D H) -> (R N D H)", R=1, N=40, D=128, H=1
        )
        new_state_dict["blocks." + str(i) + ".cross_attn.k_img.weight"] = cross_k_img_weight
        new_state_dict["blocks." + str(i) + ".cross_attn.k_img.bias"] = cross_k_img_bias

        cross_v_img_weight = mcore_dict[
            "decoder.layers." + str(i) + ".cross_attn.linear_v_img.weight"
        ]
        cross_v_img_weight = rearrange(
            cross_v_img_weight, "(N R D) H -> (R N D) H", R=1, N=40, D=128, H=5120
        )
        cross_v_img_bias = mcore_dict[
            "decoder.layers." + str(i) + ".cross_attn.linear_v_img.bias"
        ]
        cross_v_img_bias = rearrange(
            cross_v_img_bias, "(N R D H) -> (R N D H)", R=1, N=40, D=128, H=1
        )
        new_state_dict["blocks." + str(i) + ".cross_attn.v_img.weight"] = cross_v_img_weight
        new_state_dict["blocks." + str(i) + ".cross_attn.v_img.bias"] = cross_v_img_bias

    # cross_attention o
    cross_o_weight = mcore_dict[
        "decoder.layers." + str(i) + ".cross_attn.linear_proj.weight"
    ]
    cross_o_weight = rearrange(
        cross_o_weight, "(N R D) H ->(R N D) H", R=1, N=40, D=128, H=5120
    )
    new_state_dict["blocks." + str(i) + ".cross_attn.o.weight"] = cross_o_weight

    cross_o_bias = mcore_dict[
        "decoder.layers." + str(i) + ".cross_attn.linear_proj.bias"
    ]
    cross_o_bias = rearrange(
        cross_o_bias, "(N R D H) ->(R N D H)", R=1, N=40, D=128, H=1
    )
    new_state_dict["blocks." + str(i) + ".cross_attn.o.bias"] = cross_o_bias

    # 1, 6, 5120 -> 6, 1, 5120 # modulation transpose
    modulation = mcore_dict["decoder.layers." + str(i) + ".modulation"]
    modulation = rearrange(modulation, "D M L -> M D L")
    new_state_dict["blocks." + str(i) + ".modulation"] = modulation

    ## General replacement
    for key, value in inside_blk_replace_dict.items():
        key = key.replace("blocks.0", "blocks." + str(i))
        value = value.replace("decoder.layers.0", "decoder.layers." + str(i))
        new_state_dict[key] = mcore_dict[value]


for key in first_part_list:
    new_state_dict[key] = mcore_dict[key]

for key in third_part_dict:
    new_state_dict[key] = mcore_dict[key]

save_huggingface_checkpoint(new_state_dict, args.save_path)
print(f"convert success! checkpoint path: {save_path}")
