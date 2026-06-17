# pylint: skip-file
import os
import torch
import argparse
from safetensors.torch import load_file
from einops import rearrange
from pathlib import Path


parser = argparse.ArgumentParser(description="Process some checkpoints.")
parser.add_argument(
    "--model_name", type=str, required=True, help="Supported model name: [wan2_2_i2v]"
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
parser.add_argument("--num_layers", type=int, required=True, help="Number of layers")

args = parser.parse_args()

print(f"model_name: {args.model_name}")
print(f"save_path: {args.save_path}")
print(f"checkpoint_path: {args.checkpoint_path}")
print(f"num_checkpoints: {args.num_checkpoints}")
print(f"num_layers: {args.num_layers}")

assert args.checkpoint_path != args.save_path

num_layers = args.num_layers
num_checkpoints = args.num_checkpoints
checkpoint_path = args.checkpoint_path
save_path = args.save_path
model_name = args.model_name


def _save_dcp(model_state_dict, save_path, iteration=0):
    """Save model weights in DCP (fsdp_dtensor) format.

    Uses torch_save_to_dcp to convert a plain state dict to DCP without
    requiring a running distributed process group.

    Saves to <save_path>/iter_XXXXXXX/ and writes
    latest_checkpointed_iteration.txt = "<iteration>".
    """
    import tempfile
    from torch.distributed.checkpoint.format_utils import torch_save_to_dcp

    dcp_iter_path = save_path / f"iter_{iteration:07d}"
    dcp_iter_path.mkdir(parents=True, exist_ok=True)

    # torch_save_to_dcp expects a regular torch.save file as input.
    with tempfile.NamedTemporaryFile(suffix=".pt", delete=False) as tmp:
        tmp_path = tmp.name

    try:
        # Add 'module.' prefix to match the FSDP wrapper key format produced by
        # generate_state_dict() at runtime: model.module.<param_name>.
        model_sd_with_prefix = {"module." + k: v for k, v in model_state_dict.items()}
        torch.save({"model": model_sd_with_prefix}, tmp_path)
        torch_save_to_dcp(tmp_path, str(dcp_iter_path))
    finally:
        os.unlink(tmp_path)

    with open(save_path / "latest_checkpointed_iteration.txt", "w") as f:
        f.write(str(iteration))

    print(f"DCP checkpoint saved to: {dcp_iter_path}")


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
# All weight correspondence for second part
second_part_dict = {
    "decoder.layers.0.ffn.0.weight": "blocks.0.ffn.0.weight",
    "decoder.layers.0.ffn.0.bias": "blocks.0.ffn.0.bias",
    "decoder.layers.0.ffn.2.weight": "blocks.0.ffn.2.weight",
    "decoder.layers.0.ffn.2.bias": "blocks.0.ffn.2.bias",
    "decoder.layers.0.norm3.weight": "blocks.0.norm3.weight",
    "decoder.layers.0.norm3.bias": "blocks.0.norm3.bias",
    "decoder.layers.0.modulation": "blocks.0.modulation",  # Need to transpose
    "decoder.layers.0.self_attention.linear_proj.weight": "blocks.0.self_attn.o.weight",
    "decoder.layers.0.self_attention.linear_proj.bias": "blocks.0.self_attn.o.bias",
    "decoder.layers.0.self_attention.linear_qkv.weight": [
        "blocks.0.self_attn.q.weight",
        "blocks.0.self_attn.k.weight",
        "blocks.0.self_attn.v.weight",
    ],
    "decoder.layers.0.self_attention.linear_qkv.bias": [
        "blocks.0.self_attn.q.bias",
        "blocks.0.self_attn.k.bias",
        "blocks.0.self_attn.v.bias",
    ],
    "decoder.layers.0.self_attention.q_layernorm.weight": "blocks.0.self_attn.norm_q.weight",
    "decoder.layers.0.self_attention.k_layernorm.weight": "blocks.0.self_attn.norm_k.weight",
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
wan2_1_second_part_dict = {
    "decoder.layers.0.cross_attn.linear_k_img.weight": "blocks.0.cross_attn.k_img.weight",
    "decoder.layers.0.cross_attn.linear_k_img.bias": "blocks.0.cross_attn.k_img.bias",
    "decoder.layers.0.cross_attn.linear_v_img.weight": "blocks.0.cross_attn.v_img.weight",
    "decoder.layers.0.cross_attn.linear_v_img.bias": "blocks.0.cross_attn.v_img.bias",
    "decoder.layers.0.cross_attn.k_img_layernorm.weight": "blocks.0.cross_attn.norm_k_img.weight",
}
if model_name == "wan2_1_i2v":
    second_part_dict.update(wan2_1_second_part_dict)
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

    new_state_dict[f"decoder.layers.{i}.self_attention.linear_qkv.weight"] = concat_qkv_weight
    new_state_dict[f"decoder.layers.{i}.self_attention.linear_qkv.bias"] = concat_qkv_bias

    # Convert o
    o_weight = state_dict["blocks." + str(i) + ".self_attn.o.weight"]
    o_weight = rearrange(o_weight, "(R N D) H -> (N R D) H", R=1, N=40, D=128, H=5120)
    o_bias = state_dict["blocks." + str(i) + ".self_attn.o.bias"]
    o_bias = rearrange(o_bias, "(R N D H) -> (N R D H)", R=1, N=40, D=128, H=1)

    new_state_dict[f"decoder.layers.{i}.self_attention.linear_proj.weight"] = o_weight
    new_state_dict[f"decoder.layers.{i}.self_attention.linear_proj.bias"] = o_bias

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

    if model_name == "wan2_1_i2v":
        cross_k_img_w = state_dict["blocks." + str(i) + ".cross_attn.k_img.weight"]
        cross_k_img_w = rearrange(cross_k_img_w, "(R N D) H -> (N R D) H", R=1, N=40, D=128, H=5120)
        cross_k_img_b = state_dict["blocks." + str(i) + ".cross_attn.k_img.bias"]
        cross_k_img_b = rearrange(cross_k_img_b, "(R N D H) -> (N R D H)", R=1, N=40, D=128, H=1)
        new_state_dict["decoder.layers." + str(i) + ".cross_attn.linear_k_img.weight"] = cross_k_img_w
        new_state_dict["decoder.layers." + str(i) + ".cross_attn.linear_k_img.bias"] = cross_k_img_b

        cross_v_img_w = state_dict["blocks." + str(i) + ".cross_attn.v_img.weight"]
        cross_v_img_w = rearrange(cross_v_img_w, "(R N D) H -> (N R D) H", R=1, N=40, D=128, H=5120)
        cross_v_img_b = state_dict["blocks." + str(i) + ".cross_attn.v_img.bias"]
        cross_v_img_b = rearrange(cross_v_img_b, "(R N D H) -> (N R D H)", R=1, N=40, D=128, H=1)
        new_state_dict["decoder.layers." + str(i) + ".cross_attn.linear_v_img.weight"] = cross_v_img_w
        new_state_dict["decoder.layers." + str(i) + ".cross_attn.linear_v_img.bias"] = cross_v_img_b

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

save_path = Path(args.save_path)
_save_dcp(mcore_dict, save_path, iteration=1)
print(f"convert success! checkpoint path: {save_path}")
