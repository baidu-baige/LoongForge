"""Simple checkpoint conversion utility for WAN2.1 model."""

import os
import torch
from safetensors.torch import load_file
from einops import rearrange
import re

tp = 1
pp = 2
num_layers = 2


def load(load_path):
    """Load distributed checkpoint from specified path."""
    state_dict = []
    for p in range(pp):
        state_dict.append([])
        for t in range(tp):
            state_dict[p].append({})
            sub_dir_name = f"mp_rank_{t:02d}" if pp == 1 else f"mp_rank_{t:02d}_{p:03d}"
            checkpoint_path = os.path.join(
                load_path, sub_dir_name, "model_optim_rng.pt"
            )
            state_dict[p][t] = torch.load(checkpoint_path, map_location="cpu")
    return state_dict


def save(state_dict, save_path):
    """Save distributed checkpoint to specified path."""
    for p in range(pp):
        for t in range(tp):
            sub_dir_name = f"mp_rank_{t:02d}" if pp == 1 else f"mp_rank_{t:02d}_{p:03d}"
            os.makedirs(os.path.join(save_path, sub_dir_name), exist_ok=True)
            checkpoint_path = os.path.join(
                save_path, sub_dir_name, "model_optim_rng.pt"
            )
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


inside_blk_replace = [
    "blocks.0.cross_attn.k.bias",
    "blocks.0.cross_attn.k.weight",
    "blocks.0.cross_attn.k_img.bias",
    "blocks.0.cross_attn.k_img.weight",
    "blocks.0.cross_attn.norm_k.weight",
    "blocks.0.cross_attn.norm_k_img.weight",
    "blocks.0.cross_attn.norm_q.weight",
    "blocks.0.cross_attn.o.bias",
    "blocks.0.cross_attn.o.weight",
    "blocks.0.cross_attn.q.bias",
    "blocks.0.cross_attn.q.weight",
    "blocks.0.cross_attn.v.bias",
    "blocks.0.cross_attn.v.weight",
    "blocks.0.cross_attn.v_img.bias",
    "blocks.0.cross_attn.v_img.weight",
    "blocks.0.ffn.0.bias",
    "blocks.0.ffn.0.weight",
    "blocks.0.ffn.2.bias",
    "blocks.0.ffn.2.weight",
    "blocks.0.modulation",
    "blocks.0.norm3.bias",
    "blocks.0.norm3.weight",
    "blocks.0.self_attn.k.bias",
    "blocks.0.self_attn.k.weight",
    "blocks.0.self_attn.norm_k.weight",
    "blocks.0.self_attn.norm_q.weight",
    "blocks.0.self_attn.o.bias",
    "blocks.0.self_attn.o.weight",
    "blocks.0.self_attn.q.bias",
    "blocks.0.self_attn.q.weight",
    "blocks.0.self_attn.v.bias",
    "blocks.0.self_attn.v.weight",
]

outside_blk_replace = [
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
    "head.modulation",
    "head.head.weight",
    "head.head.bias",
    "img_emb.proj.0.weight",
    "img_emb.proj.0.bias",
    "img_emb.proj.1.weight",
    "img_emb.proj.1.bias",
    "img_emb.proj.3.weight",
    "img_emb.proj.3.bias",
    "img_emb.proj.4.weight",
    "img_emb.proj.4.bias",
]
state_dict = load_huggingface_chekckpoints(
    "/ssd1/models/huggingface.co/Wan2.1-I2V-14B-480P/", 7
)

new_state_dict = {}
for key in outside_blk_replace:
    new_state_dict[key] = state_dict[key]

for i in range(num_layers):
    ## 一般性替换
    for key in inside_blk_replace:
        new_key = key.replace("blocks.0", "blocks." + str(i))
        new_value = key.replace("blocks.0", "decoder.layers." + str(i))
        new_state_dict[new_value] = state_dict[new_key]

mcore_dict = new_state_dict

# tp1_pp2  = load("/ssd1/tmp/wan2.1_ori_pp8/iter_0000005")
tp1_pp2 = load("/mnt/cluster/aiak-training-llm/wan2.1/iter_0000005/")

# set 到tp_pp dict 中
used = set()
for p in range(pp):
    for t in range(tp):
        print(f" ----------------- [pp{p}, tp{t}] ----------------")
        for k, v in tp1_pp2[p][t]["model"].items():
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
                        tp1_pp2[p][t]["model"][k] = mcore_dict[real_k].clone()
                        used.add(real_k)
                    else:
                        raise ValueError
                else:
                    tp1_pp2[p][t]["model"][k] = mcore_dict[k].clone()
                    used.add(k)

                print("【-】", k, v.shape)

            # elif v.shape[0] * tp ==  mcore_dict[k].shape[0] and v.shape[1:] == mcore_dict[k].shape[1:]:
            #     print("【0】", k, v.shape)
            #     tp1_pp2[p][t]['model'][k] = mcore_dict[k].chunk(tp, dim = 0)[t].clone()
            # elif v.shape[0] == mcore_dict[k].shape[0] and v.shape[1] * tp ==  mcore_dict[k].shape[1]:
            #     print("【1】", k, v.shape)
            #     tp1_pp2[p][t]['model'][k] = mcore_dict[k].chunk(tp, dim = 1)[t].clone()
            else:
                assert False, f"{k} {v.shape} {mcore_dict[k].shape}"

for k in mcore_dict.keys():
    assert k in used, k

save_path = "/mnt/cluster/aiak-training-llm/wan2.1/release/"
save(tp1_pp2, save_path)
with open(
    "/mnt/cluster/aiak-training-llm/wan2.1/latest_checkpointed_iteration.txt", "w"
) as f:
    f.write("release")

print(f"convert success! checkpoint path: {save_path}")
