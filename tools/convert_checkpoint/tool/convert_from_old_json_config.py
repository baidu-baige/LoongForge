#!/usr/bin/env python3
import json
from collections import defaultdict
import shutil

mla_map_dict = {
    "attention.query_a": "attention.q_down",
    "attention.query_b": "attention.q_up",
    "attention.q_a_layernorm": "attention.q_up_layernorm",
    "attention.kv_a": "attention.kv_down",
    "attention.kv_b": "attention.kv_up",
    "attention.kv_a_layernorm": "attention.kv_up_layernorm",
    "attention.query": "attention.q"
}

mla_layernorm_list = {
    "attention.q_a_layernorm",
    "attention.kv_a_layernorm"
}

dense_map_dict = {
    "attention.q_a_layernorm": "attention.q_a_layernorm",
    "attention.kv_a_layernorm": "attention.kv_a_layernorm"
}

attn_norm_dict = {
    "attention.q_norm": "attention.q_a_layernorm",
    "attention.k_norm": "attention.kv_a_layernorm",
}

moe_dense_dict = {
    "mlp.dense_h_to_4h": "moe.expert_h_to_4h",
    "mlp.dense_4h_to_h": "moe.expert_4h_to_h",
}

normal_keys = [
    "word_embeddings",
    "word_position_embeddings",
    "transformer",
    "transformer_tpl",
    "layer_prefix",
    "mtp_layer_prefix",
    "input_layernorm",
    "attention.query_key_value",
    "attention.dense",
    "attention.rotary_emb.inv_freq",
    "post_attention_layernorm",
    "post_mlp_layernorm",
    "moe.gate",
    "moe.gate.bias",
    "moe.expert",
    "moe.shared_expert",
    "moe.groupedgemm.expert",
    "final_layernorm",
    "word_embeddings_for_head",
    "mtp_word_embeddings",
    "mtp_enorm",
    "mtp_hnorm",
    "mtp_eh_proj",
    "mtp_shared_head_norm",
    "mtp_shared_head_head"
]

direct_name_dict = [
    "post_attention_layerscale",
    "post_mlp_layerscale"
]

TENSOR_PARALLEL_DIM = {
    "word_embeddings.weight": 0,
    "attention.query_key_value.weight": 0,
    "attention.query_key_value.bias": 0,
    "attention.q_down.weight": 0,
    "attention.q_up.weight": 0,
    "attention.kv_down.weight": 0,
    "attention.kv_up.weight": 0,
    "attention.dense.weight" : 1,
    "mlp.dense_h_to_4h.weight": 0,
    "mlp.dense_h_to_4h.bias": 0,
    "mlp.dense_4h_to_h.weight": 1,
    "moe.expert_h_to_4h.weight": 0,
    "moe.expert_h_to_4h.bias": 0,
    "moe.expert_4h_to_h.weight": 1,
    "word_embeddings_for_head.weight": 0,
    "mtp_word_embeddings.weight": 0,
    "mtp_shared_head_head.weight": 0
}

mcore_args = [
    "untie_embeddings_and_output_weights",
    "use_rotary_position_embeddings",
    "add_embedding_padding",
    "transpose_mlp_dense",
    "transpose_query_key_value",
    "rotary_base"
]

invalid_mcore_args = [
    "num_layers_per_virtual_pipeline_stage",
    "virtual_pipeline_model_parallel_size",
    "make_vocab_size_divisible_by",
    "fp8_keys",
    "ignore_tp_keys",
    "first_k_dense_replace",
    "tensor_model_parallel_size",
    "pipeline_model_parallel_size",
    "data_parallel_size",
    "use_distributed_optimizer"
]

INVALID_K1 = ["separate_dtype"]

def convert_json(config_path):
    with open(config_path, 'r', encoding='utf-8') as f:
        data = json.loads(f.read())

    assert 'args' in data.keys() and 'name_map' in data.keys(), f"invalid file: {config_path=}.\ndata:{json.dumps(data, indent=4)}"
    args = data['args']
    name_map = data['name_map']
    torch_dtype = data["torch_dtype"]

    if 'mcore' not in name_map:
        print(f"file not converted: {config_path}")
        return None
    assert 'common' in args.keys() and 'huggingface' in args.keys() and 'mcore' in args.keys(), f"invalid file: {config_path=}.\nargs:{json.dumps(args, indent=4)}"
    args_common = args['common']
    args_hf = args['huggingface']
    args_mcore = args['mcore']

    tensor_parallel_dim = None
    if "tensor_parallel_dim" in data:
        for k, v in data.get("tensor_parallel_dim").items():
            if k == "attention.qkv_map.weight":
                continue
            elif k == "attention.q_norm.weight":
                k = "attention.q_a_layernorm.weight"
            elif k == "attention.k_norm.weight":
                k = "attention.kv_a_layernorm.weight"
            if k not in TENSOR_PARALLEL_DIM or (k in TENSOR_PARALLEL_DIM and TENSOR_PARALLEL_DIM[k] != v):
                if tensor_parallel_dim is None:
                    tensor_parallel_dim = {}
                tensor_parallel_dim[k] = v

    if "num_nextn_predict_layers" in args_hf.keys():
        args_common["num_nextn_predict_layers"] = args_hf["num_nextn_predict_layers"]

    if "first_k_dense_replace" in args_hf.keys():
        first_k_dense_replace = args_hf["first_k_dense_replace"]
    else:
        first_k_dense_replace = None

    if "num_experts" in args_hf.keys():
        num_experts = args_hf["num_experts"]
        args_common["num_experts"] = num_experts
    elif "n_routed_experts" in args_hf.keys():
        num_experts = args_hf["n_routed_experts"]
        args_common["num_experts"] = num_experts
    elif "num_local_experts" in args_hf.keys():
        num_experts = args_hf["num_local_experts"]
        args_common["num_experts"] = num_experts
    else:
        num_experts = None

    new_args_mcore = {}
    for key, value in args_mcore.items():
        if key in mcore_args:
            new_args_mcore[key] = value
        elif key in invalid_mcore_args:
            continue
        else:
            raise Exception(f"invalid key: {key} in args_mcore. {config_path=}")

    assert 'huggingface' in name_map.keys() and 'mcore' in name_map.keys(), f"invalid file: {config_path=}.\n{json.dumps(name_map, indent=4)}"
    name_hf = name_map["huggingface"]
    name_mcore = name_map["mcore"]

    is_mla = "attention.qkv_map" in name_hf and "attention.kv_a" in name_hf["attention.qkv_map"]

    new_name_hf = {}
    for key, value in name_hf.items():
        if key == "moe.mlp":
            continue
        elif key == "attention.q_norm":
            new_name_hf["attention.q_a_layernorm"] = value
        elif key == "attention.k_norm":
            new_name_hf["attention.kv_a_layernorm"] = value
        elif key == "attention.qkv_map":
            qkv_map = value
            if "attention.kv_a" in qkv_map:
                for k, v in qkv_map.items():
                    if k in mla_map_dict:
                        new_name_hf[mla_map_dict[k]] = v
                    else:
                        raise Exception(f"invalid key: {k}, in name_hf. {config_path=}")
            else:
                for k, v in qkv_map.items():
                    if k in dense_map_dict:
                        new_name_hf[dense_map_dict[k]] = v
                    else:
                        raise Exception(f"invalid key: {k}, in name_hf. {config_path=}")
        elif key in attn_norm_dict.keys():
            new_name_hf[attn_norm_dict[key]] = value
        elif key in moe_dense_dict.keys():
            if num_experts is not None:
                new_name_hf[moe_dense_dict[key]] = value
                if first_k_dense_replace is not None:
                    assert "moe.mlp" in name_hf.keys(), f"moe.mlp is not found in name_hf. please check config file. {config_path=}"
                    moe_mlp = name_hf["moe.mlp"]
                    if isinstance(name_hf[key], list):
                        new_dense_list = []
                        for v in name_hf[key]:
                            new_dense_list.append(f"{moe_mlp}.{v}")
                        new_name_hf[key] = new_dense_list
                    else:
                        new_name_hf[key] = f"{moe_mlp}.{value}"
            else:
                # mlp dense
                new_name_hf[key] = value
        elif key in normal_keys:
            new_name_hf[key] = value
        elif key in direct_name_dict:
            new_name_hf[key] = {"name": value, "is_direct_name": True}
        else:
            raise Exception(f"invalid key: {key}, in name_hf. {config_path=}")

    new_name_mcore = {}
    for key, value in name_mcore.items():
        if key == "moe.mlp":
            continue
        elif key == "attention.q_norm":
            new_name_mcore["attention.q_a_layernorm"] = value
        elif key == "attention.k_norm":
            new_name_mcore["attention.kv_a_layernorm"] = value
        elif key == "attention.qkv_map":
            qkv_map = value
            if "attention.kv_a" in qkv_map:
                for k, v in qkv_map.items():
                    if isinstance(v, str):
                        v_name = v
                    else:
                        v_name = v["name"]
                    v = {"name": v_name}
                    if k in mla_layernorm_list:
                        v["is_layernorm"] = True
                    else:
                        v["extra"] = True
                    if "fp8_keys" in args_mcore:
                        if k in mla_map_dict:
                            if k in ["attention.query_a", "attention.kv_a"]:
                                v["fp8"] = True
                                v["fp8_ignore_tp"] = True
                                new_name_mcore[mla_map_dict[k]] = v
                            elif k in ["attention.query_b", "attention.kv_b"]:
                                v["fp8"] = True
                                new_name_mcore[mla_map_dict[k]] = v
                            else:
                                new_name_mcore[mla_map_dict[k]] = v
                        else:
                            raise Exception(f"invalid key: {k}, in name_hf. {config_path=}")
                    else:
                        if k in mla_map_dict:
                            new_name_mcore[mla_map_dict[k]] = v
                        else:
                            raise Exception(f"invalid key: {k}, in name_hf. {config_path=}")
            else:
                for k, v in qkv_map.items():
                    if k in dense_map_dict:
                        new_name_mcore[dense_map_dict[k]] = v
                    else:
                        raise Exception(f"invalid key: {k}, in name_mcore. {config_path=}")
        elif key in attn_norm_dict.keys():
            new_name_mcore[attn_norm_dict[key]] = value
        elif key in moe_dense_dict.keys():
            if num_experts is not None:
                v = {}
                v["name"] = value
                if "fp8_keys" in args_mcore:
                    v["fp8"] = True
                v["extra"] = True
                new_name_mcore[moe_dense_dict[key]] = v
                if first_k_dense_replace is not None:
                    assert "moe.mlp" in name_mcore.keys(), f"moe.mlp is not found in name_mcore. please check config file. {config_path=}"
                    moe_mlp = name_mcore["moe.mlp"]
                    v = {}
                    v["name"] = f"{moe_mlp}.{value}"
                    if "fp8_keys" in args_mcore:
                        v["fp8"] = True
                    v["extra"] = True
                    new_name_mcore[key] = v
            else:
                new_name_mcore[key] = {"name": value, "extra": True}
        elif key in direct_name_dict or key in normal_keys:
            if not is_mla and key in ["input_layernorm", "attention.query_key_value", "post_attention_layernorm"]:
                if key == "input_layernorm":
                    new_name_mcore[key] = {"name": name_mcore["attention.query_key_value"], "is_layernorm": True}
                elif key == "attention.query_key_value":
                    new_name_mcore[key] = {"name": value, "extra": True}
                elif key == "post_attention_layernorm":
                    if num_experts is None:
                        new_name_mcore[key] = {"name": name_mcore["mlp.dense_h_to_4h"], "is_layernorm": True}
                    else:
                        new_name_mcore[key] = value
            else:
                if key == "attention.dense":
                    v = {}
                    v["name"] = value
                    if "fp8_keys" in args_mcore:
                        v["fp8"] = True
                    v["extra"] = True
                    new_name_mcore[key] = v
                else:
                    new_name_mcore[key] = value
        else:
            raise Exception(f"invalid key: {key}, in name_mcore. {config_path=}")

    new_args = {"common": args_common, "mcore": new_args_mcore}
    new_name_map = {"huggingface": new_name_hf, "mcore": new_name_mcore}

    new_data = {"args": new_args, "name_map": new_name_map, "torch_dtype": torch_dtype}
    if tensor_parallel_dim is not None:
        new_data["tensor_parallel_dim"] = tensor_parallel_dim

    # check new_data invalid
    for k1, v1 in data.items():
        # k1: args, name_map, torch_dtype
        if k1 in INVALID_K1:
            continue
        if not isinstance(v1, dict):
            if k1 == "torch_dtype":
                if k1 not in new_data:
                    raise Exception(f"{k1} is missing, {config_path=}")
            continue
        if k1 == "tensor_parallel_dim":
            for k2, v2 in v1.items():
                if k2 == "attention.q_norm.weight":
                    k2 = "attention.q_a_layernorm.weight"
                elif k2 == "attention.k_norm.weight":
                    k2 = "attention.kv_a_layernorm.weight"
                if k2 in ["attention.qkv_map.weight"]:
                    continue
                if k2 not in TENSOR_PARALLEL_DIM:
                    if tensor_parallel_dim is None or k2 not in tensor_parallel_dim:
                        raise Exception(f"{k1}-{k2} is missing, {config_path=}")
                if (k2 in TENSOR_PARALLEL_DIM and v2 != TENSOR_PARALLEL_DIM[k2]) or \
                        (tensor_parallel_dim is not None and k2 in tensor_parallel_dim and v2 != tensor_parallel_dim[k2]):
                    raise Exception(f"{k1}-{k2} is wrong, {config_path=}")
            continue
        for k2, v2 in v1.items():
            # k2 common huggingface, name_map
            if k1 == "args" and k2 not in ["common", "mcore"]:
                continue
            if k1 == "name_map" and k2 not in ["huggingface", "mcore"]:
                continue
            if k1 == "args" and k2 == "mcore":
                for k3, v3 in v2.items():
                    if k3 not in mcore_args and k3 not in invalid_mcore_args:
                        raise Exception(f"{k1}-{k2}-{k3} is missing, {config_path=}")
                continue
            for k3, v3 in v2.items():
                if k3 in ["attention.qkv_map", "moe.mlp"]:
                    continue
                if num_experts is not None:
                    if k3 in moe_dense_dict.keys():
                        if moe_dense_dict[k3] not in new_data[k1][k2]:
                            raise Exception(f"{k1}-{k2}-{moe_dense_dict[k3]} is missing, {config_path=}")
                        if first_k_dense_replace is not None:
                            if k3 not in new_data[k1][k2]:
                                raise Exception(f"{k1}-{k2}-{k3} is missing, {config_path=}")
                        continue
                if k3 == "attention.q_norm":
                    k3 = "attention.q_a_layernorm"
                elif k3 == "attention.k_norm":
                    k3 = "attention.kv_a_layernorm"
#                print(f"{num_experts=}, {new_data[k1][k2].keys()=}")
                if k3 not in new_data[k1][k2]:
                    print(json.dumps(new_data, indent=4))
                    raise Exception(f"{k1}-{k2}-{k3} is missing, {config_path=}")
    return new_data
    #print(json.dumps(new_data, indent=4))

def compare_dicts(d1, d2):
    diff = defaultdict(dict)
    
    for key in set(d1.keys()).union(d2.keys()):
        if key in d1 and key in d2:
            if isinstance(d1[key], dict) and isinstance(d2[key], dict):
                sub_diff = compare_dicts(d1[key], d2[key])
                if sub_diff:  # 如果子字典有差异，则记录
                    diff[key].update(sub_diff)
            elif d1[key] != d2[key]:
                diff[key]['value'] = (d1[key], d2[key])
        elif key in d1:
            diff[key]['removed'] = d1[key]
        elif key in d2:
            diff[key]['added'] = d2[key]
    
    return diff

base_path = "/Users/pengxiangyu/workspace/baidu/hac-aiacc/AIAK-Training-LLM/tools/convert_checkpoint/config"
new_base_path = "/Users/pengxiangyu/workspace/baidu/hac-aiacc/AIAK-Training-Omni/tools/convert_checkpoint/config"

import os
for filename in os.listdir(base_path):
    if filename in ["deepseek-v3-lite.json", "deepseek-v3-lite-new.json"]:
        continue
    old_path = os.path.join(base_path, filename)
    new_path = os.path.join(new_base_path, filename)
    if os.path.isfile(old_path):
        new_data = convert_json(old_path)
        if new_data is None:
            if os.path.exists(new_path):
                os.remove(new_path)
        else:
            with open(new_path, 'w', encoding='utf-8') as f:
                json.dump(new_data, f, indent=4)
        # if filename == "deepseek-v3.json":
        #     old_config_path = "/Users/pengxiangyu/workspace/baidu/hac-aiacc/AIAK-Training-Omni/tools/convert_checkpoint/config/deepseek-v3.json"
        #     with open(old_config_path, 'r', encoding='utf-8') as f:
        #         old_data = json.loads(f.read())
        #     diff = compare_dicts(old_data, new_data)
        #     print(filename)
        #     print(json.dumps(diff, indent=4))
    else:
        for vlname in os.listdir(old_path):
            vl_path = os.path.join(old_path, vlname)
            new_vl_path = os.path.join(new_path, vlname)
            os.makedirs(os.path.dirname(new_vl_path), exist_ok=True)
            if vlname in ["adapter.json", "vision-patch.json"]:
                print(f"{vl_path=}, {new_vl_path=}")
                shutil.copyfile(vl_path, new_vl_path)
            else:
                new_data = convert_json(vl_path)
                if new_data is None:
                    if os.path.exists(new_path):
                        os.remove(new_path)
                else:
                    with open(new_vl_path, 'w', encoding='utf-8') as f:
                        json.dump(new_data, f, indent=4)
