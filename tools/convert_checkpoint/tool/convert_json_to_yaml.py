#!/usr/bin/env python3
import json
import os
import re

from omegaconf import OmegaConf

current_path = os.path.dirname(os.path.abspath(__file__))

base_path = f"{current_path}/../config"
new_base_path = f"{current_path}/../../../configs/models"

def get_module_name(filename):
    dot_index = filename.rfind('.')
    if dot_index != -1:
        filename = filename[:dot_index]
    module_name = filename.replace("-", "_").replace(".", "_")
    return module_name

def get_base_module_name(filename):
    patterns = [
        r'^a(\d+|\d+\.\d+)b$',  # 格式：a数字b
        r'^(\d+|\d+\.\d+)b$',   # 格式：数字b
        r'^(\d+|\d+\.\d+)m$'    # 格式：数字m
    ]
    dot_index = filename.rfind('.')
    if dot_index != -1:
        filename = filename[:dot_index]
    s = filename.replace("_", ".").replace("-", "_")
    parts = s.split('_')
    for i in range(len(parts)):
        if any(re.match(pattern, parts[i]) for pattern in patterns):
            return '_'.join(parts[0:i])
    return s

def is_module_name(filename):
    patterns = [
        r'^a(\d+|\d+\.\d+)b$',  # 格式：a数字b
        r'^(\d+|\d+\.\d+)b$',   # 格式：数字b
        r'^(\d+|\d+\.\d+)m$'    # 格式：数字m
    ]
    dot_index = filename.rfind('.')
    if dot_index != -1:
        filename = filename[:dot_index]
    s = filename.replace("_", ".").replace("-", "_")
    parts = s.split('_')
    for i in range(len(parts)):
        if any(re.match(pattern, parts[i]) for pattern in patterns):
            return True
    return False

def is_moe(filename):
    patterns = [
        r'^a(\d+|\d+\.\d+)b$',  # 格式：a数字b
    ]
    dot_index = filename.rfind('.')
    if dot_index != -1:
        filename = filename[:dot_index]
    s = filename.replace("_", ".").replace("-", "_")
    parts = s.split('_')
    for i in range(len(parts)):
        if any(re.match(pattern, parts[i]) for pattern in patterns):
            return True
    return False

def remove_quotes(file_path):
    with open(file_path, 'r', encoding='utf-8') as file:
        content = file.read()
    content = content.replace("'", "").replace('"', "")
    with open(file_path, 'w', encoding='utf-8') as file:
        file.write(content)

def merge_files(file_list, output_path):
    with open(output_path, 'w', encoding='utf-8') as outfile:
        for file_path in file_list:
            if os.path.exists(file_path):
                with open(file_path, 'r', encoding='utf-8') as infile:
                    outfile.write(infile.read() + '\n')
                os.remove(file_path)
            else:
                print(f"Warning: {file_path} not found and skipped.")

def compare_configs(file_path, old_config):
    # 加载文件配置
    new_config = OmegaConf.load(file_path)

    # 比较两个配置是否完全一致
    ret = OmegaConf.is_config_equal(new_config, old_config)
    if ret:
        return True
    else:
        raise Exception("Configs not equal, please check the new converted yaml file: {old_config=}, {new_config=}")

def convert_adapter_yaml(old_path, convert_config_path, module_name):
    os.makedirs(os.path.dirname(convert_config_path), exist_ok=True)

    with open(old_path, 'r', encoding='utf-8') as f:
        json_data = json.loads(f.read())

    convert_config = {}
    convert_config["hydra"] = {"searchpath": ["file:///workspace/AIAK-Training-Omni/configs/models/"]}
    convert_config["defaults"] = [f"image_projector@module: ???", "_self_"]
    for k1, v1 in json_data.items():
        convert_config[k1] = v1
    convert_config["name_map"] = {"mcore": {"layer_prefix": "adapter."}}
    # if os.path.exists(convert_config_path) and compare_configs(convert_config_path, convert_config):
    #     return
    OmegaConf.save(convert_config, convert_config_path)
    print(f"[{module_name}] Save converted yaml to {convert_config_path}")

def convert_vision_yaml(vision_model_path, vision_patch_path, vision_config_path, vision_data_path, module_name):
    os.makedirs(os.path.dirname(vision_config_path), exist_ok=True)

    with open(vision_model_path, 'r', encoding='utf-8') as f:
        model_data = json.loads(f.read())

    with open(vision_patch_path, 'r', encoding='utf-8') as f:
        vision_data = json.loads(f.read())

    if os.path.exists(vision_config_path):
        ori_yaml_config = OmegaConf.load(vision_config_path)
    else:
        ori_yaml_config = None

    convert_config = {}
    data_config = {}
    convert_config["hydra"] = {"searchpath": ["file:///workspace/AIAK-Training-Omni/configs/models/"]}
    convert_config["defaults"] = [f"image_encoder@module: ???", "_self_"]
    for k1, v1 in model_data.items():
        if k1 == "args":
            if ori_yaml_config is None:
                convert_config[k1] = {}
            else:
                convert_config[k1] = ori_yaml_config[k1]
            for k2, v2 in v1.items():
                # k2: common, huggingface, mcore
                convert_config[k1][k2] = {}
                for k3, v3 in v2.items():
                    if ori_yaml_config is None:
                        if k2 == "mcore":
                            convert_config[k1][k2][k3] = v3
                        else:
                            convert_config[k1][k2][k3] = f"${{module.{k3}}}"
                    data_config[k3] = v3
        else:
            convert_config[k1] = v1
    for k1, v1 in vision_data.items():
        convert_config[k1] = v1
    OmegaConf.save(convert_config, vision_config_path)
    if not os.path.exists(vision_data_path):
        OmegaConf.save(data_config, vision_data_path)
    print(f"[{module_name}] Save converted yaml to {vision_config_path}")
    print(f"[{module_name}] Save data yaml to {vision_data_path}")


def convert_base_yaml(old_path, convert_config_path, data_path, model_base_name, module_name):
    os.makedirs(os.path.dirname(convert_config_path), exist_ok=True)
    os.makedirs(os.path.dirname(data_path), exist_ok=True)

    with open(old_path, 'r', encoding='utf-8') as f:
        json_data = json.loads(f.read())
    if os.path.exists(convert_config_path):
        ori_yaml_config = OmegaConf.load(convert_config_path)
    else:
        ori_yaml_config = None

    data_config = {}
    hydra_convert_config = {}
    defaults_convert_config = {}
    convert_config = {}
    hydra_config_path = f"{convert_config_path}_hydra"
    defaults_config_path = f"{convert_config_path}_defaults"
    json_config_path = f"{convert_config_path}_json"
    hydra_convert_config["hydra"] = {"searchpath": ["file:///workspace/AIAK-Training-Omni/configs/models/"]}
    OmegaConf.save(hydra_convert_config, hydra_config_path)
    defaults_convert_config["defaults"] = [f"{model_base_name}@module: ???", "_self_"]
    OmegaConf.save(defaults_convert_config, defaults_config_path, resolve=True)
    remove_quotes(defaults_config_path)
    for k1, v1 in json_data.items():
        if k1 == "args":
            if ori_yaml_config is None:
                convert_config[k1] = {}
            else:
                convert_config[k1] = ori_yaml_config[k1]
            for k2, v2 in v1.items():
                # k2: common, huggingface, mcore
                if k2 not in convert_config[k1]:
                    convert_config[k1][k2] = {}
                for k3, v3 in v2.items():
                    if ori_yaml_config is None:
                        if k2 == "mcore":
                            convert_config[k1][k2][k3] = v3
                        else:
                            convert_config[k1][k2][k3] = f"${{module.{k3}}}"
                    data_config[k3] = v3
        else:
            convert_config[k1] = v1
    OmegaConf.save(convert_config, json_config_path)
    merge_files([hydra_config_path, defaults_config_path, json_config_path], convert_config_path)
    if not os.path.exists(data_path):
        OmegaConf.save(data_config, data_path)
    print(f"[{module_name}] Save converted yaml to {convert_config_path}")
    print(f"[{module_name}] Save data yaml to {data_path}")


rename_base_module_dict = {
    "qwen3_coder": "qwen3"
}

def rename_base_module(base_module_name):
    for k, v in rename_base_module_dict.items():
        if k in base_module_name:
            return base_module_name.replace(k, v)
    return base_module_name


for filename in os.listdir(base_path):
    module_name = get_module_name(filename)
    dir_is_module_name = is_module_name(filename)
    base_module_name = get_base_module_name(filename)
    is_vl = base_module_name.endswith("vl")
    base_module_name = rename_base_module(base_module_name)
    base_convert_name = f"{base_module_name.replace(".", "_")}_moe" if is_moe(filename) else base_module_name.replace(".", "_")
    old_path = os.path.join(base_path, filename)

    if os.path.isfile(old_path):
        config_path = os.path.join(new_base_path, base_module_name, "ckpt_convert", f"{base_convert_name}_convert.yaml")
        data_path = os.path.join(new_base_path, base_module_name, f"{module_name}.yaml")
        convert_base_yaml(old_path, config_path, data_path, base_module_name, module_name)
        continue
    else:
        adapter_path = os.path.join(old_path, "adapter.json")
        base_convert_name = f"{base_module_name.replace(".", "_")}_mlp_adapter_convert.yaml"
        adapter_config_path = os.path.join(new_base_path, "image_projector", "ckpt_convert", base_convert_name)
        convert_adapter_yaml(adapter_path, adapter_config_path, module_name)

        # vit begin
        vision_model_path = os.path.join(old_path, "vision-model.json")
        vision_patch_path = os.path.join(old_path, "vision-patch.json")
        base_convert_name = f"{base_module_name.replace(".", "_")}_vit_convert.yaml"
        vision_config_path = os.path.join(new_base_path, "image_encoder", "ckpt_convert", base_convert_name)
        if os.path.exists(vision_model_path):
            vision_data_path = os.path.join(new_base_path, "image_encoder", f"{module_name}_vit.yaml")
            convert_vision_yaml(vision_model_path, vision_patch_path, vision_config_path, vision_data_path, module_name)
        else:
            for vlname in os.listdir(old_path):
                if not vlname.startswith("vision-model"):
                    continue
                vl_path = os.path.join(old_path, vlname)
                module_name = get_module_name(vlname)
                vision_data_path = os.path.join(new_base_path, "image_encoder", f"{module_name}_vit.yaml")
                convert_vision_yaml(vl_path, vision_patch_path, vision_config_path, vision_data_path, module_name)
        # vit end

        # base begin
        for vlname in os.listdir(old_path):
            vl_path = os.path.join(old_path, vlname)
            if not vlname.endswith(".json") or vlname in ["adapter.json", "vision-patch.json", "vision-model.json"] or vlname.startswith("vision-model"):
                continue
            if not dir_is_module_name:
                module_name = get_module_name(vlname)
            if is_vl:
                base_module_name = get_base_module_name(vlname)
                module_name = module_name.replace("_vl", "")
            base_convert_name = f"{base_module_name.replace(".", "_")}_convert.yaml"
            config_path = os.path.join(new_base_path, base_module_name, "ckpt_convert", base_convert_name)
            data_path = os.path.join(new_base_path, base_module_name, f"{module_name}.yaml")
            convert_base_yaml(vl_path, config_path, data_path, base_module_name, module_name)
        # base end
