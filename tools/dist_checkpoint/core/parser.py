import yaml
import os
import sys
from hydra import compose, initialize_config_dir
from hydra.core.global_hydra import GlobalHydra
from omegaconf import OmegaConf

project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
sys.path.insert(0, project_root)

from tools.dist_checkpoint.config.parallel_config import ParallelConfig
from tools.convert_checkpoint.common.common_config import CommonConfig



class Parser:
    """Parse YAML configuration files into dictionaries"""

    def __init__(self, yaml_file: str = None):
        """
        Initialize Parser and automatically parse YAML file 

        Args:
            yaml_file: Path to YAML file 
        """
        assert yaml_file is not None, "yaml_file is required"
        self.data = None
        self.type = None
        self.config = None
        self.param_dict = None
        self.mappings = {}
        self.module_names = {}

        self._parse_yaml(yaml_file)
        self.config = self.data['config']
        self.param_dict = self.data['param_dict']
        
        if len(self.config) == 4: # vlm
            self.type = 'vlm'
        elif len(self.config) == 2: # llm
            self.type = 'llm'
        else:
            raise ValueError("Currently only support vlm and llm")
        model_cfg = load_config(self.config['model_config_file'])
        if self.type == 'vlm':
            config_file = self.config['model_config_file']
            with open(config_file, 'r') as f:
                self.module_names = parse_at_configs(f.readlines())
            for module_type in self.module_names.keys():
                module_convert_file = self._get_convert_file(module_type)
                cfg = self._get_module_config(module_type, module_convert_file)
                OmegaConf.set_struct(cfg, False)
                # Convert to CommonConfig
                c_config = CommonConfig()
                c_config.load_convert_data(cfg)
                update_overwrite(model_cfg, c_config, module_type)
                if module_type not in ['image_encoder', 'image_projector']:
                    self.mappings['foundation'] = c_config
                else:
                    self.mappings[module_type] = c_config
        else: # llm
            module_type = self.config['convert_file'].split('/')[-3]
            cfg = load_config(self.config['convert_file'], hydra_overrides={module_type+'@module='+self.config['model_config_file'].split("/")[-1].split(".")[0]})
            OmegaConf.set_struct(cfg, False)
            # Convert to CommonConfig
            c_config = CommonConfig()
            c_config.load_convert_data(cfg)
            update_overwrite(model_cfg, c_config, module_type)
            self.mappings['language_model'] = c_config

    
    def _get_convert_file(self, module_type):
        if module_type in ['image_encoder', 'image_projector']:
            return self.config[module_type + '_convert_file']
        else: # foundation
            return self.config['foundation_convert_file']
            
    def _get_module_config(self, module_type, module_convert_file):
        module_config = load_config(module_convert_file, hydra_overrides={module_type+'@module='+self.module_names[module_type]})
        return module_config      
            
    def _parse_yaml(self, yaml_file: str):
        """
        Parse a YAML file and store as self.data

        Args:
            yaml_file: Path to YAML file

        Returns:
            Dictionary containing the parsed YAML content
        """
        with open(yaml_file, 'r', encoding='utf-8') as f:
            self.data = yaml.safe_load(f)
        return self.data
    
    def get_language_model_cfg(self):
        return self.mappings['language_model']
    def get_foundation_model_cfg(self):
        return self.mappings['foundation']

    def get_image_encoder_cfg(self):
        return self.mappings['image_encoder']

    def get_image_projector_cfg(self):
        return self.mappings['image_projector']

    def get_parallel_config(self):
        return ParallelConfig(**self.param_dict)
    
def update_overwrite(model_cfg, module_cfg, module_type):

    if module_type != 'foundation':
        for key in module_cfg.data.module.keys():
            try:
                module_cfg.module[key] = model_cfg['model'][module_type][key]
            except:
                continue
    else:
        for key in module_cfg.data.module.keys():
            try:
                module_cfg.module[key] = model_cfg['model']['foundation'][key]
            except:
                continue

def parse_at_configs(yaml_lines):
    """
    Parse configuration lines with @ symbol in YAML to extract key-value pairs.

    Args:
        yaml_lines (list): List of YAML file content lines

    Returns:
        dict: Dictionary containing configuration values in format:
            {
                'image_encoder': 'qwen_vit_rmsnorm_test',
                'image_projector': 'mlp_adapter_test',
                'qwen': 'qwen2_5_7b_test'
                ...
            }
    """
    result = {}
    for line in yaml_lines:
        line = line.strip()
        if line.startswith('- ') and '@' in line and ':' in line:
            # Remove the leading "- "
            config_str = line[2:].strip()
            # Split key and value
            key_part, value = config_str.split(':', 1)
            key_part = key_part.strip()
            value = value.strip()
            # Extract the part before @ as the key
            if '@' in key_part:
                config_key = key_part.split('@')[0].split('/')[-1]
                result[config_key.strip()] = value
    return result

def load_config(config_path, config_name=None, hydra_overrides=None):
    """
    Load configuration using the Hydra API
    Args:
        config_path: Yaml file path
        config_name: Config file name without .yaml (required if config_path is directory)
        hydra_overrides: Optional list of override strings
    
    Returns:
        Hydra config object
    """
    # Convert to absolute path
    config_path = os.path.abspath(config_path)
    
    # get config dir and name
    config_dir = os.path.dirname(config_path)
    config_name = os.path.basename(config_path)[:-5]  # remove .yaml


    if hydra_overrides is None:
        hydra_overrides = []

    # Clear previous Hydra instance
    GlobalHydra.instance().clear()

    # Initialize and compose config
    with initialize_config_dir(config_dir=config_dir, version_base=None):
        cfg = compose(config_name=config_name, overrides=hydra_overrides)

    return cfg

def test_vlm():
    p = Parser('tools/dist_checkpoint/demo/vlm_demo.yaml')
    print(p.get_foundation_model_cfg())
    # {'module': {'_target_': 'aiak_training_omni.models.foundation.Qwen2Config', 'num_layers': 28, 'hidden_size': 3584, 'ffn_hidden_size': 18944, 'num_attention_heads': 28, 'vocab_size_in_config_file': 152064, 'make_vocab_size_divisible_by': 128, 'group_query_attention': True, 'num_query_groups': 4, 'position_embedding_type': 'rope', 'add_position_embedding': False, 'rotary_interleaved': False, 'normalization': 'RMSNorm', 'swiglu': True, 'attention_dropout': 0, 'hidden_dropout': 0, 'add_bias_linear': False, 'add_qkv_bias': True, 'qk_layernorm': False, 'untie_embeddings_and_output_weights': True, 'word_embeddings_for_head': 'lm_head', 'kv_channels': None, 'num_experts': None, 'moe_ffn_hidden_size': None, 'rotary_emb_func': 'RotaryEmbedding', 'rotary_base': 1000000, 'model_type': 'qwen'}, 'args': {'common': {'num_layers': '${module.num_layers}', 'hidden_size': '${module.hidden_size}', 'ffn_hidden_size': '${module.ffn_hidden_size}', 'num_attention_heads': '${module.num_attention_heads}', 'num_key_value_heads': '${module.num_query_groups}', 'vocab_size': '${module.vocab_size_in_config_file}'}, 'huggingface': {'architectures': ['Qwen2_5_VLForConditionalGeneration'], 'model_type': 'qwen2_5_vl', 'hidden_size': '${module.hidden_size}', 'intermediate_size': '${module.ffn_hidden_size}', 'num_attention_heads': '${module.num_attention_heads}', 'num_hidden_layers': '${module.num_layers}', 'num_key_value_heads': '${module.num_query_groups}'}, 'mcore': {'use_rotary_position_embeddings': True, 'add_embedding_padding': True, 'transpose_mlp_dense': True, 'transpose_query_key_value': True, 'untie_embeddings_and_output_weights': '${module.untie_embeddings_and_output_weights}', 'make_vocab_size_divisible_by': '${module.make_vocab_size_divisible_by}'}}, 'name_map': {'huggingface': {'word_embeddings': 'model.embed_tokens', 'transformer': 'model', 'layer_prefix': 'layers', 'input_layernorm': 'input_layernorm', 'attention.query_key_value': ['self_attn.q_proj', 'self_attn.k_proj', 'self_attn.v_proj'], 'attention.dense': 'self_attn.o_proj', 'post_attention_layernorm': 'post_attention_layernorm', 'mlp.dense_h_to_4h': ['mlp.gate_proj', 'mlp.up_proj'], 'mlp.dense_4h_to_h': 'mlp.down_proj', 'final_layernorm': 'model.norm', 'word_embeddings_for_head': '${module.word_embeddings_for_head}'}, 'mcore': {'word_embeddings': 'language_model.embedding.word_embeddings', 'transformer': 'model', 'layer_prefix': 'language_model.decoder.layers', 'input_layernorm': {'name': 'self_attention.linear_qkv', 'is_layernorm': True}, 'attention.query_key_value': {'name': 'self_attention.linear_qkv', 'extra': True}, 'attention.dense': {'name': 'self_attention.linear_proj', 'extra': True}, 'post_attention_layernorm': {'name': 'mlp.linear_fc1', 'is_layernorm': True}, 'mlp.dense_h_to_4h': {'name': 'mlp.linear_fc1', 'extra': True}, 'mlp.dense_4h_to_h': {'name': 'mlp.linear_fc2', 'extra': True}, 'final_layernorm': 'language_model.decoder.final_layernorm', 'word_embeddings_for_head': 'language_model.output_layer', 'transformer_tpl': 'model%d'}}, 'torch_dtype': 'bfloat16'}
    print(p.get_image_encoder_cfg())
    # {'module': {'_target_': 'aiak_training_omni.models.encoder.Qwen2VisionRMSNormConfig', 'num_layers': 32, 'hidden_size': 1280, 'kv_channels': 80, 'ffn_hidden_size': 3420, 'patch_size': 14, 'num_attention_heads': 16, 'num_query_groups': 16, 'image_size': [1344, 1344], 'activation_func': '${act:silu}', 'add_bias_linear': True, 'add_qkv_bias': True, 'swiglu': True, 'gated_linear_unit': True, 'position_embedding_type': 'none', 'bias_activation_fusion': False, 'hidden_dropout': 0, 'attention_dropout': 0, 'normalization': 'RMSNorm', 'apply_rope_fusion': True, 'model_type': 'qwen2_5_vit'}, 'args': {'common': {'num_layers': '${module.num_layers}', 'hidden_size': '${module.hidden_size}', 'ffn_hidden_size': '${module.ffn_hidden_size}', 'num_attention_heads': '${module.num_attention_heads}', 'num_key_value_heads': '${module.num_query_groups}'}, 'huggingface': {'out_hidden_size': '${module.hidden_size}', 'fullatt_block_indexes': [7, 15, 23, 31], 'depth': '${module.num_layers}', 'hidden_size': '${module.hidden_size}', 'intermediate_size': '${module.ffn_hidden_size}', 'num_heads': '${module.num_attention_heads}', 'patch_size': '${module.patch_size}'}, 'mcore': {'use_rotary_position_embeddings': True, 'use_distributed_optimizer': False, 'transpose_mlp_dense': True, 'transpose_query_key_value': True}}, 'name_map': {'huggingface': {'transformer': 'visual', 'layer_prefix': 'blocks', 'input_layernorm': 'norm1', 'attention.query_key_value': 'attn.qkv', 'attention.dense': 'attn.proj', 'post_attention_layernorm': 'norm2', 'mlp.dense_h_to_4h': ['mlp.gate_proj', 'mlp.up_proj'], 'mlp.dense_4h_to_h': 'mlp.down_proj'}, 'mcore': {'transformer': 'model', 'layer_prefix': 'vision_model.decoder.layers', 'input_layernorm': {'name': 'self_attention.linear_qkv', 'is_layernorm': True}, 'attention.query_key_value': {'name': 'self_attention.linear_qkv', 'extra': True}, 'attention.dense': {'name': 'self_attention.linear_proj', 'extra': True}, 'post_attention_layernorm': {'name': 'mlp.linear_fc1', 'is_layernorm': True}, 'mlp.dense_h_to_4h': {'name': 'mlp.linear_fc1', 'extra': True}, 'mlp.dense_4h_to_h': {'name': 'mlp.linear_fc2', 'extra': True}, 'transformer_tpl': 'model%d'}}, 'torch_dtype': 'bfloat16', 'vision_patch': {'vision_model.patch_embed.proj.weight': 'visual.patch_embed.proj.weight'}}
    print(p.get_image_projector_cfg())
    # {'module': {'_target_': 'aiak_training_omni.models.encoder.MLPAdapterConfig', 'normalization': 'RMSNorm', 'add_bias_linear': True, 'model_type': 'qwen2_5_vl_adapter'}, 'adapter.layernorm.weight': 'visual.merger.ln_q.weight', 'adapter.linear_fc1.weight': 'visual.merger.mlp.0.weight', 'adapter.linear_fc1.bias': 'visual.merger.mlp.0.bias', 'adapter.linear_fc2.weight': 'visual.merger.mlp.2.weight', 'adapter.linear_fc2.bias': 'visual.merger.mlp.2.bias', 'name_map': {'mcore': {'layer_prefix': 'adapter.'}}}
    print(p.get_parallel_config())
    # ParallelConfig(tp_size=4, pp_size=2, ep_size=None, etp_size=None, vpp_size=2, custom_pipeline_layers=[6, 8, 6, 8], decoder_first_pipeline_num_layers=None, decoder_last_pipeline_num_layers=None, moe_grouped_gemm=False, vpp_scheduler=None, tp_ranks=None, pp_ranks=None, ep_ranks=None, etp_ranks=None, vpp_ranks=None, safetensors=True)

def test_llm():
    p = Parser('tools/dist_checkpoint/demo/llm_demo.yaml')
    print(p.get_language_model_cfg())
    # {'module': {'_target_': 'aiak_training_omni.models.foundation.Qwen2Config', 'num_layers': 24, 'hidden_size': 896, 'ffn_hidden_size': 4864, 'num_attention_heads': 14, 'vocab_size_in_config_file': 151936, 'make_vocab_size_divisible_by': 128, 'group_query_attention': True, 'num_query_groups': 2, 'position_embedding_type': 'rope', 'add_position_embedding': False, 'rotary_interleaved': False, 'normalization': 'RMSNorm', 'swiglu': True, 'attention_dropout': 0, 'hidden_dropout': 0, 'add_bias_linear': False, 'add_qkv_bias': True, 'qk_layernorm': False, 'untie_embeddings_and_output_weights': False, 'word_embeddings_for_head': None, 'kv_channels': None, 'num_experts': None, 'moe_ffn_hidden_size': None, 'rotary_base': 1000000, 'rotary_emb_func': 'RotaryEmbedding', 'model_type': 'qwen'}, 'args': {'common': {'num_layers': '${module.num_layers}', 'hidden_size': '${module.hidden_size}', 'ffn_hidden_size': '${module.ffn_hidden_size}', 'num_attention_heads': '${module.num_attention_heads}', 'num_key_value_heads': '${module.num_query_groups}', 'vocab_size': '${module.vocab_size_in_config_file}'}, 'huggingface': {'architectures': ['Qwen2_5_VLForConditionalGeneration'], 'model_type': 'qwen2_5_vl', 'hidden_size': '${module.hidden_size}', 'intermediate_size': '${module.ffn_hidden_size}', 'num_attention_heads': '${module.num_attention_heads}', 'num_hidden_layers': '${module.num_layers}', 'num_key_value_heads': '${module.num_query_groups}'}, 'mcore': {'use_rotary_position_embeddings': True, 'add_embedding_padding': True, 'transpose_mlp_dense': True, 'transpose_query_key_value': True, 'untie_embeddings_and_output_weights': '${module.untie_embeddings_and_output_weights}', 'make_vocab_size_divisible_by': '${module.make_vocab_size_divisible_by}'}}, 'name_map': {'huggingface': {'word_embeddings': 'model.embed_tokens', 'transformer': 'model', 'layer_prefix': 'layers', 'input_layernorm': 'input_layernorm', 'attention.query_key_value': ['self_attn.q_proj', 'self_attn.k_proj', 'self_attn.v_proj'], 'attention.dense': 'self_attn.o_proj', 'post_attention_layernorm': 'post_attention_layernorm', 'mlp.dense_h_to_4h': ['mlp.gate_proj', 'mlp.up_proj'], 'mlp.dense_4h_to_h': 'mlp.down_proj', 'final_layernorm': 'model.norm', 'word_embeddings_for_head': '${module.word_embeddings_for_head}'}, 'mcore': {'word_embeddings': 'embedding.word_embeddings', 'word_position_embeddings': 'model.embedding.position_embeddings', 'transformer': 'model', 'layer_prefix': 'decoder.layers', 'input_layernorm': {'name': 'self_attention.linear_qkv', 'is_layernorm': True}, 'attention.query_key_value': {'name': 'self_attention.linear_qkv', 'extra': True}, 'attention.dense': {'name': 'self_attention.linear_proj', 'extra': True}, 'post_attention_layernorm': {'name': 'mlp.linear_fc1', 'is_layernorm': True}, 'mlp.dense_h_to_4h': {'name': 'mlp.linear_fc1', 'extra': True}, 'mlp.dense_4h_to_h': {'name': 'mlp.linear_fc2', 'extra': True}, 'final_layernorm': 'decoder.final_layernorm', 'word_embeddings_for_head': 'output_layer', 'transformer_tpl': 'model%d'}}, 'torch_dtype': 'bfloat16'}
    print(p.get_parallel_config())
    # ParallelConfig(tp_size=2, pp_size=2, ep_size=None, etp_size=None, vpp_size=None, custom_pipeline_layers=None, decoder_first_pipeline_num_layers=None, decoder_last_pipeline_num_layers=None, moe_grouped_gemm=False, vpp_scheduler=None, tp_ranks=None, pp_ranks=None, ep_ranks=None, etp_ranks=None, vpp_ranks=None, safetensors=True)

if __name__ == "__main__":
    test_vlm()
    test_llm()
