#!/usr/bin/env python3
# -*- coding: UTF-8 -*-
################################################################################
#
# Copyright (c) 2023 Baidu.com, Inc. All Rights Reserved
#
# 从 examples 目录结构自动生成 optional_configs 下的测试配置文件
# 
# 使用方法:
#   python tools/generate_optional_configs.py
#   python tools/generate_optional_configs.py --models qwen2.5 llama3
#   python tools/generate_optional_configs.py --dry_run
#
################################################################################

import os
import re
import argparse
import yaml
from pathlib import Path
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass, field


@dataclass
class ModelInfo:
    """模型信息数据类"""
    family: str          # 模型家族，如 qwen2.5, llama3
    size: str            # 模型尺寸，如 7b, 14b
    model_name: str      # 完整模型名，如 qwen2.5_7b
    script_path: str     # 原始脚本路径
    training_type: str   # 训练类型: pretrain 或 sft
    
    
class OptionalConfigGenerator:
    """可选配置生成器"""
    
    # 模型家族到配置目录的映射
    MODEL_CONFIG_MAP = {
        'qwen2.5': 'qwen2.5',
        'qwen2': 'qwen2',
        'qwen3': 'qwen3',
        'qwen3_vl': 'qwen3_vl',
        'qwen2.5_vl': 'qwen2.5vl',
        'llama2': 'llama2',
        'llama3': 'llama3',
        'llama3.1': 'llama3',
        'deepseek_v2': 'deepseek2',
        'deepseek_v3': 'deepseek3',
        'internvl2.5': 'internvl2.5',
        'internvl3.5': 'internvl3.5',
        'llavaov_1.5': 'llavaov1.5',
    }
    
    # 模型家族默认的 tokenizer 路径映射
    TOKENIZER_PATH_MAP = {
        'qwen2.5': '$pfs_path/huggingface.co/Qwen/Qwen2.5-{size}',
        'qwen2': '$pfs_path/huggingface.co/Qwen/Qwen2-{size}',
        'qwen3': '$pfs_path/huggingface.co/Qwen/Qwen3-{size}',
        'llama2': '$pfs_path/huggingface.co/meta-llama/Llama-2-{size}-hf',
        'llama3': '$pfs_path/huggingface.co/meta-llama/Llama-3-{size}',
        'llama3.1': '$pfs_path/huggingface.co/meta-llama/Llama-3.1-{size}',
    }
    
    def __init__(self, 
                 examples_dir: str,
                 output_dir: str,
                 configs_models_dir: str,
                 dry_run: bool = False):
        """
        Args:
            examples_dir: examples 目录路径
            output_dir: 输出的 optional_configs 目录路径
            configs_models_dir: configs/models 目录路径
            dry_run: 是否只打印而不实际生成文件
        """
        self.examples_dir = Path(examples_dir)
        self.output_dir = Path(output_dir)
        self.configs_models_dir = Path(configs_models_dir)
        self.dry_run = dry_run
        
    def scan_examples(self, filter_families: List[str] = None) -> List[ModelInfo]:
        """扫描 examples 目录，提取所有模型信息
        
        Args:
            filter_families: 只扫描指定的模型家族
            
        Returns:
            ModelInfo 列表
        """
        models = []
        
        for family_dir in self.examples_dir.iterdir():
            if not family_dir.is_dir():
                continue
                
            family_name = family_dir.name
            
            # 过滤模型家族
            if filter_families and family_name not in filter_families:
                continue
            
            # 扫描 pretrain 和 finetuning/sft 目录
            for train_type in ['pretrain', 'finetuning', 'sft']:
                train_dir = family_dir / train_type
                if not train_dir.exists():
                    continue
                    
                # 扫描脚本文件
                for script_file in train_dir.glob('*.sh'):
                    model_info = self._parse_script_name(
                        script_file, family_name, train_type
                    )
                    if model_info:
                        models.append(model_info)
        
        return models
    
    def _parse_script_name(self, 
                           script_path: Path, 
                           family: str,
                           train_type: str) -> Optional[ModelInfo]:
        """解析脚本文件名，提取模型信息
        
        Args:
            script_path: 脚本文件路径
            family: 模型家族名
            train_type: 训练类型目录名
            
        Returns:
            ModelInfo 或 None
        """
        filename = script_path.name
        
        # 跳过非模型脚本
        if filename.startswith('preprocess'):
            return None
        
        # 提取模型尺寸，支持多种格式:
        # pretrain_qwen2.5_7b.sh -> 7b
        # sft_internvl2_5_8b.sh -> 8b
        # pretrain_llama3_70b.sh -> 70b
        size_pattern = r'_(\d+(?:\.\d+)?b)(?:_|\.sh)'
        match = re.search(size_pattern, filename, re.IGNORECASE)
        
        if not match:
            # 尝试另一种模式，如 _a3b (MoE模型)
            moe_pattern = r'_(\d+b_a\d+b)(?:_|\.sh)'
            match = re.search(moe_pattern, filename, re.IGNORECASE)
            
        if not match:
            return None
            
        size = match.group(1).lower()
        
        # 标准化训练类型
        training_type = 'sft' if train_type in ['finetuning', 'sft'] else 'pretrain'
        
        # 构建模型名称
        # 将 . 替换为 _，如 qwen2.5 -> qwen2_5
        family_normalized = family.replace('.', '_')
        model_name = f"{family_normalized}_{size}"
        
        return ModelInfo(
            family=family,
            size=size,
            model_name=model_name,
            script_path=str(script_path),
            training_type=training_type
        )
    
    def _find_model_config_path(self, family: str, size: str) -> Optional[str]:
        """查找模型配置文件路径
        
        Args:
            family: 模型家族名
            size: 模型尺寸
            
        Returns:
            配置文件路径或 None
        """
        config_dir_name = self.MODEL_CONFIG_MAP.get(family, family.replace('.', ''))
        config_dir = self.configs_models_dir / config_dir_name
        
        if not config_dir.exists():
            return None
            
        # 尝试查找匹配的配置文件
        patterns = [
            f"*{size}*.yaml",
            f"*_{size}.yaml",
            f"*-{size}.yaml",
        ]
        
        for pattern in patterns:
            matches = list(config_dir.glob(pattern))
            if matches:
                # 返回相对路径
                return f"$aiak_training_path/configs/models/{config_dir_name}/{matches[0].name}"
        
        return None
    
    def generate_yaml_content(self, model_info: ModelInfo) -> str:
        """生成 YAML 配置内容
        
        Args:
            model_info: 模型信息
            
        Returns:
            YAML 字符串
        """
        # 查找模型配置文件
        model_config_path = self._find_model_config_path(
            model_info.family, model_info.size
        )
        
        # 获取 tokenizer 路径模板
        tokenizer_template = self.TOKENIZER_PATH_MAP.get(
            model_info.family,
            f"$pfs_path/huggingface.co/{model_info.family}/{model_info.size}"
        )
        tokenizer_path = tokenizer_template.format(size=model_info.size.upper())
        
        # 根据模型尺寸推断并行配置
        tp_size, pp_size = self._infer_parallel_config(model_info.size)
        
        config = {
            'model_name': model_info.model_name,
            'description': f'{model_info.model_name}模型自动化测试',
            'MODEL_RUNNABLE': True,
            'TOTAL_K8S_NODES': 1,
            
            # 路径配置
            'HF_CKPT_PATH': f'$pfs_path/huggingface.co/{model_info.family.title()}/{model_info.size.upper()}/',
            'TOKENIZER_PATH': tokenizer_path,
            'CHECKPOINT_PATH': '$step1_output_path',
            'TENSORBOARD_PATH': f'/workspace/tensorboard/$model_name.log',
            'MODEL_CONFIG_PATH': model_config_path or f'$aiak_training_path/configs/models/{model_info.family}/{model_info.model_name}.yaml',
            'BASELINE_PATH': '$aiak_training_path/tests/baseline',
            
            # 并行配置
            'TENSOR_MODEL_PARALLER_SIZE': tp_size,
            'PIPELINE_MODEL_PARALLER_SIZE': pp_size,
            
            # 训练参数
            'log_interval': 1,
            'micro_batch_size': 1,
            'global_batch_size': 32,
            'seq_length': 4096,
            
            # 场景配置 (基础模板)
            'scenarios': self._generate_scenarios_template(model_info),
            
            # 任务配置
            'tasks': {
                'check_correctness_task': True,
            }
        }
        
        # 转换为 YAML 字符串
        yaml_str = yaml.dump(config, default_flow_style=False, allow_unicode=True, sort_keys=False)
        
        # 添加注释头
        header = f"""# Auto-generated optional config for {model_info.model_name}
# Source: {model_info.script_path}
# Training type: {model_info.training_type}
# 
# 注意: 这是自动生成的模板配置，需要根据实际情况调整以下内容:
#   1. HF_CKPT_PATH - HuggingFace 权重路径
#   2. MODEL_CONFIG_PATH - 模型配置文件路径
#   3. scenarios 下的具体步骤配置
#
"""
        return header + yaml_str
    
    def _infer_parallel_config(self, size: str) -> Tuple[int, int]:
        """根据模型尺寸推断并行配置
        
        Args:
            size: 模型尺寸，如 '7b', '70b'
            
        Returns:
            (tensor_parallel_size, pipeline_parallel_size)
        """
        # 提取数字部分
        match = re.match(r'(\d+(?:\.\d+)?)', size)
        if not match:
            return (1, 1)
            
        size_num = float(match.group(1))
        
        # 根据模型大小推断
        if size_num <= 3:
            return (1, 1)
        elif size_num <= 8:
            return (2, 1)
        elif size_num <= 14:
            return (4, 1)
        elif size_num <= 32:
            return (4, 2)
        elif size_num <= 72:
            return (8, 2)
        else:
            return (8, 4)
    
    def _generate_scenarios_template(self, model_info: ModelInfo) -> List[Dict]:
        """生成场景配置模板
        
        Args:
            model_info: 模型信息
            
        Returns:
            场景配置列表
        """
        training_type = model_info.training_type
        
        return [
            {
                'function': {
                    training_type: {
                        'Step1': {
                            'comment': '# 权重转换步骤，根据实际模型结构配置',
                            'CONVERT_ARGS': '''
            --load_platform=huggingface
            --save_platform=mcore
            --config_file=${MODEL_CONFIG_PATH}
            --tensor_model_parallel_size=${TENSOR_MODEL_PARALLER_SIZE}
            --pipeline_model_parallel_size=${PIPELINE_MODEL_PARALLER_SIZE}
            --load_ckpt_path=${HF_CKPT_PATH}
            --save_ckpt_path=${CHECKPOINT_PATH}
            --safetensors
            --no_save_optim
            --no_load_optim'''
                        },
                        'Step2': {
                            'DATA_ARGS': f'''
            --tokenizer-type HFTokenizer
            --hf-tokenizer-path $TOKENIZER_PATH
            --data-path $DATA_PATH
            --split 99,1,0''',
                            'TRAINING_ARGS': f'''
            --training-phase {training_type}
            --seq-length ${{seq_length}}
            --max-position-embeddings 32768
            --micro-batch-size ${{micro_batch_size}}
            --global-batch-size ${{global_batch_size}}
            --lr 1.0e-4
            --min-lr 1.0e-6
            --clip-grad 1.0
            --weight-decay 0
            --optimizer adam
            --train-iters ${{train_iters}}
            --lr-decay-style cosine
            --bf16''',
                            'MODEL_PARALLEL_ARGS': '''
            --attention-backend flash
            --pipeline-model-parallel-size ${PIPELINE_MODEL_PARALLER_SIZE}
            --tensor-model-parallel-size ${TENSOR_MODEL_PARALLER_SIZE}
            --use-distributed-optimizer
            --distributed-backend nccl''',
                            'MODEL_CONFIG_ARGS': '''
            --config-file $MODEL_CONFIG_PATH''',
                            'LOGGING_ARGS': '''
            --log-interval ${log_interval}
            --tensorboard-dir $TENSORBOARD_PATH'''
                        }
                    }
                }
            }
        ]
    
    def generate(self, filter_families: List[str] = None) -> Dict[str, str]:
        """执行配置生成
        
        Args:
            filter_families: 只处理指定的模型家族
            
        Returns:
            生成的文件路径到内容的映射
        """
        models = self.scan_examples(filter_families)
        generated = {}
        
        print(f"\n{'='*60}")
        print(f"Scanning examples directory: {self.examples_dir}")
        print(f"Found {len(models)} model configurations")
        print(f"{'='*60}\n")
        
        for model_info in models:
            output_file = self.output_dir / f"{model_info.model_name}.yaml"
            yaml_content = self.generate_yaml_content(model_info)
            
            generated[str(output_file)] = yaml_content
            
            if self.dry_run:
                print(f"[DRY RUN] Would generate: {output_file}")
                print(f"  Family: {model_info.family}")
                print(f"  Size: {model_info.size}")
                print(f"  Training type: {model_info.training_type}")
                print(f"  Source: {model_info.script_path}")
                print()
            else:
                # 创建输出目录
                self.output_dir.mkdir(parents=True, exist_ok=True)
                
                # 写入文件
                with open(output_file, 'w', encoding='utf-8') as f:
                    f.write(yaml_content)
                print(f"✓ Generated: {output_file}")
        
        print(f"\n{'='*60}")
        print(f"Total: {len(generated)} configurations {'would be ' if self.dry_run else ''}generated")
        print(f"{'='*60}\n")
        
        return generated


def main():
    parser = argparse.ArgumentParser(
        description='Generate optional test configs from examples directory'
    )
    parser.add_argument(
        '--examples_dir',
        type=str,
        default='../examples',
        help='Path to examples directory'
    )
    parser.add_argument(
        '--output_dir',
        type=str,
        default='optional_configs',
        help='Output directory for generated configs'
    )
    parser.add_argument(
        '--configs_models_dir',
        type=str,
        default='../configs/models',
        help='Path to configs/models directory'
    )
    parser.add_argument(
        '--models',
        type=str,
        nargs='*',
        help='Only generate configs for specific model families (e.g., qwen2.5 llama3)'
    )
    parser.add_argument(
        '--dry_run',
        action='store_true',
        help='Only print what would be generated without creating files'
    )
    
    args = parser.parse_args()
    
    # 获取脚本所在目录作为基准
    script_dir = Path(__file__).parent.parent
    
    # 解析路径
    examples_dir = Path(args.examples_dir)
    if not examples_dir.is_absolute():
        examples_dir = script_dir / examples_dir
    
    output_dir = Path(args.output_dir)
    if not output_dir.is_absolute():
        output_dir = script_dir / output_dir
        
    configs_models_dir = Path(args.configs_models_dir)
    if not configs_models_dir.is_absolute():
        configs_models_dir = script_dir / configs_models_dir
    
    generator = OptionalConfigGenerator(
        examples_dir=str(examples_dir),
        output_dir=str(output_dir),
        configs_models_dir=str(configs_models_dir),
        dry_run=args.dry_run
    )
    
    generator.generate(filter_families=args.models)


if __name__ == '__main__':
    main()
