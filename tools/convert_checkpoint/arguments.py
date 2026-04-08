# Copyright 2026 The LoongForge Authors.
# SPDX-License-Identifier: Apache-2.0

"""Argument definitions for checkpoint conversion tools."""

import argparse

_GLOBAL_ARGS = None


def set_args(args):
    global _GLOBAL_ARGS
    _GLOBAL_ARGS = args


def parse_args(title=None):
    global _GLOBAL_ARGS
    if _GLOBAL_ARGS is not None:
        return _GLOBAL_ARGS
    """Parse all arguments."""
    parser = argparse.ArgumentParser(description='LoongForge-Tool Arguments',
                                     allow_abbrev=False)
    _add_checkpoint_args(parser)
    _add_common_args(parser)
    _add_huggingface_args(parser)
    _add_megatron_args(parser)

    args = parser.parse_args()
    if args.convert_to_fp8:
        assert args.load_platform == 'mcore' and args.save_platform == 'huggingface', \
                "convert_to_fp8 only support mcore to huggingface"
    if title is None:
        _GLOBAL_ARGS = args
        return _GLOBAL_ARGS
    else:
        for group in parser._action_groups:
            if group.title == title:
                group_dict={item.dest: getattr(args, item.dest, None) for item in group._group_actions}
                _GLOBAL_ARGS = argparse.Namespace(**group_dict)
                return _GLOBAL_ARGS
        _GLOBAL_ARGS = argparse.Namespace()
        return _GLOBAL_ARGS


def _add_checkpoint_args(parser):
    group = parser.add_argument_group(title='checkpoint')

    group.add_argument('--load_platform', type=str, default=None,
                       choices=['huggingface', 'mcore'])
    group.add_argument('--save_platform', type=str, default=None,
                       choices=['huggingface', 'mcore'])
    group.add_argument('--load_ckpt_path', type=str, default=None,
                       help='path to load checkpoint')
    group.add_argument('--save_ckpt_path', type=str, default=None,
                       help='path to save checkpoint')
    group.add_argument('--common_config_path', type=str, default=None,
                       help='path to common config')
    group.add_argument("--megatron_path", type=str, default=None,
                       help="Base directory of Megatron repository")
    group.add_argument('--no_load_optim', action='store_true',
                       help='do not convert optimizer')
    group.add_argument('--no_save_optim', action='store_true',
                       help='do not save optimizer')
    group.add_argument('--model_type_custom', type=str, default=None,
                       help='custom model type')
    group.add_argument('--safetensors', action='store_true',
                       help='Use [safetensors](https://huggingface.co/docs/safetensors).')
    group.add_argument('--convert_to_fp8', action='store_true',
                       help='Convert float16 weights to fp8')
    group.add_argument('--quant_method', type=str, default='te', choices=['te', 'pt'],
                       help='The quantization method to use. Choices: [te, pt].')

    # Arguments for defining manner of converting FP8 checkpoint
    group.add_argument('--fp8_force_no_requant', action='store_true',
                       help=("If enabled, in converting FP8 checkpoint, skip the `dequantize + re-quantize`, "
                             "directly chunk/concate the quantized data.")
    )
    group.add_argument('--force_pow_2_scales', action='store_true',
                       help=("Define whether to force destination checkpoint's scale to be power-of-two.")
    )
    group.add_argument('--amax_epsilon', type=float, default=0.0,
                       help=("Epsilon value added to the amax calculation to avoid divised by zero "
                             "when converting to FP8. Only used in Transformer Engine FP8 conversion.")
    )

    group.add_argument('--pretrain_as_fp8', action='store_true',
                       help='Run pretrain as fp8, only used for hf to mcore when '
                       'saved checkpoint is bf16 and pretrain as fp8')

    group.add_argument('--fp8_quant_transfer_type', type=str, default="float32", choices=["float32", "bfloat16"],
                       help='The transfer dtype when convert from hf fp8 to mcore fp8')
    group.add_argument('--distributed_convert', action='store_true',
                       help='Convert checkpoint in distributed mode')

    group.add_argument('--config_file', type=str, help="Config file for model configuration.")
    group.add_argument('--convert_file', type=str, help="Convert file for checkpoint conversion.")
    # group.add_argument('--config_name', type=str, help="Config file name for model configuration.")
    # group.add_argument('--module', type=str, default="language", help="Module type, default: language", choices=["language", "vit"])
    group.add_argument('--mtp_num_layers', type=int, default=None,
                       help='Number of Multi-Token Prediction (MTP) Layers.'
                       'MTP extends the prediction scope to multiple future tokens at each position.'
                       'This MTP implementation sequentially predict additional tokens '
                       'by using D sequential modules to predict D additional tokens.')
    group.add_argument('--load_lora_ckpt_path', type=str, default=None, help='path to load lora checkpoint')
    group.add_argument('--lora_alpha', type=int, help="Lora alpha for LoRA fine tuning.")
    group.add_argument('--lora_dim', type=int, help="Lora dim for LoRA fine tuning.")
    group.add_argument('--adapter_convert_file', type=str, default=None, help="Convert file for adapter.")
    group.add_argument('--vision_patch_convert_file', type=str, default=None, help="Convert file for vision_patch.")
    group.add_argument("--encoder_tensor_model_parallel_size", type=int, default=None, help="Tensor parallel size for encoder.")
    group.add_argument(
        '--enable-full-hetero-dp',
        default=False,
        action="store_true",
        help="Enable full heterogeneous data parallelism. Default: False"
    )
    group.add_argument('--hf_checkpoint_device', type=str, default="cpu", help='Device used to load and save HF checkpoint, default is cpu')

def _add_common_args(parser):
    group = parser.add_argument_group(title='common')

    group.add_argument('--torch_dtype', type=str, choices=["float16", "float32", "bfloat16"],
                       help='target torch dtype')
    group.add_argument('--vocab_size', type=int, default=None, help='vocab size')
    group.add_argument('--vpp-scheduler', type=str, default=None,
                       choices=["dualpipev"],
                       help='By default, the original 1F1B scheduling method is used. When selecting DualPipeV, '
                            'the effect can be referred to at https://hackmd.io/@ufotalent/r1lVXsa9Jg')
    group.add_argument('--num-virtual-stages-per-pipeline-rank', type=int, default=None,
                       help='Number of virtual pipeline stages per pipeline parallelism rank')
    group.add_argument('--decoder-first-pipeline-num-layers',
                       type=int, default=None,
                       help=('The number of transformer layers on the first pipeline stage of the decoder. '
                       'Default None is even split of transformer layers across all pipeline stages'))
    group.add_argument('--decoder-last-pipeline-num-layers',
                       type=int, default=None,
                       help=('The number of transformer layers on the last pipeline stage of the decoder. '
                       'Default None is even split of transformer layers across all pipeline stages'))
    group.add_argument('--vit_in_first_virtual_stage_only', action='store_true',
                       help=('When virtual pipeline is enabled, VIT is only in the first virtual stage,'
                       'ensuring the integrity of VIT when transitioning from mcore to hf.'
                       'When VPP is enabled, it must be enabled, default is False'))

def _add_megatron_args(parser):
    """
    Add MegaTron related parameters to the parser.

    Args:
        parser (ArgumentParser, str): ArgumentParser object or parameter string, used to add MegaTron related parameters.

    Returns:
        None, void: No return value, directly modify the passed ArgumentParser object.
    """
    group = parser.add_argument_group(title='megatron')

    group.add_argument('--use_distributed_optimizer', action='store_true',
                       help='use distributed optimizer')
    group.add_argument('--tensor_model_parallel_size', type=int, default=1,
                       help='target tensor model parallel size')
    group.add_argument('--pipeline_model_parallel_size', type=int, default=1,
                       help='target pipeline model parallel size')
    group.add_argument('--data_parallel_size', type=int, default=1,
                       help='target data parallel size')
    group.add_argument('--expert_parallel_size', type=int, default=None,
                       help='target expert parallel size')
    group.add_argument('--expert_tensor_parallel_size', type=int, default=None,
                       help='Degree of expert model parallelism. Default is None, '
                       'which will be set to the value of --tensor-model-paralle-size.')
    group.add_argument('--pad_vocab_size_to', type=int, default=None,
                       help='Pad the vocab size to this value.'
                            'This value must be greater than the initial size of the tokenizer'
                            ', needs to be divisible by TP size and `make-vocab-size-divisible-by`.')
    group.add_argument('--add_embedding_padding', type=bool, default=True,
                       help='Whether to add embedding padding.')
    group.add_argument('--make_vocab_size_divisible_by', type=int, default=128,
                       help='Pad the vocab size to this value.'
                            'This value must be greater than the initial size of the tokenizer'
                            ', needs to be divisible by TP size and `make-vocab-size-divisible-by`.')
    group.add_argument('--custom_pipeline_layers', type=str, default=None,
                       help='add by loongforge for pp layer imbalance.For example 19,20,20,21.'
                       '19 for stage0 layers, 20 for stage1 layers...')
    group.add_argument('--pipeline_model_parallel_layout',
                       type=str, default=None,
                       help=('Note: This argument will be converted into --custom_pipeline_layers.'
                             'A string that describes a custom pipeline model parallel layout. '
                             'e.g., "E|(t|)*3,m|m||L". E, L, t, m denotes embedding, loss, transformer '
                             'decoder layer, and mtp layer, respectively. Stages are split by "|". '
                             'Replicated stages or layers can be described with multiplication. '
                             'Commas can be used cosmetically. '
                             'Default None is not using this argument to set the layout.'))
    group.add_argument('--num_layers_per_virtual_pipeline_stage', type=int, default=None,
                       help='Number of layers per virtual pipeline stage')
    group.add_argument('--transformer_impl', default='transformer_engine',
                       choices=['local', 'transformer_engine'],
                       help='Which Transformer implementation to use when load or save mcore checkpoint.'
                            'Only support `transformer_engine` now.')
    group.add_argument('--num_experts', type=int, default=None,
                       help='Number of Experts in MoE (None means no MoE)')
    group.add_argument('--checkpoint-format', type=str, default=None,
                       help='hf checkpoint format end with safetensors')
    group.add_argument('--max_workers', type=int, default=1,
                       help='thread for checkpoint converting')
    group.add_argument('--no-te', '--input-layernorm-not-in-attn', action='store_true',
                       help='SelfAttn does not contain input_layernorm in mcore.')
    group.add_argument('--moe-grouped-gemm', action='store_true',
                       help='use grouped gemm in moe')
    group.add_argument('--resume-convert', action='store_true',
                       help='resume checkpoint converting when failed')
    group.add_argument('--cache-path', type=str, default=None,
                       help='cache path used during conversion')
    group.add_argument('--layer-for-test', type=str, default=None,
                       help='get specific layer from checkpoint for test')
    group.add_argument('--num-experts-for-test', type=int, default=None,
                       help='Number of Experts in MoE for test')
    group.add_argument('--sub-num-layers-for-save', type=int, default=None,
                       help='number of layers for saving each time')
    group.add_argument('--save-sub-checkpoint-by-pp', action='store_true',
                       help='save sub checkpoints by pipeline parallel')
    group.add_argument('--sub_file_tag', type=int, default=None,
                       help='sub file index when saving huggingface checkpoint')


def _add_huggingface_args(parser):
    group = parser.add_argument_group(title='huggingface')
    pass
