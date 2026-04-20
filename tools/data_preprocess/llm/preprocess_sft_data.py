# Copyright 2026 The LoongForge Authors.
# SPDX-License-Identifier: Apache-2.0

"""preprocess sft data"""

import argparse

from transformers import AutoProcessor
from datasets import DatasetDict

from megatron.core.datasets.utils import get_blend_from_list, Split

from loongforge.data.sft_dataset import SFTDatasetConfig, SFTDataset
from loongforge.data import ChatTemplate, get_support_templates
from loongforge.tokenizer import build_tokenizer
from loongforge.utils import constants
from loongforge.utils.utils import get_default_sft_dataset_config
from loongforge.train.sft.utils import get_dataset_blend_from_list


def build_sft_dataset(args):
    """build sft dataset"""
    tokenizer = build_tokenizer(args, chat_template=args.template)
    processor = AutoProcessor.from_pretrained(args.hf_tokenizer_path, trust_remote_code=True)
    if args.image_resolution:
        setattr(processor, "image_resolution", args.image_resolution)

    config = SFTDatasetConfig(
        random_seed=args.seed,
        sequence_length=args.seq_length, # max sequence length
        enable_discard_sample=args.enable_discard_sample,
        blend=get_blend_from_list([args.input]),
        split=args.split,
        path_to_cache=None,
        tokenizer=tokenizer,
        dataset=get_dataset_blend_from_list([args.sft_dataset]),
        dataset_config_file=args.sft_dataset_config,
        streaming=False,
        chat_template=args.template,
        processor=processor,
        num_preprocess_workers=args.workers,
        train_on_prompt=args.train_on_prompt,
        ignore_index=constants.IGNORE_INDEX,
        eod_mask_loss=args.eod_mask_loss,
        history_mask_loss=args.history_mask_loss,
        is_tokenized=False,
        packing=args.packing_sft_data,
        sort_batch=args.sft_sort_batch,
        packing_buffer_size=args.packing_buffer_size,
        context_parallel_size=args.context_parallel_size,
    )
    
    dataset = SFTDataset(args.sft_dataset, args.input, config)
    split_dataset = dataset.split(config.split_matrix)

    num_samples = 0

    dataset_dict = DatasetDict()
    for i, split in enumerate(Split):
        if split_dataset[i] is not None:
            dataset_dict[split.name] = split_dataset[i]
            num_samples += len(split_dataset[i])

    dataset_dict.save_to_disk(args.output_path)

    print(f">>> Saved preprocessed dataset to {args.output_path} with total samples: {num_samples}")
    for i, split in enumerate(Split):
        print(f">>> {split.name} samples: {len(dataset_dict[split.name]) if split.name in dataset_dict else 0 }")

    print(f"NOTE: Please run sft with `--data-path {args.output_path}` and `--is-tokenized-data`")


def _add_arguments(parser: argparse.ArgumentParser):
    """Add arguments"""
    group = parser.add_argument_group(title='input-data')
    group.add_argument('--input', type=str, required=True, help='Path to input JSON')

    group.add_argument('--seq-length', type=int, default=None, help='max sequence length')

    group.add_argument('--output-path', type=str, required=True,
                       help='Output directory where the processed dataset will be saved')
    
    group.add_argument('--split', type=str, default="100,0,0",
                    help='Comma-separated list of proportions for training,'
                    ' validation, and test split. For example the split '
                    '`90,5,5` will use 90%% of data for training, 5%% for '
                    'validation and 5%% for test.')

    group.add_argument("--context-parallel-size",
                       type=int, default=None,
                       help="If packing is enabled, and context-parallel is enabled during the training phase, "
                            "it is necessary to set the corresponding context_parallel_size "
                            "to correctly pad the data.")

    group.add_argument('--tokenizer-type', type=str, default='HFTokenizer',
                       choices=['HFTokenizer'],
                       help='What type of tokenizer to use.')
    
    group.add_argument('--hf-tokenizer-path', type=str, required=True,
                       help='HuggingFace tokenizer path: '
                            '1) A string, the *model id* of a predefined tokenizer hosted inside a model repo '
                            'on huggingface.co'
                            '2) A path to a *directory* containing vocabulary files required by the tokenizer')

    group.add_argument('--use-fast-tokenizer', action='store_true',
                       help='Whether to use the fast tokenizer when --tokenizer-type=HFTokenizer. Default: False')
    
    group.add_argument('--split-special-tokens', action='store_true',
                       help="Whether the special tokens should be split during the tokenization process " 
                            "when --tokenizer-type=HFTokenizer. Default: False")

    group.add_argument("--additional-special-tokens",
                       type=str,
                       default=None,
                       help="Additional special tokens to add to the tokenizer. Use commas to separate multiple tokens")

    group = parser.add_argument_group(title='extra-sft')
    group.add_argument('--chat-template', type=str, required=True,
                       choices=get_support_templates(),
                       help='The template to apply to instruction data.')

    group.add_argument('--sft-dataset-config', type=str, default=None,
                       help="A json file that contains the dataset configuration."
                            "default: configs/dataset_config.jsoin")

    group.add_argument('--sft-dataset', type=str, default="default",
                       help='the dataset name should be defined in the dataset config file (--sft-dataset-config)')

    group.add_argument("--packing-sft-data",
                       action='store_true',
                       help="Whether to pack multiple sft data into one.")

    group.add_argument("--packing-buffer-size",
                       type=int,
                       default=10000,
                       help="Perform packing in batches, deciding how many samples each batch contains;"
                            "if the --sft-sort-batch option is enabled, the samples will be sorted after packing.")

    group.add_argument('--sft-sort-batch',
                       action='store_true',
                       help='Sort the entire dataset from smallest to largest; '
                            'if the --packing-sft-data option is enabled, sort the data after packing. Default: False')

    group.add_argument('--image-resolution', type=int, help='Resolution of image inputs')

    group.add_argument('--train-on-prompt', action='store_true',
                       help='Whether compute loss on prompt. Default: False')

    group.add_argument('--history-mask-loss',
                       action='store_true',
                       help='Only compute loss on last turn response, instead of full history. Default: False')

    group.add_argument('--eod-mask-loss', action='store_true',
                       help='Mask loss for the end of document tokens.')

    group.add_argument("--enable-discard-sample",
                       action='store_true',
                       help="Whether to discard sample when its length is greater than seq-length.")

    group = parser.add_argument_group(title='preprocess-runtime')
    group.add_argument('--workers', type=int, required=True, help='Number of worker processes to launch.')
    return parser


def parse_args():
    """arguments"""
    parser = argparse.ArgumentParser()
    _add_arguments(parser)
    
    args = parser.parse_args()

    # some default/dummy values
    args.rank = 0
    args.training_phase = constants.TrainingPhase.SFT
    args.make_vocab_size_divisible_by = 128
    args.tensor_model_parallel_size = 1
    args.vocab_extra_ids = 0
    args.seed = 42
    args.padding_side = "right"
    
    if args.sft_dataset_config is None:
        args.sft_dataset_config = get_default_sft_dataset_config()
        assert args.sft_dataset_config is not None, "No default sft dataset config found, please specify one"
    
    assert args.chat_template is not None, "chat_template not specified"
    template = ChatTemplate.from_name(args.chat_template)
    assert template is not None, f"chat_template {args.chat_template} not supported."
    args.template = template
 
    args.variable_seq_lengths = True
    if args.packing_sft_data:
        print(f"Enable to pack multiple sft data with max length {args.seq_length} ...")

    if args.train_on_prompt and args.history_mask_loss:
        raise ValueError('--train-on-prompt and --history-mask-loss cannot both be True at the same time')
            
    return args


def main():
    """main function"""
    args = parse_args()
    build_sft_dataset(args)


if __name__ == '__main__':
    main()
