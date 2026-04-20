# Copyright 2026 The LoongForge Authors.
# SPDX-License-Identifier: Apache-2.0

""" Convert dataset into WebDataset (WDS) format """

import argparse
import uuid
import json
import os
import yaml
import webdataset as wds
from tqdm import tqdm
from megatron.energon.epathlib import EPath
from megatron.energon.flavors import BaseWebdatasetFactory
from megatron.energon.flavors.webdataset import MAIN_FOLDER_NAME
from megatron.energon.flavors.webdataset.prepare import WebdatasetPreparator
from megatron.energon.flavors.webdataset.structs import ShardInfo, WebdatasetInfo, WebdatasetSplits


def construct_sample(args, vision, paths, index, entry):
    """ construct webdataset sample """
    assert vision == 'image' or vision == 'video'
    directory = args.image_dir if vision == 'image' else args.video_dir

    vision_data = {}
    vision_name = []

    # paths can be ["a/b.mp4", "c/d.mp4"] or [["a/b.mp4", "c/d.mp4"]].
    # Without flattening the nested list, a single string would be iterated char-by-char,
    if len(paths) == 1 and isinstance(paths[0], (list, tuple)):
        paths = paths[0]
    for i, path in enumerate(iterable=paths):
        with open(os.path.join(directory, path), "rb") as vision_file:
            vision_data.update({str(i) + '_' + os.path.basename(path) : vision_file.read()})
            vision_name.append(str(i) + '_' + os.path.basename(path))

    content = {
        "texts": entry[args.columns_messages],
        "media": vision,
        "name": vision_name
    }
    sample = {
        "__key__": vision + '_' + str(index),
        **vision_data,
        "json": json.dumps(content).encode("utf-8"),
    }
    return sample

def convert_to_wds(args):
    """ Convert dataset to wds format """
    assert args.media in ['video', 'image', 'mix'], "Invalid media type: {args.media}"

    if args.media == "video":
        assert args.video_dir is not None
    if args.media == "image":
        assert args.image_dir is not None

    if not os.path.exists(args.output_dir):
        os.mkdir(args.output_dir)

    if args.json_file.endswith('.jsonl'):
        data = [json.loads(line) for line in open(args.json_file, 'r')]
    elif args.json_file.endswith('.json'):
        with open(args.json_file, 'r') as f:
            data = json.load(f)
    else:
        raise ValueError("Unsupported file extension.")

    tar = os.path.join(args.output_dir, 'pretrain-%d.tar')
    with wds.ShardWriter(tar, maxcount=args.maxcount, maxsize=args.maxsize) as shard_writer:
        for index, entry in enumerate(tqdm(data)):
            if args.sample_type  == 'vqa' or args.sample_type == 'caption':
                image_path = entry.get('image') or entry.get('images')[0]
                with open(os.path.join(args.image_dir, image_path), "rb") as img_file:
                    image_data = img_file.read()
                sample = {
                    "__key__": entry.get('id', image_path).replace('.', '_'),
                    "jpg": image_data,
                    "json": json.dumps(entry[args.columns_messages]).encode("utf-8"),
                }
            else:
                video_paths = [entry.get('video')] if entry.get('video') is not None else entry.get('videos')
                image_paths = [entry.get('image')] if entry.get('image') is not None else entry.get('images')

                if video_paths is not None:
                    sample = construct_sample(args, 'video', video_paths, index, entry)
                elif image_paths is not None:
                    sample = construct_sample(args, 'image', image_paths, index, entry)
                else:   # for pure text
                    content = {
                        "texts": entry[args.columns_messages],
                        "media": "text",
                    }
                    sample = {
                        "__key__": 'text_' + str(index),
                        "json": json.dumps(content).encode("utf-8"),
                    }
            shard_writer.write(sample)
    write_config(
        EPath(args.output_dir),
        args.media,
        args.sample_type
    )
    print(f"Dataset successfully converted to wds")

def write_config(path: EPath, media: str=None, sample_type: bool=False):
    """ Write config to path """
    (path / MAIN_FOLDER_NAME).mkdir()
    all_tars = list(path.glob("**/*.tar")) + list(path.glob("**/*.tgz"))
    all_tars = [str(p.relative_to(path)) for p in sorted(all_tars)]
    class_type = "MultiMixQASample" if media == 'mix' else "MultiVidQASample"

    # Construct dataset configuration based on sample_type
    if sample_type == "vqa":
        # VQA sample type with field mapping
        dataset_definition = {
            "sample_type": {
                "__module__": "megatron.energon",
                "__class__": "VQASample"
            },
            "field_map": {
                "image": "jpg",
                "context": "json[0][content]",
                "answers": "json[1][content]"
            }
        }
    elif sample_type == "caption":
        # Captioning sample type with field mapping
        dataset_definition = {
            "sample_type": {
                "__module__": "megatron.energon",
                "__class__": "CaptioningSample"
            },
            "field_map": {
                "image": "jpg",
                "caption": "json[captions][0][content]"
            }
        }
    else:
        # Wrap in CrudeWebdataset
        dataset_definition = {
            "__module__": "megatron.energon",
            "__class__": "CrudeWebdataset",
            "subflavors": {
                "sample_type": sample_type
            }
        }
    
    with (path / MAIN_FOLDER_NAME / "dataset.yaml").open("w") as f:
        yaml.dump(dataset_definition, f, sort_keys=False)

    BaseWebdatasetFactory.prepare_dataset(
        path,
        all_tars,
        split_parts_ratio=[("train", 1.0), ("val", 0), ("test", 0)],
        tar_index_only=False,
        workers=32,
    )


def _add_arguments(parser: argparse.ArgumentParser):
    """Add arguments"""
    group = parser.add_argument_group(title='wds')
    group.add_argument('--output_dir', type=str, required=True, help='Output directory')
    group.add_argument('--json_file', type=str, required=True, help='Json file')
    group.add_argument('--image_dir', type=str, required=False, help='Image directory')
    group.add_argument('--video_dir', type=str, required=False, help='Video directory')
    group.add_argument('--maxcount', type=int, default=10000, help='Number of samples per shard')
    group.add_argument('--maxsize', type=int, default=3000000000, help='Maximum size of each shard')
    group.add_argument('--media', type=str, choices=["mix", "image", "video"], default="image", help='Media type')
    group.add_argument('--columns_messages', type=str, default="messages", help='Column name for messages')
    group.add_argument("--sample_type", type=str, required=True, help="Data sample type")

    return parser


def parse_args():
    """arguments"""
    parser = argparse.ArgumentParser()
    _add_arguments(parser)
    args = parser.parse_args()

    return args


def main():
    """main function"""
    args = parse_args()
    convert_to_wds(args)


if __name__ == '__main__':
    main()
