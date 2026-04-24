# Copyright 2026 The LoongForge Authors.
# SPDX-License-Identifier: Apache-2.0

"""Script for preparing raw samples using packed bins."""

import bisect
import os
import json
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
from utils import get_init_file, parse_args, get_cfg, VALID_MEDIA_EXT
from collections import defaultdict

args = parse_args()
cfg = get_cfg(args.config)
input_token_file, max_token_len, packed_files_dir, wds_dir = get_init_file(cfg)
SAMPLE_TYPE = cfg.get("sample", {}).get("sample_type", "").lower()
SINGLE_MEDIA_INFER_TYPES = {"packed_vqa", "packed_captioning"}

SRC_DST_EXTENSIONS = ("jpg", "json")
SRC_DIR_JSONS = wds_dir  # The storage location of json data
SRC_DIR_IMGS = wds_dir

dst_dir_json = os.path.join(packed_files_dir, "row_packing_jsons")
if os.path.exists(dst_dir_json) is False:
    os.makedirs(dst_dir_json)
MAX_WORKERS = 96

# TODO Determine the task type based on the input JSON content.
task_type = "sft"


def _load_messages(data: Union[list, dict, None]) -> List:
    """
    Normalize and return message list from various schema roots.
    Supports:
    - pure list JSON (e.g., [{"role": ..., "content": ...}, ...])
    - dict JSON with `messages` or `texts` keys.
    """
    if isinstance(data, list):
        return normalize_messages_schema(data) or []
    if isinstance(data, dict):
        return normalize_messages_schema(data.get("messages") or data.get("texts") or []) or []
    return []


def normalize_messages_schema(text_data: Union[list, dict, None]) -> Union[list, dict, None]:
    """
    Normalize message keys so both role/content and from/value variants are usable.
    Mirrors logic from get_sample_len.normalize_messages_schema but kept local
    to avoid heavy imports.
    """
    if not isinstance(text_data, list):
        return text_data

    normalized = []
    for message in text_data:
        if not isinstance(message, dict):
            normalized.append(message)
            continue

        merged = dict(message)
        if "from" not in merged and "role" in message:
            merged["from"] = message["role"]
        if "value" not in merged and "content" in message:
            merged["value"] = message["content"]
        normalized.append(merged)

    return normalized


def extract_assistant_response(json_path):
    try:
        with open(json_path, "r", encoding="utf-8") as f:
            data = json.load(f)

        messages = _load_messages(data)
        if isinstance(messages, list):
            for msg in messages:
                role = msg.get("from") or msg.get("role")
                if role in ("assistant", "gpt", "bot"):
                    content = msg.get("value") or msg.get("content")
                    if content is not None:
                        return content

        if task_type == "sft" and isinstance(data, dict):
            try:
                assistant_content = next(
                    msg["content"]
                    for msg in data["messages"]
                    if msg["role"] == "assistant"
                )
                return assistant_content
            except Exception as e:
                pass
            try:
                assistant_content = next(
                    msg["value"] for msg in data["texts"] if msg["from"] == "gpt"
                )
                return assistant_content
            except Exception as e:
                pass

        elif task_type == "pretrain" and isinstance(data, dict):
            if data.get("captions") and len(data["captions"]) > 0:
                return data["captions"][0].get("content", "")
            else:
                assert 0, "No valid caption content found"

    except FileNotFoundError:
        return f" Error: File {json_path} does not exist"
    except json.JSONDecodeError:
        return f" Error: File {json_path} is not in valid JSON format"
    except Exception as e:
        return f"An error occurred during the extraction process: {str(e)}"


def extract_user_prompt(json_path):
    try:
        with open(json_path, "r", encoding="utf-8") as f:
            data = json.load(f)

        messages = _load_messages(data)
        if isinstance(messages, list):
            for msg in messages:
                role = msg.get("from") or msg.get("role")
                if role in ("user", "human"):
                    content = msg.get("value") or msg.get("content")
                    if content is not None:
                        return content

        if isinstance(data, dict):
            try:
                user_content = next(
                    msg["content"] for msg in data["messages"] if msg["role"] == "user"
                )
                return user_content
            except Exception as e:
                pass

            try:
                user_content = next(
                    msg["value"] for msg in data["texts"] if msg["from"] == "human"
                )
                return user_content
            except Exception as e:
                pass

    except FileNotFoundError:
        return f" Error: File {json_path} does not exist"
    except json.JSONDecodeError:
        return f" Error: File {json_path} is not in valid JSON format"
    except Exception as e:
        return f"An error occurred during the extraction process: {str(e)}"


def extract_media_files(json_path):
    json_path = Path(json_path)
    sample_stem = json_path.stem
    json_dir = json_path.parent
    try:
        with open(json_path, "r", encoding="utf-8") as f:
            data = json.load(f)
        if not isinstance(data, dict):
            # Some datasets store pure list-style messages; treat as missing metadata.
            data = {}

        declared_media_type = data.get("media_type") or data.get("media")
        media_files = data.get("media_files") or data.get("name") or []
        if isinstance(media_files, (str, Path)):
            media_files = [str(media_files)]
        if not media_files and SAMPLE_TYPE in SINGLE_MEDIA_INFER_TYPES:
            # For packed_vqa/packed_captioning, infer sibling media by stem (e.g., 0001.jpg for 0001.json)
            inferred_media = defaultdict(list)
            for media_type, ext_list in VALID_MEDIA_EXT.items():
                for ext in ext_list:
                    candidate = json_dir / f"{sample_stem}{ext}"
                    if candidate.exists():
                        inferred_media[media_type + "s"].append(candidate.name)
                        break  # Prefer the first matching extension per media type
            if inferred_media:
                return inferred_media
        if not media_files:
            return defaultdict(list)
        if not isinstance(media_files, list):
            raise ValueError(
                f"`media_files`/`name` must be a list in {json_path}, "
                f"got {type(media_files).__name__}"
            )

        media_sources = defaultdict(list)

        def infer_media_type(media_file: str) -> Optional[str]:
            ext = os.path.splitext(media_file)[1].lower()
            for m_type, ext_list in VALID_MEDIA_EXT.items():
                if ext in ext_list:
                    return m_type
            return None

        for media_file in media_files:
            # Prefer declared type if it is supported; otherwise fall back to file extension.
            if declared_media_type in VALID_MEDIA_EXT:
                media_type = declared_media_type
            else:
                media_type = infer_media_type(media_file)
                if not media_type and declared_media_type not in (None, "mix"):
                    raise ValueError(
                        f"Unsupported media type '{declared_media_type}' in {json_path}"
                    )

            if not media_type:
                raise ValueError(
                    f"Cannot infer media type for '{media_file}' in {json_path}"
                )

            media_sources[media_type + "s"].append(media_file)

        return media_sources

    except (FileNotFoundError, json.JSONDecodeError):
        raise
    except Exception as e:
        raise ValueError(
            f"Failed to extract media files from {json_path}: {str(e)}"
        ) from e


def dataset_tokinfo_generator(f_name):
    """
    Dataset token information generator, reading and parsing file content line by line

    Parameter:
        f_name (str): The file path containing token information

    Generated:
        tuple: (base_name, token_len) - The basic file name and token length after parsing
    """
    try:
        with open(f_name, "r", encoding="utf-8") as f:
            for line in f:
                stripped_line = line.strip()
                if not stripped_line:
                    continue

                parts = stripped_line.split(":")
                if len(parts) == 2:
                    base_name = parts[0].strip()
                    token_len_str = parts[1].strip()

                    try:
                        token_len = int(token_len_str)
                        yield (base_name, token_len)
                    except ValueError:
                        print(
                            f"Warning: '{token_len_str}' cannot be converted to an integer. This line has been skipped",
                            file=sys.stderr,
                        )
                        continue

    except FileNotFoundError:
        print(f" error: file '{f_name}' does not exist ", file=sys.stderr)
        return
    except Exception as e:
        print(f"Error occurred while processing file: {str(e)}", file=sys.stderr)
        return


class TokenInfoReader:
    """
    Token information reader

    It supports batch reading, full reading and breakpoint resumption functions, and is suitable for processing text files containing token information.
    File format requirements: One record per line, in the format of "base_name: token_len"
    """

    def __init__(self, f_name):
        """
        Initialize the reader

        Parameter
            f_name (str): The file path containing token information
        """
        self.f_name = f_name
        self.generator = dataset_tokinfo_generator(f_name)
        self._current_position = 0

    def read(self, count=None):
        """
        Read the record

        Parameter:
            count (int, optional): The number of records to be read, default to None (read all remaining records)

        Return:
            tuple: (base_names list, token_lens list, actual read quantity)
        """
        base_names = []
        token_lens = []
        read_count = 0

        while True:
            if count is not None and read_count >= count:
                break

            try:
                base_name, token_len = next(self.generator)
                base_names.append(base_name)
                token_lens.append(token_len)
                read_count += 1
                self._current_position += 1

            except StopIteration:
                break

        return base_names, token_lens, read_count

    def get_current_position(self):
        return self._current_position


def _normalize_media(packed_media, sample_count, box_index):
    """Convert collected media into cooker friendly layout."""
    media_types = [k for k, v in packed_media.items() if v]
    if not media_types:
        raise ValueError(f"[box {box_index}] no media collected for packed samples")
    if len(media_types) > 1:
        raise ValueError(
            f"[box {box_index}] multiple media types found {media_types}, "
            "but cooker expects a single media_type"
        )

    media_key = media_types[0]
    media_files = packed_media[media_key]
    if len(media_files) != sample_count:
        raise ValueError(
            f"[box {box_index}] media/sample count mismatch: {len(media_files)} vs {sample_count}"
        )

    # cooker expects singular media_type: "image" or "video"
    media_type = media_key[:-1] if media_key.endswith("s") else media_key
    return media_type, media_files


def process_box(box_index, samples_in_box, dst_dir_json):
    packed_media = defaultdict(list)
    packed_assist_responses = []
    packed_info = []
    packed_sample_names = (sample["name"] for sample in samples_in_box)

    for sample_name in packed_sample_names:
        json_path = os.path.join(SRC_DIR_JSONS, f"{sample_name}.json")
        if task_type == "pretrain":
            packed_info.append(
                (extract_media_files(json_path), extract_assistant_response(json_path))
            )
        elif task_type == "sft":
            packed_info.append(
                (
                    extract_media_files(json_path),
                    extract_user_prompt(json_path),
                    extract_assistant_response(json_path),
                )
            )

    packed_json_path = os.path.join(dst_dir_json, f"ps_{box_index:08d}.json")
    if task_type == "pretrain":
        for media_src, cap_src in packed_info:
            for media_type, media_file in media_src.items():
                packed_media[media_type].append(media_file)
            packed_assist_responses.append(cap_src)
        packed_user_prompts = []

    elif task_type == "sft":
        packed_user_prompts = []
        for media_src, prompt_src, cap_src in packed_info:
            for media_type, media_file in media_src.items():
                packed_media[media_type].append(media_file)
            packed_assist_responses.append(cap_src)
            packed_user_prompts.append(prompt_src)

    texts = {"captions": packed_assist_responses, "prompts": packed_user_prompts}

    media_type, media_files = _normalize_media(
        packed_media, len(samples_in_box), box_index
    )

    json_data = {
        "media_files": media_files,
        "media_type": media_type,
        "texts": texts,
    }
    if SAMPLE_TYPE in SINGLE_MEDIA_INFER_TYPES:
        # Downstream caption/VQA packing expects an `images` field.
        json_data["images"] = media_files
    try:
        with open(packed_json_path, "w", encoding="utf-8") as f:
            json.dump(json_data, f, indent=4, ensure_ascii=False)
    except Exception as e:
        print(
            f" thread {threading.current_thread().name} failed to generate JSON file {packed_json_path} : {str(e)}"
        )
    return box_index


if __name__ == "__main__":
    print(
        "Step1-----------------Read the tokenlen information of the original ds-----------------Start"
    )
    info_reader = TokenInfoReader(input_token_file)
    base_names, token_lens, n_count = info_reader.read()

    print(f" read {n_count} datas ")
    print(
        "Step1-----------------Read the tokenlen information of the original ds-----------------Stop\n\n"
    )

    print("Step2-----------------packing grouping-----------------Start")

    import pickle

    def load_bin_boxes(file_path: str):
        with open(file_path, "rb") as f:
            bin_boxes = pickle.load(f)
        print(f"The packing result has been loaded: {file_path}")
        return bin_boxes

    bin_boxs = os.path.join(packed_files_dir, "bins_boxs.pkl")
    bin_boxs = load_bin_boxes(bin_boxs)
    num_bin_boxs = len(bin_boxs)

    print(
        f"raw data number----{n_count}----,after packing number----{num_bin_boxs}----"
    )
    print("Step2-----------------packing grouping-----------------Stop\n\n")

    print(
        "Step3----------------- Start building the new dataset -----------------Start"
    )
    print(
        f" starts processing the {num_bin_boxs} group of data using {MAX_WORKERS} threads "
    )

    with ThreadPoolExecutor(
        max_workers=MAX_WORKERS, thread_name_prefix="PackThread"
    ) as executor:

        futures = {
            executor.submit(
                process_box, box_index, samples_in_box, dst_dir_json
            ): box_index
            for box_index, samples_in_box in enumerate(bin_boxs)
        }

        from tqdm import tqdm

        try:
            tty = open(os.devnull, "w") if os.name == "nt" else open("/dev/tty", "w")
        except OSError:
            tty = None
        for future in tqdm(
            as_completed(futures),
            total=len(futures),
            desc="Packing progress",
            unit="pack",
            file=tty,
        ):
            try:
                future.result()
            except Exception as e:
                box_index = futures[future]
                print(
                    f"an error occurred when processing the {box_index} th group of data: {e}"
                )

    print(
        "----------------- The new dataset was successfully constructed -----------------Stop"
    )
