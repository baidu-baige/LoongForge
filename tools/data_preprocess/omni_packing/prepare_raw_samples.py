import bisect
import os
import json
import sys
from typing import Dict, List, Optional, Tuple, Union
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
from utils import get_init_file, parse_args, get_cfg, VALID_MEDIA_EXT
from collections import defaultdict

args = parse_args()
cfg = get_cfg(args.config)
input_token_file, max_token_len, packed_files_dir, wds_dir = get_init_file(cfg)

SRC_DST_EXTENSIONS = ("jpg", "json")
SRC_DIR_JSONS = wds_dir  # The storage location of json data
SRC_DIR_IMGS = wds_dir

dst_dir_json = os.path.join(packed_files_dir, "row_packing_jsons")
if os.path.exists(dst_dir_json) is False:
    os.makedirs(dst_dir_json)
MAX_WORKERS = 96

# TODO Determine the task type based on the input JSON content.
task_type = "sft"

PROMPTS = [
    "What about this picture?",
    "Please provide a vivid description of the image.",
    "Please Depict the image in words."
    "Could you please transcribe thr image into a descriptive paragraph?"
    "What is the content of this figure?",
    "What do you see here?",
    "Tell me about this image.",
    "What's going on in this artwork?",
    "What is depicted in this painting?",
    "What is the subject matter here?",
    "What can you make out in this picture?",
    "What's the main thing shown in this image?",
    "What's the gist of this artwork?",
    "What's the essence of this figure?",
    "What's the general idea here?",
    "What does this image show?",
    "What's the core element in this painting?",
    "What's the overview of this scene?",
    "What's the primary focus of this artwork?",
    "What's the fundamental subject matter?",
    "What's the general view presented?",
    "What's the main impression given by this picture?",
    "What's the central theme shown?",
    "What's the overall presentation here?",
    "What's the key element you notice?",
    "What's the fundamental concept in this image?",
    "What's the overall content?",
    "What's the main thing you get from this?",
    "What's the general subject?",
    "What's the core idea conveyed?",
    "What's the basic representation?",
    "What's the main point of this figure?",
]


def extract_assistant_response(json_path):
    try:
        with open(json_path, "r", encoding="utf-8") as f:
            data = json.load(f)

        if task_type == "sft":
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

        elif task_type == "pretrain":
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
    try:
        with open(json_path, "r", encoding="utf-8") as f:
            data = json.load(f)
        media_type = data["media_type"]
        assert media_type in VALID_MEDIA_EXT, (
            f"Unsupported media type '{media_type}'. "
            f"Supported types are: {list(VALID_MEDIA_EXT.keys())}"
        )
        media_files = defaultdict(list)
        for media_file in data["media_files"]:
            media_files[media_type + "s"].append(media_file)
        return media_files

    except FileNotFoundError:
        return f" Error: File {json_path} does not exist"
    except json.JSONDecodeError:
        return f" Error: File {json_path} is not in valid JSON format"
    except Exception as e:
        return f"An error occurred during the extraction process: {str(e)}"


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

    json_data = {**packed_media, "texts": texts}
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

        tty = open(os.devnull, "w") if os.name == "nt" else open("/dev/tty", "w")
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