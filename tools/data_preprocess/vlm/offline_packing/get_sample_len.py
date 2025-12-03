import os
import json
import logging
import tempfile
import threading
import multiprocessing
import argparse
import yaml
from pathlib import Path
from jinja2 import Template
from collections import defaultdict
from contextlib import ExitStack
from transformers import AutoProcessor
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import List, Dict, Tuple, Optional, Union, Generator, Callable
from heapq import merge
from natsort import natsorted
from queue import Empty
from multiprocessing import Pool, Manager, Value
import psutil

import media_preprocess_utils
from utils import (
    TEMPLATES,
    VALID_MEDIA_EXT,
    parse_args,
    get_cfg,
    get_temp_dir,
    get_sample_record_path,
    get_token_info_report_path,
    get_log_file_path
)
logger = logging.getLogger(__name__)

# ----------------- Configuration Global Variables -----------------

# Dict[str, Callable]: Stores media preprocessing functions.
# Filled in the main function using settings from the config file.
MEDIA_PREPROCESS = {}

# will be set by the main function from the config file
TEMPLATE_TEXT_KEY = ""
GLOBAL_PROCESSED_SAMPLE_COUNT = multiprocessing.Value("i", 0)


# ----------------- Utility Functions -----------------
def get_chat_template(sample_type: str, model_type: str) -> Template:
    """
    Retrieve a chat template string and wrap it as a Template object.

    Args:
        sample_type (str): The logical task category, e.g. "packed_captioning", "packed_vqa", "packed_multi_mix_qa", etc.
        model_type (str): The model identifier, used for selecting model-specific templates.

    Returns:
        Template: A Template object corresponding to the given sample_type and model_type.

    Raises:
        ValueError: If the specified sample_type or model_type template is not defined.
    """

    supported_sample_types = list(TEMPLATES.keys())

    task_templates = TEMPLATES.get(sample_type)
    if task_templates is None:
        raise ValueError(
            f"Unsupported sample_type '{sample_type}'. "
            f"Available sample types: {supported_sample_types}"
        )

    if isinstance(task_templates, str):
        # Simple case: directly defined template string
        template_str = task_templates
    elif isinstance(task_templates, dict):
        # Model-specific templates
        template_str = task_templates.get(model_type)
        if template_str is None:
            raise ValueError(
                f"No template found for model_type '{model_type}' under sample_type '{sample_type}'. "
                f"Available model types: {list(task_templates.keys())}"
            )
    else:
        raise TypeError(
            f"Invalid template format for sample_type '{sample_type}': expected str or dict, got {type(task_templates)}"
        )

    return template_str


def fetch_media_data(media_paths: List[Tuple[str, str]]) -> Dict[str, list]:
    """
    Process a list of media files and prepare them for model input.

    Args:
        media_paths (List[Tuple[str, str]]): List of tuples (media_type, media_file_path)

    Returns:
        Dict[str, list]: Dictionary mapping each media type to a list of processed inputs

    Raises:
        ValueError: If media type is unsupported or file extension is invalid
    """
    media_inputs: Dict[str, list] = defaultdict(list)

    for media_type, media_path in media_paths:
        # Check if media type is supported
        if media_type not in MEDIA_PREPROCESS:
            raise ValueError(f"Unsupported media type: '{media_type}'")

        media_path_obj = Path(media_path)
        ext = media_path_obj.suffix.lower()  # Support uppercase extensions, e.g., .JPG

        # Check if file extension is supported
        if ext not in VALID_MEDIA_EXT.get(media_type, []):
            raise ValueError(
                f"Unsupported file extension '{ext}' for media type '{media_type}'. "
                f"Supported: {VALID_MEDIA_EXT.get(media_type)}"
            )

        # Call the corresponding processing function
        process_func: Callable[[Path], any] = MEDIA_PREPROCESS[media_type]
        media_inputs[media_type + "s"].append(process_func(media_path_obj))
    return media_inputs


def find_sample_names(webdataset_dir: Union[str, Path]) -> List[str]:
    """Return a list of JSON file stems from the directory"""
    webdataset_dir = Path(webdataset_dir)
    sample_names = [f.stem for f in webdataset_dir.glob("*.json")]
    logger.info(f"Found {len(sample_names)} samples in {webdataset_dir}")
    return sample_names


def record_samples_to_file(sample_names: List[str], output_file: Union[str, Path]):
    """Save sorted sample names to a file"""
    try:
        content = "\n".join(natsorted(sample_names)) + "\n"
        with open(output_file, "w", encoding="utf-8") as f:
            f.write(content)
        logger.info(f"Recorded {len(sample_names)} samples to {output_file}")
    except Exception as e:
        logger.error(f"Failed to write {output_file}: {e}")
        raise


def count_valid_lines(file_path: Union[str, Path]) -> int:
    """
    Count the number of non-empty lines in a text file.

    Args:
        file_path (Union[str, Path]): Path to the file to be counted.

    Returns:
        int: The number of non-empty (non-whitespace) lines.
    """
    file_path = Path(file_path)

    if not file_path.exists():
        logger.error(f"File not found: {file_path}")
        return 0

    if file_path.stat().st_size == 0:
        logger.warning(f"File is empty: {file_path}")
        return 0

    try:
        with file_path.open("r", encoding="utf-8") as f:
            valid_lines = sum(1 for line in f if line.strip())

        logger.info(f"Counted {valid_lines} non-empty lines in {file_path.name}")
        return valid_lines

    except Exception as e:
        logger.error(f"Failed to read {file_path}: {e}", exc_info=True)
        return 0


def read_lines_by_chunk(
    file_path: Union[str, Path], chunk_size: int
) -> Generator[List[str], None, None]:
    """
    Read a text file and yield non-empty lines in fixed-size chunks.

    Args:
        file_path (Union[str, Path]): Path to the input text file.
        chunk_size (int): Number of lines per chunk to yield.

    Yields:
        List[str]: A list of non-empty lines for each chunk.

    Raises:
        FileNotFoundError: If the file does not exist.
        ValueError: If chunk_size is not positive.
    """
    file_path = Path(file_path)

    if not file_path.exists():
        raise FileNotFoundError(f"File not found: {file_path}")
    if chunk_size <= 0:
        raise ValueError(f"Invalid chunk_size: {chunk_size}. Must be > 0.")

    try:
        with file_path.open("r", encoding="utf-8") as f:
            chunk = []
            for line in f:
                line = line.strip()
                if not line:
                    continue
                chunk.append(line)
                if len(chunk) >= chunk_size:
                    logger.debug(
                        f"Yielding chunk of {len(chunk)} lines from {file_path.name}"
                    )
                    yield chunk
                    chunk = []

            # Yield remaining lines (if any)
            if chunk:
                logger.debug(
                    f"Yielding final chunk of {len(chunk)} lines from {file_path.name}"
                )
                yield chunk

    except Exception as e:
        logger.error(f"Error reading {file_path}: {e}", exc_info=True)
        raise


def get_adaptive_workers(min_workers=8, max_workers=256, base_ratio=4):
    """
    Dynamically adjust thread count based on system load.

    Args:
        min_workers (int): Minimum thread count
        max_workers (int): Maximum thread count
        base_ratio (int): Base thread multiplier per CPU core

    Returns:
        int: Recommended thread count
    """
    try:
        # Get CPU core count; some systems may return None, so default to 4 cores
        cpu_count = os.cpu_count() or 4
        cpu_usage = psutil.cpu_percent(interval=0.3)
        mem_usage = psutil.virtual_memory().percent

        dynamic_max = min(max_workers, cpu_count * base_ratio)

        # Adjust ratio based on system load
        if cpu_usage > 85 or mem_usage > 90:
            ratio = 0.4
        elif cpu_usage > 70 or mem_usage > 80:
            ratio = 0.6
        elif cpu_usage > 50 or mem_usage > 70:
            ratio = 0.8
        else:
            ratio = 1.0

        adjusted = int(min(dynamic_max, max(min_workers, dynamic_max * ratio)))

        logger.info(
            f"System load (CPU={cpu_usage:.1f}%, MEM={mem_usage:.1f}%) → "
            f"Recommended threads: {adjusted} (cores={cpu_count}, max={dynamic_max})"
        )

        return adjusted

    except Exception as e:
        logger.warning(
            f"Failed to compute adaptive workers, fallback to min_workers={min_workers}: {e}"
        )
        return min_workers


# ----------------- Core Processing Functions -----------------
def process_sample(
    json_path: Union[str, Path], chat_template, processor
) -> Tuple[Union[int, None], str]:
    """
    Process a single multimodal sample:
    1. Load JSON metadata
    2. Render text using chat template
    3. Load and process associated media
    4. Generate tokenized input and return token length

    Returns:
        Tuple[token_length or None, sample_name or error_msg]
    """
    try:
        json_path = Path(json_path)
        sample_name = json_path.stem
        if not json_path.exists():
            raise FileNotFoundError(f"JSON file not found: {json_path}")

        # --- Step 1: Load JSON content ---
        with open(json_path, "r", encoding="utf-8") as f:
            json_data = json.load(f)

        # Normalize list-style JSON (e.g. [{"role": ..., "content": ...}, ...]) to dict format.
        # This keeps downstream logic untouched and allows config-defined TEMPLATE_TEXT_KEY usage.
        if isinstance(json_data, list):
            json_data = {TEMPLATE_TEXT_KEY: json_data}

        if not isinstance(json_data, dict):
            raise ValueError(f"Invalid JSON format in {json_path}")

        # --- Step 2: Render text input ---
        # @ref convert_to_webdataset.construct_sample_for_wds
        text_data = json_data.get(TEMPLATE_TEXT_KEY)
        if not text_data:
            raise ValueError(
                f"Missing '{TEMPLATE_TEXT_KEY}' field in {json_path}"
            )
        text_input = chat_template.render(**{TEMPLATE_TEXT_KEY: text_data})

        # --- Step 3: Collect media paths ---
        media_names = json_data.get("name", [])
        main_dir = json_path.parent
        media_paths = []

        for media_name in media_names:
            media_type = media_name.split("_")[0]
            if media_type not in MEDIA_PREPROCESS:
                logger.warning(
                    f"Skipping unsupported media type '{media_type}' in {sample_name}"
                )
                continue

            media_path = main_dir / f"{sample_name}.{media_name}"
            media_paths.append((media_type, str(media_path)))

        # --- Step 4: Load and process media data ---
        media_inputs = fetch_media_data(media_paths) if media_paths else {}
        # --- Step 5: Build model input ---
        model_inputs = processor(
            text=[text_input], **media_inputs, padding=True, return_tensors="pt"
        )

        token_len = int(model_inputs["input_ids"].shape[1])
        logger.debug(f"model_inputs = {model_inputs}")
        logger.debug(f"Processed sample {sample_name}: {token_len} tokens")

        return token_len, sample_name

    except Exception as e:
        error_msg = f"Processing failed [{json_path.name}]: {e}"
        logger.error(error_msg)
        return None, error_msg


def process_chunk(
    chunk_idx: int,
    sample_record_chunk: list,
    queue_to_merge: multiprocessing.Queue,
    processor_kwargs: dict,
    chat_template_str: str,
    temp_dir: Union[str, Path],
    webdataset_dir: Union[str, Path],
) -> Union[str, None]:
    """
    Process one chunk of samples (JSON + media) in a single process,
    using internal multi-threading for efficiency.

    Args:
        chunk_idx: Index of the chunk
        sample_record_chunk: List of sample names (no .json suffix)
        queue_to_merge: Cross-process queue to store temporary file paths
        processor_kwargs: Args for `AutoProcessor.from_pretrained`
        chat_template_str: Chat template string
        temp_dir: Directory for temporary files
        webdataset_dir: Base directory containing JSON samples

    Returns:
        str: Path to temporary file containing sorted token lengths
        None: If processing failed
    """

    local_processed_count = 0
    token_len_results: List[Tuple[int, str]] = []
    chat_template = Template(chat_template_str)

    try:
        webdataset_dir = Path(webdataset_dir)
        temp_dir = Path(temp_dir)
        json_paths = [webdataset_dir / f"{fn}.json" for fn in sample_record_chunk]
        n_samples = len(json_paths)
        logger.info(
            f"Process {multiprocessing.current_process().name} starts processing chunk {chunk_idx}, containing {n_samples} samples"
        )

        # 1. Prepare processor & worker pool
        processor = AutoProcessor.from_pretrained(**processor_kwargs)
        n_workers = get_adaptive_workers()

        with ThreadPoolExecutor(
            max_workers=n_workers, thread_name_prefix=f"chunk{chunk_idx:02d}"
        ) as executor:
            future_map = {
                executor.submit(process_sample, path, chat_template, processor): path
                for path in json_paths
            }

            for future in as_completed(future_map):
                json_path = future_map[future]
                try:
                    token_len, sample_name = future.result()
                    if token_len is not None:
                        token_len_results.append((token_len, sample_name))
                        local_processed_count += 1
                    else:
                        logger.warning(f"Skipped: {sample_name}")
                except Exception as e:
                    logger.error(f"Error processing {json_path}: {e}")

        if not token_len_results:
            logger.warning(f"Chunk {chunk_idx}: No valid samples processed.")
            return None

        # 2. Sort by token length
        token_len_results_sorted = natsorted(token_len_results, key=lambda x: x[0])

        # 3. Write temporary file
        with tempfile.NamedTemporaryFile(
            mode="w+",
            delete=False,
            prefix=f"chunk{chunk_idx:03d}_",
            encoding="utf-8",
            dir=temp_dir,
        ) as f:
            temp_file_path = f.name
            for token_len, sample_name in token_len_results_sorted:
                f.write(f"{sample_name}:{token_len}\n")

        # 4. Put temporary file path into cross-process queue
        queue_to_merge.put(temp_file_path)

        # 5. Update global counter
        with GLOBAL_PROCESSED_SAMPLE_COUNT.get_lock():
            GLOBAL_PROCESSED_SAMPLE_COUNT.value += local_processed_count

        logger.info(
            f"Process {multiprocessing.current_process().name} finished chunk {chunk_idx}: "
            f"{local_processed_count}/{n_samples} valid samples processed, temp file: {temp_file_path}"
        )
        return temp_file_path

    except Exception as e:
        logger.error(
            f"Process {multiprocessing.current_process().name} failed for chunk {chunk_idx}: {e}"
        )
        return None


def merge_by_batch(
    queue_to_merge: multiprocessing.JoinableQueue,
    merge_batch_size: int,
    merged_outputs_per_batch: List[str],
    stop_event: multiprocessing.Event,
    temp_dir: Path,
    max_token_len: int,
):
    """
    Merging thread: repeatedly merges small batches of token length files into larger merged files.

    Args:
        queue_to_merge: JoinableQueue containing paths to input files.
        merge_batch_size: Number of input files to merge per batch.
        merged_outputs_per_batch: Shared list to store merged file paths.
        stop_event: Event to signal termination.
        temp_dir: Directory for temporary merged files.
        max_token_len: Maximum allowed token length to filter samples.
    """

    buffer = []
    batch_count = 0
    thread_name = threading.current_thread().name
    logger.info(f"{thread_name} started — merging every {merge_batch_size} files.")

    try:
        # Loop condition: the queue has files, or the buffer has files, or the stop signal has not been received
        # The loop only exits when the queue is empty, the buffer is empty, and the stop signal has been received
        while (not queue_to_merge.empty()) or buffer or (not stop_event.is_set()):
            # Get file from queue (with timeout to avoid blocking indefinitely)
            if not queue_to_merge.empty():
                try:
                    # Fill buffer by fetching files from queue_to_merge
                    file_path = queue_to_merge.get(
                        timeout=1
                    )  # Fetch files from queue_to_merge to fill the buffer
                    buffer.append(file_path)
                    queue_to_merge.task_done()
                    logger.debug(
                        f"merge_by_batch 接收文件{Path(file_path)}, 当前缓冲区: {len(buffer)}/{merge_batch_size}"
                    )

                    # Start merging when the number of files in the buffer reaches batch_size
                    if len(buffer) >= merge_batch_size:
                        batch_label = "merged_batch"
                        batch_count += 1
                        temp_file_path = Path(
                            tempfile.NamedTemporaryFile(
                                mode="w",
                                delete=False,
                                prefix=f"{batch_label}{batch_count:03d}_",
                                suffix=".txt",
                                dir=temp_dir,
                            ).name
                        )

                        # merge
                        result_path, line_count = merge_files_by_token(
                            buffer, temp_file_path, max_token_len=max_token_len
                        )
                        if result_path and line_count > 0:
                            merged_outputs_per_batch.append(result_path)

                        # flush buffer
                        buffer = []
                except Empty:
                    # Continue looping while the queue is empty
                    continue
                except Exception as e:
                    logger.error(f"Error in merge thread: {e}", exc_info=True)
            else:
                # When the queue is empty, check if remaining files need to be force-merged
                if buffer and stop_event.is_set():
                    # Force merge if stop signal is received and buffer has files
                    batch_label = "merged_remains"
                    batch_count += 1
                    temp_file_path = Path(
                        tempfile.NamedTemporaryFile(
                            mode="w",
                            delete=False,
                            prefix=f"{batch_label}{batch_count:03d}_",
                            suffix=".txt",
                            dir=temp_dir,
                        ).name
                    )
                    result_path, line_count = merge_files_by_token(
                        buffer, temp_file_path, max_token_len=max_token_len
                    )

                    if result_path and line_count > 0:
                        merged_outputs_per_batch.append(result_path)
                    buffer = []

                else:
                    threading.Event().wait(0.5)

        # Final check to ensure buffer is empty (to prevent omissions)
        if buffer:
            logger.error(
                f"merge_by_batch线程推出时缓冲区仍有 {len(buffer)} 个文件未处理！数据丢失"
            )
    except Exception as e:
        logger.error(f"merge_by_batch线程异常退出: {str(e)}", exc_info=True)
    finally:
        logger.info(
            f"merge_by_batch线程退出，共生成 {len(merged_outputs_per_batch)} 个文件"
        )


def merge_files_by_token(
    input_files: List[Path], output_file: Path, max_token_len: int
) -> Tuple[Optional[Path], int]:
    """
    Merge multiple sorted token length files while filtering out entries exceeding max_token_len.
    Each line format: "sample_name:token_length".

    Args:
        input_files: List of sorted sample files (each line "sample_name:token_length")
        output_file: Path to the final merged output file
        max_token_len: Maximum token length allowed; lines exceeding this will be skipped

    Returns:
        Tuple of:
            - Path to merged output file (or None if merge failed)
            - Number of lines kept after filtering
    """

    if not input_files:
        logger.warning("No input files provided for merging.")
        return None, 0

    # 1. Filter out empty or invalid files
    valid_files = []
    total_records = 0
    for file_path in input_files:
        count = count_valid_lines(file_path)
        if count > 0:
            valid_files.append(file_path)
            total_records += count
        else:
            logger.warning(f"Skipping empty or invalid file: {file_path}")

    if not valid_files:
        logger.warning("No valid files to merge after filtering.")
        return None, 0

    def parse_line(line: str) -> Tuple[int, str]:
        """Parse a line into token length and raw line"""
        sample_name, token_len_str = line.strip().split(":", 1)
        return int(token_len_str), line

    kept_count, filtered_count = 0, 0
    try:
        with ExitStack() as stack:
            # Open all files safely
            file_handles = [
                stack.enter_context(open(fpath, "r", encoding="utf-8"))
                for fpath in valid_files
            ]
            iterators = [(parse_line(line) for line in fh) for fh in file_handles]

            # Merge all iterators by token length
            with open(output_file, "w", encoding="utf-8") as fout:
                for token_len, line in merge(*iterators, key=lambda x: x[0]):
                    if token_len <= max_token_len:
                        fout.write(line)
                        kept_count += 1
                    else:
                        filtered_count += 1
        logger.info(
            f"Merged {len(valid_files)} files → {output_file.name}, "
            f"kept {kept_count} samples (filtered {filtered_count} over-limit)"
        )
        return output_file, kept_count

    except Exception as e:
        logger.error(f"Failed to merge token files: {e}", exc_info=True)
        # Delete output file if partially written
        if output_file.exists():
            output_file.unlink(missing_ok=True)
        return None, 0


def main():
    args = parse_args()
    config_path = args.config
    config = get_cfg(config_path)

    # sample_config
    max_token_len = config["sample"]["max_token_len"]
    sample_type = config["sample"]["sample_type"]

    # data_config
    wds_dir = Path(config["data"]["wds_dir"])
    global TEMPLATE_TEXT_KEY
    TEMPLATE_TEXT_KEY = config["data"]["template_text_key"]

    # model_config
    model_type = config["model"]["model_type"]
    processor_kwargs = config["model"]["processor_kwargs"]

    # process_config
    chunk_size = config["process"]["chunk_size"]
    time_out = config["process"]["time_out"]
    merge_batch_size = config["process"]["merge_batch_size"]
    
    temp_dir = get_temp_dir(wds_dir)
    sample_record = get_sample_record_path(wds_dir)
    token_info_report = get_token_info_report_path(wds_dir)

    # log_config
    log_level = config["log"]["level"]
    log_file = get_log_file_path(wds_dir)

    # fill MEDIA_PREPROCESS
    global MEDIA_PREPROCESS
    for media_type, func_name in config.get("media_preprocess", {}).items():
        preprocess_func = getattr(media_preprocess_utils, func_name)
        if preprocess_func is None:
            raise ValueError(
                f"No preprocessing function found for '{func_name}' of media type '{media_type}'"
            )
        MEDIA_PREPROCESS[media_type] = preprocess_func

    # ======== Setup logging ========
    logging.basicConfig(
        level=getattr(logging, log_level.upper()),
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        handlers=[logging.FileHandler(log_file), logging.StreamHandler()],
    )

    # ======== Initialize variables ========
    ready_to_batch_merge_files = []
    intermediate_merged_files = []
    chat_template_str = get_chat_template(sample_type, model_type)

    # Pipeline: record_samples_to_file → process_chunk → merge_by_batch → merge_files_by_token
    try:
        logger.info(f"--------------Starting data processing pipeline--------------")

        # 1. Collect samples from webdataset_dir and record to sample_record
        sample_names = find_sample_names(wds_dir)
        original_sample_count = len(sample_names)
        logger.info(f"Found {original_sample_count} sample files")
        if original_sample_count == 0:
            logger.warning("No sample files found, exiting program")
            return
        record_samples_to_file(sample_names, sample_record)

        # 2. Initialize cross-process queues
        manager = Manager()
        # produced by process_chunk, consumed by merge_by_batch
        ready_to_batch_merge_queue = manager.Queue()
        stop_event = manager.Event()

        # 3 Start background merge thread
        merge_thread = threading.Thread(
            # consume the files produced by process_chunk
            target=merge_by_batch,
            args=(
                ready_to_batch_merge_queue,
                merge_batch_size,
                intermediate_merged_files,
                stop_event,
                temp_dir,
                max_token_len,
            ),
            daemon=True,
        )
        merge_thread.start()
        logger.info("Batch_Merge thread started")

        # 4. Parallel chunk processing by process_pool
        # 4.1 Split sample record into chunks
        all_chunks = list(read_lines_by_chunk(sample_record, chunk_size))
        total_chunks = len(all_chunks)
        n_processes = min(multiprocessing.cpu_count(), total_chunks)
        logger.info(
            f"Divided into {total_chunks} chunks, starting {n_processes} processes for processing"
        )

        # 4.2 Build process args for each chunk
        chunk_process_args = [
            (
                idx + 1,  # chunk index
                chunk,  # chunk data
                ready_to_batch_merge_queue,  # cross-process queue
                processor_kwargs,  # processor initialization parameters
                chat_template_str,
                temp_dir,
                wds_dir,
            )
            for idx, chunk in enumerate(all_chunks)
        ]

        # 4.3 Start multiprocessing pool
        with Pool(processes=n_processes) as process_pool:
            async_result = process_pool.starmap_async(process_chunk, chunk_process_args)
            try:
                ready_to_batch_merge_files = async_result.get(timeout=time_out)
            except multiprocessing.TimeoutError:
                logger.error("Process pool timeout, terminating workers")
                process_pool.terminate()
                ready_to_batch_merge_files = []

        # 5. Wait for batch merge completion
        ready_to_batch_merge_files = [
            f for f in ready_to_batch_merge_files if f is not None
        ]
        logger.info(
            f"Chunk processing completed, generated {len(ready_to_batch_merge_files)} temp files"
        )

        total_processed = GLOBAL_PROCESSED_SAMPLE_COUNT.value
        logger.info(
            f"Original sample count: {original_sample_count}, Valid processed samples: {total_processed}"
        )

        if total_processed != original_sample_count:
            logger.warning(
                f"Data incomplete! Original {original_sample_count}, valid processed {total_processed}, difference {original_sample_count - total_processed}"
            )
        else:
            logger.info(
                "Data integrity verification passed, all samples processed successfully"
            )

        # Wait for all batch merge queue tasks to complete
        logger.info("Waiting for batch merge queue to finish...")
        ready_to_batch_merge_queue.join()
        logger.info("Batch merge completed")

        # Signal merge thread to finalize remaining files
        stop_event.set()
        timeout_counter = 0
        while merge_thread.is_alive() and timeout_counter < 60:
            threading.Event().wait(1)
            timeout_counter += 1

        if merge_thread.is_alive():
            logger.warning("Merge thread timedout")
        else:
            logger.info("Merge thread exited normally")

        # Verify intermediate merge output count
        expected_batch_count = (
            len(ready_to_batch_merge_files) + merge_batch_size - 1
        ) // merge_batch_size
        if len(intermediate_merged_files) != expected_batch_count:
            logger.warning(
                f"Unexpected merged file count: expected {expected_batch_count}, got {len(intermediate_merged_files)}"
            )
        else:
            logger.info(f"Merged file count verified: {len(intermediate_merged_files)}")

        # 6. Final merge to output file
        if not intermediate_merged_files:
            logger.warning("No merged files generated, skipping final merge")
            return

        total_final_records = sum(
            count_valid_lines(f) for f in intermediate_merged_files
        )
        logger.info(
            f"Starting final merge: {len(intermediate_merged_files)} batch files, total {total_final_records} records"
        )

        # Merge to final file
        final_merged_file, final_merged_count = merge_files_by_token(
            intermediate_merged_files,
            Path(token_info_report),
            max_token_len=max_token_len,
        )

        if final_merged_file and final_merged_count > 0:
            logger.info(
                f"Final result generated: {token_info_report} ({final_merged_count} records)"
            )
            if final_merged_count != total_processed:
                logger.error(
                    f"Data mismatch: processed {total_processed}, final {final_merged_count}"
                )
            else:
                logger.info(logger.info("Final record count verified"))
        else:
            logger.error("Final merge failed")

    except Exception as e:
        logger.error(f"Pipeline exception: {e}", exc_info=True)

    finally:
        # cleanup
        stop_event.set()

        if merge_thread and merge_thread.is_alive():
            merge_thread.join(timeout=2)
        threading.Event().wait(2)

        # Remove temporary files
        all_temp_files = ready_to_batch_merge_files + intermediate_merged_files
        for fpath in all_temp_files:
            if fpath != str(token_info_report) and os.path.exists(fpath):
                try:
                    os.remove(fpath)
                    logger.debug(f"Cleaned up temp file: {os.path.basename(fpath)}")
                except Exception as e:
                    logger.warning(
                        f"Failed to clean temp file {os.path.basename(fpath)}: {str(e)}"
                    )

        logger.info("Program completed successfully")


if __name__ == "__main__":
    main()