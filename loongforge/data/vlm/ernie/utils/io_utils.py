# Copyright (c) 2025 PaddlePaddle Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
""" Download utils """

import base64
import datetime
import hashlib
import io
import os
import threading
import uuid
from pathlib import Path

import requests
from PIL import Image
from decord import VideoReader

RAW_VIDEO_DIR = "./download_tmp/raw_video/"
RAW_IMAGE_DIR = "./download_tmp/raw_images/"
EXTRACTED_FRAME_DIR = "./download_tmp/extracted_frames/"
TMP_DIR = "./download_tmp/upload_tmp/"


def file_download(url, download_dir, save_to_disk=False, retry=0, retry_interval=3):
    """
    Description: Download url, if url is PIL, return directly
    Args:
        url(str, PIL): http/local path/io.Bytes, note that io.Bytes is the image byte stream
        download_path: when save_to_disk=True, return the saved address
        save_to_disk: whether to save in the local path

    """

    if isinstance(url, Image.Image):
        return url
    elif isinstance(url, VideoReader):
        return url
    elif url.startswith("http"):
        response = requests.get(url)
        bytes_data = response.content
    elif os.path.isfile(url):
        if save_to_disk:
            return url
        bytes_data = open(url, "rb").read()
    else:
        bytes_data = base64.b64decode(url)
    if not save_to_disk:
        return bytes_data

    download_path = os.path.join(download_dir, get_filename(url))
    Path(download_path).parent.mkdir(parents=True, exist_ok=True)
    with open(download_path, "wb") as f:
        f.write(bytes_data)
    return download_path


def get_filename(url=None):
    """
    Get Filename
    """
    if url is None:
        return str(uuid.uuid4()).replace("-", "")
    t = datetime.datetime.now()
    if not isinstance(url, bytes):
        url = url.encode("utf-8")

    md5_hash = hashlib.md5(url).hexdigest()
    pid = os.getpid()
    tid = threading.get_ident()

    # Remove the suffix to prevent save-jpg from reporting errors
    image_filname = f"{t.year}-{t.month:02d}-{t.day:02d}-{pid}-{tid}-{md5_hash}"
    return image_filname


def get_downloadable(
    url, download_dir=RAW_VIDEO_DIR, save_to_disk=False, retry=0, retry_interval=3
):
    """download video and store it in the disk

    return downloaded **path** if save_to_disk is set to true
    return downloaded **bytes** if save_to_disk is set to false
    """

    if not os.path.exists(download_dir):
        os.makedirs(download_dir)
    downloaded_path = file_download(
        url,
        download_dir,
        save_to_disk=save_to_disk,
        retry=retry,
        retry_interval=retry_interval,
    )
    return downloaded_path


def str2hash(url):
    """
    str2hash
    """
    return hashlib.sha256(url.encode()).hexdigest()


def pil2hash(pil):
    """
    PIL.Image to hash
    """
    byte_io = io.BytesIO()
    pil.save(byte_io, format="PNG")  # avoid compression effects
    image_bytes = byte_io.getvalue()

    return hashlib.sha256(image_bytes).hexdigest()


def get_hashable(to_be_hashed):
    """get hashable"""
    if isinstance(to_be_hashed, bytes):
        return to_be_hashed
    elif isinstance(to_be_hashed, Image.Image):
        return to_be_hashed.tobytes()
    elif isinstance(to_be_hashed, str):
        return to_be_hashed.encode("utf-8")
    else:
        raise ValueError(f"not support type: {type(to_be_hashed)}")

def image_info_2_hash(img_one):
    """
    image info to hash
    """
    if isinstance(img_one["image_url"], str):
        return str2hash(img_one["image_url"])
    elif isinstance(img_one["image_url"], Image.Image):
        return pil2hash(img_one["image_url"])
    else:
        raise ValueError("only support str or PIL.Image now.")
