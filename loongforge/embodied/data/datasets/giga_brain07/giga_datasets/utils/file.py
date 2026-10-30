import json
import os
import pickle
import shutil
import stat
from typing import Any

import yaml

try:
    from yaml import CDumper as Dumper
    from yaml import CLoader as Loader
except ImportError:
    from yaml import Dumper, Loader


def empty_dir(root_dir: str) -> None:
    """Remove a directory if it exists and recreate it empty.

    Args:
        root_dir (str): Path to the directory that will be (re)created.

    Notes:
        This function deletes the whole directory tree when it exists, so use
        with care if the path may contain important files.
    """
    if os.path.exists(root_dir):
        shutil.rmtree(root_dir)
    os.makedirs(root_dir)


def list_dir(root_dir: str, recursive: bool = False, exts: set[str] | None = None) -> list[str]:
    """List file paths under a directory with optional recursion and filtering.

    Args:
        root_dir (str): Root directory to search.
        recursive (bool): If True, walk subdirectories recursively. If False,
            only list files directly under ``root_dir``.
        exts (set[str] | None): Optional set of file extensions to keep
            (lowercased, including the leading dot, e.g., {'.json', '.yml'}).

    Returns:
        list[str]: Sorted list of file paths that match the criteria.
    """
    file_paths = []
    if recursive:
        for cur_dir, _, file_names in os.walk(root_dir, followlinks=True):
            for file_name in file_names:
                file_path = os.path.join(cur_dir, file_name)
                if os.path.isfile(file_path):
                    if exts is None:
                        file_paths.append(file_path)
                    else:
                        suffix = os.path.splitext(file_name)[1].lower()
                        if suffix in exts:
                            file_paths.append(file_path)
    else:
        for file_name in sorted(os.listdir(root_dir)):
            file_path = os.path.join(root_dir, file_name)
            if os.path.isfile(file_path):
                if exts is None:
                    file_paths.append(file_path)
                else:
                    suffix = os.path.splitext(file_name)[1].lower()
                    if suffix in exts:
                        file_paths.append(file_path)
    file_paths.sort()
    return file_paths


def chmod_dir(root_dir: str, mode: int = stat.S_IRWXU | stat.S_IRWXG | stat.S_IRWXO) -> None:
    """Recursively change permissions for a directory and its files.

    Args:
        root_dir (str): Directory whose contents will be chmod'ed.
        mode (int): Permission bits, e.g., ``stat.S_IRWXU | stat.S_IRWXG``.

    Notes:
        Symbolic links are followed due to ``followlinks=True`` when walking.
    """
    for cur_dir, _, file_names in os.walk(root_dir, followlinks=True):
        os.chmod(cur_dir, mode)
        for file_name in file_names:
            file_path = os.path.join(cur_dir, file_name)
            if os.path.isfile(file_path):
                os.chmod(file_path, mode)


def load_file(file_path: str, **kwargs: Any) -> Any:
    """Load a structured file based on its extension.

    Args:
        file_path (str): Path to file ending with one of {'.pkl', '.pickle',
            '.json', '.yaml', '.yml'}.
        **kwargs (Any): Extra keyword arguments forwarded to the corresponding
            loader:
            - pickle.load(..., **kwargs)
            - json.load(..., **kwargs)
            - yaml.load(..., Loader=..., **kwargs)  (``Loader`` defaulted to CLoader if available)

    Returns:
        Any: The data structure loaded from the file.

    Raises:
        AssertionError: If the extension is unsupported.
    """
    if file_path.endswith('.pkl') or file_path.endswith('.pickle'):
        data = pickle.load(open(file_path, 'rb'), **kwargs)
    elif file_path.endswith('.json'):
        data = json.load(open(file_path, 'r'), **kwargs)
    elif file_path.endswith('.yaml') or file_path.endswith('yml'):
        kwargs.setdefault('Loader', Loader)
        data = yaml.load(open(file_path, 'r'), **kwargs)
    else:
        assert False
    return data


def save_file(file_path: str, data: Any, **kwargs: Any) -> None:
    """Save a Python object to disk based on file extension.

    Args:
        file_path (str): Destination path ending with one of {'.pkl', '.pickle',
            '.json', '.yaml', '.yml'}.
        data (Any): Object to serialize.
        **kwargs (Any): Extra keyword arguments forwarded to the corresponding
            dumper:
            - pickle.dump(obj, file, **kwargs)
            - json.dump(obj, file, indent=4, **kwargs)  (``indent`` defaulted to 4)
            - yaml.dump(obj, file, Dumper=..., **kwargs)  (``Dumper`` defaulted to CDumper if available)

    Raises:
        AssertionError: If the extension is unsupported.
    """
    os.makedirs(os.path.dirname(file_path), exist_ok=True)
    if file_path.endswith('.pkl') or file_path.endswith('.pickle'):
        pickle.dump(data, open(file_path, 'wb'), **kwargs)
    elif file_path.endswith('.json'):
        kwargs.setdefault('indent', 4)
        json.dump(data, open(file_path, 'w'), **kwargs)
    elif file_path.endswith('.yaml') or file_path.endswith('yml'):
        kwargs.setdefault('Dumper', Dumper)
        yaml.dump(data, open(file_path, 'w'), **kwargs)
    else:
        assert False
