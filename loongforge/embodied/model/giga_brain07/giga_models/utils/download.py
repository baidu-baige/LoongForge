import os
import shutil
import subprocess
import sys
from typing import Optional, Sequence

import gdown
from huggingface_hub import hf_hub_download, snapshot_download
from huggingface_hub.file_download import repo_folder_name
from torch.hub import download_url_to_file

from .paths import get_repo_dir
from .utils import wrap_call

python = sys.executable


def run(command: str, folder_path: Optional[str] = None, try_until_success: bool = True, **kwargs) -> None:
    """Run shell command with retry logic and cleanup on failure.

    Args:
        command: Shell command to execute.
        folder_path: Path to folder that will be deleted on failure (if try_until_success=True).
        try_until_success: If True, retry indefinitely and clean up folder_path on failure; if False, raise on first failure.
        **kwargs: Additional arguments passed to subprocess.run().
    """
    run_kwargs = {
        'args': command,
        'shell': True,
        'env': os.environ,
        'encoding': 'utf8',
        'errors': 'ignore',
        'stdout': subprocess.PIPE,
        'stderr': subprocess.PIPE,
    }
    run_kwargs.update(kwargs)
    while True:
        print(command)
        result = subprocess.run(**run_kwargs)
        if result.returncode == 0:
            break
        else:
            if try_until_success:
                print(result.stderr, result.stdout)
                if folder_path is not None and os.path.exists(folder_path):
                    shutil.rmtree(folder_path)
            else:
                raise ValueError(result.stderr, result.stdout)


def git_clone(
    url_path: str,
    repo_dir: Optional[str] = None,
    repo_name: Optional[str] = None,
    force: bool = False,
    recursive: bool = False,
    **kwargs,
) -> str:
    """Clone a git repository to local directory.

    Args:
        url_path: Git repository URL (should end with .git).
        repo_dir: Parent directory for cloned repo; defaults to get_repo_dir().
        repo_name: Name for local repo directory; auto-extracted from url_path if None.
        force: If True, remove existing repo and re-clone; if False, return existing path.
        recursive: If True, use --recursive flag to clone submodules.
        **kwargs: Additional arguments passed to run().

    Returns:
        str: Path to the cloned repository.
    """
    if repo_dir is None:
        repo_dir = get_repo_dir()
    if repo_name is None:
        repo_name = url_path.split('/')[-1]
        assert repo_name.endswith('.git')
        repo_name = repo_name[:-4]
    repo_path = os.path.join(repo_dir, repo_name)
    if os.path.exists(repo_path):
        if force:
            shutil.rmtree(repo_path)
        else:
            return repo_path
    if recursive:
        command = f'git clone --recursive {url_path} {repo_path}'
    else:
        command = f'git clone {url_path} {repo_path}'
    run(command, folder_path=repo_path, try_until_success=True, **kwargs)
    return repo_path


def run_pip(command: str, **kwargs) -> None:
    """Run pip command using current Python executable.

    Args:
        command: Pip command arguments (e.g., "install package").
        **kwargs: Additional arguments passed to run().
    """
    run(f'{python} -m pip {command}', try_until_success=False, **kwargs)


def run_python(command: str, **kwargs) -> None:
    """Run Python script or command using current Python executable.

    Args:
        command: Python command or script path with arguments.
        **kwargs: Additional arguments passed to run().
    """
    run(f'{python} {command}', try_until_success=False, **kwargs)


def download_from_huggingface(
    repo_id: str,
    root_dir: Optional[str] = None,
    local_dir: Optional[str] = None,
    filenames: Optional[Sequence[str] | str] = None,
    folders: Optional[Sequence[str] | str] = None,
    repo_type: Optional[str] = None,
    force: bool = False,
    **kwargs,
) -> str:
    """Download files or entire repository from Hugging Face Hub.

    Args:
        repo_id: Repository ID on Hugging Face (e.g., "username/repo-name").
        root_dir: Root directory for downloads; uses HF cache env vars if None.
        local_dir: Specific local directory path; computed from root_dir and repo_id if None.
        filenames: Specific file(s) to download; downloads entire repo if None.
        folders: Specific folder(s) to download; downloads entire repo if None.
        repo_type: Type of repo ("model", "dataset", or "space"); defaults to "model".
        force: If True, re-download even if files exist locally.
        **kwargs: Additional arguments passed to hf_hub_download or snapshot_download.

    Returns:
        str: Path to downloaded file or directory.
    """
    if repo_type is None:
        repo_type = 'model'
    if local_dir is None:
        if root_dir is None:
            if repo_type == 'model':
                root_dir = os.environ.get('HF_HUB_CACHE', '~/.cache/huggingface/models')
            elif repo_type == 'dataset':
                root_dir = os.environ.get('HF_DATASETS_CACHE', '~/.cache/huggingface/datasets')
            elif repo_type == 'space':
                root_dir = os.environ.get('HF_HUB_CACHE', '~/.cache/huggingface/spaces')
            else:
                assert False
        local_dir = os.path.join(root_dir, repo_folder_name(repo_id=repo_id, repo_type=repo_type))
    cache_dir = os.path.join(local_dir, '.cache')
    if filenames is not None:
        filenames = [filenames] if isinstance(filenames, str) else filenames
        local_paths = []
        for filename in filenames:
            local_path = os.path.join(local_dir, filename)
            local_paths.append(local_path)
            if os.path.exists(local_path):
                if force:
                    os.remove(local_path)
                else:
                    continue
            print(f'download from {repo_id} to {local_path}')
            wrap_call(hf_hub_download)(
                repo_id=repo_id,
                filename=filenames[0],
                cache_dir=cache_dir,
                local_dir=local_dir,
                **kwargs,
            )
        local_path = local_paths[0] if len(local_paths) == 1 else local_paths
    elif folders is not None:
        folders = [folders] if isinstance(folders, str) else folders
        local_paths = []
        for folder in folders:
            local_path = os.path.join(local_dir, folder)
            local_paths.append(local_path)
            if os.path.exists(local_path):
                if force:
                    os.remove(local_path)
                else:
                    continue
            print(f'download from {repo_id} to {local_path}')
            wrap_call(snapshot_download)(
                repo_id=repo_id,
                cache_dir=cache_dir,
                local_dir=local_dir,
                allow_patterns=f'{folder}/*',
                **kwargs,
            )
        local_path = local_paths[0] if len(local_paths) == 1 else local_paths
    else:
        download_full = 'allow_patterns' not in kwargs and 'ignore_patterns' not in kwargs
        if os.path.exists(local_dir) and download_full:
            if force:
                shutil.rmtree(local_dir)
            else:
                return local_dir
        print(f'download from {repo_id} to {local_dir}')
        local_path = wrap_call(snapshot_download)(
            repo_id=repo_id,
            cache_dir=cache_dir,
            local_dir=local_dir,
            **kwargs,
        )
    if os.path.exists(cache_dir):
        shutil.rmtree(cache_dir)
    return local_path


def download_from_url(url: str, local_path: str, force: bool = False) -> str:
    """Download file from URL to local path.

    Args:
        url: URL to download from.
        local_path: Local file path to save downloaded file.
        force: If True, re-download even if file exists locally.

    Returns:
        str: Path to downloaded file.
    """
    if os.path.exists(local_path):
        if force:
            os.remove(local_path)
        else:
            return local_path
    os.makedirs(os.path.dirname(local_path), exist_ok=True)
    print(f'download from {url} to {local_path}')
    wrap_call(download_url_to_file)(url, local_path)
    return local_path


def download_from_gdrive(file_id: str, local_path: str, force: bool = False, **kwargs) -> str:
    """Download file from Google Drive to local path.

    Args:
        file_id: Google Drive file ID.
        local_path: Local file path to save downloaded file.
        force: If True, re-download even if file exists locally.
        **kwargs: Additional arguments passed to gdown.download().

    Returns:
        str: Path to downloaded file.
    """
    if os.path.exists(local_path):
        if force:
            os.remove(local_path)
        else:
            return local_path
    os.makedirs(os.path.dirname(local_path), exist_ok=True)
    print(f'download from {file_id} to {local_path}')
    wrap_call(gdown.download)(id=file_id, output=local_path, **kwargs)
    return local_path
