# Copyright 2026 The LoongForge Authors.
# SPDX-License-Identifier: Apache-2.0

"""Utilities for the LoongForge VLA evaluation module."""

from __future__ import annotations

import argparse
import os
import signal
import subprocess
import time
from pathlib import Path
from urllib.request import urlopen


class ManagedServer:
    """Provide ManagedServer behavior."""

    def __init__(self, args: argparse.Namespace) -> None:
        """Run __init__."""
        self.args = args
        self.process: subprocess.Popen | None = None

    @property
    def health_url(self) -> str:
        """Run health_url."""
        return f"http://{self.args.host}:{self.args.health_port}/healthz"

    def start(self) -> None:
        """Run start."""
        self.process = start_server(self.args)
        wait_healthz(self.health_url, self.args.server_start_timeout_sec, self.process)

    def ensure_running(self) -> None:
        """Run ensure_running."""
        if self.process is None or self.process.poll() is not None:
            self.stop()
            self.start()
            return
        try:
            with urlopen(self.health_url, timeout=2) as response:
                if response.status == 200:
                    return
        except Exception:
            pass
        self.stop()
        self.start()

    def stop(self) -> None:
        """Run stop."""
        if self.process is None:
            return
        if self.process.poll() is None:
            try:
                os.killpg(self.process.pid, signal.SIGTERM)
            except ProcessLookupError:
                pass
            try:
                self.process.wait(timeout=15)
            except subprocess.TimeoutExpired:
                try:
                    os.killpg(self.process.pid, signal.SIGKILL)
                except ProcessLookupError:
                    pass
                self.process.wait(timeout=15)
        self.process = None


def wait_healthz(url: str, timeout_sec: float, process: subprocess.Popen | None) -> None:
    """Run wait_healthz."""
    deadline = time.time() + timeout_sec
    last_error = None
    while time.time() < deadline:
        if process is not None and process.poll() is not None:
            raise RuntimeError(f"Policy server exited before healthy: {process.returncode}")
        try:
            with urlopen(url, timeout=2) as response:
                if response.status == 200:
                    return
        except Exception as exc:
            last_error = exc
        time.sleep(1)
    raise TimeoutError(f"Policy server did not become healthy at {url}: {last_error}")


def ensure_libero_config(args: argparse.Namespace) -> None:
    """Run ensure_libero_config."""
    if not args.libero_config_path:
        return
    config_dir = Path(args.libero_config_path)
    config_dir.mkdir(parents=True, exist_ok=True)
    config_file = config_dir / "config.yaml"
    if not config_file.exists():
        root = Path(os.environ.get("LIBERO_ROOT", "/workspace/LIBERO")) / "libero" / "libero"
        config_file.write_text(
            f"benchmark_root: {root}\n"
            f"bddl_files: {root}/bddl_files\n"
            f"init_states: {root}/init_files\n"
            f"datasets: {root}/../datasets\n"
            f"assets: {root}/assets\n",
            encoding="utf-8",
        )
    os.environ["LIBERO_CONFIG_PATH"] = str(config_dir)


def apply_runtime_env(args: argparse.Namespace, env: dict[str, str]) -> None:
    """Run apply_runtime_env."""
    if args.ld_library_path:
        env["LD_LIBRARY_PATH"] = f"{args.ld_library_path}:{env.get('LD_LIBRARY_PATH', '')}"
    env["LD_LIBRARY_PATH"] = f"/usr/lib64:{env.get('LD_LIBRARY_PATH', '')}"
    if args.mujoco_gl:
        env["MUJOCO_GL"] = args.mujoco_gl
    if args.pyopengl_platform:
        env["PYOPENGL_PLATFORM"] = args.pyopengl_platform
    if args.libero_config_path:
        env["LIBERO_CONFIG_PATH"] = args.libero_config_path


def start_server(args: argparse.Namespace) -> subprocess.Popen:
    """Run start_server."""
    eval_root = Path(args.eval_root).resolve()
    loongforge_root = Path(args.loongforge_root).resolve() if args.loongforge_root else eval_root.parent
    env = os.environ.copy()
    env["PYTHONPATH"] = f"{loongforge_root}:{eval_root}:{env.get('PYTHONPATH', '')}"
    apply_runtime_env(args, env)
    ensure_libero_config(args)

    if args.server_kind in {"loongforge", "pi05", "loongforge_pi05"}:
        command = [
            args.server_python,
            "-m",
            "loongforge.embodied.eval.servers.loongforge_server",
            "--config",
            args.config,
        ]
    elif args.server_kind == "mock":
        command = [
            args.server_python,
            "-m",
            "loongforge.embodied.eval.servers.mock_server",
            "--port",
            str(args.port),
            "--health-port",
            str(args.health_port),
            "--action-dim",
            str(args.action_dim),
        ]
    else:
        raise ValueError(f"Unsupported model backend for LoongForge eval: {args.server_kind!r}")
    Path(args.server_log).parent.mkdir(parents=True, exist_ok=True)
    log = open(args.server_log, "a", buffering=1)
    return subprocess.Popen(
        command,
        cwd=str(eval_root),
        env=env,
        stdin=subprocess.DEVNULL,
        stdout=log,
        stderr=subprocess.STDOUT,
        start_new_session=True,
    )


def build_argparser() -> argparse.ArgumentParser:
    """Run build_argparser."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="")
    parser.add_argument("--server-kind", default="mock")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=10093)
    parser.add_argument("--health-port", type=int, default=10094)
    parser.add_argument("--server-start-timeout-sec", type=float, default=180.0)
    parser.add_argument(
        "--server-log",
        default="/workspace/LoongForge-VLA/loongforge/embodied/eval/reports/manual/policy_server.log",
    )
    parser.add_argument("--action-dim", type=int, default=7)
    parser.add_argument("--server-python", default="/workspace/miniconda3/envs/loongforge/bin/python")
    parser.add_argument("--loongforge-root", default="/workspace/LoongForge-VLA")
    parser.add_argument("--eval-root", default="/workspace/LoongForge-VLA/loongforge/embodied/eval")
    parser.add_argument("--libero-config-path", default="")
    parser.add_argument("--mujoco-gl", default="")
    parser.add_argument("--pyopengl-platform", default="")
    parser.add_argument("--ld-library-path", default="")
    return parser
