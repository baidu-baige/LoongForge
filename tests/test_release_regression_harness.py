# Copyright 2026 The LoongForge Authors.
# SPDX-License-Identifier: Apache-2.0

import subprocess
import os
import shlex
import ast
import importlib.util
import json
import math
from pathlib import Path

import pytest


REPOSITORY_ROOT = Path(__file__).parent.parent
RELEASE_REGRESSION_SCRIPT = (
    REPOSITORY_ROOT
    / ".github"
    / "scripts"
    / "self_runner"
    / "run_release_regression.sh"
)
CREATE_CONTAINER_SCRIPT = (
    REPOSITORY_ROOT
    / ".github"
    / "scripts"
    / "self_runner"
    / "create_container.sh"
)


def test_release_regression_script_is_valid_shell():
    result = subprocess.run(
        ["bash", "-n", str(RELEASE_REGRESSION_SCRIPT)],
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode == 0
    assert result.stdout == ""
    assert result.stderr == ""


def test_plot_output_expands_the_configured_environment(monkeypatch, tmp_path):
    import yaml
    module = ast.parse((REPOSITORY_ROOT / "tests/llm_vlm/tasks/base_task.py").read_text())
    cls = next(node for node in module.body if isinstance(node, ast.ClassDef) and node.name == "BaseTask")
    method = next(node for node in cls.body if isinstance(node, ast.FunctionDef) and node.name == "_resolve_training_log_path")
    method.decorator_list = []
    namespace = {"os": os, "yaml": yaml}
    exec(compile(ast.Module(body=[method], type_ignores=[]), "<plot-path>", "exec"), namespace)
    config = tmp_path / "common.yaml"
    config.write_text("training_log_path: ${TRAINING_LOG_PATH}\npfs_path: ${PFS_PATH}\n")
    monkeypatch.setenv("TRAINING_LOG_PATH", str(tmp_path / "mounted-output"))
    assert namespace["_resolve_training_log_path"](str(config)) == str(tmp_path / "mounted-output")


def test_release_metrics_archive_matches_the_baseline_and_preserves_failures(tmp_path):
    spec = importlib.util.spec_from_file_location("release_evidence", REPOSITORY_ROOT / ".github/scripts/release_evidence.py")
    evidence = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(evidence)
    llm_log = tmp_path / "llm.log"
    llm_log.write_text(
        "training_phase ... pretrain\n"
        "iteration 1/ 20 | elapsed time per iteration (ms): 100 | lm loss: 2.5 | grad norm: 1.2\n"
        "training_phase ... sft\n"
        "iteration 1/ 20 | elapsed time per iteration (ms): 150 | lm loss: 1.5 | grad norm: 0.8\n"
    )
    vla_results = tmp_path / "results.json"
    vla_results.write_text(json.dumps({"results": [{
        "model_name": "pi05_ddp", "passed": False, "failed_metrics": ["action_loss"],
        "warnings": ["throughput soft check"], "metrics": [
            {"iteration": 1, "action_loss": 0.4, "grad_norm": float("nan"), "secret": "discard"},
        ],
    }]}))
    result = evidence.collect(
        source=REPOSITORY_ROOT, llm_log=llm_log, llm_model="deepseek_v2_lite",
        llm_chip="a", vla_results=vla_results, vla_chip="p", passed=True,
    )
    llm, sft, vla = result["results"]
    assert llm["actual"][0]["lm_loss"] == 2.5
    assert llm["baseline"][0]["lm_loss"] == 2.556303
    assert llm["grad_norm_checked"] is False
    assert len(llm["actual"]) == 1
    assert sft["training_type"] == "sft"
    assert sft["actual"] == [{
        "iteration": 1, "elapsed_time_ms": 150, "lm_loss": 1.5, "grad_norm": 0.8,
    }]
    assert sft["baseline"] == []
    assert vla["passed"] is False
    assert vla["failed_metrics"] == ["action_loss"]
    assert "grad_norm" not in vla["actual"][0]
    assert "secret" not in vla["actual"][0]


def test_release_regression_sets_single_node_distributed_environment():
    script = RELEASE_REGRESSION_SCRIPT.read_text(encoding="utf-8")

    for name, value in (
        ("RANK", "0"),
        ("WORLD_SIZE", "1"),
        ("MASTER_ADDR", "127.0.0.1"),
    ):
        assert f'-e "{name}={value}"' in script

    assert '-e "MASTER_PORT=$master_port"' in script
    assert "master_port=$((20000 + ($$ % 30000)))" in script
    assert '"source_sha":"%s"' in script
    assert '"execution_slot":"%s"' in script
    assert '"image_digest":"%s"' in script


def test_remote_regression_preserves_run_attempt_and_archives_failed_evidence(tmp_path):
    source = tmp_path / "source"
    source.mkdir()
    bin_dir = tmp_path / "bin"
    bin_dir.mkdir()
    command_log = tmp_path / "ssh.log"
    sync_log = tmp_path / "rsync.log"
    for name, text in {
        "ssh": (
            'printf "%s\\n" "$*" >> "$TEST_COMMAND_LOG"\n'
            '[[ "$*" != *run_release_regression.sh* ]] || exit 7\n'
        ),
        "rsync": 'printf "%s\\n" "$*" >> "$TEST_SYNC_LOG"\n',
    }.items():
        executable = bin_dir / name
        executable.write_text("#!/usr/bin/env bash\n" + text)
        executable.chmod(0o755)
    env = os.environ.copy()
    env.update({
        "PATH": f"{bin_dir}:{env['PATH']}",
        "TEST_COMMAND_LOG": str(command_log),
        "TEST_SYNC_LOG": str(sync_log),
        "LOONGFORGE_EXECUTION_MODE": "ssh",
        "LOONGFORGE_EXECUTION_ALIAS": "executor",
        "LOONGFORGE_EXECUTION_ROOT": "/release-root",
        "LOONGFORGE_EXECUTION_SOURCE": "/release-source",
        "LOONGFORGE_EXECUTION_CONFIG": "/release-config.env",
        "LOONGFORGE_RELEASE_EVIDENCE_ROOT": str(tmp_path / "evidence"),
        "GITHUB_RUN_ID": "123",
        "GITHUB_RUN_ATTEMPT": "2",
    })
    result = subprocess.run(
        [str(RELEASE_REGRESSION_SCRIPT), "--image", "registry.example/image:test",
         "--source", str(source), "--sha", "a" * 40, "--baseline-chip", "p"],
        env=env, cwd=tmp_path, capture_output=True, text=True, check=False,
    )
    assert result.returncode == 7
    commands = command_log.read_text()
    assert "GITHUB_RUN_ID='123' GITHUB_RUN_ATTEMPT='2'" in commands
    assert "--baseline-chip 'p'" in commands
    assert str(tmp_path / "evidence/123/loongforge-release-artifacts/") in sync_log.read_text()


def test_local_regression_uses_requested_chip_for_both_suites_and_evidence(tmp_path):
    source = tmp_path / "source"
    marker = source / "third_party/Loong-Megatron/megatron/core/transformer"
    marker.mkdir(parents=True)
    (marker / "hyper_connection.py").touch()
    log = tmp_path / "docker.log"
    docker = tmp_path / "docker"
    docker.write_text(
        '#!/usr/bin/env bash\n'
        'printf "%s\\n" "$*" >> "$TEST_DOCKER_LOG"\n'
        'if [[ "$1 $2" == "container inspect" ]]; then exit 1; fi\n'
        'if [[ "$1 $2" == "image inspect" ]]; then echo sha256:test; fi\n'
        'if [[ "$*" == *LOONGFORGE_TEST_SUITE=llm_vlm* && "$1" == exec ]]; then\n'
        '  echo "training_phase ... pretrain"\n'
        '  echo "iteration 1/ 20 | lm loss: 2.5 | grad norm: 1.2"\n'
        'fi\n'
        'exit 0\n'
    )
    docker.chmod(0o755)
    config = tmp_path / "image.env"
    settings = {
        "DOCKER_BIN": str(docker),
        "LOONGFORGE_DEFAULT_IMAGE": "example/image:test",
        "LOONGFORGE_HOST_DATA_ROOT": str(tmp_path),
        "LOONGFORGE_HOST_LLM_DATA_ROOT": str(tmp_path),
        "LOONGFORGE_HOST_OUTPUT_ROOT": str(tmp_path / "output"),
        "LOONGFORGE_RUNNER_LOG_ROOT": str(tmp_path / "logs"),
        "LOONGFORGE_CONTAINER_DATA_ROOT": "/ci-data",
        "LOONGFORGE_CONTAINER_LLM_DATA_ROOT": "/ci-llm-data",
        "LOONGFORGE_CONTAINER_OUTPUT_ROOT": "/ci-output",
        "LOONGFORGE_CONTAINER_SOURCE": "/ci-source",
        "TRITON_LIBCUDA_PATH": str(tmp_path),
        "LOONGFORGE_GPU_DEVICE": "/dev/null",
        "LOONGFORGE_MIN_FREE_GPU_MB": "",
        # Deliberately disagree with the explicit request.
        "LOONGFORGE_BASELINE_LLM_VLM": "a",
        "LOONGFORGE_BASELINE_EMBODIED": "a",
    }
    config.write_text("".join(f"{key}={shlex.quote(value)}\n" for key, value in settings.items()))
    env = os.environ.copy()
    env.update({
        "CI_CONFIG_PATH_IMAGE": str(config),
        "LOONGFORGE_EXECUTION_MODE": "local",
        "TEST_DOCKER_LOG": str(log),
        "LOONGFORGE_RELEASE_EVIDENCE_ROOT": "",
        "GITHUB_RUN_ID": "123", "GITHUB_RUN_ATTEMPT": "1",
    })
    # Evidence reads baselines/parser from the tested source, as in production.
    (source / "tests").symlink_to(REPOSITORY_ROOT / "tests", target_is_directory=True)
    result = subprocess.run(
        [str(RELEASE_REGRESSION_SCRIPT), "--image", "example/image:test",
         "--source", str(source), "--sha", "a" * 40, "--baseline-chip", "p"],
        env=env, cwd=tmp_path, capture_output=True, text=True, check=False,
    )
    assert result.returncode == 0, result.stderr
    commands = log.read_text().splitlines()
    training = [line for line in commands if line.startswith("exec ") and "--chip " in line]
    assert len(training) == 2
    assert all("--chip p " in line for line in training)
    metrics = json.loads(
        (tmp_path / "loongforge-release-artifacts/regression-metrics.json").read_text()
    )
    case = metrics["results"][0]
    assert case["baseline_chip"] == "p"
    assert case["baseline"][0]["lm_loss"] == 2.55532


def test_p_chip_deepseek_baseline_has_complete_numeric_pretrain_metrics():
    path = REPOSITORY_ROOT / "tests/llm_vlm/baseline/default/p/deepseek_v2_lite.json"
    baseline = json.loads(path.read_text())
    assert set(baseline) == {"pretrain"}
    records = baseline["pretrain"]
    assert [record["iteration"] for record in records] == list(range(1, 21))
    fields = {
        "iteration", "lm_loss", "grad_norm", "elapsed_time_ms", "throughput",
        "mem_allocated_avg_MB", "mem_max_allocated_avg_MB",
    }
    assert all(set(record) == fields for record in records)
    assert all(math.isfinite(value) for record in records for value in record.values())
    assert records[0]["lm_loss"] == 2.55532
    assert records[-1]["lm_loss"] == 2.250846


def test_baseline_reader_never_falls_back_to_another_chip_or_phase(monkeypatch, tmp_path):
    monkeypatch.syspath_prepend(str(REPOSITORY_ROOT / "tests/llm_vlm"))
    monkeypatch.chdir(tmp_path)
    spec = importlib.util.spec_from_file_location(
        "release_config_manager", REPOSITORY_ROOT / "tests/llm_vlm/tools/config_manager.py",
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    monkeypatch.setattr(module, "__file__", str(tmp_path / "tools/config_manager.py"))
    baseline_root = tmp_path / "baseline/default"
    (baseline_root / "a").mkdir(parents=True)
    (baseline_root / "a/model.json").write_text('{"pretrain": [{"iteration": 1}]}')
    with pytest.raises(FileNotFoundError):
        module.ConfigManager.get_baseline_data(None, {}, "model", "pretrain", "p")
    (baseline_root / "p/model.json").write_text('{"pretrain": [{"iteration": 1}]}')
    with pytest.raises(ValueError, match="sft"):
        module.ConfigManager.get_baseline_data(None, {}, "model", "sft", "p")


def test_release_container_copies_megatron_into_writable_source(tmp_path):
    source = tmp_path / "source"
    megatron = source / "third_party" / "Loong-Megatron"
    megatron_marker = megatron / "megatron" / "core" / "transformer"
    megatron_marker.mkdir(parents=True)
    (megatron_marker / "hyper_connection.py").write_text("stub\n", encoding="utf-8")
    (source / "loongforge.txt").write_text("stub\n", encoding="utf-8")

    host_data = tmp_path / "host-data"
    host_llm_data = tmp_path / "host-llm-data"
    host_output = tmp_path / "host-output"
    container_source = tmp_path / "container-source"
    for directory in (host_data, host_llm_data, host_output):
        directory.mkdir()

    docker_log = tmp_path / "docker.log"
    fake_docker = tmp_path / "fake-docker"
    fake_docker.write_text(
        "\n".join(
            [
                "#!/usr/bin/env bash",
                "set -e",
                'if [[ "$1" == "container" && "$2" == "inspect" ]]; then exit 1; fi',
                'if [[ "$1" == "run" ]]; then',
                f'    printf \'%s\\n\' "$*" >> {shlex.quote(str(docker_log))}',
                "    exit 0",
                "fi",
                'if [[ "$1" == "exec" ]]; then',
                "    shift 2",
                '    exec "$@"',
                "fi",
                "exit 2",
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    fake_docker.chmod(0o755)

    env = os.environ.copy()
    env.update(
        {
            "DOCKER_BIN": str(fake_docker),
            "LOONGFORGE_PULL_IMAGE": "false",
            "LOONGFORGE_HOST_DATA_ROOT": str(host_data),
            "LOONGFORGE_CONTAINER_DATA_ROOT": "/ci-data",
            "LOONGFORGE_HOST_LLM_DATA_ROOT": str(host_llm_data),
            "LOONGFORGE_CONTAINER_LLM_DATA_ROOT": "/ci-llm-data",
            "LOONGFORGE_HOST_OUTPUT_ROOT": str(host_output),
            "LOONGFORGE_CONTAINER_OUTPUT_ROOT": "/ci-output",
            "LOONGFORGE_CONTAINER_SOURCE": str(container_source),
            "LOONGFORGE_CONTAINER_SOURCE_MOUNT": str(source),
            "TRITON_LIBCUDA_PATH": "/opt/triton/libcuda",
            "LOONGFORGE_GPU_DEVICE": "/dev/null",
        }
    )

    result = subprocess.run(
        [
            str(CREATE_CONTAINER_SCRIPT),
            "registry.example/loongforge:test",
            str(source),
            "release",
            "loongforge-test-container",
        ],
        env=env,
        capture_output=True,
        text=True,
        check=False,
    )

    assert result.returncode == 0
    assert result.stdout == ""
    assert result.stderr == ""
    assert (container_source / "loongforge.txt").is_file()
    assert (
        container_source
        / "third_party"
        / "Loong-Megatron"
        / "megatron"
        / "core"
        / "transformer"
        / "hyper_connection.py"
    ).is_file()

    docker_args = docker_log.read_text(encoding="utf-8")
    assert "--device=/dev/null" in docker_args
    assert f"{source}:{source}:ro" in docker_args
    assert "/workspace/Loong-Megatron:ro" not in docker_args
