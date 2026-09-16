# Copyright 2026 The LoongForge Authors.
# SPDX-License-Identifier: Apache-2.0

import json
import os
import shutil
import subprocess
import sys
from pathlib import Path


def run_redactor(source, target, values):
    env = os.environ.copy()
    env.update(values)
    return subprocess.run(
        [sys.executable, "ci/redact_ci_artifact.py", "--input", str(source), "--output", str(target)],
        env=env, capture_output=True, text=True, check=False,
    )


def run_preflight(
    tmp_path, values, suite="llm_vlm", build_image="false", extra_env=None
):
    config = tmp_path / "ci.env"
    config.write_text("\n".join(f"{key}={value}" for key, value in values.items()) + "\n", encoding="utf-8")
    env = os.environ.copy()
    env.update({
        "CI_CONFIG_PATH": str(config),
        "DOCKER_BIN": str(tmp_path / "docker"),
        # Keep preflight tests host-independent: the free-GPU-memory check
        # depends on the machine's current occupancy.
        "LOONGFORGE_MIN_FREE_GPU_MB": "",
    })
    if extra_env:
        for name, value in extra_env.items():
            if value is None:
                env.pop(name, None)
            else:
                env[name] = value
    return subprocess.run(
        ["bash", ".github/scripts/self_runner/preflight.sh", suite, build_image],
        env=env, capture_output=True, text=True, check=False,
    )


def preflight_contract(tmp_path):
    docker = tmp_path / "docker"
    docker.write_text(
        "#!/usr/bin/env bash\n"
        'if [[ "$1" == info ]]; then printf "%s\\n" "${FAKE_DOCKER_ROOT:?}"; exit 0; fi\n'
        'case "$1" in version|buildx|run) exit 0;; esac\n'
        "exit 1\n",
        encoding="utf-8",
    )
    docker.chmod(0o755)
    directories = {
        name: tmp_path / path
        for name, path in {
            "LOONGFORGE_HOST_DATA_ROOT": "data",
            "LOONGFORGE_HOST_OUTPUT_ROOT": "output",
            "LOONGFORGE_RUNNER_LOG_ROOT": "logs",
            "TRITON_LIBCUDA_PATH": "triton",
        }.items()
    }
    for directory in directories.values():
        directory.mkdir()
    return {
        "LOONGFORGE_DEFAULT_IMAGE": "default-image",
        "LOONGFORGE_CONTAINER_DATA_ROOT": "/container/data",
        "LOONGFORGE_CONTAINER_OUTPUT_ROOT": "/container/output",
        "LOONGFORGE_CONTAINER_SOURCE": "/container/source",
        "LOONGFORGE_GPU_DEVICE": "device-token",
        "LOONGFORGE_ALLOW_PR_IMAGE_BUILD": "true",
        **{name: str(path) for name, path in directories.items()},
    }


def fake_host_tools(tmp_path, gpu_output="70000", gpu_status=0, free_kb=400_000_000):
    fake_bin = tmp_path / "host-bin"
    fake_bin.mkdir(parents=True, exist_ok=True)
    nvidia_smi = fake_bin / "nvidia-smi"
    nvidia_smi.write_text(
        "#!/usr/bin/env bash\n"
        f"printf '%b\\n' {gpu_output!r}\n"
        f"exit {gpu_status}\n",
        encoding="utf-8",
    )
    nvidia_smi.chmod(0o755)
    df = fake_bin / "df"
    df.write_text(
        "#!/usr/bin/env bash\n"
        "printf '%s\\n' 'Filesystem 1024-blocks Used Available Capacity Mounted on'\n"
        f"printf '%s\\n' 'fake 500000000 1 {free_kb} 1% /docker'\n",
        encoding="utf-8",
    )
    df.chmod(0o755)
    return str(fake_bin) + os.pathsep + os.environ["PATH"]


def fake_gpu_target_tool(tmp_path, output="8.0", status=0):
    fake_bin = tmp_path / "gpu-target-bin"
    fake_bin.mkdir(parents=True, exist_ok=True)
    nvidia_smi = fake_bin / "nvidia-smi"
    nvidia_smi.write_text(
        "#!/usr/bin/env bash\n"
        f"printf '%b\\n' {output!r}\n"
        f"exit {status}\n",
        encoding="utf-8",
    )
    nvidia_smi.chmod(0o755)
    return str(nvidia_smi)


def test_detect_gpu_target_maps_compute_capability_without_echoing_hardware(tmp_path):
    detector = ".github/scripts/self_runner/detect_gpu_target.sh"
    for compute_cap, expected in (("8.0", "a"), ("12.0", "p")):
        case_dir = tmp_path / compute_cap.replace(".", "_")
        env = os.environ.copy()
        env["NVIDIA_SMI_BIN"] = fake_gpu_target_tool(case_dir, compute_cap)
        result = subprocess.run(
            ["bash", detector], env=env, capture_output=True, text=True, check=False,
        )
        assert result.returncode == 0
        assert result.stdout.strip() == expected
        assert compute_cap not in result.stderr


def test_detect_gpu_target_rejects_mixed_or_unknown_architectures(tmp_path):
    detector = ".github/scripts/self_runner/detect_gpu_target.sh"
    cases = (("8.0\n12.0", "runner-gpu: mixed targets"), ("9.0", "runner-gpu: unsupported"))
    for output, message in cases:
        case_dir = tmp_path / message.rsplit(": ", 1)[-1].replace(" ", "-")
        env = os.environ.copy()
        env["NVIDIA_SMI_BIN"] = fake_gpu_target_tool(case_dir, output)
        result = subprocess.run(
            ["bash", detector], env=env, capture_output=True, text=True, check=False,
        )
        assert result.returncode == 1
        assert result.stderr.strip() == message
        assert "8.0" not in result.stderr and "12.0" not in result.stderr


def test_self_runner_builder_rejects_target_mismatch(tmp_path):
    source = tmp_path / "source"
    source.mkdir()
    config = tmp_path / "builder.env"
    config.write_text("LOONGFORGE_ALLOW_PR_IMAGE_BUILD=true\n", encoding="utf-8")
    env = os.environ.copy()
    env.update({
        "CI_CONFIG_PATH": str(config),
        "DOCKER_BIN": "/bin/false",
        "NVIDIA_SMI_BIN": fake_gpu_target_tool(tmp_path / "p-card", "12.0"),
    })
    result = subprocess.run(
        [
            "bash", ".github/scripts/self_runner/build_candidate_image.sh",
            "--target", "a", "--sha", "a" * 40, "--tree-sha", "b" * 40,
            "--pr", "7", "--source", str(source),
        ], env=env, capture_output=True, text=True, check=False,
    )
    assert result.returncode == 2
    assert result.stderr.strip() == "runner-gpu: target mismatch"


def test_redactor_ignores_short_flag_values_from_environment(tmp_path):
    source = tmp_path / "run.log"
    target = tmp_path / "safe.log"
    source.write_text(
        "[2026-08-28 15:49:04] [INFO] exit code 1, duration 725.7s, 1/1 models\n",
        encoding="utf-8",
    )
    result = run_redactor(source, target, {
        "RUNNER_ALLOW_RUNASROOT": "1",
        "RUNNER_ENVIRONMENT": "0",
        "RUNNER_TEMP": "/long/runner/temp/path",
    })
    assert result.returncode == 0
    text = target.read_text(encoding="utf-8")
    assert "15:49:04" in text
    assert "exit code 1" in text
    assert "1/1 models" in text
    assert "/long/runner/temp/path" not in text


def test_redactor_keeps_result_fields_and_removes_runner_values(tmp_path):
    source = tmp_path / "run.log"
    target = tmp_path / "safe.log"
    source.write_text(
        "status=passed model=pi05_ddp SECRET_VALUE /private/runner/data "
        + "http" + "s://" + "runner" + ".example" + ".invalid/api\n",
        encoding="utf-8",
    )
    result = run_redactor(source, target, {
        "SECRET_VALUE": "SECRET_VALUE",
        "RUNNER_PRIVATE_ROOT": "/private/runner/data",
    })
    assert result.returncode == 0
    text = target.read_text(encoding="utf-8")
    assert "status=passed" in text and "model=pi05_ddp" in text
    assert "SECRET_VALUE" not in text
    assert "/private/runner/data" not in text
    assert "runner" + ".example" + ".invalid" not in text


def test_redactor_filters_json_and_redacts_physical_device(tmp_path):
    source = tmp_path / "run.json"
    target = tmp_path / "safe.json"
    source.write_text(json.dumps({
        "status": "passed",
        "suite": "llm_vlm",
        "model": "pi05_ddp",
        "exit_code": 0,
        "log": "logs/run.log",
        "secret": "do-not-publish",
        "device": "runner-local-gpu",
        "absolute": "/private/runner/data/result.json",
    }), encoding="utf-8")
    result = run_redactor(source, target, {"LOONGFORGE_GPU_DEVICE": "runner-local-gpu"})
    assert result.returncode == 0
    payload = json.loads(target.read_text(encoding="utf-8"))
    assert payload["status"] == "passed"
    assert payload["suite"] == "llm_vlm"
    assert payload["model"] == "pi05_ddp"
    assert payload["exit_code"] == 0
    assert payload["log"] == "logs/run.log"
    assert "secret" not in payload
    assert "runner-local-gpu" not in target.read_text(encoding="utf-8")


def test_redactor_keeps_suite_results_and_redacts_nested_paths(tmp_path):
    source = tmp_path / "results.json"
    target = tmp_path / "safe-results.json"
    source.write_text(json.dumps({
        "finished_at": "2026-08-28 12:12:39",
        "chip": "p",
        "log_dir": "/private/runner/output/native/run_1",
        "auto_collect_baseline": False,
        "results": [
            {
                "model_name": "pi05_ddp",
                "script": "/workspace/source/examples/vla/pi05/run.sh",
                "passed": True,
                "failed_metrics": [],
                "warnings": ["throughput degraded 10% > 5% (soft check, warning only)"],
                "error": "",
                "log_dir": "/private/runner/output/native/run_1/pi05_ddp",
                "duration_sec": 716.2,
                "metrics": [
                    {"iteration": 1, "action_loss": 0.4305, "grad_norm": 1.6127},
                    {"iteration": 2, "action_loss": 0.4516, "grad_norm": 1.7270},
                ],
            },
        ],
        "secret": "do-not-publish",
    }), encoding="utf-8")
    result = run_redactor(source, target, {})
    assert result.returncode == 0
    payload = json.loads(target.read_text(encoding="utf-8"))
    assert payload["chip"] == "p"
    assert payload["auto_collect_baseline"] is False
    assert "secret" not in payload
    entry = payload["results"][0]
    assert entry["model_name"] == "pi05_ddp"
    assert entry["passed"] is True
    assert entry["duration_sec"] == 716.2
    assert entry["metrics"][0]["action_loss"] == 0.4305
    assert "degraded 10%" in entry["warnings"][0]
    # Nested runner-local paths must not survive.
    assert "/private/runner" not in target.read_text(encoding="utf-8")
    assert "/workspace/source" not in target.read_text(encoding="utf-8")


def test_redactor_removes_corporate_hosts_private_ips_and_device_labels(tmp_path):
    source = tmp_path / "hardware.log"
    target = tmp_path / "safe.log"
    private_a = ".".join(("10", "2", "3", "4"))
    private_b = ".".join(("172", "20", "4", "5"))
    private_c = ".".join(("192", "168", "8", "9"))
    source.write_text(
        "host=build01.corp.example " + private_a + " " + private_b + " " + private_c + " "
        + "amp" + "ere " + "sm_" + "80 " + "gfx" + "90a "
        + "GPU-12345678-cafe " + "P" + "6K\n",
        encoding="utf-8",
    )
    result = run_redactor(source, target, {})
    assert result.returncode == 0
    text = target.read_text(encoding="utf-8")
    assert "build01.corp.example" not in text
    assert private_a not in text
    assert private_b not in text
    assert private_c not in text
    assert "amp" + "ere" not in text
    assert "sm_" + "80" not in text
    assert "gfx" + "90a" not in text
    assert "GPU-12345678-cafe" not in text
    assert "P" + "6K" not in text


def test_redactor_rejects_invalid_json(tmp_path):
    source = tmp_path / "broken.json"
    target = tmp_path / "safe.json"
    source.write_text("{not-json", encoding="utf-8")
    result = run_redactor(source, target, {})
    assert result.returncode == 2


def test_preflight_reports_missing_required_variable_without_value(tmp_path):
    docker = tmp_path / "docker"
    docker.write_text("#!/usr/bin/env bash\nexit 0\n", encoding="utf-8")
    docker.chmod(0o755)
    values = {
        "LOONGFORGE_DEFAULT_IMAGE": "ubuntu:24.04",
        "LOONGFORGE_HOST_DATA_ROOT": str(tmp_path / "data"),
        "LOONGFORGE_CONTAINER_DATA_ROOT": "/workspace/data",
        "LOONGFORGE_HOST_OUTPUT_ROOT": str(tmp_path / "output"),
        "LOONGFORGE_CONTAINER_OUTPUT_ROOT": "/workspace/output",
        "LOONGFORGE_CONTAINER_SOURCE": "/workspace/source",
        "LOONGFORGE_RUNNER_LOG_ROOT": str(tmp_path / "logs"),
        "TRITON_LIBCUDA_PATH": "/opt/triton/libcuda",
    }
    for name in ("LOONGFORGE_HOST_DATA_ROOT", "LOONGFORGE_HOST_OUTPUT_ROOT", "LOONGFORGE_RUNNER_LOG_ROOT"):
        directory_names = {
            "LOONGFORGE_HOST_DATA_ROOT": "data",
            "LOONGFORGE_HOST_OUTPUT_ROOT": "output",
            "LOONGFORGE_RUNNER_LOG_ROOT": "logs",
        }
        (tmp_path / directory_names[name]).mkdir()
    result = run_preflight(tmp_path, values)
    assert result.returncode == 2
    assert result.stderr.strip() == "LOONGFORGE_GPU_DEVICE"
    assert str(tmp_path) not in result.stdout + result.stderr


def test_preflight_accepts_a_runner_local_contract_without_echoing_values(tmp_path):
    docker = tmp_path / "docker"
    docker.write_text(
        "#!/usr/bin/env bash\n"
        'case "$1" in version|buildx|run) exit 0;; esac\n'
        "exit 1\n",
        encoding="utf-8",
    )
    docker.chmod(0o755)
    data = tmp_path / "data"
    output = tmp_path / "output"
    logs = tmp_path / "logs"
    triton = tmp_path / "triton"
    for directory in (data, output, logs, triton):
        directory.mkdir()
    source = tmp_path / "source"
    (source / "third_party/Loong-Megatron/megatron/core/transformer").mkdir(parents=True)
    hyper_connection = source / "third_party/Loong-Megatron/megatron/core/transformer/hyper_connection.py"
    hyper_connection.write_text("", encoding="utf-8")
    (source / "tests/llm_vlm").mkdir(parents=True)
    (source / "tests/llm_vlm/main.py").write_text("", encoding="utf-8")
    values = {
        "LOONGFORGE_DEFAULT_IMAGE": "default-image",
        "LOONGFORGE_HOST_DATA_ROOT": str(data),
        "LOONGFORGE_CONTAINER_DATA_ROOT": "/container/data",
        "LOONGFORGE_HOST_OUTPUT_ROOT": str(output),
        "LOONGFORGE_CONTAINER_OUTPUT_ROOT": "/container/output",
        "LOONGFORGE_CONTAINER_SOURCE": "/container/source",
        "LOONGFORGE_RUNNER_LOG_ROOT": str(logs),
        "TRITON_LIBCUDA_PATH": str(triton),
        "LOONGFORGE_GPU_DEVICE": "device-token",
    }
    result = run_preflight(tmp_path, values)
    assert result.returncode == 0
    assert "docker: ok" in result.stdout
    assert "mounts: ok" in result.stdout
    assert "image-device: ok" in result.stdout
    assert "default-image" not in result.stdout
    assert "device-token" not in result.stdout


def test_preflight_fails_closed_when_gpu_query_fails(tmp_path):
    values = preflight_contract(tmp_path)
    path = fake_host_tools(tmp_path, gpu_status=1)
    result = run_preflight(tmp_path, values, extra_env={
        "PATH": path,
        "FAKE_DOCKER_ROOT": str(tmp_path),
        "LOONGFORGE_MIN_FREE_GPU_MB": "60000",
    })

    assert result.returncode == 1
    assert result.stderr.strip() == "gpu-memory: unavailable"
    assert "gpu-memory: ok" not in result.stdout


def test_preflight_gpu_messages_do_not_expose_device_counts_or_memory(tmp_path):
    values = preflight_contract(tmp_path)
    path = fake_host_tools(tmp_path, gpu_output="12345\n70000")
    result = run_preflight(tmp_path, values, extra_env={
        "PATH": path,
        "FAKE_DOCKER_ROOT": str(tmp_path),
        "LOONGFORGE_MIN_FREE_GPU_MB": "60000",
    })

    assert result.returncode == 1
    assert result.stderr.strip() == "gpu-memory: insufficient"
    assert "12345" not in result.stdout + result.stderr
    assert "70000" not in result.stdout + result.stderr
    assert "gpu0" not in result.stdout + result.stderr


def test_preflight_rejects_empty_gpu_inventory(tmp_path):
    values = preflight_contract(tmp_path)
    path = fake_host_tools(tmp_path, gpu_output="")
    result = run_preflight(tmp_path, values, extra_env={
        "PATH": path,
        "FAKE_DOCKER_ROOT": str(tmp_path),
        "LOONGFORGE_MIN_FREE_GPU_MB": "60000",
    })

    assert result.returncode == 1
    assert result.stderr.strip() == "gpu-memory: unavailable"


def test_preflight_uses_default_gpu_threshold_and_reports_generic_success(tmp_path):
    values = preflight_contract(tmp_path)
    path = fake_host_tools(tmp_path, gpu_output="70000")
    result = run_preflight(tmp_path, values, extra_env={
        "PATH": path,
        "FAKE_DOCKER_ROOT": str(tmp_path),
        "LOONGFORGE_MIN_FREE_GPU_MB": None,
    })

    assert result.returncode == 0
    assert "gpu-memory: ok" in result.stdout
    assert "70000" not in result.stdout + result.stderr


def test_preflight_rejects_invalid_gpu_threshold_without_echoing_it(tmp_path):
    values = preflight_contract(tmp_path)
    marker = "invalid-private-value"
    result = run_preflight(tmp_path, values, extra_env={
        "FAKE_DOCKER_ROOT": str(tmp_path),
        "LOONGFORGE_MIN_FREE_GPU_MB": marker,
    })

    assert result.returncode == 2
    assert result.stderr.strip() == "gpu-memory: invalid configuration"
    assert marker not in result.stdout + result.stderr


def test_preflight_checks_docker_storage_before_building_an_image(tmp_path):
    values = preflight_contract(tmp_path)
    path = fake_host_tools(tmp_path, free_kb=100_000_000)
    result = run_preflight(tmp_path, values, build_image="true", extra_env={
        "PATH": path,
        "FAKE_DOCKER_ROOT": str(tmp_path),
    })

    assert result.returncode == 1
    assert result.stderr.strip() == "docker-storage: insufficient"
    assert str(tmp_path) not in result.stdout + result.stderr
    assert "100000000" not in result.stdout + result.stderr


def test_preflight_allows_explicitly_disabling_docker_storage_check(tmp_path):
    values = preflight_contract(tmp_path)
    path = fake_host_tools(tmp_path, free_kb=1)
    result = run_preflight(tmp_path, values, build_image="true", extra_env={
        "PATH": path,
        "FAKE_DOCKER_ROOT": str(tmp_path),
        "LOONGFORGE_MIN_DOCKER_FREE_GB": "",
    })

    assert result.returncode == 0
    assert "docker-storage:" not in result.stdout + result.stderr


def test_preflight_accepts_sufficient_docker_storage(tmp_path):
    values = preflight_contract(tmp_path)
    path = fake_host_tools(tmp_path, free_kb=400_000_000)
    result = run_preflight(tmp_path, values, build_image="true", extra_env={
        "PATH": path,
        "FAKE_DOCKER_ROOT": str(tmp_path),
    })

    assert result.returncode == 0
    assert "docker-storage: ok" in result.stdout


def test_preflight_rejects_invalid_docker_storage_threshold(tmp_path):
    values = preflight_contract(tmp_path)
    marker = "invalid-private-value"
    result = run_preflight(tmp_path, values, build_image="true", extra_env={
        "FAKE_DOCKER_ROOT": str(tmp_path),
        "LOONGFORGE_MIN_DOCKER_FREE_GB": marker,
    })

    assert result.returncode == 2
    assert result.stderr.strip() == "docker-storage: invalid configuration"
    assert marker not in result.stdout + result.stderr


def test_runner_errors_do_not_echo_config_paths(tmp_path):
    missing_config = tmp_path / "private-config.env"
    env = os.environ.copy()
    env.update({"CI_CONFIG_PATH": str(missing_config)})
    result = subprocess.run(
        ["bash", ".github/scripts/self_runner/preflight.sh", "llm_vlm", "false"],
        env=env, capture_output=True, text=True, check=False,
    )
    assert result.returncode == 2
    assert result.stderr.strip() == "config: missing"
    assert str(missing_config) not in result.stderr

    source = tmp_path / "source"
    source.mkdir()
    config = tmp_path / "builder.env"
    config.write_text(
        "LOONGFORGE_ALLOW_PR_IMAGE_BUILD=true\n"
        f"IMAGE_DOCKERFILE={tmp_path / 'private-missing.Dockerfile'}\n",
        encoding="utf-8",
    )
    env.update({
        "CI_CONFIG_PATH": str(config),
        "NVIDIA_SMI_BIN": fake_gpu_target_tool(tmp_path / "a-card", "8.0"),
    })
    result = subprocess.run(
        ["bash", ".github/scripts/self_runner/build_candidate_image.sh", "--target", "a",
         "--sha", "a" * 40, "--tree-sha", "b" * 40, "--pr", "1", "--source", str(source)],
        env=env, capture_output=True, text=True, check=False,
    )
    assert result.returncode == 2
    assert result.stderr.strip() == "dockerfile: missing"
    assert str(source) not in result.stderr

    runner = tmp_path / "private-runner"
    runner.mkdir()
    env.update({
        "SOURCE_DIR": str(source),
        "LOONGFORGE_REGRESSION_RUNNER": str(runner),
        "HEAD_SHA": "a" * 40,
    })
    result = subprocess.run(
        ["bash", ".github/scripts/run_regression.sh", "llm_vlm"],
        env=env, capture_output=True, text=True, check=False,
    )
    assert result.returncode == 2
    assert result.stderr.strip() == "regression runner is not executable"
    assert str(runner) not in result.stdout + result.stderr


def test_internal_source_download_failure_is_explicit_and_redacted(tmp_path):
    manifest = tmp_path / "source-manifest.env"
    internal_url = "https://signed-source.example.invalid/private/archive.tar.gz"
    manifest.write_text(
        "TEST_SOURCE_URL=" + internal_url + "\n"
        "TEST_SOURCE_SHA256=" + "a" * 64 + "\n",
        encoding="utf-8",
    )
    fake_bin = tmp_path / "bin"
    fake_bin.mkdir()
    wget = fake_bin / "wget"
    wget.write_text("#!/usr/bin/env bash\nexit 4\n", encoding="utf-8")
    wget.chmod(0o755)
    env = os.environ.copy()
    env.update({
        "PATH": str(fake_bin) + os.pathsep + env["PATH"],
        "SOURCE_MANIFEST_PATH": str(manifest),
    })
    result = subprocess.run(
        ["bash", "docker/fetch_source.sh", "TEST_SOURCE", str(tmp_path / "source"),
         "source-root", "false"],
        env=env, capture_output=True, text=True, check=False,
    )
    assert result.returncode == 1
    assert result.stderr.strip() == "failed to download internal source: TEST_SOURCE"
    assert internal_url not in result.stdout + result.stderr


def test_regression_outputs_only_a_stable_relative_artifact_name():
    script = Path(".github/scripts/self_runner/run_regression.sh").read_text(encoding="utf-8")
    assert "result_json=%s" not in script
    assert "log_file=%s" not in script
    assert "artifact_dir=loongforge-artifacts" in script
    assert "RUNNER_TEMP" not in script
    assert '"status":"failed"' not in script


def test_candidate_builder_uses_a_trusted_dockerfile_and_scans_the_image():
    script = Path(
        ".github/scripts/self_runner/build_candidate_image.sh"
    ).read_text(encoding="utf-8")
    assert 'trusted_dockerfile="$script_dir/../../../docker/Dockerfile"' in script
    assert '$source_dir/docker/Dockerfile' not in script
    assert '"IMAGE_APT_SOURCES:apt_sources"' in script
    assert '"IMAGE_PIP_CONFIG:pip_config"' in script
    assert '"IMAGE_SOURCE_MANIFEST:source_manifest"' in script
    assert "IMAGE_BUILD_ARG_" in script
    assert 'image_policy.py" "$image_ref"' in script


def test_candidate_builder_passes_runner_secrets_as_buildkit_secrets(tmp_path):
    source = tmp_path / "source"
    source.mkdir()
    (source / "pyproject.toml").write_text("[project]\nname='fixture'\n", encoding="utf-8")
    manifest = tmp_path / "source-manifest.env"
    manifest.write_text("FIXTURE_SOURCE=private-placeholder\n", encoding="utf-8")
    config = tmp_path / "builder.env"
    config.write_text(
        "LOONGFORGE_ALLOW_PR_IMAGE_BUILD=true\n"
        f"IMAGE_SOURCE_MANIFEST={manifest}\n",
        encoding="utf-8",
    )
    docker_log = tmp_path / "docker.log"
    docker = tmp_path / "docker"
    docker.write_text(
        "#!/usr/bin/env bash\n"
        "printf '%s\\n' \"$*\" >> \"$FAKE_DOCKER_LOG\"\n"
        'if [[ "$1 $2" == "image inspect" ]]; then printf "%s" "[{}]"; fi\n'
        "exit 0\n",
        encoding="utf-8",
    )
    docker.chmod(0o755)
    env = os.environ.copy()
    env.update({
        "CI_CONFIG_PATH": str(config),
        "DOCKER_BIN": str(docker),
        "FAKE_DOCKER_LOG": str(docker_log),
        "NVIDIA_SMI_BIN": fake_gpu_target_tool(tmp_path / "a-card", "8.0"),
    })
    result = subprocess.run(
        [
            "bash",
            ".github/scripts/self_runner/build_candidate_image.sh",
            "--target", "a",
            "--sha", "a" * 40,
            "--tree-sha", "b" * 40,
            "--pr", "7",
            "--source", str(source),
        ],
        env=env,
        capture_output=True,
        text=True,
        check=False,
    )

    assert result.returncode == 0
    assert result.stdout.strip().startswith("loongforge-ci:pr-7-head-")
    commands = docker_log.read_text(encoding="utf-8")
    assert "--secret id=source_manifest,src=" in commands
    assert str(manifest) in commands


def test_release_builder_reuses_candidate_build_contract_with_release_tag(tmp_path):
    source = tmp_path / "source"
    source.mkdir()
    (source / "pyproject.toml").write_text("[project]\nname='fixture'\n", encoding="utf-8")
    pip_config = tmp_path / "pip.conf"
    pip_config.write_text("[global]\nindex-url=https://example.invalid/simple\n", encoding="utf-8")
    config = tmp_path / "builder.env"
    config.write_text(
        "LOONGFORGE_ALLOW_RELEASE_IMAGE_BUILD=true\n"
        f"IMAGE_PIP_CONFIG={pip_config}\n",
        encoding="utf-8",
    )
    docker_log = tmp_path / "docker.log"
    docker = tmp_path / "docker"
    docker.write_text(
        "#!/usr/bin/env bash\n"
        "printf '%s\\n' \"$*\" >> \"$FAKE_DOCKER_LOG\"\n"
        'if [[ "$1 $2" == "image inspect" ]]; then printf "%s" "[{}]"; fi\n'
        "exit 0\n",
        encoding="utf-8",
    )
    docker.chmod(0o755)
    env = os.environ.copy()
    env.update({
        "CI_CONFIG_PATH": str(config),
        "DOCKER_BIN": str(docker),
        "FAKE_DOCKER_LOG": str(docker_log),
        "NVIDIA_SMI_BIN": fake_gpu_target_tool(tmp_path / "p-card", "12.0"),
    })
    image_ref = "docker.io/loongforge/loongforge:1.2.3"
    result = subprocess.run(
        [
            "bash",
            ".github/scripts/self_runner/build_candidate_image.sh",
            "--release",
            "--image-ref", image_ref,
            "--target", "auto",
            "--sha", "a" * 40,
            "--tree-sha", "b" * 40,
            "--source", str(source),
        ],
        env=env,
        capture_output=True,
        text=True,
        check=False,
    )

    assert result.returncode == 0
    assert result.stdout.strip() == image_ref
    commands = docker_log.read_text(encoding="utf-8")
    assert f"-t {image_ref}" in commands
    assert "--build-arg COMPILE_ENV=blackwell" in commands
    assert "--secret id=pip_config,src=" in commands
    assert "io.loongforge.release=true" in commands
    assert "io.loongforge.candidate=true" not in commands
    assert "image prune" not in commands


def test_candidate_builder_rejects_proxy_credentials_without_echoing_them(tmp_path):
    source = tmp_path / "source"
    source.mkdir()
    config = tmp_path / "builder.env"
    config.write_text("LOONGFORGE_ALLOW_PR_IMAGE_BUILD=true\n", encoding="utf-8")
    marker = "fixture-password"
    env = os.environ.copy()
    env.update({
        "CI_CONFIG_PATH": str(config),
        "DOCKER_BIN": "/bin/false",
        "HTTP_PROXY": f"http://fixture-user:{marker}@proxy.example.invalid:8080",
        "NVIDIA_SMI_BIN": fake_gpu_target_tool(tmp_path / "a-card", "8.0"),
    })
    result = subprocess.run(
        [
            "bash",
            ".github/scripts/self_runner/build_candidate_image.sh",
            "--target", "a",
            "--sha", "a" * 40,
            "--tree-sha", "b" * 40,
            "--pr", "7",
            "--source", str(source),
        ],
        env=env,
        capture_output=True,
        text=True,
        check=False,
    )

    assert result.returncode == 2
    assert result.stderr.strip() == "proxy: embedded credentials are not allowed"
    assert marker not in result.stdout + result.stderr


def test_regression_clears_stale_workspace_artifacts_between_runs(tmp_path):
    docker = tmp_path / "docker"
    docker.write_text(
        "#!/usr/bin/env bash\n"
        "if [[ \"$1\" == container && \"$2\" == inspect ]]; then exit 1; fi\n"
        "exit 0\n",
        encoding="utf-8",
    )
    docker.chmod(0o755)
    data = tmp_path / "data"
    output = tmp_path / "output"
    logs = tmp_path / "logs"
    triton = tmp_path / "triton"
    for directory in (data, output, logs, triton):
        directory.mkdir()
    source = tmp_path / "source"
    (source / "third_party/Loong-Megatron/megatron/core/transformer").mkdir(parents=True)
    hyper_connection = source / "third_party/Loong-Megatron/megatron/core/transformer/hyper_connection.py"
    hyper_connection.write_text("", encoding="utf-8")
    (source / "tests/llm_vlm").mkdir(parents=True)
    (source / "tests/llm_vlm/main.py").write_text("", encoding="utf-8")
    config = tmp_path / "ci.env"
    config.write_text(
        "\n".join([
            "LOONGFORGE_DEFAULT_IMAGE=default-image",
            f"LOONGFORGE_HOST_DATA_ROOT={data}",
            "LOONGFORGE_CONTAINER_DATA_ROOT=/container/data",
            f"LOONGFORGE_HOST_OUTPUT_ROOT={output}",
            "LOONGFORGE_CONTAINER_OUTPUT_ROOT=/container/output",
            "LOONGFORGE_CONTAINER_SOURCE=/container/source",
            f"LOONGFORGE_RUNNER_LOG_ROOT={logs}",
            f"TRITON_LIBCUDA_PATH={triton}",
            "LOONGFORGE_GPU_DEVICE=device-token",
            "LOONGFORGE_PULL_IMAGE=false",
        ]) + "\n",
        encoding="utf-8",
    )
    env = os.environ.copy()
    env.update({"CI_CONFIG_PATH": str(config), "DOCKER_BIN": str(docker), "SOURCE_DIR": str(source),
                "MODELS": "test_model", "LOONGFORGE_MIN_FREE_GPU_MB": ""})
    artifact_root = Path("loongforge-artifacts")
    try:
        first = subprocess.run(
            ["bash", ".github/scripts/run_regression.sh", "llm_vlm"],
            env={**env, "HEAD_SHA": "a" * 40}, capture_output=True, text=True, check=False,
        )
        assert first.returncode == 0
        (artifact_root / "stale-marker").write_text("old", encoding="utf-8")
        second = subprocess.run(
            ["bash", ".github/scripts/run_regression.sh", "llm_vlm"],
            env={**env, "HEAD_SHA": "b" * 40}, capture_output=True, text=True, check=False,
        )
        assert second.returncode == 0
        assert not (artifact_root / "stale-marker").exists()
        assert list(artifact_root.glob("*.result.json"))
    finally:
        shutil.rmtree(artifact_root, ignore_errors=True)
