"""Lightweight subprocess tests for the four-GPU rendering dispatcher."""

from __future__ import annotations

import os
from pathlib import Path
import shutil
import signal
import subprocess
import time

import pytest


REPO_ROOT = Path(__file__).resolve().parents[3]
PARALLEL_LAUNCHER = REPO_ROOT / "project/jepa_distill/scripts/render_carla_4gpu.sh"


FAKE_SINGLE_LAUNCHER = r'''#!/usr/bin/env bash
set -euo pipefail
K=""
OUTPUT_DIR=""
DRY_RUN=0
while (( $# )); do
    case "$1" in
        --k) K="$2"; shift 2 ;;
        --output-dir) OUTPUT_DIR="$2"; shift 2 ;;
        --dry-run) DRY_RUN=1; shift ;;
        *) shift ;;
    esac
done
if [[ "${DRY_RUN}" == 1 || "${DRY_RUN:-0}" == 1 ]]; then
    printf 'preflight|%s|%s\n' "${K}" "${OUTPUT_DIR}" >>"${MOCK_RECORD}"
    if [[ "${K}" == "${MOCK_FAIL_PREFLIGHT_K:-}" ]]; then
        echo "mock preflight failure k=${K}" >&2
        exit 19
    fi
    exit 0
fi
printf 'job|%s|%s|%s\n' "${K}" "${CUDA_VISIBLE_DEVICES}" "${OUTPUT_DIR}" >>"${MOCK_RECORD}"
echo "mock renderer k=${K} gpu=${CUDA_VISIBLE_DEVICES} output=${OUTPUT_DIR}"
if [[ "${K}" == "${MOCK_FAIL_K:-}" ]]; then
    echo "mock failure k=${K}" >&2
    exit 23
fi
if [[ "${K}" == "${MOCK_HOLD_K:-}" ]]; then
    trap 'echo terminated >"${MOCK_TERM_FILE}"; exit 143' TERM
    sleep 60 &
    echo "$!" >"${MOCK_CHILD_PID_FILE}"
    wait
fi
'''


def _launcher_fixture(tmp_path: Path, *, cuda_visible_devices: str = "2,3,4,5"):
    script_dir = tmp_path / "scripts"
    script_dir.mkdir()
    launcher_copy = script_dir / PARALLEL_LAUNCHER.name
    shutil.copy2(PARALLEL_LAUNCHER, launcher_copy)
    single_launcher = script_dir / "render_carla.sh"
    single_launcher.write_text(FAKE_SINGLE_LAUNCHER, encoding="utf-8")
    single_launcher.chmod(0o755)

    eval_root = tmp_path / "evaluation"
    eval_root.mkdir()
    output_dir = tmp_path / "render_output"
    record = tmp_path / "launches.txt"
    env = os.environ.copy()
    env.update(
        {
            "CUDA_VISIBLE_DEVICES": cuda_visible_devices,
            "MOCK_RECORD": str(record),
            "OMP_NUM_THREADS": "1",
            "MKL_NUM_THREADS": "1",
            "OPENBLAS_NUM_THREADS": "1",
        }
    )
    return launcher_copy, eval_root, output_dir, record, env


def _command(script: Path, eval_root: Path, output_dir: Path, *extra_args: str):
    return [
        "bash",
        str(script),
        "--eval-root",
        str(eval_root),
        "--output-dir",
        str(output_dir),
        *extra_args,
    ]


@pytest.mark.parametrize("dry_run_source", ["flag", "environment"])
def test_four_gpu_dry_run_preflights_without_writes_or_render_jobs(tmp_path, dry_run_source):
    script, eval_root, output_dir, record, env = _launcher_fixture(tmp_path)
    extra_args = ("--dry-run",)
    if dry_run_source == "environment":
        env["DRY_RUN"] = "1"
        extra_args = ()

    result = subprocess.run(
        _command(script, eval_root, output_dir, "--failure-mode", "offroad", *extra_args),
        env=env,
        text=True,
        capture_output=True,
        check=False,
        timeout=10,
    )

    assert result.returncode == 0, result.stderr
    assert not output_dir.exists()
    assert record.read_text(encoding="utf-8").splitlines() == [
        f"preflight|{execution_k}|{output_dir}/run_k{execution_k}"
        for execution_k in range(1, 5)
    ]
    for execution_k, gpu_id in enumerate(("2", "3", "4", "5"), start=1):
        assert f"k={execution_k} GPU={gpu_id} log={output_dir}/k{execution_k}.log" in result.stdout
        assert f"--output-dir {output_dir}/run_k{execution_k}" in result.stdout
        assert f"--k {execution_k} --device cuda:0" in result.stdout
    assert "--failure-mode offroad" in result.stdout


def test_four_gpu_dispatch_maps_devices_outputs_and_logs_independently(tmp_path):
    script, eval_root, output_dir, record, env = _launcher_fixture(tmp_path)

    result = subprocess.run(
        _command(script, eval_root, output_dir),
        env=env,
        text=True,
        capture_output=True,
        check=False,
        timeout=15,
    )

    assert result.returncode == 0, result.stderr
    records = record.read_text(encoding="utf-8").splitlines()
    assert records[:4] == [
        f"preflight|{execution_k}|{output_dir}/run_k{execution_k}"
        for execution_k in range(1, 5)
    ]
    jobs = {
        line.split("|", 2)[1]: line.split("|", 2)[2]
        for line in records[4:]
    }
    assert len(jobs) == 4
    for execution_k, gpu_id in enumerate(("2", "3", "4", "5"), start=1):
        assert jobs[str(execution_k)].startswith(f"{gpu_id}|{output_dir}/run_k{execution_k}")
        log_path = output_dir / f"k{execution_k}.log"
        assert log_path.is_file()
        assert f"mock renderer k={execution_k} gpu={gpu_id}" in log_path.read_text(encoding="utf-8")


def test_four_gpu_dispatch_propagates_child_failure_and_preserves_other_logs(tmp_path):
    script, eval_root, output_dir, record, env = _launcher_fixture(tmp_path)
    env["MOCK_FAIL_K"] = "2"

    result = subprocess.run(
        _command(script, eval_root, output_dir),
        env=env,
        text=True,
        capture_output=True,
        check=False,
        timeout=15,
    )

    assert result.returncode != 0
    assert "k=2 failed" in result.stderr
    assert (output_dir / "k2.log").is_file()
    assert "mock failure k=2" in (output_dir / "k2.log").read_text(encoding="utf-8")
    assert len([line for line in record.read_text(encoding="utf-8").splitlines() if line.startswith("job|")]) == 4


def test_four_gpu_dispatch_stops_when_any_preflight_fails(tmp_path):
    script, eval_root, output_dir, record, env = _launcher_fixture(tmp_path)
    env["MOCK_FAIL_PREFLIGHT_K"] = "3"

    result = subprocess.run(
        _command(script, eval_root, output_dir),
        env=env,
        text=True,
        capture_output=True,
        check=False,
        timeout=10,
    )

    assert result.returncode != 0
    assert "mock preflight failure k=3" in result.stderr
    assert record.read_text(encoding="utf-8").splitlines() == [
        f"preflight|{execution_k}|{output_dir}/run_k{execution_k}"
        for execution_k in range(1, 4)
    ]
    assert not output_dir.exists()
    assert not any(line.startswith("job|") for line in record.read_text(encoding="utf-8").splitlines())


@pytest.mark.parametrize("gpu_ids", ["2,3,4", "2,3,4,2"])
def test_four_gpu_dispatch_rejects_non_four_or_duplicate_gpu_ids(tmp_path, gpu_ids):
    script, eval_root, output_dir, record, env = _launcher_fixture(
        tmp_path, cuda_visible_devices=gpu_ids
    )

    result = subprocess.run(
        _command(script, eval_root, output_dir, "--dry-run"),
        env=env,
        text=True,
        capture_output=True,
        check=False,
        timeout=10,
    )

    assert result.returncode != 0
    assert "CUDA_VISIBLE_DEVICES" in result.stderr or "Duplicate GPU index" in result.stderr
    assert not record.exists()
    assert not output_dir.exists()


@pytest.mark.parametrize("interrupt_signal", [signal.SIGINT, signal.SIGTERM])
def test_interrupt_terminates_render_child_process_group(tmp_path, interrupt_signal):
    script, eval_root, output_dir, record, env = _launcher_fixture(tmp_path)
    term_file = tmp_path / "term.txt"
    child_pid_file = tmp_path / "child.pid"
    env.update(
        {
            "MOCK_HOLD_K": "1",
            "MOCK_TERM_FILE": str(term_file),
            "MOCK_CHILD_PID_FILE": str(child_pid_file),
        }
    )
    process = subprocess.Popen(
        _command(script, eval_root, output_dir),
        env=env,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )
    child_pid = None
    deadline = time.monotonic() + 10
    while time.monotonic() < deadline:
        if child_pid_file.is_file():
            child_pid = int(child_pid_file.read_text(encoding="utf-8"))
            break
        if process.poll() is not None:
            break
        time.sleep(0.05)
    assert child_pid is not None, "the k=1 fixture child did not start"

    process.send_signal(interrupt_signal)
    stdout, stderr = process.communicate(timeout=10)

    assert process.returncode == 130, (stdout, stderr)
    assert term_file.is_file(), "the child shell did not receive the process-group signal"
    assert _process_is_stopped(child_pid), f"fixture child process {child_pid} survived interruption"


def _process_is_stopped(process_id: int) -> bool:
    """A process is stopped when it is absent or is only a reaped zombie."""

    proc_stat = Path(f"/proc/{process_id}/stat")
    deadline = time.monotonic() + 3
    while time.monotonic() < deadline:
        if not proc_stat.exists():
            return True
        fields = proc_stat.read_text(encoding="utf-8").split()
        if len(fields) > 2 and fields[2] == "Z":
            return True
        try:
            os.kill(process_id, 0)
        except ProcessLookupError:
            return True
        time.sleep(0.05)
    return False
