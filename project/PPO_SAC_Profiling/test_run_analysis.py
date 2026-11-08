"""CPU tests for matrix cancellation cleanup and run-summary accounting."""

import json
import os
from pathlib import Path
import subprocess
import sys
import time

import pytest

from project.PPO_SAC_Profiling import analyze_runs, analyze_traces, run_matrix


def test_trace_activity_excludes_annotations_and_retains_late_work(tmp_path):
    path = tmp_path / "trace.json"
    path.write_text(json.dumps({"traceEvents": [
        {"ph": "X", "cat": "user_annotation", "name": "profile_training", "ts": 10, "dur": 20},
        {"ph": "X", "cat": "gpu_user_annotation", "name": "profile_training", "ts": 10, "dur": 20},
        {"ph": "X", "cat": "kernel", "name": "gemm", "ts": 12, "dur": 5},
        {"ph": "X", "cat": "gpu_memcpy", "name": "HtoD", "ts": 20, "dur": 2, "args": {"bytes": 64}},
        {"ph": "X", "cat": "kernel", "name": "late", "ts": 31, "dur": 3},
    ]}))
    result = analyze_traces.summarize_trace(path)
    assert sum(row["duration_us"] for row in result["activities"]) == 10
    assert sum(row["count"] for row in result["activities"]) == 3
    assert sum(row["bytes"] for row in result["activities"]) == 64
    assert result["top_training_kernels_us"] == [("gemm", 5)]
    assert any(row["phase"] == "outside_cpu_scopes" for row in result["activities"])


def _process_state(pid):
    """Return Linux process state, or None once the pid no longer exists."""
    try:
        fields = Path(f"/proc/{pid}/stat").read_text().split()
    except FileNotFoundError:
        return None
    return fields[2]


def _wait_until(predicate, timeout_seconds):
    deadline = time.monotonic() + timeout_seconds
    while time.monotonic() < deadline:
        result = predicate()
        if result:
            return result
        time.sleep(0.01)
    raise AssertionError(f"condition was not met within {timeout_seconds} seconds")


@pytest.mark.skipif(not sys.platform.startswith("linux"), reason="requires Linux process groups and /proc")
def test_stop_owned_group_kills_term_resistant_child_only(tmp_path):
    ready_path = tmp_path / "child-ready"
    child_code = (
        "import os, signal, sys, time; "
        "signal.signal(signal.SIGTERM, signal.SIG_IGN); "
        "open(sys.argv[1], 'w').write(str(os.getpid())); "
        "time.sleep(30)"
    )
    owner_code = (
        "import subprocess, sys, time; "
        "subprocess.Popen([sys.executable, '-c', sys.argv[2], sys.argv[1]]); "
        "time.sleep(30)"
    )
    owned = subprocess.Popen(
        [sys.executable, "-c", owner_code, str(ready_path), child_code],
        start_new_session=True,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )
    unrelated = subprocess.Popen(
        [sys.executable, "-c", "import time; time.sleep(30)"],
        start_new_session=True,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )

    try:
        child_pid = int(_wait_until(
            lambda: (ready_path.read_text().strip() or None) if ready_path.is_file() else None,
            timeout_seconds=2,
        ))
        assert _process_state(child_pid) == "S"

        run_matrix.stop_owned_group(owned)

        assert owned.poll() is not None
        _wait_until(
            lambda: _process_state(child_pid) in (None, "Z"),
            timeout_seconds=2,
        )
        assert unrelated.poll() is None
    finally:
        if owned.poll() is None:
            run_matrix.stop_owned_group(owned)
        if unrelated.poll() is None:
            unrelated.terminate()
            try:
                unrelated.wait(timeout=1)
            except subprocess.TimeoutExpired:
                unrelated.kill()
                unrelated.wait(timeout=1)


def _write_cycle(directory, cycle_index, *, included, intrusive, received,
                 critic_rows, policy_rows, loss_exposures, collection_seconds,
                 optimization_seconds, global_sampled=None):
    global_metrics = {
        "collection_seconds": collection_seconds,
        "optimization_seconds": optimization_seconds,
        "received_rows": received,
        "peak_gpu_allocated_bytes": 1024,
    }
    if global_sampled is not None:
        global_metrics["sampled_transitions"] = global_sampled
    path = directory / f"cycle_{cycle_index:03d}.json"
    path.write_text(json.dumps({
        "cycle_index": cycle_index,
        "measurement_included": included,
        "torch_profiler_intrusive": intrusive,
        "global": global_metrics,
        "ranks": [{
            "critic_loss_rows_local": critic_rows,
            "policy_loss_rows_local": policy_rows,
            "loss_row_exposures_local": loss_exposures,
        }],
    }))
    return path.name


def test_sac_summary_counts_critic_samples_once_and_excludes_warmup_and_trace(tmp_path):
    cycle_files = [
        _write_cycle(
            tmp_path, 0, included=False, intrusive=False, received=1000,
            critic_rows=10000, policy_rows=10000, loss_exposures=20000,
            collection_seconds=10, optimization_seconds=20,
        ),
        _write_cycle(
            tmp_path, 1, included=True, intrusive=False, received=10,
            critic_rows=10, policy_rows=10, loss_exposures=20,
            collection_seconds=1, optimization_seconds=2, global_sampled=10,
        ),
        # Trace flag excludes this cycle even if an upstream record marks it included.
        _write_cycle(
            tmp_path, 2, included=True, intrusive=True, received=1000,
            critic_rows=10000, policy_rows=10000, loss_exposures=20000,
            collection_seconds=10, optimization_seconds=20,
        ),
        _write_cycle(
            tmp_path, 3, included=True, intrusive=False, received=30,
            critic_rows=90, policy_rows=90, loss_exposures=180,
            collection_seconds=3, optimization_seconds=4,
        ),
    ]
    summary_path = tmp_path / "summary.json"
    summary_path.write_text(json.dumps({
        "algorithm": "sac",
        "variant": "stock",
        "world_size": 1,
        "profile_settings": {"device": "cpu"},
        "cycle_files": cycle_files,
    }))

    result = analyze_runs.summarize_run(summary_path)

    assert result["measured_cycles"] == 2
    assert result["received_rows"] == 40
    # A replay row appears in both actor and critic losses, so count critic rows once.
    assert result["sampled_transitions"] == 100
    # Aggregate ratio is sum(samples) / sum(received), not a mean of cycle ratios.
    assert result["training_exposures_per_received_row"] == 2.5
    assert result["learner_microseconds_per_sample"] == 60_000
    assert result["collection_seconds"] == 4
    assert result["training_seconds"] == 6
    assert result["cycle_files"] == [
        str(tmp_path / "cycle_001.json"),
        str(tmp_path / "cycle_003.json"),
    ]
