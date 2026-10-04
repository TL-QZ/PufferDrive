"""nuPlan benchmark protocol and launcher tests without simulator execution."""

import os
import signal
import shutil
import subprocess
import time
from pathlib import Path

import yaml

import project.jepa_distill.evaluate_carla as evaluate_carla
from project.jepa_distill.evaluate_carla import build_protocol_config


REPO_ROOT = Path(__file__).resolve().parents[3]
BENCHMARK_CONFIG = (
    REPO_ROOT
    / "project/baseline_run_sync_2026-08-24/override_config/evaluation_benchmarks.yaml"
)
LAUNCHER = (
    REPO_ROOT / "project/jepa_distill/scripts/evaluate_nuplan_4gpu.sh"
)
CARLA_LAUNCHER = (
    REPO_ROOT / "project/jepa_distill/scripts/evaluate_carla_4gpu.sh"
)


def _checkpoint():
    teacher_args = {
        "env": {"dt": 0.3},
        "policy": {},
        "eval": {},
        "train": {"device": "cpu", "amp": False, "compile": False},
        "vec": {"seed": 0, "num_envs": 1},
        "package": "ocean",
        "policy_name": "Drive",
    }
    return {
        "format": "condition_b_v1",
        "model_config": {"chunk_length": 4, "num_action_classes": 12},
        "config": {"teacher_config": teacher_args},
    }


def _baseline_catalog():
    with BENCHMARK_CONFIG.open(encoding="utf-8") as config_file:
        return yaml.safe_load(config_file)


def test_nuplan_protocol_matches_baseline_replay_settings_and_output_metadata():
    from pufferlib.ocean.evaluation_utils import evaluation_utils as drive_benchmark

    baseline = _baseline_catalog()
    for benchmark_name in ("nuplan_single", "nuplan_multi"):
        benchmark_entry = next(
            item for item in baseline["benchmarks"] if item["name"] == benchmark_name
        )
        protocol = build_protocol_config(
            _checkpoint(),
            execution_k=3,
            checkpoint_path="student.pt",
            output_dir="nuplan/" + benchmark_name,
            device="cpu",
            benchmark=benchmark_name,
        )
        evaluation = protocol["evaluation"]
        resolved_args = drive_benchmark.build_benchmark_args(
            protocol["puffer_args"], benchmark_entry, baseline["env"]
        )

        assert evaluation["benchmark_config"] == str(BENCHMARK_CONFIG.resolve())
        assert evaluation["benchmarks"] == [benchmark_name]
        assert evaluation["num_scenarios"] == benchmark_entry["num_scenarios"] == 1000
        assert evaluation["episode_timesteps"] == benchmark_entry["env"]["scenario_length"] == 200
        assert evaluation["env_overrides"]["dt"] == baseline["env"]["dt"] == 0.1
        assert evaluation["action_selection"] == "mean"
        assert evaluation["num_agents"] == protocol["puffer_args"]["eval"]["num_agents"] == 300
        assert evaluation["vec_overrides"]["num_envs"] == protocol["puffer_args"]["vec"]["num_envs"] == 20
        assert evaluation["output_name"] == "act_k3"
        assert evaluation["output_subdir"] == "standalone"
        assert evaluation["run_id"] == "k3"
        assert resolved_args["env"]["dt"] == 0.1
        assert resolved_args["env"]["resample_replay_to_dt"] is False
        for key in (
            "control_mode",
            "sdc_controller",
            "non_sdc_controller",
            "non_vehicle_controller",
        ):
            assert resolved_args["env"][key] == benchmark_entry["env"][key]
        assert resolved_args["env"]["num_agents"] == 300

        record = evaluate_carla._protocol_record(
            protocol,
            evaluate_carla.validate_checkpoint_metadata(_checkpoint()),
            smoke=False,
        )
        assert record["benchmark"] == benchmark_name
        assert record["scenario_length_steps"] == 200
        assert record["num_scenarios"] == 1000
        assert record["replan_interval_steps"] == 9
        assert record["num_agents"] == 300
        assert record["num_workers"] == 20


def test_run_evaluation_dry_run_accepts_benchmark_without_creating_output(tmp_path, monkeypatch):
    checkpoint_path = tmp_path / "student.pt"
    checkpoint_path.write_bytes(b"stub")
    output_path = tmp_path / "nuplan_single"
    monkeypatch.setattr(
        evaluate_carla,
        "_load_student_checkpoint",
        lambda *args, **kwargs: (None, _checkpoint()),
    )

    records = evaluate_carla.run_evaluation(
        checkpoint_path,
        execution_ks=(2,),
        output_dir=output_path,
        device="cpu",
        dry_run=True,
        benchmark="nuplan_single",
    )

    assert records[0]["benchmark"] == "nuplan_single"
    assert records[0]["scenario_length_steps"] == 200
    assert not output_path.exists()


def test_nuplan_multi_smoke_keeps_all_required_agent_slots():
    protocol = build_protocol_config(
        _checkpoint(),
        execution_k=1,
        checkpoint_path="student.pt",
        output_dir="nuplan/smoke",
        device="cpu",
        smoke=True,
        benchmark="nuplan_multi",
    )
    assert protocol["evaluation"]["num_scenarios"] == 2
    assert protocol["evaluation"]["episode_timesteps"] == 13
    assert protocol["evaluation"]["num_agents"] == 300
    assert protocol["evaluation"]["vec_overrides"]["num_envs"] == 1


def test_nuplan_launcher_dry_run_is_sequential_and_rejects_existing_root(tmp_path):
    launcher_dir = tmp_path / "launcher"
    launcher_dir.mkdir()
    copied_launcher = launcher_dir / "evaluate_nuplan_4gpu.sh"
    shutil.copy2(LAUNCHER, copied_launcher)
    stub = launcher_dir / "evaluate_carla_4gpu.sh"
    stub.write_text(
        "#!/usr/bin/env bash\n"
        "set -euo pipefail\n"
        "{ printf call; for arg in \"$@\"; do printf '\\t%s' \"$arg\"; done; printf '\\n'; } >> \"${LAUNCHER_TRACE}\"\n",
        encoding="utf-8",
    )
    stub.chmod(0o755)

    checkpoint = tmp_path / "checkpoints/student.pt"
    checkpoint.parent.mkdir()
    checkpoint.write_text("stub checkpoint", encoding="utf-8")
    output_root = tmp_path / "results"
    trace = tmp_path / "launcher.trace"
    environment = {
        **os.environ,
        "CUDA_VISIBLE_DEVICES": "2,3,4,5",
        "LAUNCHER_TRACE": str(trace),
    }
    result = subprocess.run(
        [
            "bash",
            str(copied_launcher),
            "--checkpoint",
            "checkpoints/student.pt",
            "--output-dir",
            "results",
            "--smoke",
            "--dry-run",
        ],
        cwd=tmp_path,
        env=environment,
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode == 0, result.stderr
    assert not output_root.exists()
    calls = trace.read_text(encoding="utf-8").splitlines()
    assert len(calls) == 2
    first_call = calls[0].split("\t")
    second_call = calls[1].split("\t")
    assert first_call == [
        "call",
        "--checkpoint",
        str(checkpoint.resolve()),
        "--output-dir",
        str(output_root / "nuplan_single"),
        "--benchmark",
        "nuplan_single",
        "--smoke",
        "--dry-run",
    ]
    assert second_call == [
        "call",
        "--checkpoint",
        str(checkpoint.resolve()),
        "--output-dir",
        str(output_root / "nuplan_multi"),
        "--benchmark",
        "nuplan_multi",
        "--smoke",
        "--dry-run",
    ]

    trace.unlink()
    output_root.mkdir()
    rejected = subprocess.run(
        [
            "bash",
            str(copied_launcher),
            "--checkpoint",
            str(checkpoint),
            "--output-dir",
            str(output_root),
            "--dry-run",
        ],
        cwd=tmp_path,
        env=environment,
        capture_output=True,
        text=True,
        check=False,
    )
    assert rejected.returncode == 2
    assert "Output path already exists" in rejected.stderr
    assert not trace.exists()


def test_carla_4gpu_launcher_forwards_benchmark_with_carla_default(tmp_path):
    launcher_dir = tmp_path / "launcher"
    launcher_dir.mkdir()
    copied_launcher = launcher_dir / "evaluate_carla_4gpu.sh"
    shutil.copy2(CARLA_LAUNCHER, copied_launcher)
    child = launcher_dir / "evaluate_carla.sh"
    child.write_text("#!/usr/bin/env bash\nexit 0\n", encoding="utf-8")
    child.chmod(0o755)
    checkpoint = tmp_path / "student.pt"
    checkpoint.write_text("stub checkpoint", encoding="utf-8")
    environment = {**os.environ, "CUDA_VISIBLE_DEVICES": "2,3,4,5"}

    default_root = tmp_path / "carla-results"
    default_result = subprocess.run(
        [
            "bash",
            str(copied_launcher),
            "--checkpoint",
            str(checkpoint),
            "--output-dir",
            str(default_root),
            "--dry-run",
        ],
        cwd=tmp_path,
        env=environment,
        capture_output=True,
        text=True,
        check=False,
    )
    assert default_result.returncode == 0, default_result.stderr
    assert "--benchmark carla" in default_result.stdout
    assert not default_root.exists()

    nuplan_root = tmp_path / "nuplan-results"
    nuplan_result = subprocess.run(
        [
            "bash",
            str(copied_launcher),
            "--checkpoint",
            str(checkpoint),
            "--output-dir",
            str(nuplan_root),
            "--benchmark",
            "nuplan_multi",
            "--dry-run",
        ],
        cwd=tmp_path,
        env=environment,
        capture_output=True,
        text=True,
        check=False,
    )
    assert nuplan_result.returncode == 0, nuplan_result.stderr
    assert "--benchmark nuplan_multi" in nuplan_result.stdout
    assert not nuplan_root.exists()


def test_nuplan_launcher_forwards_sigint_as_term_to_active_child(tmp_path):
    launcher_dir = tmp_path / "launcher"
    launcher_dir.mkdir()
    copied_launcher = launcher_dir / "evaluate_nuplan_4gpu.sh"
    shutil.copy2(LAUNCHER, copied_launcher)
    child = launcher_dir / "evaluate_carla_4gpu.sh"
    child.write_text(
        "#!/usr/bin/env bash\n"
        "trap '' INT\n"
        "trap 'printf term > \"${SIGNAL_TRACE}\"; exit 0' TERM\n"
        "printf ready > \"${READY_TRACE}\"\n"
        "for tick in {1..100}; do sleep 0.1; done\n",
        encoding="utf-8",
    )
    child.chmod(0o755)
    checkpoint = tmp_path / "student.pt"
    checkpoint.write_text("stub checkpoint", encoding="utf-8")
    ready_trace = tmp_path / "child.ready"
    signal_trace = tmp_path / "child.signal"
    environment = {
        **os.environ,
        "CUDA_VISIBLE_DEVICES": "2,3,4,5",
        "READY_TRACE": str(ready_trace),
        "SIGNAL_TRACE": str(signal_trace),
    }
    process = subprocess.Popen(
        [
            "bash",
            str(copied_launcher),
            "--checkpoint",
            str(checkpoint),
            "--output-dir",
            str(tmp_path / "results"),
        ],
        cwd=tmp_path,
        env=environment,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.PIPE,
        text=True,
        start_new_session=True,
    )
    try:
        for _ in range(100):
            if ready_trace.exists():
                break
            if process.poll() is not None:
                raise AssertionError("launcher exited before starting the benchmark child")
            time.sleep(0.02)
        assert ready_trace.exists()
        process.send_signal(signal.SIGINT)
        assert process.wait(timeout=5) == 130
        assert signal_trace.read_text(encoding="utf-8") == "term"
    finally:
        if process.poll() is None:
            os.killpg(process.pid, signal.SIGTERM)
            process.wait(timeout=5)
