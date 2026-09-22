"""Focused monitoring contracts with a fake W&B SDK."""

import json
import sys
import types

import pytest

from project.jepa_distill.monitoring import MetricProgress, WandbMonitor


class FakeRun:
    def __init__(self, run_id):
        self.id = run_id
        self.url = f"https://wandb.invalid/{run_id}"
        self.logs = []
        self.finish_calls = []

    def log(self, payload, step=None):
        self.logs.append((dict(payload), step))

    def finish(self, **kwargs):
        self.finish_calls.append(kwargs)


def fake_wandb(monkeypatch, run_id="student-1"):
    run = FakeRun(run_id)
    calls = []

    def init(**kwargs):
        calls.append(kwargs)
        return run

    sdk = types.SimpleNamespace(
        init=init,
        run=run,
        util=types.SimpleNamespace(generate_id=lambda: "generated-student"),
    )
    monkeypatch.setitem(sys.modules, "wandb", sdk)
    return run, calls


def progress(step=3):
    return MetricProgress(step, 12, 0, 1)


def test_disabled_mode_logs_locally_without_importing_sdk(tmp_path, monkeypatch):
    monkeypatch.setitem(sys.modules, "wandb", None)
    monitor = WandbMonitor(
        {"enabled": False, "mode": "disabled"},
        {"teacher": {"wandb_run_id": "teacher-run"}},
        tmp_path,
    )
    monitor.log_metrics({"train/loss": 1.5}, progress=progress())
    state = monitor.state_dict()
    monitor.finish()

    event = json.loads((tmp_path / "metrics.jsonl").read_text().splitlines()[0])
    assert event["event_index"] == 0
    assert event["progress/optimizer_step"] == 3
    assert state["mode"] == "disabled"
    assert state["run_id"] is None


def test_fresh_and_resume_keep_student_identity_and_event_cursor(tmp_path, monkeypatch):
    run, calls = fake_wandb(monkeypatch)
    config = {"enabled": True, "mode": "online", "project": "toy-student"}
    monitor = WandbMonitor(config, {"teacher": {"run_id": "teacher-run"}}, tmp_path)
    monitor.log_metrics({"train/loss": 2.0}, progress=progress(4))
    monitor.log_metrics({"validation/loss": 1.0}, progress=progress(4))
    state = monitor.state_dict()
    monitor.finish()

    assert calls[0]["id"] == "generated-student"
    assert calls[0]["project"] == "toy-student"
    assert state["run_id"] == run.id
    assert state["event_index"] == 1
    assert state["event_cursor"] == 2
    assert [step for _, step in run.logs] == [0, 1]

    resumed = WandbMonitor(config, {}, tmp_path, checkpoint_state=state)
    resumed.log_evaluation(
        {"collision_rate": 0.25},
        benchmark_name="toy",
        output_name="native_dt03",
        policy_role="student",
        progress=progress(2),
    )
    resumed_state = resumed.state_dict()
    resumed.finish()
    assert calls[1]["id"] == run.id
    assert calls[1]["resume"] == "must"
    assert resumed_state["event_index"] == 2
    assert run.logs[-1][0]["eval/native_dt03/toy/student/collision_rate"] == 0.25


def test_nonfinite_payload_and_invalid_policy_role_fail(tmp_path, monkeypatch):
    fake_wandb(monkeypatch)
    monitor = WandbMonitor({"enabled": True}, {}, tmp_path)
    with pytest.raises(ValueError, match="non-finite"):
        monitor.log_metrics({"train/loss": float("nan")}, progress=progress())
    with pytest.raises(ValueError, match="policy_role"):
        monitor.log_evaluation(
            {"collision_rate": 0.0},
            benchmark_name="toy",
            output_name="native_dt03",
            policy_role="invalid",
            progress=progress(),
        )
    monitor.finish()


def test_resume_reconciles_local_history_ahead_of_checkpoint(tmp_path, monkeypatch):
    run, calls = fake_wandb(monkeypatch)
    config = {"enabled": True, "mode": "online", "project": "toy-student"}
    monitor = WandbMonitor(config, {}, tmp_path)
    monitor.log_metrics({"train/loss": 1.0}, progress=progress(1))
    state = monitor.state_dict()
    monitor.finish()

    with (tmp_path / "metrics.jsonl").open("a", encoding="utf-8") as metrics_file:
        metrics_file.write(json.dumps({"event_index": 5}) + "\n")
    resumed = WandbMonitor(config, {}, tmp_path, checkpoint_state=state)
    resumed.log_metrics({"train/loss": 0.5}, progress=progress(2))
    assert resumed.state_dict()["reconciled_local_history_ahead"] is True
    assert run.logs[-1][1] == 6
    resumed.finish()
    assert calls[1]["id"] == state["run_id"]


def test_resume_reconciles_remote_history_event_cursor(tmp_path, monkeypatch):
    run, _ = fake_wandb(monkeypatch)
    config = {"enabled": True, "mode": "online", "project": "toy-student"}
    monitor = WandbMonitor(config, {}, tmp_path)
    monitor.log_metrics({"train/loss": 1.0}, progress=progress(1))
    state = monitor.state_dict()
    monitor.finish()

    run.event_cursor = 8  # Next remote history event; last event index is 7.
    resumed = WandbMonitor(config, {}, tmp_path, checkpoint_state=state)
    resumed.log_metrics({"train/loss": 0.5}, progress=progress(2))
    assert resumed.state_dict()["remote_last_event_index"] == 7
    assert run.logs[-1][1] == 8
    resumed.finish()
