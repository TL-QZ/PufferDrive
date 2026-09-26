"""Remote selection preserves complete local diagnostics and resume cursors."""
import json

from project.jepa_distill.monitoring import WandbMonitor
from project.jepa_distill.observation_probe.monitoring import KEY_METRICS, ProbeMonitor
from project.jepa_distill.tests.test_monitoring import fake_wandb, progress


def test_probe_remote_selection_preserves_local_diagnostics_and_resume(tmp_path, monkeypatch):
    run, _ = fake_wandb(monkeypatch)
    config = {"enabled": True, "project": "toy-student"}
    metrics = {name: 0.5 for name in KEY_METRICS}
    diagnostics = {
        "probe/windows": 64,
        "probe/prediction/partners/position_mae_m/elements": 128,
        "probe/prediction/context/mae": 0.1,
        "train/gradient_norm": 1.2,
        "train/loss_ego": 0.2,
    }
    metrics.update(diagnostics)
    monitor = ProbeMonitor(config, {}, tmp_path)
    monitor.log_metrics(metrics, progress=progress(3))
    state = monitor.state_dict()
    monitor.finish()

    local = json.loads((tmp_path / "metrics.jsonl").read_text().splitlines()[0])
    assert local["metrics"] == metrics
    uploaded, step = run.logs[0]
    assert len(KEY_METRICS) == 12
    assert {key for key in uploaded if key.startswith(("train/", "probe/"))} == KEY_METRICS
    assert not set(diagnostics).intersection(uploaded)
    assert uploaded["progress/optimizer_step"] == 3
    assert "progress/simulator_transitions" not in uploaded
    assert step == 0

    resumed = ProbeMonitor(config, {}, tmp_path, checkpoint_state=state)
    resumed.log_metrics({"probe/prediction/loss_total": 0.4}, progress=progress(4))
    resumed.finish()
    assert run.logs[-1][1] == 1
    assert len((tmp_path / "metrics.jsonl").read_text().splitlines()) == 2


def test_probe_test_split_keeps_namespace_and_missing_metrics_absent(tmp_path, monkeypatch):
    run, _ = fake_wandb(monkeypatch)
    monitor = ProbeMonitor({"enabled": True}, {}, tmp_path)
    monitor.log_metrics({
        "test/probe/prediction/loss_total": 2.0,
        "test/probe/prediction/loss_total/elements": 8,
    }, progress=progress())
    monitor.finish()
    payload, _ = run.logs[0]
    assert payload["test/probe/prediction/loss_total"] == 2.0
    assert "test/probe/prediction/loss_total/elements" not in payload
    assert "test/probe/prediction/traffic_controls.state/accuracy" not in payload


def test_shared_condition_b_monitor_still_uploads_all_metrics(tmp_path, monkeypatch):
    run, _ = fake_wandb(monkeypatch)
    monitor = WandbMonitor({"enabled": True}, {}, tmp_path)
    monitor.log_metrics({"validation/teacher_kl": 0.2}, progress=progress())
    monitor.finish()
    assert run.logs[0][0]["validation/teacher_kl"] == 0.2
