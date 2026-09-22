"""Student adapter and bounded evaluator tests."""

import torch
from torch import nn
import yaml

from project.jepa_distill.evaluate import (
    StudentPolicyAdapter,
    _checkpoint_output_defaults,
    evaluate_student,
)


class ToyStudent(nn.Module):
    def __init__(self):
        super().__init__()
        self.head = nn.Linear(3, 48)

    def forward(self, observations):
        return self.head(observations).reshape(observations.shape[0], 4, 12)


class ToyEnv:
    def __init__(self):
        self.reset_count = 0
        self.step_count = 0

    def reset(self):
        self.reset_count += 1
        return torch.zeros(3).numpy(), {}

    def step(self, action):
        self.step_count += 1
        done = self.step_count % 2 == 0
        return torch.zeros(3).numpy(), 0.0, done, {"progress": float(self.step_count)}


class CaptureMonitor:
    def __init__(self):
        self.events = []

    def log_evaluation(self, metrics, **kwargs):
        self.events.append((dict(metrics), kwargs))


def jerk_tables():
    physical = torch.tensor(
        [[longitudinal, lateral] for longitudinal in (-15.0, -4.0, 0.0, 4.0) for lateral in (-4.0, 0.0, 4.0)]
    )
    normalized = physical.clone()
    normalized[:, 0] = torch.where(normalized[:, 0] < 0, normalized[:, 0] / 15.0, normalized[:, 0] / 4.0)
    normalized[:, 1] /= 4.0
    return normalized, physical


def test_adapter_uses_slot_zero_and_drive_piecewise_mean_conversion():
    normalized, physical = jerk_tables()
    adapter = StudentPolicyAdapter(
        ToyStudent(), action_table=normalized, action_table_physical=physical
    )
    (logits,), value = adapter.forward_eval(torch.zeros(2, 3))
    assert logits.shape == (2, 12)
    assert value.shape == (2, 1)
    assert adapter.is_continuous is False
    torch.testing.assert_close(
        adapter.discrete_actions_to_continuous(torch.tensor([0, 11])),
        normalized[[0, 11]],
    )
    probabilities = torch.zeros(1, 12)
    probabilities[0, 0] = 1.0
    torch.testing.assert_close(
        adapter.discrete_probs_to_continuous_mean(probabilities),
        normalized[[0]],
    )


def test_adapter_rejects_recurrent_state():
    adapter = StudentPolicyAdapter(ToyStudent())
    try:
        adapter.forward_eval(torch.zeros(1, 3), {"lstm_h": torch.zeros(1, 1)})
    except ValueError as exc:
        assert "feed-forward" in str(exc)
    else:
        raise AssertionError("expected recurrent state rejection")


def test_injected_evaluation_is_bounded_and_logs_student_namespace():
    monitor = CaptureMonitor()
    env = ToyEnv()
    results = evaluate_student(
        {
            "num_scenarios": 2,
            "episode_timesteps": 3,
            "benchmarks": ["toy"],
            "output_name": "native_dt03",
            "action_selection": "mean",
        },
        student=ToyStudent(),
        env=env,
        monitor=monitor,
    )
    assert results["toy"]["summary"]["num_scenarios"] == 2
    assert env.reset_count == 2
    assert env.step_count == 4
    assert monitor.events[0][1]["policy_role"] == "student"
    assert monitor.events[0][1]["benchmark_name"] == "toy"
    assert monitor.events[0][0]["num_timesteps"] == 4


def test_checkpoint_teacher_config_reconstructs_eval_args_without_teacher_logging(tmp_path, monkeypatch):
    checkpoint = {
        "model": ToyStudent(),
        "config": {
            "teacher_config": {
                "env": {},
                "policy": {},
                "eval": {"action_selection": "mean"},
                "train": {},
                "vec": {},
                "package": "ocean",
                "policy_name": "Drive",
                "wandb": True,
                "neptune": True,
                "tb": True,
                "load_id": "teacher-run",
                "load_model_path": "teacher.pt",
            }
        },
        "step": 7,
    }
    checkpoint_path = tmp_path / "student.pt"
    torch.save(checkpoint, checkpoint_path)
    captured = {}

    def fake_eval(**kwargs):
        captured.update(kwargs)
        return {"toy": {"summary": {"metrics_mean": {"goal_rate": 1.0}}}}

    import pufferlib.pufferl

    monkeypatch.setattr(pufferlib.pufferl, "eval", fake_eval)
    monitor = CaptureMonitor()
    evaluate_student(
        {
            "student_checkpoint": str(checkpoint_path),
            "num_scenarios": 1,
            "benchmarks": ["toy"],
            "output_name": "native_dt03",
            "wandb": {"enabled": False, "mode": "disabled"},
        },
        monitor=monitor,
    )
    assert captured["args"]["wandb"] is False
    assert captured["args"]["load_id"] is None
    assert captured["args"]["load_model_path"] is None
    assert monitor.events[0][1]["policy_role"] == "student"


def test_evaluation_runtime_overrides_teacher_device_precision_and_seed(tmp_path, monkeypatch):
    benchmark_config = tmp_path / "benchmarks.yaml"
    benchmark_config.write_text(
        yaml.safe_dump(
            {
                "env": {"dt": 0.1},
                "benchmarks": [
                    {
                        "name": "toy",
                        "seed": 3,
                        "num_scenarios": 99,
                        "env": {
                            "simulation_mode": "gigaflow",
                            "dt": 0.1,
                            "scenario_length": 12,
                        },
                    }
                ],
            }
        )
    )
    captured = {}

    def fake_eval(**kwargs):
        captured.update(kwargs)
        return {"toy": {"summary": {"metrics_mean": {"goal_rate": 1.0}}}}

    import pufferlib.pufferl

    monkeypatch.setattr(pufferlib.pufferl, "eval", fake_eval)
    monitor = CaptureMonitor()
    evaluate_student(
        {
            "puffer_args": {
                "env": {},
                "policy": {},
                "eval": {},
                "train": {
                    "device": "cuda:7",
                    "amp": True,
                    "compile": True,
                    "seed": 2,
                },
                "vec": {"seed": 2},
                "package": "ocean",
                "policy_name": "Drive",
                "wandb": False,
            },
            "num_scenarios": 2,
            "benchmarks": ["toy"],
            "num_agents": 32,
            "vec": {"num_envs": 2, "num_workers": 2, "batch_size": 2},
            "benchmark_config": str(benchmark_config),
            "env_overrides": {"dt": 0.3},
            "output_root": str(tmp_path),
            "run_id": "eval",
            "device": "cpu",
            "amp": False,
            "compile": False,
            "seed": 17,
        },
        student=ToyStudent(),
        monitor=monitor,
    )
    assert captured["args"]["train"]["device"] == "cpu"
    assert captured["args"]["train"]["amp"] is False
    assert captured["args"]["train"]["compile"] is False
    assert captured["args"]["train"]["seed"] == 17
    assert captured["args"]["vec"]["seed"] == 17
    assert captured["args"]["eval"]["num_agents"] == 32
    assert captured["args"]["vec"]["num_envs"] == 2
    assert captured["args"]["vec"]["num_workers"] == 2
    assert captured["args"]["vec"]["batch_size"] == 2
    with open(captured["args"]["eval"]["benchmark_config"], encoding="utf-8") as config_file:
        resolved = yaml.safe_load(config_file)
    assert resolved["benchmarks"][0]["seed"] == 17
    assert resolved["benchmarks"][0]["num_scenarios"] == 2
    assert resolved["benchmarks"][0]["env"]["scenario_length"] == 4


def test_null_output_location_comes_from_checkpoint_training_config():
    resolved = _checkpoint_output_defaults(
        {"run_id": None},
        {"run_id": None, "output_root": None},
        {"config": {"training": {"run_id": "student-run", "output_root": "/tmp/student-runs"}}},
        None,
    )
    assert resolved["run_id"] == "student-run"
    assert resolved["output_root"] == "/tmp/student-runs"


def test_unsupported_evaluation_protocols_fail_fast():
    for key, value, message in (
        ("population", "teacher_self_play", "population"),
        ("execution_horizon", 2, "execution_horizon"),
        ("transfer", {"enabled": True}, "transfer.enabled"),
    ):
        config = {"num_scenarios": 1, "episode_timesteps": 1, key: value}
        try:
            evaluate_student(config, student=ToyStudent(), env=ToyEnv(), monitor=CaptureMonitor())
        except ValueError as exc:
            assert message in str(exc)
        else:
            raise AssertionError(f"expected {key} validation to fail")
