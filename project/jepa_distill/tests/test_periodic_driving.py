"""Periodic student-driving evaluation contracts for Condition B training."""

from __future__ import annotations

import importlib
import copy
import random
from pathlib import Path

import numpy as np
import pytest
import torch
import yaml
from torch import nn

from project.jepa_distill import evaluate as evaluate_module
from project.jepa_distill.evaluate import evaluate_student
from project.jepa_distill.monitoring import MetricProgress
from project.jepa_distill.runtime import load_config


TRAIN_MODULE = importlib.import_module("project.jepa_distill.train")
ORCHESTRATION_MODULE = importlib.import_module(
    "project.jepa_distill.tests.test_training_orchestration"
)


class _ToyStudent(nn.Module):
    """Small deterministic policy used at the evaluator boundary."""

    def __init__(self):
        super().__init__()
        self.head = nn.Linear(3, 48)

    def forward(self, observations):
        return self.head(observations).reshape(observations.shape[0], 4, 12)


class _CaptureMonitor:
    def __init__(self):
        self.events = []
        self.finished = False

    def log_evaluation(self, metrics, **kwargs):
        self.events.append((dict(metrics), kwargs))

    def finish(self, **kwargs):
        self.finished = True


def _evaluation_config(tmp_path: Path) -> dict:
    return {
        "puffer_args": {
            "env": {},
            "policy": {},
            "eval": {},
            "train": {"device": "cuda:7", "amp": True, "compile": True, "seed": 2},
            "vec": {"seed": 2},
            "package": "ocean",
            "policy_name": "Drive",
            "wandb": False,
        },
        "num_scenarios": 2,
        "benchmarks": ["carla"],
        "output_name": "training_native",
        "run_id": "student_run",
        "output_root": str(tmp_path),
        "device": "cpu",
        "action_selection": "mean",
        "env_overrides": {"dt": 0.3},
        "wandb": {"enabled": False, "mode": "disabled"},
        "progress": {
            "optimizer_step": 7,
            "simulator_transitions": 14,
            "collection_round_idx": 1,
            "update_epoch_idx": 2,
        },
    }


def _training_driving_config(base: dict, *, enabled: bool, interval_steps: int) -> None:
    base["env_overrides"].update({"action_type": "continuous", "dt": 0.3})
    base["driving_evaluation"] = {
        "enabled": enabled,
        "interval_steps": interval_steps,
        "benchmarks": ["carla"],
        "num_scenarios": 2,
        "episode_timesteps": 4,
        "action_selection": "mean",
        "output_name": "training_native",
        "seed": 42,
        "device": base["training"]["device"],
        "amp": False,
        "compile": False,
    }


def _periodic_eval_settings(payload: dict) -> dict:
    if isinstance(payload.get("driving_evaluation"), dict):
        return payload["driving_evaluation"]
    if isinstance(payload.get("evaluation"), dict):
        return payload["evaluation"]
    return payload


def _periodic_eval_step(payload: dict) -> int:
    settings = _periodic_eval_settings(payload)
    progress = settings.get("progress", payload.get("progress"))
    if isinstance(progress, dict):
        return int(progress["optimizer_step"])
    return int(progress.optimizer_step)


def _install_fake_evaluator(monkeypatch, calls, *, raise_error=False):
    def fake_evaluate_student(*args, **kwargs):
        payload = args[0] if args else kwargs["config"]
        settings = copy.deepcopy(_periodic_eval_settings(payload))
        calls.append(
            {
                "config": settings,
                "student": kwargs.get("student"),
                "monitor": kwargs.get("monitor"),
                "raw_config": copy.deepcopy(payload),
            }
        )
        if raise_error:
            raise RuntimeError("mock driving evaluator failed")
        return {
            "carla": {
                "summary": {"metrics_mean": {"goal_rate": 0.5}}
            }
        }

    monkeypatch.setattr(evaluate_module, "evaluate_student", fake_evaluate_student)
    monkeypatch.setattr(TRAIN_MODULE, "evaluate_student", fake_evaluate_student, raising=False)


def _prepare_training_test(monkeypatch, config, tmp_path):
    def collector(config, output_dir, *, teacher, env, collection_round_idx):
        return ORCHESTRATION_MODULE._write_round_manifest(
            Path(output_dir), collection_round_idx
        )

    ORCHESTRATION_MODULE._patch_training_boundaries(monkeypatch, collector)
    return ORCHESTRATION_MODULE._validation_batches()


def test_train_evaluates_at_interval_and_completion_once_with_native_step_dirs(
    tmp_path, monkeypatch
):
    config = ORCHESTRATION_MODULE._config(
        tmp_path, "periodic", max_optimizer_steps=3, num_collections=1, update_epochs=2
    )
    _training_driving_config(config, enabled=True, interval_steps=2)
    calls = []
    _install_fake_evaluator(monkeypatch, calls)
    validation = _prepare_training_test(monkeypatch, config, tmp_path)
    student = ORCHESTRATION_MODULE.RecordingStudent()
    monitor = ORCHESTRATION_MODULE.RecordingMonitor()

    result = TRAIN_MODULE.train(
        config,
        validation_batches=validation,
        teacher=ORCHESTRATION_MODULE._make_teacher(),
        student=student,
        env=ORCHESTRATION_MODULE.EnvToken(),
        monitor=monitor,
    )

    assert result["optimizer_steps"] == 3
    assert [_periodic_eval_step(call["config"]) for call in calls] == [2, 3]
    assert all(call["student"] is student for call in calls)
    assert all(call["monitor"] is monitor for call in calls)

    locations = []
    for call in calls:
        settings = call["config"]
        assert settings["env_overrides"]["dt"] == config["env_overrides"]["dt"]
        assert settings["device"] == config["training"]["device"]
        for key, value in config["vec"].items():
            if key in ("num_envs", "num_workers", "batch_size"):
                assert settings["vec_overrides"][key] == value
        output_root = Path(settings["output_root"])
        run_id = settings["run_id"]
        location = output_root / run_id / settings["output_subdir"]
        locations.append(location)
        assert output_root / run_id == Path(config["training"]["output_root"]) / config["training"]["run_id"]
        assert settings["output_subdir"].startswith("training/step_")
    assert len(set(locations)) == 2


def test_periodic_driving_does_not_leak_training_scenario_length_into_eval(
    tmp_path, monkeypatch
):
    config = ORCHESTRATION_MODULE._config(
        tmp_path, "periodic_steps", max_optimizer_steps=1, num_collections=1, update_epochs=1
    )
    config["env_overrides"]["scenario_length"] = 64
    _training_driving_config(config, enabled=True, interval_steps=1)
    config["driving_evaluation"]["episode_timesteps"] = 128
    calls = []
    _install_fake_evaluator(monkeypatch, calls)

    TRAIN_MODULE.run_student_driving_evaluation(
        config,
        student=ORCHESTRATION_MODULE.RecordingStudent(),
        resolved_teacher_config={"env": {"num_agents": 1, "dt": 0.3}, "eval": {}},
        monitor=ORCHESTRATION_MODULE.RecordingMonitor(),
        progress=MetricProgress(1, 7, 0, 0),
        step=1,
        run_dir=tmp_path / "periodic_steps",
    )

    settings = calls[0]["config"]
    assert settings["episode_timesteps"] == 128
    assert "scenario_length" not in settings["env_overrides"]


def test_explicit_periodic_episode_timesteps_bypass_old_catalog_duration_conversion(
    tmp_path,
):
    source = tmp_path / "benchmarks.yaml"
    source.write_text(
        yaml.safe_dump(
            {
                "env": {"dt": 0.1},
                "benchmarks": [
                    {
                        "name": "carla_fast",
                        "num_scenarios": 99,
                        "env": {
                            "simulation_mode": "gigaflow",
                            "dt": 0.1,
                            "scenario_length": 500,
                        },
                    }
                ],
            }
        ),
        encoding="utf-8",
    )

    output_dir = tmp_path / "resolved"
    output_dir.mkdir()
    resolved_path = evaluate_module._materialize_benchmark_config(
        source,
        output_dir=output_dir,
        benchmark_names=["carla_fast"],
        env_overrides={"dt": 0.3},
        episode_timesteps=128,
        num_scenarios=2,
        seed=42,
    )

    resolved = yaml.safe_load(Path(resolved_path).read_text(encoding="utf-8"))
    assert resolved["env"]["dt"] == 0.3
    assert resolved["benchmarks"][0]["env"]["scenario_length"] == 128


def test_disabled_periodic_driving_does_not_use_legacy_evaluation_switch(
    tmp_path, monkeypatch
):
    config = ORCHESTRATION_MODULE._config(
        tmp_path, "periodic_disabled", max_optimizer_steps=3, num_collections=1, update_epochs=2
    )
    _training_driving_config(config, enabled=False, interval_steps=2)
    config["evaluation"] = {"enabled": True}
    config["evaluation_enabled"] = True
    calls = []
    _install_fake_evaluator(monkeypatch, calls)
    validation = _prepare_training_test(monkeypatch, config, tmp_path)

    TRAIN_MODULE.train(
        config,
        validation_batches=validation,
        teacher=ORCHESTRATION_MODULE._make_teacher(),
        student=ORCHESTRATION_MODULE.RecordingStudent(),
        env=ORCHESTRATION_MODULE.EnvToken(),
        monitor=ORCHESTRATION_MODULE.RecordingMonitor(),
    )

    assert calls == []


def test_resume_skips_the_last_evaluated_step_and_evaluates_next_step_once(
    tmp_path, monkeypatch
):
    config = ORCHESTRATION_MODULE._config(
        tmp_path, "periodic_resume", max_optimizer_steps=2, num_collections=1, update_epochs=2
    )
    _training_driving_config(config, enabled=True, interval_steps=2)
    calls = []
    _install_fake_evaluator(monkeypatch, calls)
    validation = _prepare_training_test(monkeypatch, config, tmp_path)
    student = ORCHESTRATION_MODULE.RecordingStudent()
    teacher = ORCHESTRATION_MODULE._make_teacher()

    first = TRAIN_MODULE.train(
        config,
        validation_batches=validation,
        teacher=teacher,
        student=student,
        env=ORCHESTRATION_MODULE.EnvToken(),
        monitor=ORCHESTRATION_MODULE.RecordingMonitor(),
    )
    assert [_periodic_eval_step(call["config"]) for call in calls] == [2]

    config["training"]["max_optimizer_steps"] = 4
    config["training"]["resume_checkpoint"] = first["checkpoint"]
    second_monitor = ORCHESTRATION_MODULE.RecordingMonitor()
    resumed = TRAIN_MODULE.train(
        config,
        validation_batches=validation,
        teacher=teacher,
        student=student,
        env=ORCHESTRATION_MODULE.EnvToken(),
        monitor=second_monitor,
    )

    assert resumed["optimizer_steps"] == 4
    assert [_periodic_eval_step(call["config"]) for call in calls] == [2, 4]
    assert calls[-1]["student"] is student
    assert calls[-1]["monitor"] is second_monitor


def test_failed_periodic_evaluation_restores_rng_and_each_student_submodule_mode(
    tmp_path, monkeypatch
):
    config = ORCHESTRATION_MODULE._config(
        tmp_path, "periodic_failure", max_optimizer_steps=1, num_collections=1, update_epochs=1
    )
    _training_driving_config(config, enabled=True, interval_steps=1)
    validation = _prepare_training_test(monkeypatch, config, tmp_path)
    student = ORCHESTRATION_MODULE.RecordingStudent()
    entry = {}

    original_train_step = TRAIN_MODULE.train_step

    def train_step_with_mixed_modes(*args, **kwargs):
        losses = original_train_step(*args, **kwargs)
        args[0].target.eval()
        args[0].decoder.eval()
        return losses

    monkeypatch.setattr(TRAIN_MODULE, "train_step", train_step_with_mixed_modes)

    def failing_evaluate_student(*args, **kwargs):
        evaluated_student = kwargs["student"]
        entry["torch"] = torch.get_rng_state().clone()
        entry["python"] = random.getstate()
        numpy_state = np.random.get_state()
        entry["numpy"] = (
            numpy_state[0],
            numpy_state[1].copy(),
            numpy_state[2],
            numpy_state[3],
            numpy_state[4],
        )
        entry["modes"] = {
            name: module.training for name, module in evaluated_student.named_modules()
        }
        torch.rand(4)
        random.random()
        np.random.rand(4)
        evaluated_student.train()
        raise RuntimeError("mock driving evaluator failed")

    monkeypatch.setattr(evaluate_module, "evaluate_student", failing_evaluate_student)
    monkeypatch.setattr(TRAIN_MODULE, "evaluate_student", failing_evaluate_student, raising=False)

    with pytest.raises(RuntimeError, match="mock driving evaluator failed"):
        TRAIN_MODULE.train(
            config,
            validation_batches=validation,
            teacher=ORCHESTRATION_MODULE._make_teacher(),
            student=student,
            env=ORCHESTRATION_MODULE.EnvToken(),
            monitor=ORCHESTRATION_MODULE.RecordingMonitor(),
        )

    assert torch.equal(torch.get_rng_state(), entry["torch"])
    assert random.getstate() == entry["python"]
    current_numpy = np.random.get_state()
    assert current_numpy[0] == entry["numpy"][0]
    np.testing.assert_array_equal(current_numpy[1], entry["numpy"][1])
    assert current_numpy[2:] == entry["numpy"][2:]
    assert {
        name: module.training for name, module in student.named_modules()
    } == entry["modes"]


def test_training_configs_enable_native_periodic_driving_with_toy_cadence():
    base = load_config("project/jepa_distill/config/condition_b.yaml")
    toy = load_config("project/jepa_distill/config/toy.yaml")

    assert base["driving_evaluation"]["enabled"] is True
    assert base["driving_evaluation"]["interval_steps"] == 50
    assert base["driving_evaluation"]["output_name"] == "training_native"
    assert base["env_overrides"]["dt"] == 0.3

    assert toy["driving_evaluation"]["enabled"] is True
    assert toy["driving_evaluation"]["interval_steps"] == 20
    assert toy["driving_evaluation"]["num_scenarios"] == 2
    assert toy["driving_evaluation"]["episode_timesteps"] == 64


def test_evaluate_student_sends_mocked_driving_metrics_to_supplied_monitor(
    tmp_path, monkeypatch
):
    captured = {}

    def fake_evaluate_with_pufferl(
        adapter, runtime_args, evaluation_config, *, benchmark_names
    ):
        captured["adapter"] = adapter
        captured["runtime_args"] = runtime_args
        captured["evaluation_config"] = dict(evaluation_config)
        captured["benchmark_names"] = list(benchmark_names)
        return {
            "carla": {
                "summary": {"metrics_mean": {"goal_rate": 0.75, "collision_rate": 0.0}}
            }
        }

    monkeypatch.setattr(
        evaluate_module, "_evaluate_with_pufferl", fake_evaluate_with_pufferl
    )
    monitor = _CaptureMonitor()
    config = _evaluation_config(tmp_path)

    result = evaluate_student(config, student=_ToyStudent(), monitor=monitor)

    assert result["carla"]["summary"]["metrics_mean"]["goal_rate"] == 0.75
    assert captured["benchmark_names"] == ["carla"]
    assert captured["runtime_args"]["train"]["device"] == "cpu"
    assert captured["runtime_args"]["train"]["amp"] is False
    assert captured["runtime_args"]["train"]["compile"] is False
    assert captured["evaluation_config"]["action_selection"] == "mean"
    assert captured["evaluation_config"]["env_overrides"]["dt"] == 0.3
    assert captured["evaluation_config"]["progress"]["optimizer_step"] == 7
    assert monitor.finished is False
    assert len(monitor.events) == 1
    metrics, metadata = monitor.events[0]
    assert metrics == {"goal_rate": 0.75, "collision_rate": 0.0}
    assert metadata["benchmark_name"] == "carla"
    assert metadata["output_name"] == "training_native"
    assert metadata["policy_role"] == "student"
    assert metadata["progress"].optimizer_step == 7
    assert metadata["progress"].simulator_transitions == 14


def test_invalid_periodic_driving_config_fails_before_runtime_initialization(
    tmp_path, monkeypatch
):
    config = ORCHESTRATION_MODULE._config(
        tmp_path, "invalid_driving", max_optimizer_steps=1, num_collections=1, update_epochs=1
    )
    config["driving_evaluation"] = {"enabled": True, "interval_steps": 0}
    calls = []

    runtime_module = importlib.import_module("project.jepa_distill.runtime")
    monkeypatch.setattr(runtime_module, "prepare_runtime", lambda _config: calls.append("prepare"))
    monkeypatch.setattr(
        importlib.import_module("project.jepa_distill.teacher"),
        "resolve_teacher_config",
        lambda _config: calls.append("teacher_config"),
    )

    with pytest.raises(ValueError, match="driving_evaluation.interval_steps"):
        TRAIN_MODULE.train(config, validation_batches=[])

    assert calls == []
