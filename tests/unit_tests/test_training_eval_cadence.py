import pytest

import pufferlib
from pufferlib import pufferl


def _evaluation_args(*, evaluation_interval_epochs=None, evaluation_interval_agent_steps=None):
    return {
        "train": {
            "evaluation_interval_epochs": evaluation_interval_epochs,
            "evaluation_interval_agent_steps": evaluation_interval_agent_steps,
            "evaluation_benchmarks": "carla_fast",
        },
        "eval": {"benchmark_config": "benchmarks.yaml"},
    }


def _stub_benchmark_config(monkeypatch):
    calls = []

    def load_benchmark_config(config_path, selected_benchmarks):
        calls.append((config_path, selected_benchmarks))
        return {"map_dir": "maps"}, [{"name": "carla_fast"}]

    monkeypatch.setattr(pufferl.drive_benchmark, "load_benchmark_config", load_benchmark_config)
    return calls


def test_default_ppo_validation_keeps_epoch_cadence_and_rejects_agent_step_key(monkeypatch):
    calls = _stub_benchmark_config(monkeypatch)

    assert pufferl.drive_benchmark.validate_training_evaluation_config(
        _evaluation_args(evaluation_interval_epochs=3)
    ) is True
    assert calls == [("benchmarks.yaml", "carla_fast")]

    with pytest.raises(pufferlib.APIUsageError, match="evaluation_interval_agent_steps"):
        pufferl.drive_benchmark.validate_training_evaluation_config(
            _evaluation_args(evaluation_interval_agent_steps=100)
        )


def test_sac_agent_step_interval_is_accepted_when_enabled(monkeypatch):
    calls = _stub_benchmark_config(monkeypatch)

    assert pufferl.drive_benchmark.validate_training_evaluation_config(
        _evaluation_args(evaluation_interval_agent_steps=100),
        allow_agent_steps=True,
    ) is True
    assert calls == [("benchmarks.yaml", "carla_fast")]


@pytest.mark.parametrize("interval", [True, 1.0, 0, -1])
def test_sac_agent_step_interval_requires_strictly_positive_integer(monkeypatch, interval):
    _stub_benchmark_config(monkeypatch)

    with pytest.raises(pufferlib.APIUsageError, match="evaluation_interval_agent_steps.*positive integer"):
        pufferl.drive_benchmark.validate_training_evaluation_config(
            _evaluation_args(evaluation_interval_agent_steps=interval),
            allow_agent_steps=True,
        )


def test_training_evaluation_rejects_both_cadence_modes(monkeypatch):
    _stub_benchmark_config(monkeypatch)

    with pytest.raises(pufferlib.APIUsageError, match="evaluation_interval"):
        pufferl.drive_benchmark.validate_training_evaluation_config(
            _evaluation_args(evaluation_interval_epochs=2, evaluation_interval_agent_steps=100),
            allow_agent_steps=True,
        )
