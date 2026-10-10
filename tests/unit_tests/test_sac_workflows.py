import types
from pathlib import Path
import random
import warnings

import numpy as np
import pytest

from pufferlib import sac_workflows


def workflow_args(tmp_path):
    return {
        "env_name": "puffer_drive",
        "package": "ocean",
        "run_name": "sac",
        "load_model_path": "actor.pt",
        "train": {
            "data_dir": str(tmp_path / "training"),
            "fresh_run": False,
            "resume_state_path": None,
            "total_timesteps": 12,
            "device": "cpu",
        },
        "env": {"num_agents": 4},
        "vec": {"batch_size": 2, "num_envs": 20},
        "max_runs": 2,
        "sweep": {
            "method": "FakeSearch",
            "metric": "score",
            "goal": "maximize",
            "downsample": 2,
            "train": {"learning_rate": {"distribution": "uniform"}},
        },
        "controlled_exp": {
            "train": {"initial_alpha": {"values": [0.1, 0.2]}, "tau": {"values": [0.005, 0.01]}}
        },
        "profile": {"total_timesteps": 20},
        "autotune": {"max_envs": 8, "time_per_test": 1},
    }


def test_sweep_uses_fresh_trial_directories_and_observes_downsampled_logs(tmp_path, monkeypatch):
    observed = []
    suggestions = []

    class FakeSearch:
        def __init__(self, config):
            assert config["metric"] == "score"

        def suggest(self, args):
            args["train"]["learning_rate"] = random.random()
            suggestions.append(args["train"]["learning_rate"])

        def observe(self, args, score, cost):
            observed.append((args["train"]["data_dir"], score, cost))

    monkeypatch.setattr(sac_workflows.pufferlib.sweep, "FakeSearch", FakeSearch, raising=False)
    train_calls = []

    def fake_train(env_name, args):
        random.seed(42)
        train_calls.append((env_name, args))
        return [
            {"environment/score": score, "uptime": score / 10, "agent_steps": index}
            for index, score in enumerate((1.0, 2.0, 3.0, 4.0), start=1)
        ]

    previous_random_state = random.getstate()
    random.seed(1729)
    expected_suggestions = [random.random(), random.random()]
    random.seed(1729)
    monkeypatch.setattr(sac_workflows.pufferl_sac, "train", fake_train)
    sac_workflows.sweep(workflow_args(tmp_path))
    random.setstate(previous_random_state)

    assert len(train_calls) == 2
    trial_args = [args for _, args in train_calls]
    assert len({args["train"]["data_dir"] for args in trial_args}) == 2
    assert len({args["run_name"] for args in trial_args}) == 2
    assert all(args["train"]["fresh_run"] for args in trial_args)
    assert all(args["train"]["resume_state_path"] is None for args in trial_args)
    assert all(args["load_model_path"] == "actor.pt" for args in trial_args)
    assert suggestions == pytest.approx(expected_suggestions)
    assert len(observed) == 4
    assert observed[0][1:] == pytest.approx((2.0, 0.2))
    assert observed[1][1:] == pytest.approx((4.0, 0.4))


def test_controlled_exp_runs_each_grid_point_in_a_fresh_directory(tmp_path, monkeypatch):
    calls = []

    def fake_train(env_name, args):
        calls.append(args)
        return []

    monkeypatch.setattr(sac_workflows.pufferl_sac, "train", fake_train)
    results = sac_workflows.controlled_exp(workflow_args(tmp_path))

    assert len(results) == len(calls) == 4
    assert len({args["train"]["data_dir"] for args in calls}) == 4
    assert all(args["train"]["fresh_run"] for args in calls)
    assert {(args["train"]["initial_alpha"], args["train"]["tau"]) for args in calls} == {
        (0.1, 0.005), (0.1, 0.01), (0.2, 0.005), (0.2, 0.01)
    }


def test_profile_sets_received_row_budget_and_exports_trace(tmp_path, monkeypatch):
    captured = {}

    def fake_train(env_name, args):
        captured.update(args=args, env_name=env_name)
        return [{"agent_steps": 20}]

    monkeypatch.setattr(sac_workflows.pufferl_sac, "train", fake_train)
    trace_path = sac_workflows.profile(workflow_args(tmp_path))

    assert captured["args"]["train"]["total_timesteps"] == 20
    assert captured["args"]["train"]["fresh_run"] is True
    assert captured["env_name"] == "puffer_drive"
    assert trace_path.endswith("trace.json")
    assert Path(trace_path).is_file()


def test_autotune_passes_env_instance_batch_and_sac_final_observations(tmp_path, monkeypatch):
    args = workflow_args(tmp_path)
    args["vec"]["batch_size"] = "auto"
    captured = {}

    class Env:
        def __init__(self, **kwargs):
            captured["env_kwargs"] = kwargs

    monkeypatch.setattr(
        sac_workflows.importlib,
        "import_module",
        lambda _name: types.SimpleNamespace(env_creator=lambda _env_name: Env),
    )
    def fake_autotune(creator, **kwargs):
        captured.update(creator=creator, **kwargs)
        np.mean([])
        return "autotuned"

    monkeypatch.setattr(sac_workflows.pufferlib.vector, "autotune", fake_autotune)
    assert sac_workflows.autotune(args) == "autotuned"
    captured["creator"]()

    assert captured["batch_size"] == 10
    assert captured["max_envs"] == 8
    assert captured["time_per_test"] == 1
    assert captured["env_kwargs"]["capture_final_observations"] is True


def test_autotune_does_not_suppress_other_numpy_runtime_warnings(tmp_path, monkeypatch):
    args = workflow_args(tmp_path)
    monkeypatch.setattr(
        sac_workflows.importlib,
        "import_module",
        lambda _name: types.SimpleNamespace(env_creator=lambda _env_name: lambda **_kwargs: None),
    )

    def fake_autotune(_creator, **_kwargs):
        warnings.warn_explicit(
            "unexpected warning", RuntimeWarning, "numpy/core/_methods.py", 1,
            module="numpy.core._methods",
        )

    monkeypatch.setattr(sac_workflows.pufferlib.vector, "autotune", fake_autotune)
    with warnings.catch_warnings():
        warnings.simplefilter("error", RuntimeWarning)
        with pytest.raises(RuntimeWarning, match="unexpected warning"):
            sac_workflows.autotune(args)


def test_workflows_reject_distributed_invocations(tmp_path, monkeypatch):
    monkeypatch.setenv("WORLD_SIZE", "2")
    with pytest.raises(RuntimeError, match="requires one process"):
        sac_workflows.controlled_exp(workflow_args(tmp_path))


def test_workflow_rejects_explicit_resume_state(tmp_path, monkeypatch):
    args = workflow_args(tmp_path)
    args["train"]["resume_state_path"] = "trainer_state.pt"
    train_calls = []
    monkeypatch.setattr(sac_workflows.pufferl_sac, "train", lambda **kwargs: train_calls.append(kwargs))

    with pytest.raises(ValueError, match="does not support train.resume_state_path"):
        sac_workflows.controlled_exp(args)
    assert not train_calls


def test_sac_workflows_config_composes_with_sac_base_config():
    from hydra import compose, initialize_config_dir
    from omegaconf import OmegaConf

    config_dir = Path(sac_workflows.__file__).resolve().parent / "config"
    with initialize_config_dir(config_dir=str(config_dir), version_base=None):
        base_config = compose(config_name="puffer_drive_sac")
    fragment = OmegaConf.load(config_dir / "sac_workflows.yaml")
    OmegaConf.set_struct(base_config, False)
    config = OmegaConf.merge(base_config, fragment)

    assert config.policy_name == "DriveSAC"
    assert config.profile.total_timesteps == 20000
    assert config.sweep.train.replay_update_ratio.min == 0.05
    assert config.controlled_exp.train.learning_rate is None
    assert config.controlled_exp.train.ent_coef is None
