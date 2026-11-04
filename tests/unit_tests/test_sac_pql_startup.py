"""PQL configuration preflight and learner-only checkpoint compatibility."""

from types import SimpleNamespace

import pytest
import torch

from pufferlib import pufferl, pufferl_sac
from pufferlib.sac_replay import TensorReplay
from pufferlib.sac_sequence_replay import SequenceReplay
from test_sac_nstep_collection import (
    _ConstructorVecEnv, _NoopUtilization, _TinyActor, _TinyCritic,
)
from test_pufferl_sac_startup import _FakeLearner, _FakeVecEnv, _startup_args


def _make_learner(monkeypatch, directory, **overrides):
    monkeypatch.setattr(pufferl, "Utilization", _NoopUtilization)
    monkeypatch.setattr(pufferl.PuffeRL, "print_dashboard", lambda *_args, **_kwargs: None)
    config = dict(
        data_dir=str(directory), action_count=2, total_timesteps=20,
        batch_size=2, minibatch_size=2, max_minibatch_size=1,
        checkpoint_interval=None, return_estimator="pql",
    )
    config.update(overrides)
    replay = SequenceReplay(2, (1,), num_agents=1)
    return pufferl_sac.PuffeSAC(
        config, _ConstructorVecEnv(), _TinyActor(), _TinyCritic(), replay, reset_env=False,
    )


@pytest.mark.parametrize("settings,match", [
    ({"return_estimator": "gae"}, "return_estimator"),
    ({"pql_horizon": True}, "pql_horizon"),
    ({"pql_horizon": 0}, "pql_horizon"),
    ({"pql_horizon": 1.5}, "pql_horizon"),
    ({"pql_lambda": float("nan")}, "pql_lambda"),
    ({"pql_lambda": float("inf")}, "pql_lambda"),
    ({"pql_lambda": True}, "pql_lambda"),
    ({"pql_lambda": -0.1}, "pql_lambda"),
    ({"pql_lambda": 1.1}, "pql_lambda"),
    ({"pql_entropy_convention": "full_entropy"}, "pql_entropy_convention"),
    ({"n_steps": 2}, "n_steps=1"),
])
def test_invalid_settings_fail_before_process_or_environment_creation(monkeypatch, settings, match):
    monkeypatch.setattr(
        pufferl_sac, "_initialize_process_group",
        lambda *_args: pytest.fail("allocated process group before estimator validation"),
    )
    with pytest.raises(ValueError, match=match):
        pufferl_sac.train(args={"train": {"return_estimator": "pql", **settings}})


def test_hydra_accepts_estimator_overrides():
    config = pufferl_sac.load_config(overrides=[
        "train.return_estimator=pql", "train.pql_horizon=64", "train.pql_lambda=0.8",
    ])
    assert config["train"]["return_estimator"] == "pql"
    assert config["train"]["pql_horizon"] == 64
    assert config["train"]["pql_lambda"] == 0.8
    assert config["train"]["n_steps"] == 1


@pytest.mark.parametrize("nested", [True, False])
def test_behavior_factory_rejected_before_resources(monkeypatch, nested):
    monkeypatch.setattr(
        pufferl_sac, "_initialize_process_group",
        lambda *_args: pytest.fail("allocated resources"),
    )
    config = {"return_estimator": "pql", "n_steps": 1}
    with pytest.raises(ValueError, match="PQL.*behavior_policy_factory"):
        pufferl_sac.train(
            args={"train": config} if nested else config,
            behavior_policy_factory=lambda *_args: pytest.fail("created provider"),
        )


def test_startup_constructs_sequence_replay(monkeypatch, tmp_path):
    args = _startup_args(tmp_path)
    args["train"]["return_estimator"] = "pql"
    vecenv = _FakeVecEnv()
    recorded = []

    def make_learner(*arguments):
        learner = _FakeLearner(*arguments)
        recorded.append(learner)
        return learner

    monkeypatch.setattr(pufferl_sac, "PuffeSAC", make_learner)
    monkeypatch.setattr(pufferl, "_get_git_metadata", lambda: {})
    monkeypatch.setattr(pufferl, "_save_experiment_config", lambda *_args: None)
    pufferl_sac.train(args=args, vecenv=vecenv, policy=SimpleNamespace(action_count=2), critic=object())
    assert isinstance(recorded[0].replay, SequenceReplay)
    assert len(recorded[0].replay) == 0


def test_constructor_rejects_independent_replay(monkeypatch, tmp_path):
    with pytest.raises(ValueError, match="SequenceReplay"):
        pufferl_sac.PuffeSAC(
            {"return_estimator": "pql", "data_dir": str(tmp_path)},
            _ConstructorVecEnv(), _TinyActor(), _TinyCritic(), TensorReplay(2, (1,)),
            reset_env=False,
        )


def test_nonempty_injected_pql_replay_fails_before_resources(monkeypatch):
    replay = SequenceReplay(2, (1,), num_agents=1)
    replay.add_sequence(
        torch.tensor([0]), torch.tensor([[0.0]]), torch.tensor([0]), torch.tensor([1.0]),
        torch.tensor([[1.0]]), torch.tensor([0.9]), torch.tensor([False]), torch.tensor([False]),
    )
    monkeypatch.setattr(
        pufferl_sac, "_initialize_process_group",
        lambda *_args: pytest.fail("allocated resources before replay preflight"),
    )
    with pytest.raises(ValueError, match="empty initial replay"):
        pufferl_sac.train(args={"train": {"return_estimator": "pql"}}, replay=replay)


def test_pql_checkpoint_restores_learner_with_empty_replay_and_validates_contract(monkeypatch, tmp_path):
    source = _make_learner(monkeypatch, tmp_path / "source")
    source.save_checkpoint()
    path = tmp_path / "source" / "trainer_state.pt"
    state = torch.load(path, map_location="cpu", weights_only=False)
    assert state["config"]["return_estimator"] == "pql"
    assert state["config"]["pql_entropy_convention"] == "soft_each_return_endpoint"
    restored = _make_learner(monkeypatch, tmp_path / "restored")
    restored.load_training_state(path)
    assert len(restored.replay) == 0
    assert not restored.pending.any()
    for key, value in (("pql_lambda", 0.5), ("pql_horizon", 16)):
        mismatched = _make_learner(monkeypatch, tmp_path / key, **{key: value})
        with pytest.raises(ValueError, match=f"Checkpoint config mismatch: {key}"):
            mismatched.load_training_state(path)


def test_legacy_checkpoint_remains_fixed_n(monkeypatch, tmp_path):
    from test_sac_nstep_collection import _make_initialized_learner

    source = _make_initialized_learner(monkeypatch, tmp_path / "legacy")
    source.save_checkpoint()
    path = tmp_path / "legacy" / "trainer_state.pt"
    state = torch.load(path, map_location="cpu", weights_only=False)
    for key in ("return_estimator", "pql_horizon", "pql_lambda", "pql_entropy_convention"):
        state["config"].pop(key)
    torch.save(state, path)
    restored = _make_initialized_learner(monkeypatch, tmp_path / "fixed")
    restored.load_training_state(path)
    assert restored.config["return_estimator"] == "n_step"
    pql = _make_learner(monkeypatch, tmp_path / "pql", max_minibatch_size=2)
    with pytest.raises(ValueError, match="Checkpoint config mismatch: return_estimator"):
        pql.load_training_state(path)
