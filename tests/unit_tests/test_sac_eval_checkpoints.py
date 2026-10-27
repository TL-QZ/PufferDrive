import json
from pathlib import Path
from unittest.mock import Mock

import pytest
import torch
from torch import nn

from pufferlib import pufferl
from pufferlib import pufferl_sac
from test_sac_learner import _make_checkpoint_learner


class _Policy(nn.Module):
    def __init__(self):
        super().__init__()
        self.weight = nn.Parameter(torch.zeros(()))


def _trainer(tmp_path, *, rank=0):
    trainer = object.__new__(pufferl_sac.PuffeSAC)
    trainer.rank = rank
    trainer.config = {"data_dir": str(tmp_path)}
    trainer.uncompiled_policy = _Policy()
    trainer.epoch = 1
    trainer.agent_steps = 10
    trainer.best_score = 123.0
    trainer.save_checkpoint = Mock(return_value="latest-model.pt")
    return trainer


def _results(benchmark_scores):
    return {
        name: {
            "episodes": [{"scenario_id": f"{name}-scenario"}],
            "summary": {
                "num_scenarios": 1,
                "num_episodes": 1,
                "metrics_mean": {"puffer_score": score, "progress": 0.75, "bad": float("nan")},
            },
        }
        for name, score in benchmark_scores.items()
    }


def _manifest(tmp_path):
    path = tmp_path / "best_evaluation.json"
    return path, json.loads(path.read_text(encoding="utf-8"))


def _actor_path(tmp_path, entry):
    return tmp_path / entry["checkpoint"]


def test_eval_winners_are_independent_and_latest_learner_checkpoint_saves_each_time(tmp_path):
    trainer = _trainer(tmp_path)
    trainer.uncompiled_policy.weight.data.fill_(1.0)

    trainer.save_evaluation_checkpoint(_results({"carla_fast": 4.0, "nuplan_single": 20.0}))
    manifest_path, manifest = _manifest(tmp_path)
    assert manifest_path.exists()
    carla_actor_path = _actor_path(tmp_path, manifest["benchmarks"]["carla_fast"])
    nuplan_actor_path = _actor_path(tmp_path, manifest["benchmarks"]["nuplan_single"])
    carla_actor = torch.load(carla_actor_path, map_location="cpu", weights_only=True)
    assert set(carla_actor) == {"weight"}
    assert torch.equal(carla_actor["weight"], torch.tensor(1.0))
    assert manifest["benchmarks"]["carla_fast"]["score_metric"] == "puffer_score"
    assert manifest["benchmarks"]["carla_fast"]["summary"] == {
        "num_scenarios": 1,
        "num_episodes": 1,
        "metrics_mean": {"puffer_score": 4.0, "progress": 0.75},
    }

    restored = _Policy()
    pufferl_sac._load_actor_weights(restored, carla_actor_path, torch.device("cpu"), pufferl)
    assert restored.weight.item() == 1.0

    trainer.epoch = 2
    trainer.agent_steps = 20
    trainer.uncompiled_policy.weight.data.fill_(2.0)
    trainer.save_evaluation_checkpoint(_results({"carla_fast": 5.0, "nuplan_single": 19.0}))
    _, improved_manifest = _manifest(tmp_path)
    improved_carla_path = _actor_path(tmp_path, improved_manifest["benchmarks"]["carla_fast"])
    assert improved_manifest["benchmarks"]["carla_fast"]["score"] == 5.0
    assert improved_manifest["benchmarks"]["nuplan_single"]["score"] == 20.0
    assert improved_manifest["benchmarks"]["nuplan_single"]["checkpoint"] == str(
        nuplan_actor_path.relative_to(tmp_path)
    )
    improved_actor = torch.load(improved_carla_path, map_location="cpu", weights_only=True)
    assert torch.equal(improved_actor["weight"], torch.tensor(2.0))
    assert not carla_actor_path.exists()
    assert nuplan_actor_path.exists()

    manifest_before_tie = (tmp_path / "best_evaluation.json").read_bytes()
    trainer.epoch = 3
    trainer.agent_steps = 30
    trainer.save_evaluation_checkpoint(_results({"carla_fast": 5.0, "nuplan_single": 18.0}))
    assert (tmp_path / "best_evaluation.json").read_bytes() == manifest_before_tie
    assert trainer.save_checkpoint.call_count == 3
    assert trainer.best_score == 123.0


@pytest.mark.parametrize(
    "results",
    [
        {},
        {"carla_fast": None},
        {"carla_fast": {"episodes": [], "summary": {"num_episodes": 0, "metrics_mean": {"puffer_score": 1}}}},
        {"carla_fast": {"episodes": [{}], "summary": {"num_episodes": 1, "metrics_mean": {}}}},
        {"carla_fast": {"episodes": [{}], "summary": {"num_episodes": 1, "metrics_mean": {"puffer_score": float("nan")}}}},
        {"carla_fast": {"episodes": [{}], "summary": {"num_episodes": 1, "metrics_mean": {"puffer_score": float("inf")}}}},
    ],
)
def test_ineligible_eval_does_not_write_any_checkpoint(tmp_path, results):
    trainer = _trainer(tmp_path)

    assert trainer.save_evaluation_checkpoint(results) is None

    trainer.save_checkpoint.assert_not_called()
    assert not (tmp_path / "best_evaluation.json").exists()


def test_restart_uses_existing_manifest_even_when_learner_best_score_is_stale(tmp_path):
    first_trainer = _trainer(tmp_path)
    first_trainer.save_evaluation_checkpoint(_results({"carla_fast": 10.0}))
    _, initial_manifest = _manifest(tmp_path)
    existing_actor_path = _actor_path(tmp_path, initial_manifest["benchmarks"]["carla_fast"])
    existing_manifest_bytes = (tmp_path / "best_evaluation.json").read_bytes()

    resumed_trainer = _trainer(tmp_path)
    resumed_trainer.epoch = 8
    resumed_trainer.agent_steps = 80
    resumed_trainer.best_score = -float("inf")
    resumed_trainer.save_evaluation_checkpoint(_results({"carla_fast": 9.0}))

    assert (tmp_path / "best_evaluation.json").read_bytes() == existing_manifest_bytes
    assert existing_actor_path.exists()
    resumed_trainer.save_checkpoint.assert_called_once_with()


def test_actor_write_failure_preserves_existing_manifest_and_actor(tmp_path, monkeypatch):
    trainer = _trainer(tmp_path)
    trainer.save_evaluation_checkpoint(_results({"carla_fast": 1.0}))
    _, manifest = _manifest(tmp_path)
    actor_path = _actor_path(tmp_path, manifest["benchmarks"]["carla_fast"])
    old_actor_bytes = actor_path.read_bytes()
    old_manifest_bytes = (tmp_path / "best_evaluation.json").read_bytes()

    def fail_save(*args, **kwargs):
        raise OSError("injected actor write failure")

    monkeypatch.setattr(pufferl_sac.torch, "save", fail_save)
    with pytest.raises(OSError, match="injected actor write failure"):
        trainer.save_evaluation_checkpoint(_results({"carla_fast": 2.0}))

    assert actor_path.read_bytes() == old_actor_bytes
    assert (tmp_path / "best_evaluation.json").read_bytes() == old_manifest_bytes
    assert trainer.save_checkpoint.call_count == 1


def test_manifest_publish_failure_preserves_existing_manifest_and_actor(tmp_path, monkeypatch):
    trainer = _trainer(tmp_path)
    trainer.save_evaluation_checkpoint(_results({"carla_fast": 1.0}))
    _, manifest = _manifest(tmp_path)
    actor_path = _actor_path(tmp_path, manifest["benchmarks"]["carla_fast"])
    old_actor_bytes = actor_path.read_bytes()
    manifest_path = tmp_path / "best_evaluation.json"
    old_manifest_bytes = manifest_path.read_bytes()
    replace = pufferl_sac.os.replace

    def fail_manifest_replace(source, target):
        if Path(target) == manifest_path:
            raise OSError("injected manifest publish failure")
        return replace(source, target)

    monkeypatch.setattr(pufferl_sac.os, "replace", fail_manifest_replace)
    with pytest.raises(OSError, match="injected manifest publish failure"):
        trainer.save_evaluation_checkpoint(_results({"carla_fast": 2.0}))

    assert manifest_path.read_bytes() == old_manifest_bytes
    assert actor_path.read_bytes() == old_actor_bytes
    trainer.save_checkpoint.assert_called_once_with()


def test_nonzero_rank_never_writes_eval_or_latest_checkpoints(tmp_path):
    trainer = _trainer(tmp_path, rank=1)

    assert trainer.save_evaluation_checkpoint(_results({"carla_fast": 1.0})) is None

    trainer.save_checkpoint.assert_not_called()
    assert not (tmp_path / "best_evaluation.json").exists()


def test_real_learner_latest_save_keeps_eval_actor_and_writes_resume_state(monkeypatch, tmp_path):
    trainer = _make_checkpoint_learner(monkeypatch, tmp_path)

    trainer.save_evaluation_checkpoint(_results({"carla_fast": 3.5}))

    _, manifest = _manifest(tmp_path)
    actor_path = _actor_path(tmp_path, manifest["benchmarks"]["carla_fast"])
    actor_state = torch.load(actor_path, map_location="cpu", weights_only=True)
    resume_state = torch.load(tmp_path / "trainer_state.pt", map_location="cpu", weights_only=False)
    assert set(actor_state) == set(trainer.uncompiled_policy.state_dict())
    assert resume_state["scope"] == "learner_only"
    assert resume_state["epoch"] == trainer.epoch
    assert actor_path.exists()
