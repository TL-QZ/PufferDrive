"""Bounded end-to-end orchestration checks for the Condition B trainer."""

from __future__ import annotations

import copy
import importlib
import json
from pathlib import Path

import numpy as np
import pytest
import torch
from torch import nn

from project.jepa_distill.dataset import TrainingBatch
from project.jepa_distill.tests.test_train import SmallStudent


TRAIN_MODULE = importlib.import_module("project.jepa_distill.train")
TEACHER_MODULE = importlib.import_module("project.jepa_distill.teacher")
COLLECT_MODULE = importlib.import_module("project.jepa_distill.collect")


class RecordingStudent(SmallStudent):
    """SmallStudent that exposes the exact windows consumed by train()."""

    def __init__(self):
        super().__init__()
        self.training_window_starts: list[float] = []

    def forward(self, observations, controls):
        if self.training:
            self.training_window_starts.extend(
                observations[:, 0, 0].detach().cpu().tolist()
            )
        return super().forward(observations, controls)


class RecordingMonitor:
    """No-op monitor with enough checkpoint state for train()."""

    def __init__(self):
        self.events = []

    def log_metrics(self, metrics, *, progress):
        self.events.append((dict(metrics), progress))

    def state_dict(self):
        return {
            "mode": "disabled",
            "enabled": False,
            "run_id": None,
            "event_index": len(self.events) - 1,
        }

    def finish(self, *, exit_code=0):
        return None


class EnvToken:
    """Identity-only stand-in; the collector is mocked at the orchestration boundary."""


def _config(tmp_path: Path, run_id: str, *, max_optimizer_steps: int,
            num_collections: int, update_epochs: int) -> dict:
    transitions_per_round = 7
    return {
        "teacher": {"checkpoint": "unused", "config": "unused"},
        "env_overrides": {},
        "collection": {
            "num_collections": num_collections,
            "action_selection": "mean",
            "split_seeds": {"train": 11, "validation": 12, "test": 13},
            "dataset_id": None,
            "transitions_per_round": transitions_per_round,
            "max_transitions": num_collections * transitions_per_round,
            "max_disk_bytes": 10_000_000,
            "shard_transition_count": transitions_per_round,
            "output_root": str(tmp_path / "datasets"),
            "split": "train",
        },
        "model": {"chunk_length": 4, "num_action_classes": 2},
        "loss": {},
        "ema": {"tau": 0.5},
        "training": {
            "seed": 23,
            "device": "cpu",
            "batch_size": 2,
            "update_epochs": update_epochs,
            "learning_rate": 0.01,
            "weight_decay": 0.0,
            "gradient_clip_norm": 1.0,
            "optimizer": "AdamW",
            "amp": False,
            "compile": False,
            "distributed": False,
            "validation_manifest": None,
            "run_id": run_id,
            "max_optimizer_steps": max_optimizer_steps,
            "validation_interval_steps": 99,
            "checkpoint_interval_steps": 1,
            "resume_checkpoint": None,
            "output_root": str(tmp_path / "runs"),
            "validation_transitions": 8,
            "cpu_threads": 1,
        },
        "wandb": {
            "enabled": False,
            "mode": "disabled",
            "log_interval_steps": 1,
        },
        "vec": {"backend": "Serial", "num_envs": 1, "num_workers": 1, "batch_size": 1},
    }


def _validation_batches() -> list[TrainingBatch]:
    observations = torch.arange(30, dtype=torch.float32).reshape(2, 5, 3) / 10
    controls = torch.zeros(2, 4, 2)
    teacher_logits = torch.tensor(
        [
            [[1.0, -1.0], [0.5, -0.5], [-0.5, 0.5], [-1.0, 1.0]],
            [[-1.0, 1.0], [-0.5, 0.5], [0.5, -0.5], [1.0, -1.0]],
        ]
    )
    return [TrainingBatch(observations, controls, teacher_logits)]


def _write_round_manifest(output_dir: Path, round_idx: int) -> dict:
    """Write four valid K=4 windows with round-specific first observations."""

    round_dir = output_dir / f"round_{round_idx:04d}"
    shard_dir = round_dir / "shards"
    shard_dir.mkdir(parents=True)
    transition_count = 7
    observation_dim = 3
    marker = float(round_idx * 100)
    observations = (
        marker
        + np.arange(transition_count + 1, dtype=np.float32)[:, None]
        + np.array([0.0, 0.1, 0.2], dtype=np.float32)[None, :]
    )
    arrays = {
        "observations": observations,
        "executed_controls": np.zeros((transition_count, 2), dtype=np.float32),
        "teacher_logits": np.tile(np.array([[1.0, -1.0]], dtype=np.float32), (transition_count, 1)),
        "trajectory_offsets": np.array([0, transition_count], dtype=np.int64),
        "observation_offsets": np.array([0, transition_count + 1], dtype=np.int64),
        "valid_transition": np.ones(transition_count, dtype=bool),
        "state_valid": np.ones(transition_count + 1, dtype=bool),
        "terminated": np.zeros(transition_count, dtype=bool),
        "truncated": np.zeros(transition_count, dtype=bool),
        "eligibility_mask": np.ones(transition_count, dtype=bool),
    }
    shard_path = shard_dir / "shard_000000.npz"
    np.savez_compressed(shard_path, **arrays)
    reference = {"shard_idx": 0, "trajectory_idx": 0}
    splits = {
        "train": {"trajectory_refs": [reference], "trajectory_count": 1},
        "validation": {"trajectory_refs": [], "trajectory_count": 0},
        "test": {"trajectory_refs": [], "trajectory_count": 0},
    }
    manifest = {
        "schema_version": 1,
        "chunk_length": 4,
        "num_action_classes": 2,
        "observation_layout": {"observation_dim": observation_dim},
        "teacher_checkpoint_sha256": "fake-teacher-sha256",
        "effective_config": {"teacher_config": {"env": {"num_agents": 1}}, "collection": {"action_selection": "mean"}},
        "split": "train",
        "split_seed": 11,
        "shards": [
            {
                "path": "shards/shard_000000.npz",
                "trajectory_count": 1,
                "transition_count": transition_count,
                "observation_dim": observation_dim,
                "chunk_length": 4,
                "num_action_classes": 2,
            }
        ],
        "trajectory_count": 1,
        "transition_count": transition_count,
        "collection_transition_count": transition_count,
        "valid_window_count": 4,
        "stats": {
            "rejected_transition_count": 0,
            "valid_window_count": 4,
        },
        "splits": splits,
    }
    manifest_path = round_dir / "manifest.json"
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
    result = dict(manifest)
    result["manifest_path"] = str(manifest_path)
    return result


def _expected_window_starts(*, num_collections: int, update_epochs: int,
                            seed: int, windows_per_round: int = 4) -> list[float]:
    starts = []
    for round_idx in range(num_collections):
        for epoch_idx in range(update_epochs):
            generator = torch.Generator().manual_seed(
                seed + round_idx * update_epochs + epoch_idx
            )
            indices = torch.randperm(windows_per_round, generator=generator).tolist()
            starts.extend(float(round_idx * 100 + index) for index in indices)
    return starts


def _patch_training_boundaries(monkeypatch, collector):
    monkeypatch.setattr(
        TEACHER_MODULE,
        "resolve_teacher_config",
        lambda config: {"env": {"num_agents": 1}},
    )
    monkeypatch.setattr(COLLECT_MODULE, "collect_dataset", collector)


def _make_teacher() -> nn.Module:
    teacher = nn.Linear(3, 2)
    teacher.requires_grad_(False)
    teacher.condition_b_checkpoint_sha256 = "fake-teacher-sha256"
    teacher.condition_b_observation_layout = {"observation_dim": 3}
    return teacher


def test_train_keeps_frozen_teacher_env_and_optimizer_across_collection_rounds(
    tmp_path, monkeypatch
):
    config = _config(
        tmp_path,
        "two_rounds",
        max_optimizer_steps=4,
        num_collections=2,
        update_epochs=1,
    )
    student = RecordingStudent()
    teacher = _make_teacher()
    teacher_before = copy.deepcopy(teacher.state_dict())
    env = EnvToken()
    monitor = RecordingMonitor()
    validation = _validation_batches()
    collection_calls = []

    def collector(config, output_dir, *, teacher, env, collection_round_idx):
        output_dir = Path(output_dir)
        if collection_calls:
            assert not list(output_dir.glob("round_*"))
        collection_calls.append((collection_round_idx, teacher, env))
        return _write_round_manifest(output_dir, collection_round_idx)

    optimizer_objects = []
    original_train_step = TRAIN_MODULE.train_step

    def train_step_spy(student, batch, optimizer, *, config=None):
        optimizer_objects.append(optimizer)
        return original_train_step(student, batch, optimizer, config=config)

    _patch_training_boundaries(monkeypatch, collector)
    monkeypatch.setattr(TRAIN_MODULE, "train_step", train_step_spy)

    result = TRAIN_MODULE.train(
        config,
        validation_batches=validation,
        teacher=teacher,
        student=student,
        env=env,
        monitor=monitor,
    )

    assert result["optimizer_steps"] == 4
    assert result["simulator_transitions"] == 14
    assert [(round_idx, seen_teacher is teacher, seen_env is env)
            for round_idx, seen_teacher, seen_env in collection_calls] == [
                (0, True, True),
                (1, True, True),
            ]
    assert len({id(optimizer) for optimizer in optimizer_objects}) == 1
    assert not list((tmp_path / "datasets" / "two_rounds" / "train").glob("round_*"))
    for name, parameter in teacher.state_dict().items():
        torch.testing.assert_close(parameter, teacher_before[name])
    assert student.training_window_starts == _expected_window_starts(
        num_collections=2, update_epochs=1, seed=config["training"]["seed"]
    )
    validation_events = [
        metrics for metrics, _progress in monitor.events
        if any(name.startswith("validation/") for name in metrics)
    ]
    assert len(validation_events) == 1
    assert all(np.isfinite(value) for value in validation_events[0].values())


def test_train_resume_matches_uninterrupted_order_and_reuses_manifest(
    tmp_path, monkeypatch
):
    full_config = _config(
        tmp_path / "full",
        "full",
        max_optimizer_steps=4,
        num_collections=1,
        update_epochs=2,
    )
    resume_config = _config(
        tmp_path / "resume",
        "resume",
        max_optimizer_steps=1,
        num_collections=1,
        update_epochs=2,
    )
    collection_calls = []

    def collector(config, output_dir, *, teacher, env, collection_round_idx):
        collection_calls.append(Path(output_dir))
        return _write_round_manifest(Path(output_dir), collection_round_idx)

    _patch_training_boundaries(monkeypatch, collector)
    torch.manual_seed(101)
    initial = RecordingStudent()
    initial_state = copy.deepcopy(initial.state_dict())
    full_student = RecordingStudent()
    full_student.load_state_dict(initial_state)
    interrupted_student = RecordingStudent()
    interrupted_student.load_state_dict(initial_state)
    teacher = _make_teacher()
    env = EnvToken()
    validation = _validation_batches()

    full_result = TRAIN_MODULE.train(
        full_config,
        validation_batches=validation,
        teacher=teacher,
        student=full_student,
        env=env,
        monitor=RecordingMonitor(),
    )
    assert len(collection_calls) == 1

    interrupted_result = TRAIN_MODULE.train(
        resume_config,
        validation_batches=validation,
        teacher=teacher,
        student=interrupted_student,
        env=env,
        monitor=RecordingMonitor(),
    )
    assert interrupted_result["optimizer_steps"] == 1
    assert interrupted_result["simulator_transitions"] == 7
    assert len(collection_calls) == 2
    training_root = tmp_path / "resume" / "datasets" / "resume" / "train"
    assert (training_root / "round_0000" / "manifest.json").is_file()
    checkpoint_payload = torch.load(
        interrupted_result["checkpoint"], map_location="cpu", weights_only=False
    )
    stored_manifest = checkpoint_payload["sampler_state"]["manifest_path"]
    assert Path(stored_manifest).is_file()

    before_resume_calls = len(collection_calls)
    resume_config["training"]["resume_checkpoint"] = interrupted_result["checkpoint"]
    capped_result = TRAIN_MODULE.train(
        resume_config,
        validation_batches=validation,
        teacher=teacher,
        student=interrupted_student,
        env=env,
        monitor=RecordingMonitor(),
    )
    assert capped_result["optimizer_steps"] == 1
    assert len(collection_calls) == before_resume_calls
    assert (training_root / "round_0000" / "manifest.json").is_file()

    resume_config["training"]["max_optimizer_steps"] = 4
    resume_config["training"]["resume_checkpoint"] = capped_result["checkpoint"]
    resumed_result = TRAIN_MODULE.train(
        resume_config,
        validation_batches=validation,
        teacher=teacher,
        student=interrupted_student,
        env=env,
        monitor=RecordingMonitor(),
    )

    assert resumed_result["optimizer_steps"] == full_result["optimizer_steps"] == 4
    assert resumed_result["simulator_transitions"] == full_result["simulator_transitions"] == 7
    assert len(collection_calls) == before_resume_calls
    assert not list(training_root.glob("round_*"))
    assert interrupted_student.training_window_starts == full_student.training_window_starts
    assert interrupted_student.training_window_starts == _expected_window_starts(
        num_collections=1,
        update_epochs=2,
        seed=resume_config["training"]["seed"],
    )
    for name, expected in full_student.state_dict().items():
        torch.testing.assert_close(interrupted_student.state_dict()[name], expected, rtol=0, atol=0)


def test_completed_round_saves_cleanup_cursor_before_removal_and_replays_on_resume(
    tmp_path, monkeypatch
):
    config = _config(
        tmp_path,
        "cleanup_resume",
        max_optimizer_steps=2,
        num_collections=1,
        update_epochs=1,
    )
    teacher = _make_teacher()
    student = RecordingStudent()
    env = EnvToken()
    monitor = RecordingMonitor()
    validation = _validation_batches()
    collection_calls = []

    def collector(config, output_dir, *, teacher, env, collection_round_idx):
        collection_calls.append(collection_round_idx)
        return _write_round_manifest(Path(output_dir), collection_round_idx)

    _patch_training_boundaries(monkeypatch, collector)
    lifecycle_module = __import__(
        "project.jepa_distill.collection_lifecycle", fromlist=["remove_completed_collection"]
    )
    original_remove = lifecycle_module.remove_completed_collection

    def interrupt_before_remove(training_root, round_idx):
        raise RuntimeError("simulated stop after the durable cleanup checkpoint")

    monkeypatch.setattr(lifecycle_module, "remove_completed_collection", interrupt_before_remove)
    with pytest.raises(RuntimeError, match="durable cleanup checkpoint"):
        TRAIN_MODULE.train(
            config,
            validation_batches=validation,
            teacher=teacher,
            student=student,
            env=env,
            monitor=monitor,
        )

    run_dir = tmp_path / "runs" / "cleanup_resume"
    training_root = tmp_path / "datasets" / "cleanup_resume" / "train"
    checkpoint_path = run_dir / "checkpoint.pt"
    checkpoint_payload = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    assert checkpoint_payload["step"] == config["training"]["max_optimizer_steps"]
    assert checkpoint_payload["sampler_state"]["collection_round_idx"] == 1
    assert checkpoint_payload["sampler_state"]["pending_cleanup_round_idx"] == 0
    assert (training_root / "round_0000").is_dir()

    monkeypatch.setattr(lifecycle_module, "remove_completed_collection", original_remove)
    config["training"]["resume_checkpoint"] = str(checkpoint_path)
    result = TRAIN_MODULE.train(
        config,
        validation_batches=validation,
        teacher=teacher,
        student=student,
        env=env,
        monitor=RecordingMonitor(),
    )

    assert result["optimizer_steps"] == 2
    assert collection_calls == [0]
    assert not list(training_root.glob("round_*"))
    final_payload = torch.load(result["checkpoint"], map_location="cpu", weights_only=False)
    assert "pending_cleanup_round_idx" not in final_payload["sampler_state"]


def test_resume_at_last_batch_of_final_epoch_completes_cleanup_without_updates(
    tmp_path, monkeypatch
):
    config = _config(
        tmp_path,
        "last_batch_resume",
        max_optimizer_steps=3,
        num_collections=1,
        update_epochs=1,
    )
    teacher = _make_teacher()
    student = RecordingStudent()
    env = EnvToken()
    validation = _validation_batches()
    collection_calls = []

    def collector(config, output_dir, *, teacher, env, collection_round_idx):
        collection_calls.append(collection_round_idx)
        return _write_round_manifest(Path(output_dir), collection_round_idx)

    _patch_training_boundaries(monkeypatch, collector)
    original_save = TRAIN_MODULE.save_checkpoint

    def stop_after_last_batch(path, student, **kwargs):
        original_save(path, student, **kwargs)
        sampler_state = kwargs.get("sampler_state", {})
        if (
            Path(path).name == "checkpoint.pt"
            and kwargs.get("step") == 2
            and sampler_state.get("next_batch_idx") == 2
        ):
            raise RuntimeError("simulated stop at round-end batch cursor")

    monkeypatch.setattr(TRAIN_MODULE, "save_checkpoint", stop_after_last_batch)
    with pytest.raises(RuntimeError, match="round-end batch cursor"):
        TRAIN_MODULE.train(
            config,
            validation_batches=validation,
            teacher=teacher,
            student=student,
            env=env,
            monitor=RecordingMonitor(),
        )

    checkpoint_path = tmp_path / "runs" / "last_batch_resume" / "checkpoint.pt"
    training_root = tmp_path / "datasets" / "last_batch_resume" / "train"
    payload = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    assert payload["sampler_state"]["collection_round_idx"] == 0
    assert payload["sampler_state"]["update_epoch_idx"] == 0
    assert payload["sampler_state"]["next_batch_idx"] == 2
    assert (training_root / "round_0000").is_dir()

    monkeypatch.setattr(TRAIN_MODULE, "save_checkpoint", original_save)
    config["training"]["resume_checkpoint"] = str(checkpoint_path)
    result = TRAIN_MODULE.train(
        config,
        validation_batches=validation,
        teacher=teacher,
        student=student,
        env=env,
        monitor=RecordingMonitor(),
    )
    assert result["optimizer_steps"] == 2
    assert collection_calls == [0]
    assert not list(training_root.glob("round_*"))
