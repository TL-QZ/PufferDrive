"""Focused tests for the bounded-memory Condition B collector."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest
import torch

from project.jepa_distill.collect import collect_dataset
from project.jepa_distill.dataset import TrajectoryDataset
from project.jepa_distill.streaming import StreamingTrajectoryDataset, collect_streaming_dataset


class _Teacher(torch.nn.Module):
    def __init__(self, classes: int = 3) -> None:
        super().__init__()
        self.action_table = torch.tensor(
            [[-1.0, -0.5], [0.0, 0.0], [1.0, 0.5]], dtype=torch.float32
        )[:classes]
        self.action_table_physical = self.action_table * 2.0
        self.condition_b_checkpoint_sha256 = "streaming-test-teacher"
        self.condition_b_observation_layout = {"observation_dim": 2}
        self.batch_sizes: list[int] = []

    def forward_eval(self, observations: torch.Tensor):
        self.batch_sizes.append(int(observations.shape[0]))
        logits = torch.arange(self.action_table.shape[0], dtype=torch.float32, device=observations.device)
        return logits.expand(observations.shape[0], -1), torch.zeros(
            observations.shape[0], 1, device=observations.device
        )


class _Env:
    num_agents = 2

    def __init__(self, *, truncate_at: int | None = None, ineligible_at: int | None = None) -> None:
        self.truncate_at = truncate_at
        self.ineligible_at = ineligible_at
        self.step_count = 0
        self.reset_seeds: list[int | None] = []
        self.observations = np.zeros((self.num_agents, 2), dtype=np.float32)
        self.masks = np.ones(self.num_agents, dtype=bool)
        self._pending = None

    def reset(self, seed=None):
        self.reset_seeds.append(seed)
        self.step_count = 0
        self.observations[:] = 1.0
        self.masks[:] = True
        return self.observations.copy(), {}

    def step(self, actions):
        del actions
        self.step_count += 1
        terminated = np.zeros(self.num_agents, dtype=bool)
        truncated = np.zeros(self.num_agents, dtype=bool)
        self.observations += 1.0
        self.masks[:] = True
        if self.ineligible_at is not None and self.step_count == self.ineligible_at:
            self.masks[:] = False
        if self.truncate_at is not None and self.step_count == self.truncate_at:
            truncated[:] = True
            self.observations[:] = 100.0
            self.step_count = 0
        self._pending = (
            self.observations.copy(),
            np.zeros(self.num_agents, dtype=np.float32),
            terminated,
            truncated,
            [],
            np.arange(self.num_agents),
            self.masks.copy(),
        )
        return self._pending[:5]

    def send(self, actions):
        self.step(actions)

    def recv(self):
        return self._pending


def _config(*, transitions: int = 8, inference_batch_size: int = 2) -> dict:
    return {
        "teacher_config": {"env": {"dt": 0.3}},
        "model": {"chunk_length": 2, "num_action_classes": 3},
        "collection": {
            "split": "train",
            "split_seeds": {"train": 11, "validation": 22, "test": 33},
            "action_selection": "sample",
            "transitions_per_round": transitions,
            "inference_batch_size": inference_batch_size,
            "max_disk_bytes": 10_000_000,
        },
    }


def test_streaming_matches_legacy_boundary_windows_and_samples(tmp_path):
    teacher_legacy = _Teacher()
    teacher_streaming = _Teacher()
    config = _config()
    config["collection"]["action_selection"] = "mode"
    legacy = collect_dataset(
        config, tmp_path / "legacy", teacher=teacher_legacy, env=_Env(truncate_at=3), collection_round_idx=0
    )
    streaming = collect_streaming_dataset(
        config, tmp_path / "streaming", teacher=teacher_streaming, env=_Env(truncate_at=3), collection_round_idx=0
    )

    assert streaming["collection_transition_count"] == legacy["collection_transition_count"]
    assert streaming["valid_window_count"] == legacy["valid_window_count"] == 2
    legacy_dataset = TrajectoryDataset(Path(legacy["manifest_path"]).parent, split="train")
    streaming_dataset = StreamingTrajectoryDataset(streaming["manifest_path"])
    assert len(streaming_dataset) == len(legacy_dataset)
    for index in range(len(streaming_dataset)):
        actual = streaming_dataset[index]
        expected = legacy_dataset[index]
        torch.testing.assert_close(actual.observations, expected.observations)
        torch.testing.assert_close(actual.executed_controls, expected.executed_controls)
        torch.testing.assert_close(actual.teacher_logits, expected.teacher_logits)


def test_streaming_chunked_inference_and_batched_fetch(tmp_path):
    teacher = _Teacher()
    manifest = collect_streaming_dataset(
        _config(transitions=10, inference_batch_size=2),
        tmp_path,
        teacher=teacher,
        env=_Env(),
        collection_round_idx=0,
    )
    assert max(teacher.batch_sizes) <= 2
    assert len(teacher.batch_sizes) == 5
    dataset = StreamingTrajectoryDataset(manifest["manifest_path"])
    batch = dataset.get_batch(np.arange(min(3, len(dataset)), dtype=np.int64))
    assert tuple(batch.observations.shape[1:]) == (3, 2)
    assert tuple(batch.executed_controls.shape[1:]) == (2, 2)
    assert tuple(batch.teacher_logits.shape[1:]) == (2, 3)
    np.testing.assert_array_equal(
        dataset.window_indices,
        np.unique(dataset.window_indices),
    )


def test_streaming_live_round_continuity_and_ineligible_reset(tmp_path):
    env = _Env(ineligible_at=2)
    first = collect_streaming_dataset(_config(), tmp_path, teacher=_Teacher(), env=env, collection_round_idx=0)
    reset_count = len(env.reset_seeds)
    second = collect_streaming_dataset(_config(), tmp_path, teacher=_Teacher(), env=env, collection_round_idx=1)
    assert len(env.reset_seeds) == reset_count
    assert first["valid_window_count"] == 0
    assert second["valid_window_count"] == 0
    assert second["collection_round_idx"] == 1


def test_incomplete_round_is_rejected_and_manifest_is_atomic(tmp_path):
    incomplete_round = tmp_path / "round_0000"
    incomplete_round.mkdir()
    with pytest.raises(RuntimeError, match="incomplete"):
        collect_streaming_dataset(_config(), tmp_path, teacher=_Teacher(), env=_Env(), collection_round_idx=1)

    complete_root = tmp_path / "complete"
    manifest = collect_streaming_dataset(
        _config(), complete_root, teacher=_Teacher(), env=_Env(), collection_round_idx=0
    )
    on_disk = json.loads(Path(manifest["manifest_path"]).read_text())
    assert on_disk["complete"] is True
    assert "manifest_path" not in on_disk
