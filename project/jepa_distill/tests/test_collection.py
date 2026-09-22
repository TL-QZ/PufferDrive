"""Bounded collector tests with a vector-like zero-copy environment."""

from pathlib import Path
import json

import numpy as np
import pytest
import torch

from project.jepa_distill.collect import collect_dataset
from project.jepa_distill.dataset import TrajectoryDataset


class _Teacher(torch.nn.Module):
    def __init__(self, classes=3):
        super().__init__()
        self.action_table = torch.tensor(
            [[-1.0, -0.5], [0.0, 0.0], [1.0, 0.5]], dtype=torch.float32
        )[:classes]
        self.action_table_physical = self.action_table * 2.0

    def forward_eval(self, observations):
        logits = torch.arange(self.action_table.shape[0], dtype=torch.float32, device=observations.device)
        return logits.expand(observations.shape[0], -1), torch.zeros(observations.shape[0], 1, device=observations.device)


class _Env:
    """Drive-shaped env exposing masks only through send/recv."""

    num_agents = 2

    def __init__(self, *, truncate_at=None):
        self.truncate_at = truncate_at
        self.reset_seeds = []
        self.step_count = 0
        self.observations = np.zeros((self.num_agents, 2), dtype=np.float32)
        self.masks = np.ones(self.num_agents, dtype=bool)
        self.last_actions = []
        self._pending = None

    def reset(self, seed=None):
        self.reset_seeds.append(seed)
        self.step_count = 0
        self.observations[:] = 1.0
        self.masks[:] = True
        return self.observations.copy(), {}

    def step(self, actions):
        self.last_actions.append(np.asarray(actions).copy())
        self.step_count += 1
        terminated = np.zeros(self.num_agents, dtype=bool)
        truncated = np.zeros(self.num_agents, dtype=bool)
        if self.truncate_at is not None and self.step_count == self.truncate_at:
            truncated[:] = True
            self.observations[:] = 100.0
            self.step_count = 0
        else:
            self.observations += 1.0
        self.masks[:] = True
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


def _config(*, transitions=8, split="train", max_disk_bytes=10_000_000):
    return {
        "teacher": {},
        "model": {"chunk_length": 2, "num_action_classes": 3},
        "collection": {
            "split": split,
            "split_seeds": {"train": 11, "validation": 22, "test": 33},
            "action_selection": "sample",
            "transitions_per_round": transitions,
            "shard_transition_count": 4,
            "max_disk_bytes": max_disk_bytes,
        },
    }


def test_collector_writes_full_logits_actual_controls_and_manifest(tmp_path):
    env = _Env()
    manifest = collect_dataset(_config(), tmp_path, teacher=_Teacher(), env=env, collection_round_idx=0)
    assert Path(manifest["manifest_path"]).is_file()
    on_disk_manifest = json.loads(Path(manifest["manifest_path"]).read_text())
    assert "manifest_path" not in on_disk_manifest
    assert manifest["observation_layout"]["observation_dim"] == 2
    assert manifest["action_layout"]["normalized_controls"] == [[-1.0, -0.5], [0.0, 0.0], [1.0, 0.5]]
    assert manifest["stats"]["eligibility_mask_available"] is True
    assert manifest["collection_transition_count"] == 8
    assert len(env.last_actions) == 4
    np.testing.assert_array_less(np.abs(np.concatenate(env.last_actions)), 1.000001)
    dataset = TrajectoryDataset(Path(manifest["manifest_path"]).parent, split="train", manifest=on_disk_manifest)
    assert len(dataset) == 6
    sample = dataset[0]
    assert tuple(sample.observations.shape) == (3, 2)
    assert tuple(sample.executed_controls.shape) == (2, 2)
    assert tuple(sample.teacher_logits.shape) == (2, 3)


def test_terminal_truncation_closes_window_and_live_env_continues(tmp_path):
    env = _Env(truncate_at=3)
    config = _config(transitions=8)
    first = collect_dataset(config, tmp_path, teacher=_Teacher(), env=env, collection_round_idx=0)
    assert first["stats"]["rejected_reasons"]["truncated"] == 2
    # Each of the two stable vector slots contributes one valid pre-reset
    # window; the truncation transition itself is excluded.
    assert first["valid_window_count"] == 2
    reset_count = len(env.reset_seeds)
    second = collect_dataset(config, tmp_path, teacher=_Teacher(), env=env, collection_round_idx=1)
    assert len(env.reset_seeds) == reset_count
    assert second["collection_round_idx"] == 1
    assert second["stats"]["valid_window_count"] == 2


def test_split_seed_is_selected_on_first_reset(tmp_path):
    env = _Env()
    collect_dataset(_config(split="validation"), tmp_path, teacher=_Teacher(), env=env, collection_round_idx=0)
    assert env.reset_seeds == [22]


def test_disk_budget_is_checked_before_reset(tmp_path):
    env = _Env()
    with pytest.raises(RuntimeError, match="max_disk_bytes"):
        collect_dataset(_config(max_disk_bytes=1), tmp_path, teacher=_Teacher(), env=env, collection_round_idx=0)
    assert not list(tmp_path.rglob("*.npz"))
