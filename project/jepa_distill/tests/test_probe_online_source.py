"""Online source checks and a small end-to-end streaming collection."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import pytest
import torch

from project.jepa_distill.observation_probe.config import DEFAULT_CONFIG, load_probe_config
from project.jepa_distill.observation_probe.online import (
    OnlineCollectionManager,
    estimated_round_bytes,
    inspect_online_source,
    online_collection_plan,
)
from project.jepa_distill.observation_probe.train import preflight_probe
from project.jepa_distill.runtime import resolve_path


_TEACHER_HASH = hashlib.sha256(b"probe-source-test-teacher").hexdigest()
_ACTION_TABLE = [[-1.0, -0.5], [0.0, 0.0], [1.0, 0.5]]


class _FakeTeacher(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.action_table = torch.tensor(_ACTION_TABLE, dtype=torch.float32)
        self.action_table_physical = self.action_table * 2.0
        self.condition_b_checkpoint_sha256 = _TEACHER_HASH
        self.condition_b_observation_layout = {
            "observation_dim": 2,
            "dtype": "float32",
        }

    def forward_eval(self, observations: torch.Tensor):
        logits = torch.zeros((observations.shape[0], 3), dtype=torch.float32)
        logits[:, 2] = 3.0
        return logits, torch.zeros((observations.shape[0], 1), dtype=torch.float32)


class _FakeEnv:
    num_agents = 2

    def __init__(self) -> None:
        self.observations = np.zeros((self.num_agents, 2), dtype=np.float32)
        self.masks = np.ones(self.num_agents, dtype=bool)
        self._pending = None
        self.closed = False

    def reset(self, seed=None):
        self.seed = seed
        self.observations[:] = 1.0
        self.masks[:] = True
        return self.observations.copy(), {}

    def send(self, actions):
        del actions
        self.observations += 1.0
        self._pending = (
            self.observations.copy(),
            np.zeros(self.num_agents, dtype=np.float32),
            np.zeros(self.num_agents, dtype=bool),
            np.zeros(self.num_agents, dtype=bool),
            [],
            np.arange(self.num_agents),
            self.masks.copy(),
        )

    def recv(self):
        return self._pending

    def close(self):
        self.closed = True


def _source_checkpoint(tmp_path: Path) -> Path:
    teacher_checkpoint = tmp_path / "teacher.pt"
    teacher_checkpoint.write_bytes(b"metadata-only test teacher checkpoint")
    source_config = {
        "teacher": {"checkpoint": str(teacher_checkpoint)},
        "teacher_checkpoint_sha256": _TEACHER_HASH,
        "model": {"chunk_length": 2, "num_action_classes": 3},
        "vec": {"backend": "Serial", "num_envs": 1, "num_workers": 1, "batch_size": 1},
        "collection": {
            "num_collections": 1,
            "transitions_per_round": 8,
            "max_transitions": 8,
            "max_disk_bytes": 1_000_000,
            "action_selection": "mode",
            "inference_batch_size": 2,
            "split_seeds": {"train": 11, "validation": 22, "test": 33},
        },
        "training": {"validation_transitions": 8},
        "validation": {
            "env_overrides": {},
            "vec": {"backend": "Serial", "num_envs": 1, "num_workers": 1, "batch_size": 1},
        },
    }
    teacher_config = {
        "env": {
            "num_agents": 2,
            "action_type": "discrete",
            "dt": 0.3,
            "dynamics_model": "classic",
            "reward_conditioning": "none",
        },
        "policy": {},
    }
    payload = {
        "format": "condition_b_v1",
        "config": source_config,
        "model_config": {
            "model": {"chunk_length": 2, "num_action_classes": 3},
            "observation_layout": {"observation_dim": 2, "dtype": "float32"},
            "teacher_config": teacher_config,
            "action_table": _ACTION_TABLE,
            "action_table_physical": [[-2.0, -1.0], [0.0, 0.0], [2.0, 1.0]],
        },
    }
    checkpoint = tmp_path / "condition_b.pt"
    torch.save(payload, checkpoint)
    return checkpoint


def _online_config(tmp_path: Path, checkpoint: Path) -> dict:
    overrides = [
        f"checkpoint={json.dumps(str(checkpoint))}",
        f"data.collection_root={json.dumps(str(tmp_path / 'collections'))}",
        "training.run_id=source-test",
        "training.device=cpu",
        "training.world_size=1",
        "training.seed=7",
        "wandb.enabled=false",
        "collection.num_collections=1",
        "collection.transitions_per_round=8",
        "collection.max_transitions=8",
        "collection.max_disk_bytes=1000000",
        "collection.validation_transitions=8",
        "collection.action_selection=mode",
        "collection.inference_batch_size=2",
        "collection.env_overrides={num_agents: 2}",
        "collection.vec_overrides={backend: Serial, num_envs: 1, num_workers: 1, batch_size: 1}",
    ]
    return load_probe_config(DEFAULT_CONFIG, overrides)


def test_online_preflight_reads_source_metadata_without_cached_data_or_runtime(tmp_path):
    checkpoint = _source_checkpoint(tmp_path)
    config = _online_config(tmp_path, checkpoint)
    collection_root = resolve_path(config["data"]["collection_root"])

    report = preflight_probe(config)

    assert report["mode"] == "online"
    assert report["manifest_count"] == 0
    assert report["data"]["selected_manifests"] == []
    assert report["data"]["valid_windows"] == "unknown until the live simulator collection completes"
    assert report["heldout_data"]["validation"] == "collected once by rank 0 and retained"
    assert not collection_root.exists()


def test_online_plan_rejects_disk_budget_that_cannot_fit_active_round_and_heldout(tmp_path):
    checkpoint = _source_checkpoint(tmp_path)
    source = inspect_online_source(checkpoint)
    config = _online_config(tmp_path, checkpoint)
    round_bytes = estimated_round_bytes(
        transitions=8,
        slots=2,
        compatibility=source["compatibility"],
    )
    config["collection"]["max_disk_bytes"] = 2 * round_bytes - 1

    with pytest.raises(ValueError, match="cannot fit one active training round"):
        online_collection_plan(config, source)


def test_manifest_layout_allows_collector_dtype_extension_and_checks_features(tmp_path, monkeypatch):
    checkpoint = _source_checkpoint(tmp_path)
    payload = torch.load(checkpoint, map_location="cpu", weights_only=False)
    source_layout = {
        "boundary_features": 9,
        "context_dim": 26,
        "ego_features": 10,
        "goal_dim": 9,
        "goal_features": 3,
        "lane_features": 9,
        "num_reward_coefs": 17,
        "obs_slots_boundary_kept": 50,
        "obs_slots_lane_kept": 70,
        "obs_slots_partners_n": 16,
        "obs_slots_traffic_controls_n": 4,
        "obs_valid_count_features": 4,
        "observation_dim": 1292,
        "partner_features": 9,
        "traffic_control_features": 7,
    }
    payload["model_config"]["observation_layout"] = source_layout
    torch.save(payload, checkpoint)

    source = inspect_online_source(checkpoint)
    config = _online_config(tmp_path, checkpoint)
    plan = online_collection_plan(config, source)
    manager = OnlineCollectionManager(
        config,
        source,
        plan,
        rank=0,
        world_size=1,
        device=torch.device("cpu"),
    )
    expected_config, _ = manager._collection_config(
        split="validation", round_idx=0, transitions=8
    )
    assert "dtype" not in expected_config["collection"]["observation_layout"]

    manifest_path = tmp_path / "validation_manifest.json"
    manifest_path.write_text(
        json.dumps({"effective_config": expected_config}), encoding="utf-8"
    )
    inspected_info = {
        "observation_dim": 1292,
        "chunk_length": 2,
        "num_action_classes": 3,
        "observation_layout": {**source_layout, "dtype": "float32"},
        "action_layout": source["compatibility"]["action_layout"],
        "teacher_observation_recipe": source["compatibility"]["teacher_observation_recipe"],
        "teacher_checkpoint_sha256": source["teacher_checkpoint_sha256"],
        "collection_transition_count": 8,
        "split_seed": manager._expected_seed(
            split="validation", round_idx=0, source_rank=0
        ),
        "valid_window_count": 6,
    }

    def inspect_manifest(_path, *, expected_round_idx, expected_split):
        assert expected_round_idx == 0
        assert expected_split == "validation"
        return inspected_info

    monkeypatch.setattr(
        "project.jepa_distill.observation_probe.data._inspect_manifest",
        inspect_manifest,
    )

    info = manager._validate_manifest(
        manifest_path,
        round_idx=0,
        split="validation",
        transitions=8,
    )
    assert info["valid_window_count"] == 6

    inspected_info["observation_layout"]["lane_features"] += 1
    with pytest.raises(ValueError, match="disagrees with frozen source at observation_layout"):
        manager._validate_manifest(
            manifest_path,
            round_idx=0,
            split="validation",
            transitions=8,
        )


def test_manager_collects_a_real_streaming_round_from_frozen_source(tmp_path, monkeypatch):
    checkpoint = _source_checkpoint(tmp_path)
    source = inspect_online_source(checkpoint)
    config = _online_config(tmp_path, checkpoint)
    plan = online_collection_plan(config, source)
    teacher = _FakeTeacher()
    environments: list[_FakeEnv] = []

    def make_env(teacher_config, collection_config, split):
        assert teacher_config["env"]["dt"] == 0.3
        assert split == "train"
        env = _FakeEnv()
        environments.append(env)
        return env

    def load_teacher(source_config, *, resolved_config, device):
        assert source_config["teacher"]["checkpoint"] == source["source_config"]["teacher"]["checkpoint"]
        assert resolved_config["env"]["num_agents"] == 2
        assert device == "cpu"
        return teacher

    monkeypatch.setattr("project.jepa_distill.runtime.create_vecenv", make_env)
    monkeypatch.setattr("project.jepa_distill.teacher.load_teacher", load_teacher)
    manager = OnlineCollectionManager(
        config,
        source,
        plan,
        rank=0,
        world_size=1,
        device=torch.device("cpu"),
    )

    manifest_paths, infos = manager.ensure_training_manifests(round_idx=0, transitions=8)

    manifest_path = Path(manifest_paths[0])
    saved_manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    assert manifest_path == manager.training_root / "round_0000" / "manifest.json"
    assert saved_manifest["complete"] is True
    assert saved_manifest["split"] == "train"
    assert saved_manifest["collection_transition_count"] == 8
    assert saved_manifest["split_seed"] == 7
    assert saved_manifest["teacher_checkpoint_sha256"] == _TEACHER_HASH
    assert infos[0]["valid_window_count"] == 6
    assert environments[0].closed is True

    dataset = manager.open_training_dataset(manifest_paths)
    try:
        assert len(dataset) == 6
        batch = dataset.get_batch(np.asarray([0, 5], dtype=np.int64))
        assert tuple(batch.current_observations.shape) == (2, 2)
        assert tuple(batch.future_observations.shape) == (2, 2)
        assert tuple(batch.executed_controls.shape) == (2, 2, 2)
    finally:
        dataset.close()
