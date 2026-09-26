"""CPU-only online collection lifecycle tests with a mocked simulator manager."""
from __future__ import annotations

import copy
import hashlib
import json
import shutil
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch
from torch import nn

from project.jepa_distill.collection_lifecycle import remove_completed_collection
from project.jepa_distill.observation_probe.config import load_probe_config


_OBSERVATION_LAYOUT = {
    "observation_dim": 2,
    "dtype": "float32",
    "ego_features": 1,
    "num_reward_coefs": 1,
    "goal_dim": 1,
    "context_dim": 1,
    "goal_features": 1,
    "partner_features": 1,
    "lane_features": 1,
    "boundary_features": 1,
    "traffic_control_features": 1,
    "obs_slots_partners_n": 0,
    "obs_slots_lane_kept": 0,
    "obs_slots_boundary_kept": 0,
    "obs_slots_traffic_controls_n": 0,
    "obs_valid_count_features": 0,
}
_COMPATIBILITY = {
    "observation_dim": 2,
    "chunk_length": 4,
    "num_action_classes": 12,
    "observation_layout": _OBSERVATION_LAYOUT,
    "action_layout": {},
    "teacher_observation_recipe": None,
}


class _Monitor:
    def __init__(self):
        self.steps = []
        self.exit_code = None

    def log_metrics(self, metrics, *, progress):
        self.steps.append(progress.optimizer_step)

    def log_evaluation(self, *args, **kwargs):
        return None

    def state_dict(self):
        return {"steps": list(self.steps)}

    def finish(self, *, exit_code=0):
        self.exit_code = exit_code


class _Dataset:
    instances = []

    def __init__(self, manifest_paths, *, expected_split="train"):
        self.paths = tuple(map(str, manifest_paths))
        self.expected_split = expected_split
        self.round_idx = None
        for path in self.paths:
            parent_name = Path(path).parent.name
            if parent_name.startswith("round_"):
                self.round_idx = int(parent_name.removeprefix("round_"))
                break
        self.count = 4 if expected_split == "train" else 2
        self.observation_dim = 2
        self.chunk_length = 4
        self.num_action_classes = 12
        self.observation_layout = _OBSERVATION_LAYOUT
        self.action_layout = {}
        self.teacher_observation_recipe = None
        self.closed = False
        self.__class__.instances.append(self)

    def __len__(self):
        return self.count

    def get_batch(self, indices):
        values = torch.tensor([[0.25, 0.5]] * len(indices), dtype=torch.float32)
        return SimpleNamespace(future_observations=values)

    def close(self):
        self.closed = True


class _Decoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.projection = nn.Linear(2, 2, bias=False)


class _Jepa(nn.Module):
    latent_dim = 2

    def __init__(self):
        super().__init__()
        self.projection = nn.Linear(2, 2, bias=False)


def _install_online_fakes(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    *,
    round_count: int,
    transitions_per_round: int,
    epochs_per_collection: int,
    max_optimizer_steps: int,
    batch_size: int,
    run_id: str,
):
    import project.jepa_distill.observation_probe.data as data_module
    import project.jepa_distill.observation_probe.evaluate as evaluate_module
    import project.jepa_distill.observation_probe.online as online_module
    import project.jepa_distill.observation_probe.schema as schema_module
    import project.jepa_distill.observation_probe.train as train_module

    _Dataset.instances = []
    source_checkpoint = tmp_path / "frozen-jepa.pt"
    source_checkpoint.write_bytes(b"fixed source checkpoint fixture")
    validation_manifest = tmp_path / "validation-manifest.json"
    validation_manifest.write_text("{}", encoding="utf-8")

    config = load_probe_config()
    config["checkpoint"] = str(source_checkpoint)
    config["data"].update(
        mode="online",
        collection_root=str(tmp_path / "collections"),
        collection_rounds=None,
        source_ranks=None,
        validation_manifest=str(validation_manifest),
        test_manifest=None,
        max_windows_per_collection=None,
    )
    config["training"].update(
        epochs_per_collection=epochs_per_collection,
        batch_size=batch_size,
        microbatch_size=batch_size,
        learning_rate=0.01,
        world_size=1,
        device="cpu",
        max_optimizer_steps=max_optimizer_steps,
        seed=7,
        cpu_threads=1,
        run_id=run_id,
        output_root=str(tmp_path / "runs"),
        resume_checkpoint=None,
    )
    config["evaluation"].update(
        every_collection=True,
        batch_size=2,
        max_windows=2,
        render_samples=0,
        training_mean_windows=2,
    )
    config["wandb"]["enabled"] = False

    source = {
        "source_config": {},
        "model_metadata": {},
        "teacher_checkpoint_sha256": "a" * 64,
        "compatibility": _COMPATIBILITY,
    }
    plan = {
        "round_count": round_count,
        "round_transitions": [transitions_per_round] * round_count,
        "max_transitions": transitions_per_round * round_count,
    }
    shared = {"ensure_calls": [], "removed": [], "evaluated": [], "manager_instances": []}

    class _Manager:
        def __init__(self, config, source, plan, *, rank, world_size, device):
            self.run_root = Path(config["data"]["collection_root"]) / config["training"]["run_id"]
            self.training_root = self.run_root / "rank_000" / "train"
            self.run_dir = Path(config["training"]["output_root"]) / config["training"]["run_id"]
            self.plan = plan
            shared["manager_instances"].append(self)

        def collect_heldout(self):
            return str(validation_manifest)

        def validate_active(self, active_round_idx):
            rounds = sorted(
                child.name
                for child in self.training_root.glob("round_*")
                if child.is_dir()
            ) if self.training_root.exists() else []
            expected = [] if active_round_idx is None else [f"round_{active_round_idx:04d}"]
            if rounds and rounds != expected:
                raise AssertionError(f"unexpected active rounds: {rounds}, expected {expected}")

        def ensure_training_manifests(self, round_idx, transitions):
            self.validate_active(round_idx)
            round_path = self.training_root / f"round_{round_idx:04d}"
            manifest_path = round_path / "manifest.json"
            reused = manifest_path.is_file()
            if not reused:
                round_path.mkdir(parents=True, exist_ok=False)
                manifest_path.write_text(
                    json.dumps({"complete": True, "split": "train", "round": round_idx}),
                    encoding="utf-8",
                )
            shared["ensure_calls"].append((round_idx, transitions, reused))
            return [str(manifest_path)], [
                {"collection_transition_count": transitions, "valid_window_count": 4}
            ]

        def fingerprints(self, manifest_paths):
            return [
                {
                    "path": str(Path(path).resolve()),
                    "sha256": hashlib.sha256(Path(path).read_bytes()).hexdigest(),
                }
                for path in manifest_paths
            ]

        def remove_training_collection(self, round_idx):
            readers = [
                reader for reader in _Dataset.instances
                if reader.expected_split == "train" and reader.round_idx == round_idx
            ]
            assert readers and all(reader.closed for reader in readers)
            assert round_idx in shared["evaluated"]
            checkpoint = torch.load(
                self.run_dir / "checkpoint.pt", map_location="cpu", weights_only=False
            )
            assert checkpoint["progress"]["pending_cleanup_round_idx"] == round_idx
            assert checkpoint["progress"]["collection_index"] == round_idx + 1
            assert checkpoint["active_collection"]
            assert (self.training_root / f"round_{round_idx:04d}").is_dir()
            remove_completed_collection(self.training_root, round_idx)
            shared["removed"].append(round_idx)

    monkeypatch.setattr(online_module, "inspect_online_source", lambda path: source)
    monkeypatch.setattr(online_module, "online_collection_plan", lambda config, source: plan)
    monkeypatch.setattr(online_module, "OnlineCollectionManager", _Manager)
    monkeypatch.setattr(data_module, "ProbeCollectionDataset", _Dataset)
    monkeypatch.setattr(train_module, "_setup_runtime", lambda config: (
        torch, 0, 1, 0, torch.device("cpu"), False
    ))
    monkeypatch.setattr(train_module, "_validate_heldout_compatibility", lambda *args, **kwargs: "matched")
    monkeypatch.setattr(schema_module, "ObservationSchema", lambda layout: SimpleNamespace(observation_dim=2))
    monkeypatch.setattr(evaluate_module, "fit_training_mean", lambda *args, **kwargs: torch.tensor([0.25, 0.5]))
    monkeypatch.setattr(
        train_module,
        "_evaluate_split",
        lambda *args, **kwargs: (
            shared["evaluated"].append(kwargs["collection_round_idx"])
            or {"probe/reconstruction/loss_total": 0.5}
        ),
    )
    monkeypatch.setattr(
        train_module,
        "train_update",
        lambda *args, **kwargs: {"loss_total": 0.25, "gradient_norm": 0.0},
    )
    return config, {"data": {"compatibility": _COMPATIBILITY}}, shared, train_module


def test_online_probe_cleans_each_round_after_training_reader_closes(tmp_path, monkeypatch):
    config, report, shared, train_module = _install_online_fakes(
        tmp_path,
        monkeypatch,
        round_count=2,
        transitions_per_round=4,
        epochs_per_collection=1,
        max_optimizer_steps=2,
        batch_size=4,
        run_id="online_two_rounds",
    )
    monitor = _Monitor()

    result = train_module.train_probe(
        config, report, jepa=_Jepa(), decoder=_Decoder(), monitor=monitor
    )

    assert result["status"] == "completed"
    assert result["optimizer_steps"] == 2
    assert shared["ensure_calls"] == [(0, 4, False), (1, 4, False)]
    assert shared["removed"] == [0, 1]
    assert all(reader.closed for reader in _Dataset.instances)
    training_root = tmp_path / "collections" / "online_two_rounds" / "rank_000" / "train"
    assert not list(training_root.glob("round_*"))
    assert Path(config["data"]["validation_manifest"]).is_file()

    final_checkpoint = torch.load(
        Path(config["training"]["output_root"]) / "online_two_rounds" / "checkpoint.pt",
        map_location="cpu",
        weights_only=False,
    )
    assert final_checkpoint["progress"]["collection_index"] == 2
    assert final_checkpoint["progress"]["collected_transitions"] == 8
    assert final_checkpoint["progress"]["active_round_idx"] is None
    assert final_checkpoint["progress"]["pending_cleanup_round_idx"] is None
    assert final_checkpoint["active_collection"] is None
    assert final_checkpoint["training_mean_observation"] == [0.25, 0.5]
    assert monitor.exit_code == 0


def test_online_midround_cap_keeps_collection_and_higher_cap_resume_cleans_it(tmp_path, monkeypatch):
    config, report, shared, train_module = _install_online_fakes(
        tmp_path,
        monkeypatch,
        round_count=1,
        transitions_per_round=4,
        epochs_per_collection=2,
        max_optimizer_steps=2,
        batch_size=2,
        run_id="online_midround_resume",
    )
    first_result = train_module.train_probe(
        config, report, jepa=_Jepa(), decoder=_Decoder(), monitor=_Monitor()
    )

    checkpoint_path = tmp_path / "runs" / "online_midround_resume" / "checkpoint.pt"
    capped_checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    training_root = tmp_path / "collections" / "online_midround_resume" / "rank_000" / "train"
    assert first_result["status"] == "capped"
    assert capped_checkpoint["progress"]["optimizer_step"] == 2
    assert capped_checkpoint["progress"]["collection_index"] == 0
    assert capped_checkpoint["progress"]["active_round_idx"] == 0
    assert capped_checkpoint["progress"]["collected_transitions"] == 4
    assert capped_checkpoint["progress"]["pending_cleanup_round_idx"] is None
    assert capped_checkpoint["active_collection"]
    assert not shared["removed"]
    assert (training_root / "round_0000").is_dir()
    assert all(reader.closed for reader in _Dataset.instances if reader.expected_split == "train")

    resumed_config = copy.deepcopy(config)
    resumed_config["training"]["max_optimizer_steps"] = 4
    resumed_config["training"]["resume_checkpoint"] = str(checkpoint_path)
    resumed_monitor = _Monitor()
    resumed_result = train_module.train_probe(
        resumed_config,
        report,
        jepa=_Jepa(),
        decoder=_Decoder(),
        monitor=resumed_monitor,
    )

    assert resumed_result["status"] == "completed"
    assert resumed_result["optimizer_steps"] == 4
    assert shared["ensure_calls"] == [(0, 4, False), (0, 4, True)]
    assert shared["removed"] == [0]
    assert not list(training_root.glob("round_*"))
    final_checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    assert final_checkpoint["progress"]["collection_index"] == 1
    assert final_checkpoint["progress"]["collected_transitions"] == 4
    assert final_checkpoint["progress"]["active_round_idx"] is None
    assert final_checkpoint["progress"]["pending_cleanup_round_idx"] is None
    assert final_checkpoint["active_collection"] is None
    assert final_checkpoint["training_mean_observation"] == [0.25, 0.5]
    assert all(reader.closed for reader in _Dataset.instances if reader.expected_split == "train")
    assert resumed_monitor.exit_code == 0
