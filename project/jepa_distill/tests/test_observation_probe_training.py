"""CPU tests for global-batch updates, padded tails and probe resume cursors."""
from __future__ import annotations

import copy
import json
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch
from torch import nn

from project.jepa_distill.observation_probe.contracts import DecoderOutput, ReconstructionLoss
from project.jepa_distill.observation_probe.train import (
    _atomic_torch_save,
    _checkpoint_payload,
    _epoch_batch_indices,
    _load_resume_checkpoint,
    _next_progress,
    train_probe,
    train_update,
)


class _ToyDataset:
    def __init__(self, values: torch.Tensor):
        self.values = values

    def get_batch(self, indices):
        return SimpleNamespace(future_observations=self.values[list(indices)])


class _ToyJepa(nn.Module):
    latent_dim = 2

    def encode_target(self, observations):
        return observations


class _ToyDecoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.projection = nn.Linear(2, 2, bias=False)

    def forward(self, normalized_latents):
        return DecoderOutput(
            continuous={"ego": self.projection(normalized_latents)},
            categorical_logits={},
            presence_logits={},
        )


def _toy_losses(monkeypatch):
    import project.jepa_distill.observation_probe.losses as losses

    monkeypatch.setattr(losses, "match_objects", lambda *args, **kwargs: {})

    def reconstruction_loss(prediction, targets, assignments, **kwargs):
        loss = (prediction.continuous["ego"] - targets[:, :2]).square().sum(-1).mean()
        return ReconstructionLoss(loss, {"toy": loss}, {"ego": len(targets)})

    monkeypatch.setattr(losses, "reconstruction_loss", reconstruction_loss)


def _update_config(microbatch_size: int) -> dict:
    return {
        "training": {
            "microbatch_size": microbatch_size,
            "gradient_clip_norm": 1000.0,
        },
        "matching": {},
        "loss": {},
        "_observation_layout": {"observation_dim": 2},
    }


def test_accumulated_microbatches_match_one_global_sample_mean(monkeypatch):
    _toy_losses(monkeypatch)
    values = torch.tensor(
        [[1.0, 0.0], [0.0, 2.0], [3.0, 4.0], [2.0, 1.0], [5.0, 2.0]],
        dtype=torch.float32,
    )
    dataset = _ToyDataset(values)
    torch.manual_seed(11)
    accumulated = _ToyDecoder()
    reference = copy.deepcopy(accumulated)
    accumulated_optimizer = torch.optim.SGD(accumulated.parameters(), lr=0.05)
    reference_optimizer = torch.optim.SGD(reference.parameters(), lr=0.05)

    metrics = train_update(
        accumulated,
        _ToyJepa(),
        dataset,
        list(range(len(values))),
        accumulated_optimizer,
        _update_config(microbatch_size=2),
        torch.device("cpu"),
    )

    normalized = torch.nn.functional.normalize(values, dim=-1)
    reference_optimizer.zero_grad(set_to_none=True)
    reference_loss = (reference.projection(normalized) - values).square().sum(-1).mean()
    reference_loss.backward()
    torch.nn.utils.clip_grad_norm_(reference.parameters(), 1000.0)
    reference_optimizer.step()

    torch.testing.assert_close(accumulated.projection.weight, reference.projection.weight)
    assert metrics["loss_total"] == pytest.approx(float(reference_loss.detach()))
    assert all(parameter.grad is None for parameter in _ToyJepa().parameters())


def test_epoch_batch_indices_repeats_only_the_short_tail():
    shuffled_order = [31, 12, 45]

    assert _epoch_batch_indices(shuffled_order, 3, 0, 4) == [31, 12, 45, 31]
    assert _epoch_batch_indices(shuffled_order, 3, 4, 8) == [12, 45, 31, 12]
    assert _epoch_batch_indices(shuffled_order, 3, 0, 8) == [31, 12, 45, 31, 12, 45, 31, 12]


def test_progress_cursor_advances_collection_after_final_epoch_batch():
    next_cursor = _next_progress(
        collection_index=1,
        epoch_index=2,
        batch_index=3,
        batches_per_epoch=4,
        epochs_per_collection=3,
        collection_count=5,
        optimizer_step=19,
    )

    assert next_cursor == {
        "collection_index": 2,
        "epoch_index": 0,
        "next_batch_index": 0,
        "optimizer_step": 19,
    }


def test_resume_checkpoint_requires_matching_frozen_data_and_config_identity(tmp_path):
    decoder = _ToyDecoder()
    optimizer = torch.optim.AdamW(decoder.parameters(), lr=0.01)
    payload = _checkpoint_payload(
        decoder,
        optimizer,
        identity={"frozen_checkpoint_sha256": "abc", "config": {"seed": 1}},
        progress={"collection_index": 0, "epoch_index": 1, "next_batch_index": 2, "optimizer_step": 3},
        monitoring_state=None,
        last_validation_step=None,
        last_test_step=None,
        best_validation_loss=None,
    )
    path = tmp_path / "probe.pt"
    _atomic_torch_save(payload, path)

    restored = _load_resume_checkpoint(
        path,
        _ToyDecoder(),
        torch.optim.AdamW(_ToyDecoder().parameters(), lr=0.01),
        expected_identity={"frozen_checkpoint_sha256": "abc", "config": {"seed": 1}},
        map_location="cpu",
    )
    assert restored["progress"]["optimizer_step"] == 3

    with pytest.raises(ValueError, match="identity differs"):
        _load_resume_checkpoint(
            path,
            _ToyDecoder(),
            torch.optim.AdamW(_ToyDecoder().parameters(), lr=0.01),
            expected_identity={"frozen_checkpoint_sha256": "changed", "config": {"seed": 1}},
            map_location="cpu",
        )


class _FakeMonitor:
    def __init__(self, *, fail_step: int | None = None):
        self.fail_step = fail_step
        self.steps = []
        self.exit_code = None

    def log_metrics(self, metrics, *, progress):
        self.steps.append(progress.optimizer_step)
        if self.fail_step == progress.optimizer_step:
            raise RuntimeError("simulated interruption after the previous checkpoint")

    def state_dict(self):
        return {"mode": "disabled", "last_step": self.steps[-1] if self.steps else 0}

    def finish(self, *, exit_code=0):
        self.exit_code = exit_code


def test_training_checkpoint_resumes_at_next_global_batch(tmp_path, monkeypatch):
    _toy_losses(monkeypatch)
    from project.jepa_distill.observation_probe import data, evaluate, model

    checkpoint = tmp_path / "frozen.pt"
    checkpoint.write_bytes(b"fixed frozen checkpoint fixture")
    train_first = tmp_path / "train-round-4.json"
    train_second = tmp_path / "train-round-7.json"
    validation = tmp_path / "validation.json"
    for path in (train_first, train_second, validation):
        path.write_text("{}", encoding="utf-8")

    class FakeDataset:
        open_calls = []

        def __init__(self, paths, *, expected_split="train"):
            self.paths = tuple(map(str, paths))
            self.expected_split = expected_split
            self.open_calls.append((self.paths, expected_split))
            self.count = 5 if expected_split == "train" and "round-4" in self.paths[0] else (
                3 if expected_split == "validation" else 3
            )
            self.observation_dim = 2
            self.chunk_length = 20
            self.num_action_classes = 12
            self.observation_layout = {"observation_dim": 2}
            self.action_layout = {}
            self.teacher_observation_recipe = None
            self.values = torch.tensor(
                [[float(index + 1), float(index + 2)] for index in range(self.count)],
                dtype=torch.float32,
            )
            self.closed = False

        def __len__(self):
            return self.count

        def get_batch(self, indices):
            return SimpleNamespace(future_observations=self.values[list(indices)])

        def close(self):
            self.closed = True

    class TinyDecoder(_ToyDecoder):
        def __init__(self, latent_dim, observation_layout, *, hidden_sizes):
            super().__init__()

    def evaluate_probe(jepa, decoder, batches, *, config):
        count = sum(len(batch.future_observations) for batch in batches)
        return {"probe/reconstruction/loss_total": float(count)}

    monkeypatch.setattr(data, "ProbeCollectionDataset", FakeDataset, raising=False)
    monkeypatch.setattr(data, "selected_indices", lambda n, **kwargs: list(range(min(n, kwargs.get("max_windows") or n))))
    monkeypatch.setattr(model, "load_frozen_jepa", lambda path, *, device: _ToyJepa())
    monkeypatch.setattr(model, "ObservationDecoder", TinyDecoder)
    monkeypatch.setattr(evaluate, "evaluate_probe", evaluate_probe)

    from project.jepa_distill.observation_probe.config import load_probe_config

    config = load_probe_config()
    config["data"]["mode"] = "cached"
    config["checkpoint"] = str(checkpoint)
    config["data"]["validation_manifest"] = str(validation)
    config["data"]["test_manifest"] = None
    config["data"]["collection_rounds"] = [4, 7]
    config["data"]["source_ranks"] = [0]
    config["training"].update(
        {
            "epochs_per_collection": 1,
            "batch_size": 4,
            "microbatch_size": 2,
            "learning_rate": 0.01,
            "world_size": 1,
            "device": "cpu",
            "max_optimizer_steps": 2,
            "run_id": "probe-resume-test",
            "output_root": str(tmp_path / "runs"),
            "resume_checkpoint": None,
        }
    )
    config["evaluation"].update({"batch_size": 2, "max_windows": 3, "render_samples": 0})
    report = {
        "data": {
            "per_round": [
                {
                    "collection_round_idx": 4,
                    "manifest_paths": [str(train_first)],
                    "pooled_valid_window_count": 5,
                },
                {
                    "collection_round_idx": 7,
                    "manifest_paths": [str(train_second)],
                    "pooled_valid_window_count": 3,
                },
            ],
            "compatibility": {
                "observation_dim": 2,
                "chunk_length": 20,
                "num_action_classes": 12,
                "observation_layout": {"observation_dim": 2},
                "action_layout": {},
                "teacher_observation_recipe": None,
            },
        }
    }
    first_monitor = _FakeMonitor(fail_step=2)

    with pytest.raises(RuntimeError, match="simulated interruption"):
        train_probe(config, report, monitor=first_monitor)

    saved_path = tmp_path / "runs" / "probe-resume-test" / "checkpoint.pt"
    saved = torch.load(saved_path, map_location="cpu", weights_only=False)
    assert saved["format"] == "observation_probe_v1"
    assert saved["progress"] == {
        "collection_index": 0,
        "epoch_index": 0,
        "next_batch_index": 1,
        "optimizer_step": 1,
    }
    assert first_monitor.exit_code == 1

    resumed_config = copy.deepcopy(config)
    resumed_config["training"]["resume_checkpoint"] = str(saved_path)
    resumed_monitor = _FakeMonitor()
    result = train_probe(resumed_config, report, monitor=resumed_monitor)

    assert result["status"] == "capped"
    assert result["optimizer_steps"] == 2
    assert result["progress"]["collection_index"] == 1
    assert [row["split"] for row in result["evaluations"]] == ["validation"]
    assert Path(result["best_checkpoint_path"]).is_file()
    best = torch.load(result["best_checkpoint_path"], map_location="cpu", weights_only=False)
    assert best["best_validation_loss"] == pytest.approx(3.0)
    assert resumed_monitor.steps[-1] == 2
    assert resumed_monitor.exit_code == 0
    assert any(split == "validation" for _paths, split in FakeDataset.open_calls)
