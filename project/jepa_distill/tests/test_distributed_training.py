"""Small CPU/Gloo regression tests for the distributed Condition B trainer."""

from __future__ import annotations

import copy
import os
import random
from pathlib import Path

import numpy as np
import pytest
import torch
import torch.distributed as dist
import torch.multiprocessing as mp
from torch import nn

from project.jepa_distill.dataset import TrainingBatch
from project.jepa_distill.model import LossTerms


DIST = __import__("project.jepa_distill.distributed_train", fromlist=["*"])

torch.set_num_threads(1)


class LinearStudent(nn.Module):
    """Tiny student with an explicit EMA target for DDP arithmetic checks."""

    def __init__(self) -> None:
        super().__init__()
        self.online = nn.Linear(2, 1)
        self.target = copy.deepcopy(self.online)
        self.target.requires_grad_(False)

    def forward(self, observations: torch.Tensor, controls: torch.Tensor) -> torch.Tensor:
        del controls
        return self.online(observations[:, 0]).squeeze(-1)

    def compute_losses(
        self,
        outputs: torch.Tensor,
        teacher_logits: torch.Tensor,
        config=None,
    ) -> LossTerms:
        del config
        target = teacher_logits[:, 0, 0]
        total = (outputs - target).square().mean()
        zero = total.detach() * 0.0
        return LossTerms(total, total, zero, zero)

    @torch.no_grad()
    def update_target_encoder(self, tau: float = 0.5) -> None:
        for target, online in zip(self.target.parameters(), self.online.parameters()):
            target.mul_(tau).add_(online, alpha=1.0 - tau)

    def export_metadata(self):
        return {"test_model": "distributed_linear"}


class TensorDataset:
    """In-memory stand-in for the streaming dataset contract."""

    def __init__(self, observations: torch.Tensor, targets: torch.Tensor) -> None:
        self.observations = observations
        self.targets = targets

    def __len__(self) -> int:
        return int(self.observations.shape[0])

    def get_batch(self, indices) -> TrainingBatch:
        index_tensor = torch.as_tensor(list(indices), dtype=torch.long)
        count = int(index_tensor.numel())
        controls = torch.zeros(count, 1, 2)
        teacher_logits = torch.zeros(count, 1, 1)
        teacher_logits[:, 0, 0] = self.targets[index_tensor]
        return TrainingBatch(
            self.observations[index_tensor],
            controls,
            teacher_logits,
        )


def _init_gloo(rank: int, world_size: int, rendezvous: str) -> None:
    # The managed runner has no resolvable container hostname; bind Gloo to
    # the loopback interface so the test remains local and deterministic.
    os.environ.setdefault("GLOO_SOCKET_IFNAME", "lo")
    dist.init_process_group(
        backend="gloo",
        init_method=f"file://{rendezvous}",
        rank=rank,
        world_size=world_size,
    )


def _base_config(*, microbatch_size: int, variance_scope: str = "microbatch") -> dict:
    return {
        "training": {
            "microbatch_size": microbatch_size,
            "variance_scope": variance_scope,
            "gradient_clip_norm": 100.0,
        },
        "ema": {"tau": 0.5},
    }


def _fixed_student() -> LinearStudent:
    student = LinearStudent()
    with torch.no_grad():
        student.online.weight.copy_(torch.tensor([[0.3, -0.2]]))
        student.online.bias.copy_(torch.tensor([0.1]))
        student.target.load_state_dict(student.online.state_dict())
    return student


def _reference_update(
    student: LinearStudent,
    dataset: TensorDataset,
    sample_indices: list[int],
    *,
    config: dict,
) -> tuple[dict[str, torch.Tensor], LossTerms]:
    optimizer = torch.optim.AdamW(student.parameters(), lr=0.05, weight_decay=0.0)
    batch = dataset.get_batch(sample_indices)
    optimizer.zero_grad(set_to_none=True)
    losses = student.compute_losses(
        student(batch.observations, batch.executed_controls),
        batch.teacher_logits,
        config=config,
    )
    losses.total.backward()
    torch.nn.utils.clip_grad_norm_(
        [parameter for parameter in student.parameters() if parameter.requires_grad],
        config["training"]["gradient_clip_norm"],
    )
    optimizer.step()
    student.update_target_encoder(tau=config["ema"]["tau"])
    return copy.deepcopy(student.state_dict()), losses


def _accumulation_worker(rank: int, world_size: int, rendezvous: str, output: str) -> None:
    del rank
    _init_gloo(0, world_size, rendezvous)
    try:
        from torch.nn.parallel import DistributedDataParallel

        observations = torch.tensor(
            [
                [[0.0, 1.0], [0.0, 0.0]],
                [[1.0, 2.0], [0.0, 0.0]],
                [[2.0, 3.0], [0.0, 0.0]],
                [[3.0, 4.0], [0.0, 0.0]],
                [[4.0, 5.0], [0.0, 0.0]],
            ]
        )
        targets = torch.tensor([0.25, -0.5, 1.5, 2.0, -1.0])
        dataset = TensorDataset(observations, targets)
        config = _base_config(microbatch_size=2, variance_scope="disabled")

        accumulated = _fixed_student()
        reference = _fixed_student()
        optimizer = torch.optim.AdamW(accumulated.parameters(), lr=0.05, weight_decay=0.0)
        ddp_student = DistributedDataParallel(accumulated)
        losses, sample_count, _gradient_norm = DIST._training_update(
            ddp_student,
            accumulated,
            dataset,
            list(range(len(dataset))),
            optimizer,
            config=config,
            device=torch.device("cpu"),
            world_size=world_size,
        )
        reference_state, reference_losses = _reference_update(
            reference, dataset, list(range(len(dataset))), config=config
        )
        torch.save(
            {
                "accumulated": copy.deepcopy(accumulated.state_dict()),
                "reference": reference_state,
                "losses": tuple(losses),
                "reference_losses": tuple(reference_losses),
                "sample_count": sample_count,
            },
            output,
        )
    finally:
        dist.destroy_process_group()


def _ddp_parity_worker(rank: int, world_size: int, rendezvous: str, output_dir: str) -> None:
    _init_gloo(rank, world_size, rendezvous)
    try:
        from torch.nn.parallel import DistributedDataParallel

        observations = torch.tensor(
            [
                [[0.0, 1.0], [0.0, 0.0]],
                [[1.0, 2.0], [0.0, 0.0]],
                [[2.0, 3.0], [0.0, 0.0]],
                [[3.0, 4.0], [0.0, 0.0]],
            ]
        )
        targets = torch.tensor([0.25, -0.5, 1.5, 2.0])
        dataset = TensorDataset(observations, targets)
        config = _base_config(microbatch_size=2)
        student = _fixed_student()
        optimizer = torch.optim.AdamW(student.parameters(), lr=0.05, weight_decay=0.0)
        ddp_student = DistributedDataParallel(student)
        local_indices = [0, 1] if rank == 0 else [2, 3]
        losses, sample_count, gradient_norm = DIST._training_update(
            ddp_student,
            student,
            dataset,
            local_indices,
            optimizer,
            config=config,
            device=torch.device("cpu"),
            world_size=world_size,
        )
        torch.save(
            {
                "state": copy.deepcopy(student.state_dict()),
                "losses": tuple(losses),
                "sample_count": sample_count,
                "gradient_norm": gradient_norm,
            },
            Path(output_dir) / f"rank_{rank}.pt",
        )
        dist.barrier()
    finally:
        dist.destroy_process_group()


def _sampler_worker(rank: int, world_size: int, rendezvous: str, output_dir: str) -> None:
    _init_gloo(rank, world_size, rendezvous)
    try:
        first = DIST._rank_epoch_indices(
            5, seed=41, epoch_idx=3, rank=rank, world_size=world_size
        )
        second = DIST._rank_epoch_indices(
            5, seed=41, epoch_idx=3, rank=rank, world_size=world_size
        )
        assert first == second
        torch.save(
            {"indices": first[0], "padding": first[1], "rank_count": first[2]},
            Path(output_dir) / f"rank_{rank}.pt",
        )
        dist.barrier()
    finally:
        dist.destroy_process_group()


class RecordingMonitor:
    def __init__(self, output: Path) -> None:
        self.output = output
        self.events: list[dict] = []

    def log_metrics(self, metrics, *, progress) -> None:
        self.events.append({"metrics": dict(metrics), "step": progress.optimizer_step})

    def state_dict(self) -> dict:
        return {"events": len(self.events), "rank_zero": True}


class FakeStreamingDataset:
    def __init__(self, manifest_path: Path) -> None:
        self.manifest_path = Path(manifest_path)
        self.count = 4
        # The trainer stores each rank's round under ``.../rank_xxx/train``;
        # the ownership assertion does not depend on a rank-specific sample
        # value, so keep this fixture independent of that directory layout.
        base = 0.0
        self.observations = torch.tensor(
            [
                [[base + 0.0, 1.0], [0.0, 0.0]],
                [[base + 1.0, 2.0], [0.0, 0.0]],
                [[base + 2.0, 3.0], [0.0, 0.0]],
                [[base + 3.0, 4.0], [0.0, 0.0]],
            ]
        )
        self.targets = torch.tensor([0.25, -0.5, 1.5, 2.0])

    def __len__(self) -> int:
        return self.count

    def get_batch(self, indices) -> TrainingBatch:
        return TensorDataset(self.observations, self.targets).get_batch(indices)

    def close(self) -> None:
        return None


class FakeEnv:
    def close(self) -> None:
        return None


def _ownership_worker(rank: int, world_size: int, rendezvous: str, root: str) -> None:
    _init_gloo(rank, world_size, rendezvous)
    try:
        import project.jepa_distill.runtime as runtime_module
        import project.jepa_distill.streaming as streaming_module
        import project.jepa_distill.teacher as teacher_module
        import project.jepa_distill.train as train_module

        runtime_module.prepare_runtime = lambda config: None
        teacher_module.resolve_teacher_config = lambda config: {
            "env": {"num_agents": 1, "dt": 0.3},
            "eval": {},
        }
        streaming_module.StreamingTrajectoryDataset = FakeStreamingDataset

        def fake_collect(config, output_dir, *, teacher, env, collection_round_idx):
            del config, teacher, env, collection_round_idx
            output_dir = Path(output_dir)
            output_dir.mkdir(parents=True, exist_ok=True)
            path = output_dir / "manifest.json"
            path.write_text(
                '{"collection_transition_count": 4, "transition_count": 4}',
                encoding="utf-8",
            )
            return {"manifest_path": str(path), "collection_transition_count": 4}

        streaming_module.collect_streaming_dataset = fake_collect
        train_module.validate_dataset_identity = lambda *args, **kwargs: None
        train_module.validate = lambda *args, **kwargs: {"loss_total": 0.0}
        eval_calls = Path(root) / f"eval_rank_{rank}.txt"

        def fake_driving(*args, **kwargs):
            del args, kwargs
            eval_calls.write_text("called", encoding="utf-8")
            return {"fake/collision_rate": 0.0}

        train_module.run_student_driving_evaluation = fake_driving

        config = {
            "teacher": {"checkpoint": "unused", "config": "unused"},
            "teacher_config": {},
            "env_overrides": {},
            "model": {"chunk_length": 1, "num_action_classes": 1},
            "loss": {},
            "ema": {"tau": 0.5},
            "collection": {
                "num_collections": 1,
                "action_selection": "mean",
                "split_seeds": {"train": 11, "validation": 12, "test": 13},
                "dataset_id": "ownership_dataset",
                "transitions_per_round": 4,
                "max_transitions": 4,
                "max_disk_bytes": 10_000_000,
                "shard_transition_count": 4,
                "output_root": str(Path(root) / "datasets"),
                "split": "train",
            },
            "training": {
                "seed": 23,
                "device": "cpu",
                "batch_size": 2,
                "microbatch_size": 2,
                "variance_scope": "microbatch",
                "update_epochs": 1,
                "learning_rate": 0.05,
                "weight_decay": 0.0,
                "gradient_clip_norm": 100.0,
                "optimizer": "AdamW",
                "amp": False,
                "compile": False,
                "distributed": True,
                "world_size": world_size,
                "validation_manifest": None,
                "run_id": "ownership_run",
                "max_optimizer_steps": 1,
                "validation_interval_steps": 100,
                "checkpoint_interval_steps": 1,
                "resume_checkpoint": None,
                "output_root": str(Path(root) / "runs"),
                "validation_transitions": 4,
                "cpu_threads": 1,
            },
            "wandb": {
                "enabled": False,
                "mode": "disabled",
                "log_interval_steps": 1,
                "diagnostics_interval_steps": 100,
            },
            "vec": {"backend": "Serial", "num_envs": 1, "num_workers": 1, "batch_size": 1},
            "driving_evaluation": {
                "enabled": True,
                "interval_steps": 1,
                "num_scenarios": 1,
                "episode_timesteps": 1,
                "population": "student_self_play",
                "execution_horizon": 1,
                "action_selection": "mean",
                "benchmarks": ["fake"],
                "num_agents": 1,
                "env_overrides": {},
                "vec_overrides": {},
            },
        }
        student = _fixed_student()
        monitor = RecordingMonitor(Path(root)) if rank == 0 else None
        result = DIST.train_distributed(
            config,
            validation_batches=[TensorDataset(
                torch.tensor([[[0.0, 1.0], [0.0, 0.0]], [[1.0, 2.0], [0.0, 0.0]]]),
                torch.tensor([0.25, -0.5]),
            ).get_batch([0, 1])],
            teacher=nn.Identity(),
            student=student,
            env=FakeEnv(),
            monitor=monitor,
        )
        torch.save(
            {
                "result": result,
                "events": monitor.events if monitor is not None else [],
                "eval_called": eval_calls.is_file(),
            },
            Path(root) / f"ownership_rank_{rank}.pt",
        )
        dist.barrier()
    finally:
        dist.destroy_process_group()


def _spawn(worker, world_size: int, tmp_path: Path, *args) -> None:
    rendezvous = str(tmp_path / f"rendezvous_{worker.__name__}")
    mp.spawn(
        worker,
        args=(world_size, rendezvous, *args),
        nprocs=world_size,
        join=True,
    )


def test_sample_weighted_accumulation_matches_full_batch_without_variance(tmp_path):
    output = tmp_path / "accumulation.pt"
    _spawn(_accumulation_worker, 1, tmp_path, str(output))
    payload = torch.load(output, map_location="cpu", weights_only=False)

    assert payload["sample_count"] == 5
    for name, accumulated in payload["accumulated"].items():
        torch.testing.assert_close(accumulated, payload["reference"][name], rtol=1e-5, atol=1e-6)
    for accumulated, reference in zip(payload["losses"], payload["reference_losses"]):
        torch.testing.assert_close(accumulated, reference, rtol=1e-6, atol=1e-7)


def test_two_gloo_ranks_have_equal_gradients_and_ema_state(tmp_path):
    output_dir = tmp_path / "ddp_parity"
    output_dir.mkdir()
    _spawn(_ddp_parity_worker, 2, tmp_path, str(output_dir))
    rank_zero = torch.load(output_dir / "rank_0.pt", map_location="cpu", weights_only=False)
    rank_one = torch.load(output_dir / "rank_1.pt", map_location="cpu", weights_only=False)

    for name, value in rank_zero["state"].items():
        torch.testing.assert_close(value, rank_one["state"][name], rtol=1e-6, atol=1e-7)
    assert rank_zero["sample_count"] == rank_one["sample_count"] == 4
    torch.testing.assert_close(rank_zero["losses"][0], rank_one["losses"][0])
    assert rank_zero["gradient_norm"] == pytest.approx(rank_one["gradient_norm"], rel=1e-6)

    observations = torch.tensor(
        [
            [[0.0, 1.0], [0.0, 0.0]],
            [[1.0, 2.0], [0.0, 0.0]],
            [[2.0, 3.0], [0.0, 0.0]],
            [[3.0, 4.0], [0.0, 0.0]],
        ]
    )
    targets = torch.tensor([0.25, -0.5, 1.5, 2.0])
    expected, _ = _reference_update(
        _fixed_student(),
        TensorDataset(observations, targets),
        [0, 1, 2, 3],
        config=_base_config(microbatch_size=2),
    )
    for name, value in expected.items():
        torch.testing.assert_close(value, rank_zero["state"][name], rtol=1e-5, atol=1e-6)


def test_unequal_shards_are_padded_to_equal_rank_step_counts(tmp_path):
    output_dir = tmp_path / "sampler"
    output_dir.mkdir()
    _spawn(_sampler_worker, 2, tmp_path, str(output_dir))
    rank_zero = torch.load(output_dir / "rank_0.pt", map_location="cpu", weights_only=False)
    rank_one = torch.load(output_dir / "rank_1.pt", map_location="cpu", weights_only=False)

    assert rank_zero["padding"] == rank_one["padding"] == 1
    assert rank_zero["rank_count"] == rank_one["rank_count"] == 3
    assert len(rank_zero["indices"]) == len(rank_one["indices"]) == 3
    assert set(rank_zero["indices"] + rank_one["indices"]) == set(range(5))
    assert rank_zero["indices"] != rank_one["indices"]


def test_resume_restores_torch_numpy_python_rng_and_sampler_order():
    torch.manual_seed(101)
    np.random.seed(101)
    random.seed(101)
    state = DIST._capture_rng_state()
    expected = (torch.rand(5), np.random.rand(5), random.random())

    torch.manual_seed(999)
    np.random.seed(999)
    random.seed(999)
    DIST._restore_rng_state(state)
    actual = (torch.rand(5), np.random.rand(5), random.random())
    torch.testing.assert_close(actual[0], expected[0])
    np.testing.assert_array_equal(actual[1], expected[1])
    assert actual[2] == expected[2]

    first = DIST._rank_epoch_indices(11, seed=17, epoch_idx=4, rank=1, world_size=2)
    resumed = DIST._rank_epoch_indices(11, seed=17, epoch_idx=4, rank=1, world_size=2)
    assert first == resumed


def test_rank_zero_owns_logging_and_driving_evaluation(tmp_path):
    _spawn(_ownership_worker, 2, tmp_path, str(tmp_path))
    rank_zero = torch.load(tmp_path / "ownership_rank_0.pt", map_location="cpu", weights_only=False)
    rank_one = torch.load(tmp_path / "ownership_rank_1.pt", map_location="cpu", weights_only=False)

    assert rank_zero["result"] is not None
    assert rank_one["result"] is None
    assert rank_zero["events"]
    assert rank_one["events"] == []
    assert rank_zero["eval_called"] is True
    assert rank_one["eval_called"] is False
