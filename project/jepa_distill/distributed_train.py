"""Two or more rank Condition B training.

The single GPU trainer in :mod:`project.jepa_distill.train` remains the small
reference implementation.  This module owns the extra state required by a
``torchrun`` launch: rank local collection, a shared memory mapped window
pool, gradient accumulation, and rank zero side effects.
"""

from __future__ import annotations

import copy
import json
import math
import os
import random
import subprocess
import tempfile
import time
from contextlib import nullcontext
from pathlib import Path
from typing import Any, Iterable, Mapping, Optional, Sequence

import numpy as np
import torch
import torch.distributed as dist
import torch.nn as nn

from .dataset import TrainingBatch, TrainingSample
from .model import LossTerms


_DISTRIBUTED_CHECKPOINT_VERSION = 1


def _directory_size(root: Path) -> int:
    if not root.exists():
        return 0
    return sum(path.stat().st_size for path in root.rglob("*") if path.is_file())


def _as_positive_int(value: Any, *, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 1:
        raise ValueError(f"{name} must be a positive integer")
    return int(value)


def _env_int(name: str, default: int) -> int:
    raw_value = os.environ.get(name)
    if raw_value is None:
        return default
    try:
        value = int(raw_value)
    except ValueError as exc:
        raise ValueError(f"{name} must be an integer, got {raw_value!r}") from exc
    if value < 0:
        raise ValueError(f"{name} must be non-negative, got {value}")
    return value


def _rank_context(config: Mapping[str, Any]) -> tuple[int, int, int, str, bool]:
    """Return rank metadata and initialize the torch process group once."""

    settings = config.get("training", {})
    if not isinstance(settings, Mapping):
        raise ValueError("training must be a mapping")
    configured_world_size = _as_positive_int(
        settings.get("world_size"), name="training.world_size"
    )
    requested_world_size = _env_int(
        "WORLD_SIZE", dist.get_world_size() if dist.is_initialized() else 1
    )
    if requested_world_size != configured_world_size:
        raise RuntimeError(
            "torchrun world size disagrees with training.world_size: "
            f"launcher={requested_world_size}, config={configured_world_size}"
        )
    if configured_world_size < 2:
        raise ValueError("distributed training requires training.world_size >= 2")

    rank = _env_int("RANK", dist.get_rank() if dist.is_initialized() else 0)
    local_rank = _env_int("LOCAL_RANK", rank)
    if rank >= configured_world_size:
        raise RuntimeError(f"RANK={rank} is outside world size {configured_world_size}")
    if local_rank >= configured_world_size:
        raise RuntimeError(
            f"LOCAL_RANK={local_rank} is outside world size {configured_world_size}"
        )

    owns_process_group = False
    if dist.is_initialized():
        actual_world_size = dist.get_world_size()
        actual_rank = dist.get_rank()
        if actual_world_size != configured_world_size or actual_rank != rank:
            raise RuntimeError(
                "initialized process group disagrees with torchrun metadata: "
                f"group=({actual_rank}, {actual_world_size}), "
                f"environment=({rank}, {configured_world_size})"
            )
    else:
        requested_device = str(settings.get("device", "cpu"))
        use_nccl = requested_device.startswith("cuda")
        if use_nccl:
            if not torch.cuda.is_available():
                raise RuntimeError("training.device requests CUDA but CUDA is unavailable")
            backend = "nccl"
        else:
            backend = "gloo"
        dist.init_process_group(backend=backend, init_method="env://")
        owns_process_group = True
    return rank, configured_world_size, local_rank, "nccl" if torch.cuda.is_available() else "gloo", owns_process_group


def _barrier(world_size: int) -> None:
    if world_size > 1:
        dist.barrier()


def _all_gather_object(value: Any, world_size: int) -> list[Any]:
    if world_size == 1:
        return [value]
    gathered: list[Any] = [None for _ in range(world_size)]
    dist.all_gather_object(gathered, value)
    return gathered


def _broadcast_object(value: Any, *, source_rank: int, rank: int, world_size: int) -> Any:
    if world_size == 1:
        return value
    values = [value if rank == source_rank else None]
    dist.broadcast_object_list(values, src=source_rank)
    return values[0]


def _collective_invalid(local_invalid: bool, *, device: torch.device, world_size: int) -> bool:
    """Make a finite value failure visible to every rank before raising."""

    if world_size == 1:
        return bool(local_invalid)
    flag = torch.tensor(int(local_invalid), dtype=torch.int32, device=device)
    dist.all_reduce(flag, op=dist.ReduceOp.MAX)
    return bool(flag.item())


def _collective_lifecycle_action(action: Any, *, description: str, world_size: int) -> Any:
    """Run one rank-local filesystem check/action and fail every rank together."""
    result = None
    local_error = None
    try:
        result = action()
    except Exception as exc:
        local_error = f"{type(exc).__name__}: {exc}"
    errors = _all_gather_object(local_error, world_size)
    failures = [f"rank {rank}: {error}" for rank, error in enumerate(errors) if error]
    if failures:
        raise RuntimeError(
            f"distributed collection lifecycle {description} failed: " + "; ".join(failures)
        )
    _barrier(world_size)
    return result


def _all_reduce_count(local_count: int, *, device: torch.device, world_size: int) -> int:
    if isinstance(local_count, bool) or local_count < 0:
        raise ValueError("sample count must be a non-negative integer")
    if world_size == 1:
        return int(local_count)
    count = torch.tensor(local_count, dtype=torch.int64, device=device)
    dist.all_reduce(count, op=dist.ReduceOp.SUM)
    return int(count.item())


def _capture_rng_state() -> dict[str, Any]:
    numpy_state = np.random.get_state()
    return {
        "torch": torch.get_rng_state().clone(),
        "python": random.getstate(),
        "numpy": (numpy_state[0], numpy_state[1].copy(), *numpy_state[2:]),
        "cuda": [state.clone() for state in torch.cuda.get_rng_state_all()]
        if torch.cuda.is_available()
        else None,
    }


def _restore_rng_state(state: Mapping[str, Any]) -> None:
    required = ("torch", "python", "numpy", "cuda")
    if any(name not in state for name in required):
        raise ValueError("distributed checkpoint is missing a rank RNG stream")
    torch.set_rng_state(state["torch"].cpu())
    random.setstate(state["python"])
    np.random.set_state(state["numpy"])
    if state["cuda"] is not None:
        if not torch.cuda.is_available():
            raise RuntimeError("checkpoint contains CUDA RNG state but CUDA is unavailable")
        torch.cuda.set_rng_state_all([item.cpu() for item in state["cuda"]])


def _validate_resume_distributed_config(
    current: Mapping[str, Any], saved: Mapping[str, Any], *, world_size: int
) -> None:
    from .train import validate_resume_config

    validate_resume_config(current, saved)
    current_training = current.get("training", {})
    saved_training = saved.get("training", {})
    for name in ("distributed", "world_size", "microbatch_size", "variance_scope"):
        if current_training.get(name) != saved_training.get(name):
            raise ValueError(
                f"Cannot change training.{name} when resuming distributed training"
            )
    if current_training.get("world_size") != world_size:
        raise RuntimeError(
            "distributed checkpoint world size does not match this launch: "
            f"config={current_training.get('world_size')}, launch={world_size}"
        )


def _validate_config(config: Mapping[str, Any], world_size: int) -> None:
    settings = config.get("training")
    collection = config.get("collection")
    if not isinstance(settings, Mapping) or not isinstance(collection, Mapping):
        raise ValueError("distributed training requires training and collection mappings")
    if settings.get("distributed") is not True:
        raise RuntimeError(
            "multi-rank launch requires training.distributed=true; "
            "the legacy single-GPU trainer is disabled for WORLD_SIZE>1"
        )
    if settings.get("world_size") != world_size:
        raise RuntimeError(
            "training.world_size must match the initialized process group: "
            f"config={settings.get('world_size')}, group={world_size}"
        )
    _as_positive_int(settings.get("batch_size"), name="training.batch_size")
    if settings["batch_size"] < 2:
        raise ValueError("training.batch_size must be at least two")
    microbatch_size = settings.get("microbatch_size")
    _as_positive_int(microbatch_size, name="training.microbatch_size")
    if microbatch_size < 2:
        raise ValueError("training.microbatch_size must be at least two for the variance penalty")
    if microbatch_size > settings["batch_size"]:
        raise ValueError("training.microbatch_size cannot exceed training.batch_size")
    if settings.get("variance_scope", "microbatch") != "microbatch":
        raise ValueError("distributed training requires training.variance_scope='microbatch'")
    for name in (
        "update_epochs",
        "max_optimizer_steps",
        "validation_interval_steps",
        "checkpoint_interval_steps",
    ):
        _as_positive_int(settings.get(name), name=f"training.{name}")
    for name in ("num_collections", "transitions_per_round", "max_transitions", "max_disk_bytes"):
        _as_positive_int(collection.get(name), name=f"collection.{name}")
    if collection["num_collections"] * collection["transitions_per_round"] > collection["max_transitions"]:
        raise ValueError("Requested collections exceed collection.max_transitions")
    if settings.get("amp") or settings.get("compile"):
        raise ValueError("distributed Condition B training does not support AMP or compile")
    if settings.get("optimizer", "AdamW") != "AdamW":
        raise ValueError("Only the configured AdamW optimizer is supported")
    if not isinstance(settings.get("seed"), int) or isinstance(settings.get("seed"), bool):
        raise ValueError("training.seed must be an integer")
    split_seeds = collection.get("split_seeds")
    if not isinstance(split_seeds, Mapping):
        raise ValueError("collection.split_seeds must be a mapping")
    if len({split_seeds.get(name) for name in ("train", "validation", "test")}) != 3:
        raise ValueError("Training, validation, and test must use distinct collection seeds")


def _pooled_manifest_root(config: Mapping[str, Any], run_id: str, rank: int) -> Path:
    from .runtime import resolve_path

    collection = config["collection"]
    dataset_id = collection.get("dataset_id") or run_id
    if not isinstance(dataset_id, str) or not dataset_id or Path(dataset_id).name != dataset_id:
        raise ValueError("collection.dataset_id must be a directory name or null")
    return resolve_path(collection["output_root"]) / dataset_id / f"rank_{rank:03d}"


def _manifest_path(manifest: Mapping[str, Any]) -> Path:
    path = manifest.get("manifest_path")
    if not isinstance(path, (str, Path)) or not str(path):
        raise ValueError("streaming collection did not return manifest_path")
    result = Path(path).expanduser().resolve()
    if not result.is_file():
        raise FileNotFoundError(f"streaming collection manifest does not exist: {result}")
    return result


def _read_manifest(path: Path) -> Mapping[str, Any]:
    with path.open("r", encoding="utf-8") as manifest_file:
        manifest = json.load(manifest_file)
    if not isinstance(manifest, Mapping):
        raise ValueError(f"streaming manifest must be a mapping: {path}")
    return manifest


def _manifest_transition_count(manifest: Mapping[str, Any]) -> int:
    value = manifest.get("collection_transition_count", manifest.get("transition_count"))
    return _as_positive_int(value, name="manifest.collection_transition_count")


def _rank_collection_config(config: Mapping[str, Any], *, rank: int, world_size: int) -> dict[str, Any]:
    local_config = copy.deepcopy(dict(config))
    collection = local_config["collection"]
    split_seeds = dict(collection["split_seeds"])
    for split_name, seed in split_seeds.items():
        if isinstance(seed, bool) or not isinstance(seed, int):
            raise ValueError(f"collection.split_seeds.{split_name} must be an integer")
        split_seeds[split_name] = int(seed) * world_size + rank
    collection["split_seeds"] = split_seeds
    return local_config


class _PooledStreamingDataset:
    """Read a global window index from rank-local streaming manifests."""

    def __init__(self, manifest_paths: Sequence[Path]) -> None:
        if not manifest_paths:
            raise ValueError("at least one training manifest is required")
        from .streaming import StreamingTrajectoryDataset

        self.manifest_paths = tuple(Path(path).resolve() for path in manifest_paths)
        self.datasets = tuple(
            StreamingTrajectoryDataset(path) for path in self.manifest_paths
        )
        self.offsets: list[int] = [0]
        for dataset in self.datasets:
            count = len(dataset)
            if isinstance(count, bool) or not isinstance(count, int) or count < 1:
                raise ValueError("every streaming training manifest must contain valid windows")
            self.offsets.append(self.offsets[-1] + count)

    def __len__(self) -> int:
        return self.offsets[-1]

    def _locate(self, global_index: int) -> tuple[int, int]:
        if isinstance(global_index, bool) or not isinstance(global_index, int):
            raise TypeError("pooled dataset index must be an integer")
        if global_index < 0:
            global_index += len(self)
        if global_index < 0 or global_index >= len(self):
            raise IndexError(f"pooled dataset index {global_index} is out of range")
        for dataset_idx in range(len(self.datasets)):
            if global_index < self.offsets[dataset_idx + 1]:
                return dataset_idx, global_index - self.offsets[dataset_idx]
        raise RuntimeError("pooled dataset offset lookup failed")

    def __getitem__(self, global_index: int) -> TrainingSample:
        dataset_idx, local_index = self._locate(global_index)
        return self.datasets[dataset_idx][local_index]

    def get_batch(self, global_indices: Sequence[int]) -> TrainingBatch:
        if not global_indices:
            raise ValueError("streaming batch cannot be empty")
        grouped: dict[int, list[tuple[int, int]]] = {}
        for position, global_index in enumerate(global_indices):
            dataset_idx, local_index = self._locate(int(global_index))
            grouped.setdefault(dataset_idx, []).append((position, local_index))

        output_tensors: Optional[list[torch.Tensor]] = None
        for dataset_idx, entries in grouped.items():
            local_indices = [local_index for _position, local_index in entries]
            dataset = self.datasets[dataset_idx]
            get_batch = getattr(dataset, "get_batch", None)
            if get_batch is None:
                raise TypeError("StreamingTrajectoryDataset must provide get_batch(indices)")
            batch = get_batch(local_indices)
            if not isinstance(batch, TrainingBatch):
                try:
                    batch = TrainingBatch(*batch)
                except (TypeError, ValueError) as exc:
                    raise TypeError("StreamingTrajectoryDataset.get_batch must return TrainingBatch") from exc
            if output_tensors is None:
                output_tensors = [
                    torch.empty(
                        (len(global_indices), *tensor.shape[1:]),
                        dtype=tensor.dtype,
                        device=tensor.device,
                    )
                    for tensor in batch
                ]
            positions = torch.tensor(
                [position for position, _local_index in entries],
                dtype=torch.long,
                device=output_tensors[0].device,
            )
            for field, tensor in enumerate(batch):
                if tensor.shape[0] != len(entries):
                    raise ValueError("StreamingTrajectoryDataset.get_batch returned the wrong batch size")
                output_tensors[field].index_copy_(0, positions, tensor)
        if output_tensors is None:
            raise RuntimeError("streaming batch assembly produced no output")
        return TrainingBatch(*output_tensors)

    def close(self) -> None:
        for dataset in self.datasets:
            close = getattr(dataset, "close", None)
            if close is not None:
                close()
                continue
            for name in (
                "observations",
                "controls",
                "teacher_logits",
                "transition_valid",
                "eligibility_mask",
                "terminated",
                "truncated",
                "endpoint_valid",
                "generations",
                "window_indices",
            ):
                array = getattr(dataset, name, None)
                mmap = getattr(array, "_mmap", None)
                if mmap is not None:
                    mmap.close()


def _rank_epoch_indices(
    window_count: int, *, seed: int, epoch_idx: int, rank: int, world_size: int
) -> tuple[list[int], int, int]:
    if window_count < 1:
        raise ValueError("training dataset contains no valid windows")
    generator = torch.Generator(device="cpu").manual_seed(int(seed) + int(epoch_idx))
    permutation = torch.randperm(window_count, generator=generator).tolist()
    padding_count = (-window_count) % world_size
    if padding_count:
        permutation.extend(permutation[:padding_count])
    rank_indices = permutation[rank::world_size]
    if len(rank_indices) != len(permutation) // world_size:
        raise RuntimeError("rank-strided permutation produced unbalanced sample counts")
    return rank_indices, padding_count, len(permutation) // world_size


def _effective_batches(indices: Sequence[int], batch_size: int) -> list[list[int]]:
    batches = [
        list(indices[offset : offset + batch_size])
        for offset in range(0, len(indices), batch_size)
    ]
    # Keep every sample while avoiding a one-sample effective tail.  The
    # boundary is moved; singleton microbatches are never concatenated with a
    # neighboring microbatch because their local variance semantics differ.
    if len(batches) > 1 and len(batches[-1]) == 1 and len(batches[-2]) > 1:
        batches[-1].insert(0, batches[-2].pop())
    return batches


def _microbatch_indices(indices: Sequence[int], microbatch_size: int) -> list[list[int]]:
    microbatches = [
        list(indices[offset : offset + microbatch_size])
        for offset in range(0, len(indices), microbatch_size)
    ]
    # A remainder of one cannot support Condition B's variance penalty.  Move
    # one sample across the boundary, preserving both samples and the explicit
    # microbatch scope without merging the singleton into its predecessor.
    if len(microbatches) > 1 and len(microbatches[-1]) == 1 and len(microbatches[-2]) > 2:
        microbatches[-1].insert(0, microbatches[-2].pop())
    return microbatches


def _stream_batches(dataset: Any, count: int, microbatch_size: int) -> Iterable[TrainingBatch]:
    for offset in range(0, count, microbatch_size):
        indices = list(range(offset, min(offset + microbatch_size, count)))
        batch = dataset.get_batch(indices)
        if not isinstance(batch, TrainingBatch):
            batch = TrainingBatch(*batch)
        yield batch


def _training_update(
    ddp_student: nn.Module,
    student: nn.Module,
    dataset: _PooledStreamingDataset,
    indices: Sequence[int],
    optimizer: torch.optim.Optimizer,
    *,
    config: Mapping[str, Any],
    device: torch.device,
    world_size: int,
) -> tuple[LossTerms, int, float]:
    """Accumulate one per-rank effective batch without materializing it."""

    settings = config["training"]
    microbatch_size = int(settings["microbatch_size"])
    local_sample_count = len(indices)
    if local_sample_count < 1:
        raise ValueError("effective batch cannot be empty")
    microbatches = _microbatch_indices(indices, microbatch_size)
    if settings.get("variance_scope", "microbatch") == "microbatch" and any(
        len(microbatch) < 2 for microbatch in microbatches
    ) and hasattr(student, "variance_loss"):
        invalid = _collective_invalid(True, device=device, world_size=world_size)
        if invalid:
            raise ValueError(
                "a singleton microbatch cannot support the configured variance penalty; "
                "increase microbatch_size or choose a larger effective batch"
            )
    global_sample_count = _all_reduce_count(
        local_sample_count, device=device, world_size=world_size
    )
    if global_sample_count < 1:
        raise RuntimeError("global effective batch contains no samples")

    optimizer.zero_grad(set_to_none=True)
    student.train()
    weighted_totals = [0.0, 0.0, 0.0, 0.0]
    for microbatch_idx, microbatch_indices in enumerate(microbatches):
        batch = dataset.get_batch(microbatch_indices)
        batch = TrainingBatch(*(tensor.to(device) for tensor in batch))
        is_last_microbatch = microbatch_idx == len(microbatches) - 1
        if is_last_microbatch or not hasattr(ddp_student, "no_sync"):
            synchronization = nullcontext()
        else:
            synchronization = ddp_student.no_sync()
        with synchronization:
            outputs = ddp_student(batch.observations, batch.executed_controls)
            # The DDP wrapper owns only forward/backward synchronization.  Loss
            # construction stays on the underlying Condition B model so target
            # encoder and local variance semantics remain explicit.
            losses = student.compute_losses(outputs, batch.teacher_logits, config=config)
            finite_loss = all(bool(torch.isfinite(value).all().item()) for value in losses)
            if _collective_invalid(not finite_loss, device=device, world_size=world_size):
                optimizer.zero_grad(set_to_none=True)
                raise FloatingPointError("Non-finite loss; optimizer and EMA were not updated")
            sample_weight = len(microbatch_indices) * world_size / global_sample_count
            (losses.total * sample_weight).backward()
        for loss_idx, value in enumerate(losses):
            weighted_totals[loss_idx] += float(value.detach().cpu()) * len(microbatch_indices)

    trainable_parameters = [
        parameter for parameter in student.parameters() if parameter.requires_grad
    ]
    gradient_norm_tensor = torch.nn.utils.clip_grad_norm_(
        trainable_parameters,
        settings.get("gradient_clip_norm", 1.0),
        error_if_nonfinite=False,
    )
    nonfinite_gradient = not bool(torch.isfinite(gradient_norm_tensor).all().item())
    if not nonfinite_gradient:
        nonfinite_gradient = any(
            parameter.grad is not None and not bool(torch.isfinite(parameter.grad).all().item())
            for parameter in trainable_parameters
        )
    if _collective_invalid(nonfinite_gradient, device=device, world_size=world_size):
        optimizer.zero_grad(set_to_none=True)
        raise FloatingPointError("Non-finite gradient; optimizer and EMA were not updated")
    optimizer.step()
    student.update_target_encoder(tau=config.get("ema", {}).get("tau", 0.99))
    student.last_gradient_norm = float(gradient_norm_tensor.detach().cpu())

    if world_size > 1:
        totals_tensor = torch.tensor(weighted_totals, dtype=torch.float64, device=device)
        dist.all_reduce(totals_tensor, op=dist.ReduceOp.SUM)
        weighted_totals = totals_tensor.cpu().tolist()
        global_sample_count = _all_reduce_count(
            local_sample_count, device=device, world_size=world_size
        )
    return (
        LossTerms(
            *(torch.tensor(total / global_sample_count, dtype=torch.float32) for total in weighted_totals)
        ),
        global_sample_count,
        float(gradient_norm_tensor.detach().cpu()),
    )


def _validation_batches(validation_dataset: Any, microbatch_size: int) -> Iterable[TrainingBatch]:
    count = len(validation_dataset)
    if count < 1:
        raise ValueError("validation dataset contains no valid windows")
    for offset in range(0, count, microbatch_size):
        indices = list(range(offset, min(offset + microbatch_size, count)))
        batch = validation_dataset.get_batch(indices)
        if not isinstance(batch, TrainingBatch):
            batch = TrainingBatch(*batch)
        yield batch


def _atomic_attach_distributed_state(path: Path, distributed_state: Mapping[str, Any]) -> None:
    payload = torch.load(path, map_location="cpu", weights_only=False)
    if not isinstance(payload, dict) or payload.get("format") != "condition_b_v1":
        raise ValueError("save_checkpoint did not produce a Condition B checkpoint")
    payload["distributed_state"] = dict(distributed_state)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{path.name}.", suffix=".tmp", dir=path.parent
    )
    os.close(descriptor)
    try:
        torch.save(payload, temporary_name)
        os.replace(temporary_name, path)
    finally:
        if os.path.exists(temporary_name):
            os.unlink(temporary_name)


def _save_distributed_checkpoint(
    path: Path,
    student: nn.Module,
    optimizer: torch.optim.Optimizer,
    *,
    step: int,
    config: Mapping[str, Any],
    sampler_state: Mapping[str, Any],
    collection_state: Mapping[str, Any],
    monitoring_state: Mapping[str, Any],
    rank: int,
    world_size: int,
) -> None:
    from .train import save_checkpoint
    from .collection_lifecycle import durable_replace_checkpoint

    local_state = {
        "rank": rank,
        "rng_state": _capture_rng_state(),
        "sampler_state": copy.deepcopy(dict(sampler_state)),
    }
    rank_states = _all_gather_object(local_state, world_size)
    if rank != 0:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{path.name}.base.", suffix=".tmp", dir=path.parent
    )
    os.close(descriptor)
    temporary_path = Path(temporary_name)
    try:
        save_checkpoint(
            temporary_path,
            student,
            optimizer=optimizer,
            step=step,
            config=config,
            sampler_state=sampler_state,
            collection_state=collection_state,
            monitoring_state=monitoring_state,
        )
        _atomic_attach_distributed_state(
            temporary_path,
            {
                "version": _DISTRIBUTED_CHECKPOINT_VERSION,
                "world_size": world_size,
                "rank_states": rank_states,
            },
        )
        durable_replace_checkpoint(temporary_path, path)
    finally:
        if temporary_path.exists():
            temporary_path.unlink()


def _resume_rank_state(
    checkpoint_path: Path, *, rank: int, world_size: int, config: Mapping[str, Any]
) -> tuple[Mapping[str, Any], Mapping[str, Any]]:
    payload = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    if not isinstance(payload, Mapping) or payload.get("format") != "condition_b_v1":
        raise ValueError("Expected a Condition B checkpoint, not teacher PPO state")
    distributed_state = payload.get("distributed_state")
    if not isinstance(distributed_state, Mapping):
        raise ValueError("distributed resume requires a checkpoint written by the distributed trainer")
    if distributed_state.get("world_size") != world_size:
        raise RuntimeError(
            "checkpoint world size does not match this launch: "
            f"checkpoint={distributed_state.get('world_size')}, launch={world_size}"
        )
    rank_states = distributed_state.get("rank_states")
    if not isinstance(rank_states, list) or len(rank_states) != world_size:
        raise ValueError("distributed checkpoint has incomplete per-rank state")
    rank_state = rank_states[rank]
    if not isinstance(rank_state, Mapping) or rank_state.get("rank") != rank:
        raise ValueError(f"distributed checkpoint is missing rank {rank} state")
    saved_config = payload.get("config")
    if not isinstance(saved_config, Mapping):
        raise ValueError("distributed checkpoint is missing its scientific config")
    _validate_resume_distributed_config(config, saved_config, world_size=world_size)
    return payload, rank_state


def _streaming_manifest_paths(
    local_path: Path, *, rank: int, world_size: int
) -> list[Path]:
    gathered = _all_gather_object(str(local_path.resolve()), world_size)
    paths: list[Path] = []
    for gathered_path in gathered:
        if not isinstance(gathered_path, str) or not gathered_path:
            raise ValueError("all ranks must return a non-empty streaming manifest path")
        path = Path(gathered_path).expanduser().resolve()
        if not path.is_file():
            raise FileNotFoundError(f"rank {rank} cannot read gathered manifest: {path}")
        paths.append(path)
    return paths


def _validation_setup(
    config: Mapping[str, Any],
    *,
    teacher: nn.Module,
    teacher_config: Mapping[str, Any],
    student: nn.Module,
    validation_batches: Optional[Iterable[TrainingBatch]],
    state: dict[str, Any],
    run_dataset_root: Path,
    rank: int,
) -> tuple[Optional[Any], Optional[Iterable[TrainingBatch]], Optional[Path]]:
    """Create rank zero's bounded held-out stream, if one was not supplied."""

    if rank != 0:
        return None, None, None
    from .runtime import create_vecenv, resolve_path
    from .streaming import StreamingTrajectoryDataset, collect_streaming_dataset
    from .train import validate_dataset_identity

    settings = config["training"]
    configured_path = state.get("validation_manifest") or settings.get("validation_manifest")
    if validation_batches is not None:
        return None, validation_batches, Path(configured_path).resolve() if configured_path else None
    if configured_path is not None:
        manifest_path = resolve_path(configured_path)
        if not manifest_path.is_file():
            raise FileNotFoundError(f"validation manifest does not exist: {manifest_path}")
    else:
        validation_config = copy.deepcopy(dict(config))
        validation_options = validation_config.get("validation", {})
        if not isinstance(validation_options, Mapping):
            raise ValueError("validation must be a mapping")
        env_overrides = validation_options.get("env_overrides", {})
        if env_overrides is not None and not isinstance(env_overrides, Mapping):
            raise ValueError("validation.env_overrides must be a mapping")
        merged_env_overrides = dict(validation_config.get("env_overrides", {}))
        merged_env_overrides.update(dict(env_overrides or {}))
        validation_config["env_overrides"] = merged_env_overrides
        vector_config = validation_options.get("vec", validation_config.get("vec"))
        if not isinstance(vector_config, Mapping):
            raise ValueError("validation.vec must be a mapping")
        validation_config["vec"] = copy.deepcopy(dict(vector_config))
        transition_count = validation_options.get(
            "transitions", settings.get("validation_transitions")
        )
        validation_config["collection"]["split"] = "validation"
        validation_config["collection"]["transitions_per_round"] = _as_positive_int(
            transition_count, name="validation.transitions"
        )
        validation_config["collection"]["max_transitions"] = validation_config["collection"][
            "transitions_per_round"
        ]
        validation_config["collection"]["output_root"] = str(run_dataset_root)
        validation_root = run_dataset_root / "validation"
        runtime_teacher_config = copy.deepcopy(dict(teacher_config))
        runtime_teacher_env = dict(runtime_teacher_config.get("env", {}))
        runtime_teacher_env.update(merged_env_overrides)
        runtime_teacher_config["env"] = runtime_teacher_env
        validation_env = create_vecenv(runtime_teacher_config, validation_config, "validation")
        try:
            manifest = collect_streaming_dataset(
                validation_config,
                validation_root,
                teacher=teacher,
                env=validation_env,
                collection_round_idx=0,
            )
        finally:
            validation_env.close()
        manifest_path = _manifest_path(manifest)
        state["validation_manifest"] = str(manifest_path)

    manifest = _read_manifest(manifest_path)
    validate_dataset_identity(manifest, teacher, config, "validation")
    validation_dataset = StreamingTrajectoryDataset(manifest_path)
    validation_options = config.get("validation", {})
    if not isinstance(validation_options, Mapping):
        validation_options = {}
    validation_microbatch = validation_options.get(
        "microbatch_size", min(int(settings["microbatch_size"]), 1024)
    )
    validation_microbatch = _as_positive_int(
        validation_microbatch, name="validation.microbatch_size"
    )
    return (
        validation_dataset,
        _validation_batches(validation_dataset, validation_microbatch),
        manifest_path,
    )


def train_distributed(
    config: Mapping[str, Any],
    validation_batches: Optional[Iterable[TrainingBatch]] = None,
    *,
    teacher: Optional[nn.Module] = None,
    student: Optional[nn.Module] = None,
    env: Any = None,
    monitor: Any = None,
) -> Optional[Mapping[str, Any]]:
    """Run the synchronized multi-GPU Condition B trainer."""

    from .model import ConditionBModel
    from .collection_lifecycle import (
        remove_completed_collection,
        validate_active_collection,
        validate_round_manifest_path,
    )
    from .monitoring import MetricProgress, WandbMonitor
    from .runtime import create_vecenv, prepare_runtime, resolve_path
    from .teacher import load_teacher, resolve_teacher_config
    from .train import (
        representation_metrics,
        run_student_driving_evaluation,
        validate,
        validate_dataset_identity,
        validate_driving_evaluation_config,
    )
    from .streaming import collect_streaming_dataset
    from torch.nn.parallel import DistributedDataParallel

    config = copy.deepcopy(dict(config))
    rank, world_size, local_rank, _backend, owns_process_group = _rank_context(config)
    _validate_config(config, world_size)
    settings = config["training"]
    requested_device = str(settings.get("device", "cpu"))
    if requested_device.startswith("cuda"):
        rank_device = f"cuda:{local_rank}"
    else:
        rank_device = requested_device
    config["training"]["device"] = rank_device
    prepare_runtime(config)

    exit_code = 1
    owned_env = env is None
    owned_monitor = monitor is None and rank == 0
    run_dir: Optional[Path] = None
    pooled_dataset: Optional[_PooledStreamingDataset] = None
    validation_dataset: Optional[Any] = None
    try:
        run_id = settings.get("run_id")
        if rank == 0 and run_id is None:
            run_id = "condition_b_" + time.strftime("%Y%m%d_%H%M%S", time.gmtime())
        run_id = _broadcast_object(run_id, source_rank=0, rank=rank, world_size=world_size)
        if not isinstance(run_id, str) or not run_id or Path(run_id).name != run_id:
            raise ValueError("training.run_id must be a nonempty directory name")
        config["training"]["run_id"] = run_id
        run_dir = resolve_path(settings["output_root"]) / run_id
        resume_path = settings.get("resume_checkpoint")
        if resume_path is not None:
            resume_path = resolve_path(resume_path)
            if not resume_path.is_file():
                raise FileNotFoundError(f"resume checkpoint does not exist: {resume_path}")
        existing_run = run_dir.exists() and any(run_dir.iterdir())
        invalid_run = existing_run and resume_path is None
        if _collective_invalid(invalid_run, device=torch.device(rank_device), world_size=world_size):
            raise FileExistsError(f"Run directory already contains data: {run_dir}")
        if rank == 0:
            run_dir.mkdir(parents=True, exist_ok=True)
            (run_dir / "config.json").write_text(json.dumps(config, indent=2), encoding="utf-8")
        _barrier(world_size)

        config["teacher_config"] = resolve_teacher_config(config)
        teacher = teacher if teacher is not None else load_teacher(
            config, resolved_config=config["teacher_config"], device=rank_device
        )
        config["teacher_checkpoint_sha256"] = getattr(
            teacher, "condition_b_checkpoint_sha256", None
        )
        # The collector uses a rank specific seed.  Model initialization is
        # deliberately reseeded with one common value immediately beforehand.
        torch.manual_seed(int(settings["seed"]))
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(int(settings["seed"]))
        student = student if student is not None else ConditionBModel(config, teacher=teacher)
        student.to(rank_device)
        metadata = student.export_metadata()
        if any(other != metadata for other in _all_gather_object(metadata, world_size)):
            raise ValueError("student model metadata differs across distributed ranks")
        optimizer = torch.optim.AdamW(
            (parameter for parameter in student.parameters() if parameter.requires_grad),
            lr=settings["learning_rate"],
            weight_decay=settings["weight_decay"],
        )
        restored: Mapping[str, Any] = {}
        rank_resume_state: Mapping[str, Any] = {}
        if resume_path is not None:
            restored, rank_resume_state = _resume_rank_state(
                resume_path, rank=rank, world_size=world_size, config=config
            )
        ddp_student = DistributedDataParallel(
            student,
            device_ids=[local_rank] if torch.device(rank_device).type == "cuda" else None,
            broadcast_buffers=True,
        )
        if resume_path is not None:
            from .train import load_checkpoint

            load_checkpoint(resume_path, student, optimizer=optimizer, map_location=rank_device)
            _restore_rng_state(rank_resume_state["rng_state"])
        step = int(restored.get("step", 0))
        state = dict(restored.get("collection_state", {}))
        sampler = dict(rank_resume_state.get("sampler_state", restored.get("sampler_state", {})))
        transitions = int(state.get("simulator_transitions", 0))
        first_round = int(sampler.get("collection_round_idx", 0))
        if first_round < 0 or first_round > int(config["collection"]["num_collections"]):
            raise ValueError("checkpoint collection round cursor is out of range")
        dataset_root = _pooled_manifest_root(config, run_id, rank)
        training_output_root = dataset_root / "train"

        def prepare_training_collection_root() -> None:
            pending_round = sampler.get("pending_cleanup_round_idx")
            if pending_round is not None:
                if (
                    isinstance(pending_round, bool)
                    or not isinstance(pending_round, int)
                    or pending_round < 0
                    or pending_round >= first_round
                ):
                    raise ValueError(
                        "checkpoint pending cleanup round must be before its next-round cursor"
                    )
                remove_completed_collection(training_output_root, pending_round)
                sampler.pop("pending_cleanup_round_idx")
            active_round = first_round if sampler.get("manifest_paths") or sampler.get("manifest_path") else None
            validate_active_collection(training_output_root, active_round)

        _collective_lifecycle_action(
            prepare_training_collection_root,
            description="resume cleanup and active-round validation",
            world_size=world_size,
        )

        validation_dataset, validation_iterator, validation_path = _validation_setup(
            config,
            teacher=teacher,
            teacher_config=config["teacher_config"],
            student=student,
            validation_batches=validation_batches,
            state=state,
            run_dataset_root=_pooled_manifest_root(config, run_id, 0),
            rank=rank,
        )
        if rank == 0 and validation_path is not None:
            state["validation_manifest"] = str(validation_path)
        monitor = (
            monitor
            if rank == 0 and monitor is not None
            else (
                WandbMonitor(
                    config["wandb"],
                    config,
                    run_dir,
                    checkpoint_state=restored.get("monitoring_state") or None,
                )
                if rank == 0
                else None
            )
        )

        driving_config = validate_driving_evaluation_config(config)
        driving_enabled = bool(driving_config.get("enabled", False))
        driving_record = state.get("driving_evaluation")
        if not isinstance(driving_record, Mapping):
            driving_record = {}
        last_driving_step = driving_record.get("step")
        last_driving_results = copy.deepcopy(driving_record.get("results", {}))
        interval_totals: dict[str, float] = {}
        interval_windows = 0
        interval_started = time.monotonic()
        rank_config = _rank_collection_config(config, rank=rank, world_size=world_size)
        if (
            env is None
            and step < int(settings["max_optimizer_steps"])
            and first_round < int(config["collection"]["num_collections"])
        ):
            env = create_vecenv(config["teacher_config"], rank_config, "train")

        def run_validation(progress: MetricProgress) -> Mapping[str, float]:
            if rank != 0:
                return {}
            if validation_batches is not None:
                batches = validation_batches
            else:
                if validation_dataset is None:
                    raise RuntimeError("rank zero has no validation dataset")
                batches = _validation_batches(
                    validation_dataset,
                    int(config.get("validation", {}).get("microbatch_size", min(int(settings["microbatch_size"]), 1024))),
                )
            metrics = validate(student, batches, config=config)
            monitor.log_metrics(
                {f"validation/{key}": value for key, value in metrics.items()},
                progress=progress,
            )
            return dict(metrics)

        def run_driving(progress: MetricProgress, current_step: int) -> None:
            nonlocal last_driving_step, last_driving_results
            if rank != 0:
                return
            results = run_student_driving_evaluation(
                config,
                student=student,
                resolved_teacher_config=config["teacher_config"],
                monitor=monitor,
                progress=progress,
                step=current_step,
                run_dir=run_dir,
            )
            last_driving_step = current_step
            last_driving_results = copy.deepcopy(dict(results))
            state["driving_evaluation"] = {
                "step": current_step,
                "results": copy.deepcopy(dict(results)),
            }

        def checkpoint(path: Path) -> None:
            monitoring_state = monitor.state_dict() if rank == 0 else {}
            _save_distributed_checkpoint(
                path,
                student,
                optimizer,
                step=step,
                config=config,
                sampler_state=sampler,
                collection_state=state,
                monitoring_state=monitoring_state,
                rank=rank,
                world_size=world_size,
            )
            _barrier(world_size)

        last_validation: Mapping[str, float] = {}
        for round_idx in range(first_round, int(config["collection"]["num_collections"])):
            if step >= int(settings["max_optimizer_steps"]):
                break
            reuse_collection = round_idx == first_round and bool(
                sampler.get("manifest_paths") or sampler.get("manifest_path")
            )
            _collective_lifecycle_action(
                lambda: validate_active_collection(
                    training_output_root, round_idx if reuse_collection else None
                ),
                description=f"round {round_idx} pre-collection validation",
                world_size=world_size,
            )
            if reuse_collection:
                def load_resumed_collection() -> tuple[list[Path], Mapping[str, Any]]:
                    stored_paths = sampler.get("manifest_paths")
                    if not isinstance(stored_paths, list) or len(stored_paths) != world_size:
                        stored_path = sampler.get("manifest_path")
                        if not isinstance(stored_path, str):
                            raise ValueError("resume sampler state has no rank manifest paths")
                        stored_paths = [stored_path]
                    if len(stored_paths) != world_size:
                        raise ValueError(
                            "resume sampler state must contain one manifest per rank"
                        )
                    paths = [
                        validate_round_manifest_path(
                            _pooled_manifest_root(config, run_id, manifest_idx) / "train",
                            round_idx,
                            path,
                        )
                        for manifest_idx, path in enumerate(stored_paths)
                    ]
                    return paths, _read_manifest(paths[rank])

                manifest_paths, manifest = _collective_lifecycle_action(
                    load_resumed_collection,
                    description=f"round {round_idx} resume manifest validation",
                    world_size=world_size,
                )
                local_manifest_path = manifest_paths[rank]
            else:
                if env is None:
                    env = create_vecenv(config["teacher_config"], rank_config, "train")
                started = time.monotonic()
                training_config = copy.deepcopy(rank_config)
                training_config["collection"]["split"] = "train"
                training_config["collection"]["output_root"] = str(training_output_root)
                remaining_rank_transitions = (
                    int(config["collection"]["max_transitions"])
                    - round_idx * int(config["collection"]["transitions_per_round"])
                )
                if remaining_rank_transitions <= 0:
                    raise RuntimeError(
                        "collection.max_transitions has no remaining rank-local transitions"
                    )
                training_config["collection"]["max_transitions"] = remaining_rank_transitions
                if rank == 0:
                    validation_bytes = (
                        _directory_size(Path(state['validation_manifest']).parent)
                        if state.get('validation_manifest') else 0
                    )
                    configured_disk_budget = int(training_config["collection"]["max_disk_bytes"])
                    remaining_disk_budget = configured_disk_budget - validation_bytes
                    if remaining_disk_budget <= 0:
                        raise RuntimeError(
                            "held-out validation artifacts exhaust rank zero's collection.max_disk_bytes"
                        )
                    training_config["collection"]["max_disk_bytes"] = remaining_disk_budget
                local_manifest = collect_streaming_dataset(
                    training_config,
                    training_output_root,
                    teacher=teacher,
                    env=env,
                    collection_round_idx=round_idx,
                )
                local_manifest_path = _collective_lifecycle_action(
                    lambda: validate_round_manifest_path(
                        training_output_root, round_idx, _manifest_path(local_manifest)
                    ),
                    description=f"round {round_idx} collected manifest validation",
                    world_size=world_size,
                )
                manifest = _read_manifest(local_manifest_path)
                manifest_paths = _streaming_manifest_paths(
                    local_manifest_path, rank=rank, world_size=world_size
                )
                local_transition_count = _manifest_transition_count(manifest)
                global_transition_count = _all_reduce_count(
                    local_transition_count,
                    device=torch.device(rank_device),
                    world_size=world_size,
                )
                transitions += global_transition_count
                if rank == 0:
                    monitor.log_metrics(
                        {
                            "collection/seconds": time.monotonic() - started,
                            "collection/transitions": global_transition_count,
                            "collection/rank_transitions": local_transition_count,
                        },
                        progress=MetricProgress(step, transitions, round_idx, 0),
                    )
            if not reuse_collection:
                manifest_paths = _streaming_manifest_paths(
                    local_manifest_path, rank=rank, world_size=world_size
                )
            for manifest_idx, manifest_path in enumerate(manifest_paths):
                manifest_for_rank = _read_manifest(manifest_path)
                validate_dataset_identity(
                    manifest_for_rank,
                    teacher,
                    _rank_collection_config(config, rank=manifest_idx, world_size=world_size),
                    "train",
                )
            sampler["manifest_paths"] = [str(path) for path in manifest_paths]
            sampler["manifest_path"] = str(manifest_paths[rank])
            pooled_dataset = _PooledStreamingDataset(manifest_paths)
            if len(pooled_dataset) < world_size * 2:
                raise ValueError("pooled training dataset must contain at least two windows per rank")
            start_epoch = int(sampler.get("update_epoch_idx", 0)) if reuse_collection else 0
            start_batch = int(sampler.get("next_batch_idx", 0)) if reuse_collection else 0
            round_completed = True
            for epoch_idx in range(start_epoch, int(settings["update_epochs"])):
                rank_indices, repeat_count, rank_window_count = _rank_epoch_indices(
                    len(pooled_dataset),
                    seed=int(settings["seed"]) + round_idx * int(settings["update_epochs"]),
                    epoch_idx=epoch_idx,
                    rank=rank,
                    world_size=world_size,
                )
                batches = _effective_batches(rank_indices, int(settings["batch_size"]))
                if start_batch > len(batches):
                    raise ValueError("resume sampler batch cursor is outside the current epoch")
                for batch_idx in range(start_batch if epoch_idx == start_epoch else 0, len(batches)):
                    if step >= int(settings["max_optimizer_steps"]):
                        round_completed = False
                        break
                    losses, global_batch_count, gradient_norm = _training_update(
                        ddp_student,
                        student,
                        pooled_dataset,
                        batches[batch_idx],
                        optimizer,
                        config=config,
                        device=torch.device(rank_device),
                        world_size=world_size,
                    )
                    step += 1
                    progress = MetricProgress(step, transitions, round_idx, epoch_idx)
                    sampler = {
                        "collection_round_idx": round_idx,
                        "update_epoch_idx": epoch_idx,
                        "next_batch_idx": batch_idx + 1,
                        "manifest_paths": [str(path) for path in manifest_paths],
                        "manifest_path": str(manifest_paths[rank]),
                        "rank_window_count": rank_window_count,
                        "epoch_repeat_count": repeat_count,
                    }
                    state["simulator_transitions"] = transitions
                    interval_windows += global_batch_count
                    for name, value in zip(losses._fields, losses):
                        interval_totals[name] = interval_totals.get(name, 0.0) + float(value) * global_batch_count
                    if rank == 0 and (step == 1 or step % config["wandb"].get("log_interval_steps", 10) == 0):
                        metrics = {
                            f"train/loss_{name}": value / interval_windows
                            for name, value in interval_totals.items()
                        }
                        metrics["train/gradient_norm"] = gradient_norm
                        metrics["train/learning_rate"] = optimizer.param_groups[0]["lr"]
                        metrics["throughput/training_windows_per_second"] = interval_windows / max(
                            time.monotonic() - interval_started, 1e-9
                        )
                        monitor.log_metrics(metrics, progress=progress)
                        interval_totals, interval_windows = {}, 0
                        interval_started = time.monotonic()
                    if rank == 0 and step % config["wandb"].get("diagnostics_interval_steps", 100) == 0:
                        diagnostic_indices = batches[batch_idx][:128]
                        monitor.log_metrics(
                            representation_metrics(student, pooled_dataset.get_batch(diagnostic_indices)),
                            progress=progress,
                        )
                    if step % int(settings["validation_interval_steps"]) == 0:
                        _barrier(world_size)
                        if rank == 0:
                            last_validation = run_validation(progress)
                        _barrier(world_size)
                    if step % int(settings["checkpoint_interval_steps"]) == 0:
                        checkpoint(run_dir / "checkpoint.pt")
                if step >= int(settings["max_optimizer_steps"]) and epoch_idx + 1 < int(settings["update_epochs"]):
                    round_completed = False
                if not round_completed:
                    break
                completed_epoch_count = round_idx * int(settings["update_epochs"]) + epoch_idx + 1
                sampler = {
                    "collection_round_idx": round_idx,
                    "update_epoch_idx": epoch_idx + 1,
                    "next_batch_idx": 0,
                    "manifest_paths": [str(path) for path in manifest_paths],
                    "manifest_path": str(manifest_paths[rank]),
                    "rank_window_count": rank_window_count,
                    "epoch_repeat_count": repeat_count,
                }
                if driving_enabled:
                    interval_epochs = driving_config.get("interval_update_epochs", 5)
                    if isinstance(interval_epochs, int) and interval_epochs > 0 and completed_epoch_count % interval_epochs == 0:
                        progress = MetricProgress(step, transitions, round_idx, epoch_idx)
                        _barrier(world_size)
                        run_driving(progress, step)
                        _barrier(world_size)
                if rank == 0:
                    monitor.log_metrics(
                        {
                            "data/global_windows": len(pooled_dataset),
                            "data/rank_windows": rank_window_count,
                            "data/repeated_windows": repeat_count,
                            "data/rank_repeated_windows": repeat_count // max(world_size, 1),
                        },
                        progress=MetricProgress(step, transitions, round_idx, epoch_idx),
                    )
                start_batch = 0
            if not round_completed:
                break

            _collective_lifecycle_action(
                pooled_dataset.close,
                description=f"round {round_idx} reader closure",
                world_size=world_size,
            )
            pooled_dataset = None
            state["simulator_transitions"] = transitions
            sampler = {
                "collection_round_idx": round_idx + 1,
                "update_epoch_idx": 0,
                "next_batch_idx": 0,
                "pending_cleanup_round_idx": round_idx,
            }
            checkpoint(run_dir / "checkpoint.pt")
            _collective_lifecycle_action(
                lambda: remove_completed_collection(training_output_root, round_idx),
                description=f"round {round_idx} completed collection removal",
                world_size=world_size,
            )
            sampler.pop("pending_cleanup_round_idx")
            if step >= int(settings["max_optimizer_steps"]):
                break

            if step < int(settings["max_optimizer_steps"]):
                sampler = {
                    "collection_round_idx": round_idx + 1,
                    "update_epoch_idx": 0,
                    "next_batch_idx": 0,
                }

        _barrier(world_size)
        progress = MetricProgress(
            step,
            transitions,
            int(sampler.get("collection_round_idx", 0)),
            int(sampler.get("update_epoch_idx", 0)),
        )
        if rank == 0:
            if interval_windows:
                monitor.log_metrics(
                    {f"train/loss_{name}": value / interval_windows for name, value in interval_totals.items()},
                    progress=progress,
                )
            last_validation = run_validation(progress)
        _barrier(world_size)
        if driving_enabled:
            should_run_final_driving = bool(rank == 0 and last_driving_step != step)
            should_run_final_driving = bool(
                _broadcast_object(
                    should_run_final_driving,
                    source_rank=0,
                    rank=rank,
                    world_size=world_size,
                )
            )
            if should_run_final_driving:
                _barrier(world_size)
                run_driving(progress, step)
                _barrier(world_size)
        state["simulator_transitions"] = transitions
        checkpoint(run_dir / "final_model.pt")
        result = {
            "checkpoint": str(run_dir / "final_model.pt"),
            "optimizer_steps": step,
            "simulator_transitions": transitions,
            "validation": dict(last_validation) if rank == 0 else {},
            "driving_evaluation": copy.deepcopy(last_driving_results) if rank == 0 else {},
            "monitoring": dict(monitor.state_dict()) if rank == 0 else {},
        }
        if rank == 0:
            (run_dir / "result.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
        exit_code = 0
        return result if rank == 0 else None
    finally:
        if pooled_dataset is not None:
            pooled_dataset.close()
        if owned_env and env is not None:
            env.close()
        if owned_monitor and monitor is not None:
            monitor.finish(exit_code=exit_code)
        if owns_process_group and dist.is_initialized():
            dist.destroy_process_group()


train = train_distributed
