"""Read-only probe preflight and offline observation-decoder training."""
from __future__ import annotations

import argparse
import copy
import hashlib
import json
import math
import os
import tempfile
from contextlib import nullcontext
from pathlib import Path
from typing import Any, Callable, Mapping, Sequence

from .config import DEFAULT_CONFIG, load_probe_config, validate_probe_config
from .data import inspect_checkpoint_metadata, preflight_probe_data


def preflight_probe(config: Mapping[str, Any]) -> dict[str, Any]:
    """Inspect selected data/checkpoint without initializing a training runtime."""
    unresolved = validate_probe_config(config)
    if config["data"].get("mode") == "online":
        from .data import _inspect_manifest
        from .online import (
            _normalized_observation_layout,
            estimated_round_bytes,
            inspect_online_source,
            online_collection_plan,
        )
        from ..runtime import resolve_path

        source = inspect_online_source(config["checkpoint"])
        plan = online_collection_plan(config, source)
        compatibility = source["compatibility"]
        external_heldout: dict[str, Any] = {}
        for split in ("validation", "test"):
            manifest_text = config["data"].get(f"{split}_manifest")
            if manifest_text is None:
                continue
            from ..runtime import resolve_path

            info = _inspect_manifest(
                resolve_path(manifest_text).resolve(strict=True), expected_split=split
            )
            for field in ("observation_dim", "chunk_length", "num_action_classes"):
                if info[field] != compatibility[field]:
                    raise ValueError(f"external {split} manifest disagrees with frozen source at {field}")
            for field in ("observation_layout", "action_layout"):
                actual_value = info[field]
                expected_value = compatibility[field]
                if field == "observation_layout":
                    actual_value = _normalized_observation_layout(actual_value)
                    expected_value = _normalized_observation_layout(expected_value)
                if actual_value != expected_value:
                    raise ValueError(f"external {split} manifest disagrees with frozen source at {field}")
            recipe = info.get("teacher_observation_recipe")
            if recipe is not None and recipe != compatibility["teacher_observation_recipe"]:
                raise ValueError(f"external {split} manifest has incompatible teacher observation settings")
            external_heldout[split] = {
                "manifest_path": info["manifest_path"],
                "valid_window_count": info["valid_window_count"],
                "observation_recipe_compatibility": "matched" if recipe is not None else "unverified_missing_observation_recipe_metadata",
            }
        estimated_bytes = max(
            estimated_round_bytes(
                transitions=transitions,
                slots=plan["train_slots_per_rank"],
                compatibility=compatibility,
            )
            for transitions in plan["round_transitions"]
        )
        selected_rounds = plan["round_count"]
        return {
            "status": "preflight_only",
            "training_implemented": True,
            "unresolved_settings": unresolved,
            "mode": "online",
            "data": {
                "mode": "online",
                "collection_root": config["data"]["collection_root"],
                "run_collection_root": str(
                    resolve_path(config["data"]["collection_root"]).resolve()
                    / config["training"].get("run_id", "<run_id>")
                ),
                "per_round": [],
                "selected_manifests": [],
                "compatibility": compatibility,
                "scenario_separation": "unverified; collection seeds are distinct but do not establish disjoint scenarios",
                "valid_windows": "unknown until the live simulator collection completes",
            },
            "checkpoint": {
                "path": source["checkpoint_path"],
                "format": "condition_b_v1",
                "metadata_compatible": True,
                "model_state_validated": False,
                "teacher_checkpoint_sha256": source["teacher_checkpoint_sha256"],
            },
            "collection_count": selected_rounds,
            "manifest_count": 0,
            "total_valid_windows": None,
            "selected_training_windows": None,
            "max_windows_per_collection": config["data"].get("max_windows_per_collection"),
            "batch_size_unit": "global windows per optimizer update",
            "nominal_optimizer_steps": None,
            "capped_optimizer_steps": None,
            "step_count_note": "Unknown until live valid-window counts are available",
            "collection_estimates": {
                "per_rank_simulator_transitions": plan["round_transitions"],
                "total_per_rank_simulator_transitions": sum(plan["round_transitions"]),
                "estimated_active_round_bytes_per_rank": estimated_bytes,
                "max_disk_bytes_per_rank": plan["max_disk_bytes"],
                "valid_windows_per_round": "unknown until collection",
            },
            "heldout_data": {
                "external_manifests": external_heldout,
                "validation": "external manifest" if "validation" in external_heldout else "collected once by rank 0 and retained",
                "test": "external manifest" if "test" in external_heldout else "not selected",
            },
        }

    data_report = preflight_probe_data(config)
    checkpoint = config.get("checkpoint")
    checkpoint_report = (
        inspect_checkpoint_metadata(checkpoint, data_report["compatibility"])
        if checkpoint is not None else {"metadata_compatible": None, "status": "not selected"}
    )
    training = config["training"]
    epochs = training["epochs_per_collection"]
    batch_size = training["batch_size"]
    window_limit = config["data"].get("max_windows_per_collection")
    selected_window_counts = [
        min(entry["pooled_valid_window_count"], window_limit)
        if window_limit is not None
        else entry["pooled_valid_window_count"]
        for entry in data_report["per_round"]
    ]
    nominal_steps = None
    if epochs is not None and batch_size is not None:
        nominal_steps = sum(
            epochs * ((window_count + batch_size - 1) // batch_size)
            for window_count in selected_window_counts
        )
    cap = training["max_optimizer_steps"]
    capped_steps = nominal_steps
    if nominal_steps is not None and cap is not None:
        capped_steps = min(nominal_steps, cap)
    return {
        "status": "preflight_only",
        "training_implemented": True,
        "unresolved_settings": unresolved,
        "data": data_report,
        "checkpoint": checkpoint_report,
        "collection_count": len(data_report["per_round"]),
        "manifest_count": len(data_report["selected_manifests"]),
        "total_valid_windows": sum(row["pooled_valid_window_count"] for row in data_report["per_round"]),
        "selected_training_windows": sum(selected_window_counts),
        "max_windows_per_collection": window_limit,
        "batch_size_unit": "global windows per optimizer update",
        "nominal_optimizer_steps": nominal_steps,
        "capped_optimizer_steps": capped_steps,
        "step_count_note": "Full-epoch estimate; short tails are padded with deterministic duplicate windows",
        "heldout_data": {
            "validation_manifest": config["data"]["validation_manifest"],
            "test_manifest": config["data"]["test_manifest"],
            "status": "paths selected; scenario separation is not established by manifest metadata",
        },
    }


def _collective_any(flag: bool, *, device: Any, world_size: int) -> bool:
    import torch
    import torch.distributed as dist

    if world_size == 1:
        return flag
    value = torch.tensor(int(flag), dtype=torch.int32, device=device)
    dist.all_reduce(value, op=dist.ReduceOp.MAX)
    return bool(value.item())


def _collective_read_batch(dataset: Any, indices: Sequence[int], *, device: Any, world_size: int) -> Any:
    batch = None
    error: Exception | None = None
    try:
        batch = dataset.get_batch(indices)
    except Exception as exc:
        error = exc
    failed = _collective_any(error is not None, device=device, world_size=world_size)
    if failed:
        if error is not None:
            raise RuntimeError(f"could not read probe training batch: {error}") from error
        raise RuntimeError("another distributed rank could not read its probe training batch")
    return batch


def train_update(
    decoder: Any,
    jepa: Any,
    dataset: Any,
    local_indices: Sequence[int],
    optimizer: Any,
    config: Mapping[str, Any],
    device: Any,
    *,
    world_size: int = 1,
    distributed_decoder: Any = None,
) -> dict[str, float]:
    """Run one sample-weighted global optimizer update on real future latents.

    ``local_indices`` contains this rank's equal share of the configured global
    batch. The same function is used by the trainer and bounded update profilers.
    """
    import torch

    from .losses import match_objects, reconstruction_loss

    if len(local_indices) == 0:
        raise ValueError("a probe optimizer update requires at least one local window")
    training = config["training"]
    microbatch_size = int(training["microbatch_size"])
    effective_local_size = len(local_indices)
    global_sample_count = effective_local_size * world_size
    model = distributed_decoder if distributed_decoder is not None else decoder
    optimizer.zero_grad(set_to_none=True)
    decoder.train()
    jepa.eval()
    weighted_totals: dict[str, float] = {}

    for microbatch_start in range(0, effective_local_size, microbatch_size):
        microbatch_indices = local_indices[microbatch_start : microbatch_start + microbatch_size]
        batch = _collective_read_batch(
            dataset, microbatch_indices, device=device, world_size=world_size
        )
        final_microbatch = microbatch_start + len(microbatch_indices) >= effective_local_size
        synchronization = (
            nullcontext()
            if final_microbatch or not hasattr(model, "no_sync")
            else model.no_sync()
        )
        losses = None
        forward_error: Exception | None = None
        with synchronization:
            try:
                future_observations = batch.future_observations.to(device=device, dtype=torch.float32)
                with torch.no_grad():
                    target_latents = jepa.encode_target(future_observations)
                target_norms = target_latents.norm(dim=-1, keepdim=True).clamp_min(1e-6)
                normalized_target_latents = target_latents / target_norms
                prediction = model(normalized_target_latents)
                assignments = match_objects(
                    prediction,
                    future_observations,
                    observation_layout=config["_observation_layout"],
                    matching_config=config["matching"],
                )
                losses = reconstruction_loss(
                    prediction,
                    future_observations,
                    assignments,
                    observation_layout=config["_observation_layout"],
                    loss_config=config["loss"],
                )
            except Exception as exc:
                forward_error = exc
            if _collective_any(forward_error is not None, device=device, world_size=world_size):
                optimizer.zero_grad(set_to_none=True)
                if forward_error is not None:
                    raise RuntimeError(f"probe forward/loss failed: {forward_error}") from forward_error
                raise RuntimeError("probe forward/loss failed on another distributed rank")
            if losses is None:
                optimizer.zero_grad(set_to_none=True)
                raise RuntimeError("probe forward/loss returned no loss")
            finite_loss = bool(torch.isfinite(losses.total).all().item())
            if _collective_any(not finite_loss, device=device, world_size=world_size):
                optimizer.zero_grad(set_to_none=True)
                raise FloatingPointError("Non-finite probe loss; optimizer was not updated")
            sample_weight = len(microbatch_indices) * world_size / global_sample_count
            (losses.total * sample_weight).backward()

        sample_count = len(microbatch_indices)
        weighted_totals["loss_total"] = weighted_totals.get("loss_total", 0.0) + (
            float(losses.total.detach().cpu()) * sample_count
        )
        for name, value in losses.components.items():
            key = f"loss_{name}"
            weighted_totals[key] = weighted_totals.get(key, 0.0) + (
                float(value.detach().cpu()) * sample_count
            )

    trainable_parameters = [parameter for parameter in decoder.parameters() if parameter.requires_grad]
    gradient_clip_norm = float(
        training.get("gradient_clip_norm", training.get("clip_norm", 1.0))
    )
    gradient_norm = torch.nn.utils.clip_grad_norm_(
        trainable_parameters, gradient_clip_norm, error_if_nonfinite=False
    )
    invalid_gradient = not bool(torch.isfinite(gradient_norm).all().item())
    if not invalid_gradient:
        invalid_gradient = any(
            parameter.grad is not None
            and not bool(torch.isfinite(parameter.grad).all().item())
            for parameter in trainable_parameters
        )
    if _collective_any(invalid_gradient, device=device, world_size=world_size):
        optimizer.zero_grad(set_to_none=True)
        raise FloatingPointError("Non-finite probe gradient; optimizer was not updated")
    optimizer.step()

    totals = torch.tensor(
        [weighted_totals[name] for name in sorted(weighted_totals)],
        dtype=torch.float64,
        device=device,
    )
    if world_size > 1:
        torch.distributed.all_reduce(totals, op=torch.distributed.ReduceOp.SUM)
    metrics = {
        name: float(value) / global_sample_count
        for name, value in zip(sorted(weighted_totals), totals.detach().cpu().tolist())
    }
    metrics["gradient_norm"] = float(gradient_norm.detach().cpu())
    return metrics


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _identity_config(config: Mapping[str, Any]) -> dict[str, Any]:
    result = copy.deepcopy(dict(config))
    result.get("training", {}).pop("resume_checkpoint", None)
    return json.loads(json.dumps(result, sort_keys=True, allow_nan=False))


def _canonical_json(value: Any) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False)


def _validate_heldout_compatibility(
    dataset: Any,
    training_compatibility: Mapping[str, Any],
    *,
    split: str,
) -> str:
    for field in ("observation_dim", "chunk_length", "num_action_classes"):
        expected_value = training_compatibility[field]
        actual_value = getattr(dataset, field, None)
        if actual_value != expected_value:
            raise ValueError(
                f"{split} manifest disagrees with training collections at {field}: "
                f"expected {expected_value!r}, got {actual_value!r}"
            )
    for field in ("observation_layout", "action_layout"):
        expected_value = training_compatibility[field]
        actual_value = getattr(dataset, field, None)
        if field == "observation_layout":
            from .online import _normalized_observation_layout

            expected_value = _normalized_observation_layout(expected_value)
            actual_value = _normalized_observation_layout(actual_value)
        if _canonical_json(actual_value) != _canonical_json(expected_value):
            raise ValueError(f"{split} manifest has incompatible {field.replace('_', ' ')}")
    expected_recipe = training_compatibility.get("teacher_observation_recipe")
    actual_recipe = getattr(dataset, "teacher_observation_recipe", None)
    if expected_recipe is not None and actual_recipe is not None:
        if _canonical_json(actual_recipe) != _canonical_json(expected_recipe):
            raise ValueError(
                f"{split} manifest has incompatible dt/dynamics/observation normalization settings"
            )
        return "matched"
    return "unverified_missing_observation_recipe_metadata"


def _make_identity(
    config: Mapping[str, Any],
    checkpoint_path: Path,
    manifest_paths: Sequence[str],
) -> dict[str, Any]:
    from ..runtime import resolve_path

    manifest_identity = []
    for manifest_text in manifest_paths:
        manifest_path = resolve_path(manifest_text).resolve(strict=True)
        manifest_identity.append({"path": str(manifest_path), "sha256": _sha256(manifest_path)})
    heldout_identity = []
    for split in ("validation", "test"):
        manifest_text = config["data"].get(f"{split}_manifest")
        if manifest_text is not None:
            manifest_path = resolve_path(manifest_text).resolve(strict=True)
            heldout_identity.append(
                {"split": split, "path": str(manifest_path), "sha256": _sha256(manifest_path)}
            )
    return {
        "frozen_checkpoint_sha256": _sha256(checkpoint_path),
        "training_manifests": manifest_identity,
        "heldout_manifests": heldout_identity,
        "config": _identity_config(config),
    }


def _rank0_call(
    callback: Callable[[], Any],
    *,
    description: str,
    rank: int,
    world_size: int,
    broadcast_result: bool = True,
) -> Any:
    import torch.distributed as dist

    result = None
    failure: str | None = None
    if rank == 0:
        try:
            result = callback()
        except Exception as exc:
            failure = f"{type(exc).__name__}: {exc}"
            if world_size == 1:
                raise
    if world_size > 1:
        payload = [{"failure": failure, "result": result if broadcast_result else None}]
        dist.broadcast_object_list(payload, src=0)
        failure = payload[0]["failure"]
        result = payload[0]["result"]
    if failure is not None:
        raise RuntimeError(f"rank 0 {description} failed: {failure}")
    return result


def _atomic_torch_save(payload: Mapping[str, Any], path: Path) -> None:
    import torch
    from ..collection_lifecycle import durable_replace_checkpoint

    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_path = tempfile.mkstemp(
        prefix=f".{path.name}.", suffix=".tmp", dir=path.parent
    )
    os.close(descriptor)
    try:
        torch.save(dict(payload), temporary_path)
        durable_replace_checkpoint(Path(temporary_path), path)
    finally:
        if os.path.exists(temporary_path):
            os.unlink(temporary_path)


def _checkpoint_payload(
    decoder: Any,
    optimizer: Any,
    *,
    identity: Mapping[str, Any],
    progress: Mapping[str, int],
    monitoring_state: Mapping[str, Any] | None,
    last_validation_step: int | None,
    last_test_step: int | None,
    best_validation_loss: float | None,
    active_collection: Sequence[Mapping[str, str]] | None = None,
    training_mean_observation: Sequence[float] | None = None,
) -> dict[str, Any]:
    return {
        "format": "observation_probe_v1",
        "decoder_state": decoder.state_dict(),
        "optimizer_state": optimizer.state_dict(),
        "identity": dict(identity),
        "progress": dict(progress),
        "monitoring_state": dict(monitoring_state or {}),
        "last_validation_step": last_validation_step,
        "last_test_step": last_test_step,
        "best_validation_loss": best_validation_loss,
        "active_collection": [dict(item) for item in active_collection] if active_collection else None,
        "training_mean_observation": (
            list(map(float, training_mean_observation))
            if training_mean_observation is not None else None
        ),
    }


def _load_resume_checkpoint(
    path: Path,
    decoder: Any,
    optimizer: Any,
    *,
    expected_identity: Mapping[str, Any],
    map_location: Any,
) -> dict[str, Any]:
    import torch

    payload = torch.load(path, map_location="cpu", weights_only=False)
    if not isinstance(payload, Mapping) or payload.get("format") != "observation_probe_v1":
        raise ValueError("resume checkpoint must use format observation_probe_v1")
    saved_identity = copy.deepcopy(dict(payload.get("identity", {})))
    current_identity = copy.deepcopy(dict(expected_identity))
    saved_config = saved_identity.get("config", {})
    current_config = current_identity.get("config", {})
    saved_mode = saved_config.get("data", {}).get("mode") if isinstance(saved_config, Mapping) else None
    current_mode = current_config.get("data", {}).get("mode") if isinstance(current_config, Mapping) else None
    if saved_mode == current_mode == "online":
        saved_training = saved_config.get("training")
        current_training = current_config.get("training")
        if isinstance(saved_training, Mapping):
            saved_training.pop("max_optimizer_steps", None)
        if isinstance(current_training, Mapping):
            current_training.pop("max_optimizer_steps", None)
    if saved_identity != current_identity:
        raise ValueError(
            "resume checkpoint identity differs from the frozen JEPA checkpoint, manifests, or resolved config"
        )
    progress = payload.get("progress")
    if not isinstance(progress, Mapping):
        raise ValueError("resume checkpoint is missing progress state")
    for key in ("collection_index", "epoch_index", "next_batch_index", "optimizer_step"):
        value = progress.get(key)
        if isinstance(value, bool) or not isinstance(value, int) or value < 0:
            raise ValueError(f"resume checkpoint progress.{key} must be a non-negative integer")
    for key in ("collected_transitions",):
        if key in progress:
            value = progress.get(key)
            if isinstance(value, bool) or not isinstance(value, int) or value < 0:
                raise ValueError(f"resume checkpoint progress.{key} must be a non-negative integer")
    for key in ("active_round_idx", "pending_cleanup_round_idx"):
        if key in progress and progress[key] is not None:
            value = progress[key]
            if isinstance(value, bool) or not isinstance(value, int) or value < 0:
                raise ValueError(f"resume checkpoint progress.{key} must be a non-negative integer or null")
    active_collection = payload.get("active_collection")
    if active_collection is not None:
        if not isinstance(active_collection, list) or any(
            not isinstance(item, Mapping)
            or not isinstance(item.get("path"), str)
            or not isinstance(item.get("sha256"), str)
            for item in active_collection
        ):
            raise ValueError("resume checkpoint active_collection has an invalid manifest fingerprint")
    if not isinstance(payload.get("decoder_state"), Mapping):
        raise ValueError("resume checkpoint is missing decoder parameters")
    if not isinstance(payload.get("optimizer_state"), Mapping):
        raise ValueError("resume checkpoint is missing optimizer state")
    decoder.load_state_dict(payload["decoder_state"], strict=True)
    optimizer.load_state_dict(payload["optimizer_state"])
    _ = map_location
    return dict(payload)


def _setup_runtime(config: Mapping[str, Any]) -> tuple[Any, int, int, int, Any, bool]:
    import torch
    import torch.distributed as dist

    training = config["training"]
    owns_process_group = False
    env_world_size = int(os.environ.get("WORLD_SIZE", "1"))
    requested_device = str(training["device"])
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    if not dist.is_initialized() and env_world_size > 1 and requested_device.startswith("cuda"):
        if not torch.cuda.is_available():
            raise RuntimeError("training.device requests CUDA but CUDA is unavailable")
        torch.cuda.set_device(local_rank)
    if dist.is_initialized():
        rank = dist.get_rank()
        world_size = dist.get_world_size()
    elif env_world_size > 1:
        backend = "nccl" if str(training["device"]).startswith("cuda") else "gloo"
        dist.init_process_group(backend=backend, init_method="env://")
        owns_process_group = True
        rank = dist.get_rank()
        world_size = dist.get_world_size()
    else:
        rank, world_size = 0, 1

    configured_world_size = training.get("world_size")
    if configured_world_size is not None and configured_world_size != world_size:
        if owns_process_group:
            dist.destroy_process_group()
        raise ValueError(
            f"training.world_size={configured_world_size} does not match launched world size {world_size}"
        )
    local_rank = int(os.environ.get("LOCAL_RANK", rank))
    if requested_device.startswith("cuda"):
        if not torch.cuda.is_available():
            if owns_process_group:
                dist.destroy_process_group()
            raise RuntimeError("training.device requests CUDA but CUDA is unavailable")
        device_index = local_rank if world_size > 1 or requested_device == "cuda" else int(requested_device.split(":", 1)[1])
        torch.cuda.set_device(device_index)
        device = torch.device(f"cuda:{device_index}")
    else:
        device = torch.device(requested_device)
    cpu_threads = int(training.get("cpu_threads", 1))
    torch.set_num_threads(cpu_threads)
    seed = int(training["seed"])
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    return torch, rank, world_size, local_rank, device, owns_process_group


def _epoch_batch_indices(order: Sequence[int], count: int, start: int, stop: int) -> list[int]:
    """Take a slice from a lazily shuffled order, cycling only for tail padding."""
    import numpy as np

    if count < 1 or start < 0 or stop < start:
        raise ValueError("invalid padded epoch index range")
    pieces = []
    cursor = start
    while cursor < stop:
        position = cursor % count
        piece_end = min(stop - cursor, count - position)
        pieces.append(np.asarray(order[position : position + piece_end], dtype=np.int64))
        cursor += piece_end
    if not pieces:
        return []
    return np.concatenate(pieces).tolist()


def _next_progress(
    *,
    collection_index: int,
    epoch_index: int,
    batch_index: int,
    batches_per_epoch: int,
    epochs_per_collection: int,
    collection_count: int,
    optimizer_step: int,
) -> dict[str, int]:
    if batch_index + 1 < batches_per_epoch:
        return {
            "collection_index": collection_index,
            "epoch_index": epoch_index,
            "next_batch_index": batch_index + 1,
            "optimizer_step": optimizer_step,
        }
    if epoch_index + 1 < epochs_per_collection:
        return {
            "collection_index": collection_index,
            "epoch_index": epoch_index + 1,
            "next_batch_index": 0,
            "optimizer_step": optimizer_step,
        }
    return {
        "collection_index": min(collection_index + 1, collection_count),
        "epoch_index": 0,
        "next_batch_index": 0,
        "optimizer_step": optimizer_step,
    }


def _train_collection(
    decoder: Any,
    jepa: Any,
    dataset: Any,
    optimizer: Any,
    distributed_decoder: Any,
    config: Mapping[str, Any],
    *,
    device: Any,
    rank: int,
    world_size: int,
    collection_index: int,
    collection_round_idx: int,
    starting_progress: Mapping[str, int],
    collection_count: int,
    run_dir: Path,
    identity: Mapping[str, Any],
    monitor: Any,
    last_validation_step: int | None,
    last_test_step: int | None,
    best_validation_loss: float | None,
    active_collection: Sequence[Mapping[str, str]] | None = None,
    training_mean_observation: Sequence[float] | None = None,
) -> tuple[dict[str, Any], int, int, bool, int | None, int | None]:
    from .data import selected_indices
    from ..monitoring import MetricProgress

    training = config["training"]
    epochs = int(training["epochs_per_collection"])
    global_batch_size = int(training["batch_size"])
    per_rank_batch_size = global_batch_size // world_size
    microbatch_size = int(training["microbatch_size"])
    window_limit = config["data"].get("max_windows_per_collection")
    dataset_length = len(dataset)
    selected_length = min(dataset_length, int(window_limit)) if window_limit is not None else dataset_length
    if selected_length < 1:
        raise ValueError(f"training collection round {collection_round_idx} contains no selected windows")
    batches_per_epoch = math.ceil(selected_length / global_batch_size)
    start_epoch = int(starting_progress["epoch_index"])
    start_batch = int(starting_progress["next_batch_index"])
    if start_epoch >= epochs or start_batch >= batches_per_epoch:
        raise ValueError(
            f"resume cursor epoch={start_epoch}, batch={start_batch} is outside "
            f"collection round {collection_round_idx} ({epochs} epochs, {batches_per_epoch} batches)"
        )
    optimizer_step = int(starting_progress["optimizer_step"])
    duplicates_per_epoch = batches_per_epoch * global_batch_size - selected_length
    padding_duplicates = 0
    reached_cap = False
    progress = dict(starting_progress)
    selected_order = None

    for epoch_index in range(start_epoch, epochs):
        selected_order = selected_indices(
            dataset_length,
            seed=int(training["seed"]),
            epoch_index=epoch_index,
            max_windows=window_limit,
            collection_index=collection_round_idx,
        )
        if len(selected_order) != selected_length:
            raise ValueError("selected_indices returned an unexpected number of windows")
        first_batch = start_batch if epoch_index == start_epoch else 0
        for batch_index in range(first_batch, batches_per_epoch):
            global_start = batch_index * global_batch_size
            global_indices = _epoch_batch_indices(
                selected_order,
                selected_length,
                global_start,
                global_start + global_batch_size,
            )
            local_start = rank * per_rank_batch_size
            local_indices = global_indices[local_start : local_start + per_rank_batch_size]
            metrics = train_update(
                decoder,
                jepa,
                dataset,
                local_indices,
                optimizer,
                config,
                device,
                world_size=world_size,
                distributed_decoder=distributed_decoder,
            )
            optimizer_step += 1
            progress = _next_progress(
                collection_index=collection_index,
                epoch_index=epoch_index,
                batch_index=batch_index,
                batches_per_epoch=batches_per_epoch,
                epochs_per_collection=epochs,
                collection_count=collection_count,
                optimizer_step=optimizer_step,
            )
            for field in ("collected_transitions", "active_round_idx"):
                if field in starting_progress:
                    progress[field] = starting_progress[field]
            if "pending_cleanup_round_idx" in starting_progress:
                progress["pending_cleanup_round_idx"] = (
                    collection_round_idx
                    if progress["collection_index"] > collection_index
                    else None
                )
                if progress["pending_cleanup_round_idx"] is not None:
                    progress["active_round_idx"] = None
            if batch_index == 0:
                padding_duplicates += duplicates_per_epoch

            def record_step() -> None:
                if monitor is not None:
                    monitor.log_metrics(
                        {f"train/{name}": value for name, value in metrics.items()},
                        progress=MetricProgress(
                            optimizer_step=optimizer_step,
                            simulator_transitions=(
                                int(progress.get("collected_transitions", 0)) * world_size
                            ),
                            collection_round_idx=collection_round_idx,
                            update_epoch_idx=epoch_index,
                        ),
                    )
                    monitoring_state = monitor.state_dict()
                else:
                    monitoring_state = None
                _atomic_torch_save(
                    _checkpoint_payload(
                        decoder,
                        optimizer,
                        identity=identity,
                        progress=progress,
                        monitoring_state=monitoring_state,
                        last_validation_step=last_validation_step,
                        last_test_step=last_test_step,
                        best_validation_loss=best_validation_loss,
                        active_collection=active_collection,
                        training_mean_observation=training_mean_observation,
                    ),
                    run_dir / "checkpoint.pt",
                )

            _rank0_call(
                record_step,
                description="training checkpoint save",
                rank=rank,
                world_size=world_size,
                broadcast_result=False,
            )
            if training["max_optimizer_steps"] is not None and optimizer_step >= int(
                training["max_optimizer_steps"]
            ):
                reached_cap = True
                break
        if reached_cap:
            break
        start_batch = 0

    return progress, padding_duplicates, optimizer_step, reached_cap, last_validation_step, last_test_step


def _dataset_batches(dataset: Any, batch_size: int):
    for offset in range(0, len(dataset), batch_size):
        indices = list(range(offset, min(offset + batch_size, len(dataset))))
        yield dataset.get_batch(indices)


def _validation_score(metrics: Mapping[str, float]) -> float:
    score = metrics.get("probe/reconstruction/loss_total")
    if score is None or not math.isfinite(float(score)):
        raise ValueError("validation metrics must contain finite probe/reconstruction/loss_total")
    return float(score)


def _evaluate_split(
    jepa: Any,
    decoder: Any,
    dataset: Any,
    config: Mapping[str, Any],
    *,
    split: str,
    optimizer_step: int,
    collection_round_idx: int,
    epoch_index: int,
    device: Any,
    rank: int,
    world_size: int,
    run_dir: Path,
    monitor: Any,
) -> dict[str, float]:
    from .evaluate import evaluate_probe
    from ..monitoring import MetricProgress

    evaluation = config["evaluation"]
    batch_size = int(evaluation.get("batch_size") or config["training"]["microbatch_size"])

    def evaluate_on_rank_zero() -> dict[str, float]:
        evaluation_config = copy.deepcopy(dict(config))
        evaluation_config["training"]["device"] = str(device)
        evaluation_config["_render_dir"] = str(
            run_dir / "eval" / f"step_{optimizer_step:08d}" / split
        )
        metrics = evaluate_probe(
            jepa,
            decoder,
            _dataset_batches(dataset, batch_size),
            config=evaluation_config,
        )
        normalized = {str(name): float(value) for name, value in metrics.items()}
        if monitor is not None:
            logged_metrics = (
                {f"test/{name}": value for name, value in normalized.items()}
                if split == "test"
                else normalized
            )
            monitor.log_metrics(
                logged_metrics,
                progress=MetricProgress(
                    optimizer_step=optimizer_step,
                    simulator_transitions=0,
                    collection_round_idx=collection_round_idx,
                    update_epoch_idx=epoch_index,
                ),
            )
        return normalized

    return _rank0_call(
        evaluate_on_rank_zero,
        description=f"{split} evaluation",
        rank=rank,
        world_size=world_size,
        broadcast_result=True,
    )


def train_probe(
    config: Mapping[str, Any],
    report: Mapping[str, Any],
    *,
    jepa: Any = None,
    decoder: Any = None,
    monitor: Any = None,
) -> dict[str, Any]:
    """Train one persistent decoder sequentially over the selected collections."""
    import torch
    from torch.nn.parallel import DistributedDataParallel

    config = copy.deepcopy(dict(config))
    validate_probe_config(config, require_resolved=True)
    if not isinstance(report.get("data"), Mapping):
        report = preflight_probe(config)
    data_report = report["data"]
    online_mode = config["data"].get("mode") == "online"
    online_source = None
    online_plan = None
    online_manager = None
    if online_mode:
        from .online import OnlineCollectionManager, inspect_online_source, online_collection_plan

        online_source = inspect_online_source(config["checkpoint"])
        online_plan = online_collection_plan(config, online_source)
        collection_count = int(online_plan["round_count"])
        round_rows = [
            {"collection_round_idx": round_idx}
            for round_idx in range(collection_count)
        ]
    else:
        round_rows = list(data_report["per_round"])
        collection_count = len(round_rows)
        if not round_rows:
            raise ValueError("preflight report contains no selected training collections")

    torch, rank, world_size, local_rank, device, owns_process_group = _setup_runtime(config)
    training = config["training"]
    run_id = training.get("run_id")
    if not isinstance(run_id, str) or not run_id.strip() or Path(run_id).name != run_id or run_id in {".", ".."}:
        raise ValueError("training.run_id must be a non-empty directory name")
    from ..runtime import resolve_path

    run_dir = resolve_path(training["output_root"]).resolve() / run_id
    checkpoint_path = resolve_path(config["checkpoint"]).resolve(strict=True)
    resume_text = training.get("resume_checkpoint")
    resume_path = resolve_path(resume_text).resolve(strict=True) if resume_text else None
    manifest_paths = (
        [] if online_mode else [path for row in round_rows for path in row["manifest_paths"]]
    )

    def setup_run_directory() -> None:
        exists_with_contents = run_dir.exists() and any(run_dir.iterdir())
        if exists_with_contents and resume_path is None:
            raise FileExistsError(f"probe run directory already contains data: {run_dir}")
        run_dir.mkdir(parents=True, exist_ok=True)
        if resume_path is None:
            config_path = run_dir / "config.json"
            config_path.write_text(
                json.dumps(dict(config), indent=2, sort_keys=True, allow_nan=False) + "\n",
                encoding="utf-8",
            )

    _rank0_call(
        setup_run_directory,
        description="run directory setup",
        rank=rank,
        world_size=world_size,
        broadcast_result=False,
    )
    if online_mode:
        assert online_source is not None and online_plan is not None
        from .online import OnlineCollectionManager

        online_manager = OnlineCollectionManager(
            config,
            online_source,
            online_plan,
            rank=rank,
            world_size=world_size,
            device=device,
        )
        validation_manifest = config["data"].get("validation_manifest")
        if validation_manifest is None:
            validation_manifest = _rank0_call(
                online_manager.collect_heldout,
                description="online validation collection",
                rank=rank,
                world_size=world_size,
            )
            config["data"]["validation_manifest"] = str(validation_manifest)
        if rank == 0 and resume_path is None:
            (run_dir / "config.json").write_text(
                json.dumps(config, indent=2, sort_keys=True, allow_nan=False) + "\n",
                encoding="utf-8",
            )
    identity = _rank0_call(
        lambda: _make_identity(config, checkpoint_path, manifest_paths),
        description="training identity calculation",
        rank=rank,
        world_size=world_size,
    )
    if online_mode:
        assert online_source is not None
        compatibility = online_source["compatibility"]
    else:
        compatibility = data_report["compatibility"]
    observation_layout = compatibility["observation_layout"]
    runtime_config = copy.deepcopy(dict(config))
    runtime_config["_observation_layout"] = dict(observation_layout)
    if jepa is None:
        from .model import load_frozen_jepa

        jepa = load_frozen_jepa(str(checkpoint_path), device=str(device))
    jepa.to(device)
    jepa.eval()
    for parameter in jepa.parameters():
        parameter.requires_grad_(False)

    if decoder is None:
        from .model import ObservationDecoder

        decoder = ObservationDecoder(
            int(jepa.latent_dim),
            observation_layout,
            hidden_sizes=list(config["decoder"]["hidden_sizes"]),
        )
    decoder.to(device)
    optimizer = torch.optim.AdamW(
        (parameter for parameter in decoder.parameters() if parameter.requires_grad),
        lr=float(training["learning_rate"]),
        weight_decay=float(training.get("weight_decay", 0.0)),
    )

    payload: dict[str, Any] = {}
    if resume_path is not None:
        payload = _load_resume_checkpoint(
            resume_path,
            decoder,
            optimizer,
            expected_identity=identity,
            map_location=device,
        )
    progress = dict(payload.get("progress", {})) or {
        "collection_index": 0,
        "epoch_index": 0,
        "next_batch_index": 0,
        "optimizer_step": 0,
    }
    if online_mode:
        progress.setdefault("collected_transitions", 0)
        progress.setdefault("active_round_idx", None)
        progress.setdefault("pending_cleanup_round_idx", None)
    active_collection = payload.get("active_collection")
    training_mean_observation = payload.get("training_mean_observation")
    if training_mean_observation is not None:
        runtime_config["_training_mean_observation"] = list(training_mean_observation)
    if int(progress["collection_index"]) > collection_count:
        raise ValueError("resume checkpoint collection_index exceeds the configured collection count")
    if online_mode:
        pending_round = progress.get("pending_cleanup_round_idx")
        active_round = progress.get("active_round_idx")
        if pending_round is not None:
            if pending_round != int(progress["collection_index"]) - 1:
                raise ValueError("pending cleanup round must be the immediately completed round")
            if active_round is not None:
                raise ValueError("a checkpoint cannot have both an active round and pending cleanup")
        elif active_round is not None and active_round != int(progress["collection_index"]):
            raise ValueError("active round must match the collection cursor")
    last_validation_step = payload.get("last_validation_step")
    last_test_step = payload.get("last_test_step")
    best_validation_loss = payload.get("best_validation_loss")

    if world_size > 1:
        distributed_decoder = DistributedDataParallel(
            decoder,
            device_ids=[local_rank] if device.type == "cuda" else None,
            broadcast_buffers=False,
        )
    else:
        distributed_decoder = None

    if world_size > 1:
        any_monitor = _collective_any(monitor is not None, device=device, world_size=world_size)
        any_missing_monitor = _collective_any(monitor is None, device=device, world_size=world_size)
        if any_monitor and any_missing_monitor:
            raise ValueError("monitor must be supplied consistently on every distributed rank")
    if monitor is None:
        monitor_box: list[Any] = []

        def initialize_monitor() -> None:
            from .monitoring import ProbeMonitor

            monitor_box.append(
                ProbeMonitor(
                    config["wandb"], config, run_dir,
                    checkpoint_state=payload.get("monitoring_state") or None,
                )
            )

        _rank0_call(
            initialize_monitor,
            description="monitor initialization",
            rank=rank,
            world_size=world_size,
            broadcast_result=False,
        )
        if rank == 0:
            monitor = monitor_box[0]
    if rank != 0:
        monitor = None

    from .data import ProbeCollectionDataset

    heldout: dict[str, Any] = {}
    heldout_compatibility: dict[str, str] = {}
    exit_code = 1

    def open_heldout() -> None:
        validation_path = resolve_path(config["data"]["validation_manifest"])
        heldout["validation"] = ProbeCollectionDataset(
            [str(validation_path)], expected_split="validation"
        )
        heldout_compatibility["validation"] = _validate_heldout_compatibility(
            heldout["validation"], data_report["compatibility"], split="validation"
        )
        test_manifest = config["data"].get("test_manifest")
        if test_manifest is not None:
            heldout["test"] = ProbeCollectionDataset(
                [str(resolve_path(test_manifest))], expected_split="test"
            )
            heldout_compatibility["test"] = _validate_heldout_compatibility(
                heldout["test"], data_report["compatibility"], split="test"
            )

    try:
        _rank0_call(
            open_heldout,
            description="held-out dataset setup",
            rank=rank,
            world_size=world_size,
            broadcast_result=False,
        )
        if online_mode:
            assert online_manager is not None
            pending_round = progress.get("pending_cleanup_round_idx")
            if pending_round is not None:
                if world_size > 1:
                    torch.distributed.barrier()
                cleanup_error: Exception | None = None
                try:
                    online_manager.remove_training_collection(int(pending_round))
                except Exception as exc:
                    cleanup_error = exc
                cleanup_failed = _collective_any(
                    cleanup_error is not None, device=device, world_size=world_size
                )
                if cleanup_failed:
                    if cleanup_error is not None:
                        raise RuntimeError(
                            f"could not finish pending cleanup for round {pending_round}: {cleanup_error}"
                        ) from cleanup_error
                    raise RuntimeError(
                        f"another rank could not finish pending cleanup for round {pending_round}"
                    )
                if world_size > 1:
                    torch.distributed.barrier()
                progress["pending_cleanup_round_idx"] = None
                progress["active_round_idx"] = None
                active_collection = None
                _rank0_call(
                    lambda: _atomic_torch_save(
                        _checkpoint_payload(
                            decoder,
                            optimizer,
                            identity=identity,
                            progress=progress,
                            monitoring_state=monitor.state_dict() if monitor is not None else None,
                            last_validation_step=last_validation_step,
                            last_test_step=last_test_step,
                            best_validation_loss=best_validation_loss,
                            active_collection=None,
                            training_mean_observation=training_mean_observation,
                        ),
                        run_dir / "checkpoint.pt",
                    ),
                    description="resume cleanup checkpoint save",
                    rank=rank,
                    world_size=world_size,
                    broadcast_result=False,
                )
            active_round = progress.get("active_round_idx")
            cursor_round = int(progress["collection_index"])
            if active_round is not None and active_round != cursor_round:
                raise ValueError("resume active round does not match its collection cursor")
            online_manager.validate_active(
                int(active_round)
                if active_round is not None
                else (cursor_round if cursor_round < collection_count else None)
            )
        optimizer_step = int(progress["optimizer_step"])
        padding_duplicates = 0
        evaluations: list[dict[str, Any]] = []
        cap = training["max_optimizer_steps"]
        reached_cap = cap is not None and optimizer_step >= int(cap)

        for collection_index in range(int(progress["collection_index"]), len(round_rows)):
            if reached_cap:
                break
            row = round_rows[collection_index]
            round_idx = int(row["collection_round_idx"])
            active_manifest_paths: list[str] | None = None
            if online_mode:
                assert online_manager is not None and online_plan is not None
                transitions = int(online_plan["round_transitions"][collection_index])
                active_manifest_paths, active_infos = online_manager.ensure_training_manifests(
                    round_idx, transitions
                )
                active_fingerprints = online_manager.fingerprints(active_manifest_paths)
                saved_active_round = progress.get("active_round_idx")
                if saved_active_round is not None and saved_active_round != round_idx:
                    raise ValueError(
                        f"resume checkpoint expects active round {saved_active_round}, current cursor is {round_idx}"
                    )
                if active_collection is not None and active_collection != active_fingerprints:
                    raise ValueError(
                        "active online collection manifests changed since the last checkpoint"
                    )
                if saved_active_round is None:
                    cumulative_transitions = int(progress.get("collected_transitions", 0))
                    max_transitions = online_plan["max_transitions"]
                    if max_transitions is not None and cumulative_transitions + transitions > int(max_transitions):
                        raise ValueError(
                            "resumed online collection would exceed collection.max_transitions"
                        )
                    progress["collected_transitions"] = cumulative_transitions + transitions
                    progress["active_round_idx"] = round_idx
                for info in active_infos:
                    if info["collection_transition_count"] != transitions:
                        raise ValueError("rank collection transition counts differ from the online plan")
                active_collection = active_fingerprints
                row = {
                    "collection_round_idx": round_idx,
                    "manifest_paths": active_manifest_paths,
                    "pooled_valid_window_count": sum(
                        info["valid_window_count"] for info in active_infos
                    ),
                }
                _rank0_call(
                    lambda: _atomic_torch_save(
                        _checkpoint_payload(
                            decoder,
                            optimizer,
                            identity=identity,
                            progress=progress,
                            monitoring_state=monitor.state_dict() if monitor is not None else None,
                            last_validation_step=last_validation_step,
                            last_test_step=last_test_step,
                            best_validation_loss=best_validation_loss,
                            active_collection=active_collection,
                            training_mean_observation=training_mean_observation,
                        ),
                        run_dir / "checkpoint.pt",
                    ),
                    description="active collection checkpoint save",
                    rank=rank,
                    world_size=world_size,
                    broadcast_result=False,
                )
            collection_dataset = None
            open_error: Exception | None = None
            try:
                collection_dataset = ProbeCollectionDataset(
                    row["manifest_paths"], expected_split="train"
                )
                if len(collection_dataset) != int(row["pooled_valid_window_count"]):
                    raise ValueError(
                        f"pooled window count changed for training collection round {round_idx}"
                    )
            except Exception as exc:
                open_error = exc
            open_failed = _collective_any(open_error is not None, device=device, world_size=world_size)
            if open_failed:
                if collection_dataset is not None:
                    collection_dataset.close()
                if open_error is not None:
                    raise RuntimeError(f"could not open training collection round {round_idx}: {open_error}") from open_error
                raise RuntimeError(f"another rank could not open training collection round {round_idx}")

            try:
                if online_mode and training_mean_observation is None:
                    if collection_index != 0:
                        raise ValueError(
                            "online resume checkpoint is missing its fitted training_mean_observation"
                        )
                    from .evaluate import fit_training_mean
                    from .schema import ObservationSchema

                    training_mean_observation = fit_training_mean(
                        collection_dataset,
                        runtime_config,
                        ObservationSchema(observation_layout),
                    ).tolist()
                    runtime_config["_training_mean_observation"] = training_mean_observation
                    _rank0_call(
                        lambda: _atomic_torch_save(
                            _checkpoint_payload(
                                decoder,
                                optimizer,
                                identity=identity,
                                progress=progress,
                                monitoring_state=monitor.state_dict() if monitor is not None else None,
                                last_validation_step=last_validation_step,
                                last_test_step=last_test_step,
                                best_validation_loss=best_validation_loss,
                                active_collection=active_collection,
                                training_mean_observation=training_mean_observation,
                            ),
                            run_dir / "checkpoint.pt",
                        ),
                        description="training mean checkpoint save",
                        rank=rank,
                        world_size=world_size,
                        broadcast_result=False,
                    )
                starting_progress = dict(progress)
                starting_progress.update(
                    {
                        "collection_index": collection_index,
                        "epoch_index": int(progress["epoch_index"])
                        if collection_index == int(progress["collection_index"])
                        else 0,
                        "next_batch_index": int(progress["next_batch_index"])
                        if collection_index == int(progress["collection_index"])
                        else 0,
                        "optimizer_step": optimizer_step,
                    }
                )
                progress, added_duplicates, optimizer_step, reached_cap, _, _ = _train_collection(
                    decoder,
                    jepa,
                    collection_dataset,
                    optimizer,
                    distributed_decoder,
                    runtime_config,
                    device=device,
                    rank=rank,
                    world_size=world_size,
                    collection_index=collection_index,
                    collection_round_idx=round_idx,
                    starting_progress=starting_progress,
                    collection_count=collection_count,
                    run_dir=run_dir,
                    identity=identity,
                    monitor=monitor,
                    last_validation_step=last_validation_step,
                    last_test_step=last_test_step,
                    best_validation_loss=best_validation_loss,
                    active_collection=active_collection,
                    training_mean_observation=training_mean_observation,
                )
                padding_duplicates += added_duplicates
            finally:
                collection_dataset.close()

            collection_complete = progress["collection_index"] > collection_index
            if collection_complete and config["evaluation"]["every_collection"]:
                validation_metrics = _evaluate_split(
                    jepa,
                    decoder,
                    heldout.get("validation"),
                    runtime_config,
                    split="validation",
                    optimizer_step=optimizer_step,
                    collection_round_idx=round_idx,
                    epoch_index=max(int(training["epochs_per_collection"]) - 1, 0),
                    device=device,
                    rank=rank,
                    world_size=world_size,
                    run_dir=run_dir,
                    monitor=monitor,
                )
                last_validation_step = optimizer_step
                validation_score = _validation_score(validation_metrics)
                is_best = best_validation_loss is None or validation_score < float(best_validation_loss)
                if is_best:
                    best_validation_loss = validation_score
                evaluations.append({"split": "validation", "optimizer_step": optimizer_step, **validation_metrics})

                def save_validation_checkpoint() -> None:
                    checkpoint_payload = _checkpoint_payload(
                        decoder,
                        optimizer,
                        identity=identity,
                        progress=progress,
                        monitoring_state=monitor.state_dict() if monitor is not None else None,
                        last_validation_step=last_validation_step,
                        last_test_step=last_test_step,
                        best_validation_loss=best_validation_loss,
                        active_collection=active_collection,
                        training_mean_observation=training_mean_observation,
                    )
                    if is_best:
                        _atomic_torch_save(checkpoint_payload, run_dir / "best.pt")
                    _atomic_torch_save(checkpoint_payload, run_dir / "checkpoint.pt")

                _rank0_call(
                    save_validation_checkpoint,
                    description="post-validation checkpoint save",
                    rank=rank,
                    world_size=world_size,
                    broadcast_result=False,
                )
            if online_mode and collection_complete:
                assert online_manager is not None
                if world_size > 1:
                    torch.distributed.barrier()
                cleanup_error: Exception | None = None
                try:
                    online_manager.remove_training_collection(round_idx)
                except Exception as exc:
                    cleanup_error = exc
                cleanup_failed = _collective_any(
                    cleanup_error is not None, device=device, world_size=world_size
                )
                if cleanup_failed:
                    if cleanup_error is not None:
                        raise RuntimeError(
                            f"could not remove completed training round {round_idx}: {cleanup_error}"
                        ) from cleanup_error
                    raise RuntimeError(
                        f"another rank could not remove completed training round {round_idx}"
                    )
                if world_size > 1:
                    torch.distributed.barrier()
                progress["pending_cleanup_round_idx"] = None
                progress["active_round_idx"] = None
                active_collection = None
                _rank0_call(
                    lambda: _atomic_torch_save(
                        _checkpoint_payload(
                            decoder,
                            optimizer,
                            identity=identity,
                            progress=progress,
                            monitoring_state=monitor.state_dict() if monitor is not None else None,
                            last_validation_step=last_validation_step,
                            last_test_step=last_test_step,
                            best_validation_loss=best_validation_loss,
                            active_collection=None,
                            training_mean_observation=training_mean_observation,
                        ),
                        run_dir / "checkpoint.pt",
                    ),
                    description="post-cleanup checkpoint save",
                    rank=rank,
                    world_size=world_size,
                    broadcast_result=False,
                )
            if reached_cap:
                break

        optimizer_step = int(progress["optimizer_step"])
        if last_validation_step != optimizer_step:
            current_round_position = min(progress["collection_index"], len(round_rows) - 1)
            current_round = int(round_rows[current_round_position]["collection_round_idx"])
            validation_metrics = _evaluate_split(
                jepa,
                decoder,
                heldout.get("validation"),
                runtime_config,
                split="validation",
                optimizer_step=optimizer_step,
                collection_round_idx=current_round,
                epoch_index=int(progress["epoch_index"]),
                device=device,
                rank=rank,
                world_size=world_size,
                run_dir=run_dir,
                monitor=monitor,
            )
            last_validation_step = optimizer_step
            validation_score = _validation_score(validation_metrics)
            if best_validation_loss is None or validation_score < float(best_validation_loss):
                best_validation_loss = validation_score
                _rank0_call(
                    lambda: _atomic_torch_save(
                        _checkpoint_payload(
                            decoder,
                            optimizer,
                            identity=identity,
                            progress=progress,
                            monitoring_state=monitor.state_dict() if monitor is not None else None,
                            last_validation_step=last_validation_step,
                            last_test_step=last_test_step,
                            best_validation_loss=best_validation_loss,
                            active_collection=active_collection,
                            training_mean_observation=training_mean_observation,
                        ),
                        run_dir / "best.pt",
                    ),
                    description="best validation checkpoint save",
                    rank=rank,
                    world_size=world_size,
                    broadcast_result=False,
                )
            evaluations.append({"split": "validation", "optimizer_step": optimizer_step, **validation_metrics})
        if config["data"].get("test_manifest") is not None and last_test_step != optimizer_step:
            current_round_position = min(progress["collection_index"], len(round_rows) - 1)
            current_round = int(round_rows[current_round_position]["collection_round_idx"])
            test_metrics = _evaluate_split(
                jepa,
                decoder,
                heldout.get("test"),
                runtime_config,
                split="test",
                optimizer_step=optimizer_step,
                collection_round_idx=current_round,
                epoch_index=int(progress["epoch_index"]),
                device=device,
                rank=rank,
                world_size=world_size,
                run_dir=run_dir,
                monitor=monitor,
            )
            last_test_step = optimizer_step
            evaluations.append({"split": "test", "optimizer_step": optimizer_step, **test_metrics})

        _rank0_call(
            lambda: _atomic_torch_save(
                _checkpoint_payload(
                    decoder,
                    optimizer,
                    identity=identity,
                    progress=progress,
                    monitoring_state=monitor.state_dict() if monitor is not None else None,
                    last_validation_step=last_validation_step,
                    last_test_step=last_test_step,
                    best_validation_loss=best_validation_loss,
                    active_collection=active_collection,
                    training_mean_observation=training_mean_observation,
                ),
                run_dir / "checkpoint.pt",
            ),
            description="final checkpoint save",
            rank=rank,
            world_size=world_size,
            broadcast_result=False,
        )
        _rank0_call(
            lambda: {
                dataset.close()
                for dataset in heldout.values()
            },
            description="held-out dataset cleanup",
            rank=rank,
            world_size=world_size,
            broadcast_result=False,
        )
        heldout.clear()
        if world_size > 1:
            torch.distributed.barrier()
        status = "capped" if reached_cap and progress["collection_index"] < len(round_rows) else "completed"
        result = {
            "status": status,
            "run_id": run_id,
            "run_dir": str(run_dir),
            "checkpoint_path": str(run_dir / "checkpoint.pt"),
            "best_checkpoint_path": str(run_dir / "best.pt") if best_validation_loss is not None else None,
            "best_validation_loss": best_validation_loss,
            "optimizer_steps": optimizer_step,
            "progress": dict(progress),
            "training_manifest_count": (
                (int(progress["collection_index"]) + int(progress.get("active_round_idx") is not None))
                * world_size
                if online_mode else len(manifest_paths)
            ),
            "collected_transitions_per_rank": progress.get("collected_transitions"),
            "padded_duplicate_windows": padding_duplicates,
            "heldout_compatibility": dict(heldout_compatibility),
            "evaluations": evaluations,
        }
        exit_code = 0
        return result
    finally:
        for dataset in heldout.values():
            dataset.close()
        try:
            if monitor is not None:
                monitor.finish(exit_code=exit_code)
        finally:
            if owns_process_group and torch.distributed.is_initialized():
                torch.distributed.destroy_process_group()


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default=DEFAULT_CONFIG)
    parser.add_argument("--set", action="append", default=[], metavar="KEY=VALUE")
    parser.add_argument("--dry-run", action="store_true", help="Inspect saved metadata/headers without training or writing files")
    args = parser.parse_args(argv)
    try:
        config = load_probe_config(args.config, args.set)
        if not args.dry_run:
            validate_probe_config(config, require_resolved=True)
        report = preflight_probe(config)
        if args.dry_run:
            print(json.dumps(report, indent=2, allow_nan=False))
            return
        result = train_probe(config, report)
        if int(os.environ.get("RANK", "0")) == 0:
            print(json.dumps(result, indent=2, allow_nan=False))
    except (ValueError, OSError, RuntimeError, FloatingPointError, NotImplementedError) as error:
        parser.exit(2, f"Observation probe: {error}\n")
    finally:
        # Setup can fail before train_probe enters its runtime cleanup block.
        import torch.distributed as dist

        if dist.is_initialized():
            dist.destroy_process_group()


if __name__ == "__main__":
    main()
