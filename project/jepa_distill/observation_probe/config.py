"""Configuration loading and boundary checks for the observation probe."""
from __future__ import annotations

import math
import re
from collections.abc import Mapping
from typing import Any

from ..runtime import load_config


DEFAULT_CONFIG = "project/jepa_distill/config/observation_probe.yaml"
_REQUIRED_CONFIG_PATHS = (
    "checkpoint",
    "training.epochs_per_collection",
    "training.batch_size",
    "training.microbatch_size",
    "training.learning_rate",
    "training.world_size",
    "training.device",
    "training.run_id",
    "matching.costs",
    "loss.group_weights",
    "rendering.presence_threshold",
)


def _nested(config: Mapping[str, Any], dotted_path: str) -> Any:
    value: Any = config
    for part in dotted_path.split("."):
        if not isinstance(value, Mapping) or part not in value:
            raise ValueError(f"Missing configuration field: {dotted_path}")
        value = value[part]
    return value


def _positive_int(value: Any, field: str, *, allow_none: bool = False) -> None:
    if value is None and allow_none:
        return
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise ValueError(f"{field} must be a positive integer" + (" or null" if allow_none else ""))


def _nonnegative_int_list(value: Any, field: str) -> None:
    if not isinstance(value, list) or not value:
        raise ValueError(f"{field} must be a non-empty list of non-negative integers")
    for index, item in enumerate(value):
        if isinstance(item, bool) or not isinstance(item, int) or item < 0:
            raise ValueError(f"{field}[{index}] must be a non-negative integer")
    if len(set(value)) != len(value):
        raise ValueError(f"{field} entries must be unique")


def _positive_float_or_none(value: Any, field: str) -> None:
    if value is None:
        return
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{field} must be a finite positive number or null")
    if not math.isfinite(float(value)) or float(value) <= 0:
        raise ValueError(f"{field} must be a finite positive number or null")


def _weight_mapping_or_none(value: Any, field: str) -> None:
    if value is None:
        return
    if not isinstance(value, Mapping) or not value:
        raise ValueError(f"{field} must be a non-empty mapping or null")
    for name, weight in value.items():
        if not isinstance(name, str) or not name:
            raise ValueError(f"{field} keys must be non-empty strings")
        if isinstance(weight, bool) or not isinstance(weight, (int, float)):
            raise ValueError(f"{field}.{name} must be a finite non-negative number")
        if not math.isfinite(float(weight)) or float(weight) < 0:
            raise ValueError(f"{field}.{name} must be a finite non-negative number")


def validate_probe_config(
    config: Mapping[str, Any], *, require_resolved: bool = False
) -> list[str]:
    """Validate supplied values and return required fields that remain null."""

    if not isinstance(config, Mapping):
        raise ValueError("Observation probe configuration must be a mapping")
    required_sections = (
        "data", "collection", "decoder", "matching", "loss", "rendering", "training", "evaluation", "wandb"
    )
    for section in required_sections:
        if not isinstance(config.get(section), Mapping):
            raise ValueError(f"Configuration section {section!r} must be a mapping")

    checkpoint = config.get("checkpoint")
    if checkpoint is not None and (not isinstance(checkpoint, str) or not checkpoint.strip()):
        raise ValueError("checkpoint must be a non-empty path or null")

    data = config["data"]
    mode = data.get("mode")
    if mode not in {"online", "cached"}:
        raise ValueError("data.mode must be 'online' or 'cached'")
    collection_root = data.get("collection_root")
    if not isinstance(collection_root, str) or not collection_root.strip():
        raise ValueError("data.collection_root must be a non-empty path")
    if mode == "cached":
        _nonnegative_int_list(data.get("collection_rounds"), "data.collection_rounds")
        _nonnegative_int_list(data.get("source_ranks"), "data.source_ranks")
    else:
        for field in ("collection_rounds", "source_ranks"):
            value = data.get(field)
            if value is not None:
                _nonnegative_int_list(value, f"data.{field}")
    for name in ("validation_manifest", "test_manifest"):
        path = data.get(name)
        if path is not None and (not isinstance(path, str) or not path.strip()):
            raise ValueError(f"data.{name} must be a non-empty path or null")
    _positive_int(data.get("max_windows_per_collection"), "data.max_windows_per_collection", allow_none=True)

    collection = config["collection"]
    for field in (
        "num_collections", "transitions_per_round", "max_transitions", "max_disk_bytes",
        "inference_batch_size", "validation_transitions",
    ):
        _positive_int(collection.get(field), f"collection.{field}", allow_none=True)
    if collection.get("action_selection") not in {None, "sample", "mean", "mode"}:
        raise ValueError("collection.action_selection must be 'sample', 'mean', 'mode', or null")
    for field in (
        "env_overrides", "vec_overrides", "validation_env_overrides", "validation_vec_overrides"
    ):
        if not isinstance(collection.get(field, {}), Mapping):
            raise ValueError(f"collection.{field} must be a mapping")

    hidden_sizes = config["decoder"].get("hidden_sizes")
    if not isinstance(hidden_sizes, list) or not hidden_sizes:
        raise ValueError("decoder.hidden_sizes must be a non-empty list of positive integers")
    for index, value in enumerate(hidden_sizes):
        _positive_int(value, f"decoder.hidden_sizes[{index}]")

    matching = config["matching"]
    if matching.get("method") != "permutation_aware":
        raise ValueError("matching.method must be 'permutation_aware'")
    _weight_mapping_or_none(matching.get("costs"), "matching.costs")
    _weight_mapping_or_none(config["loss"].get("group_weights"), "loss.group_weights")

    presence_threshold = config["rendering"].get("presence_threshold")
    if presence_threshold is not None:
        _positive_float_or_none(presence_threshold, "rendering.presence_threshold")
        if float(presence_threshold) >= 1:
            raise ValueError("rendering.presence_threshold must be less than 1")

    training = config["training"]
    _positive_int(training.get("epochs_per_collection"), "training.epochs_per_collection", allow_none=True)
    _positive_int(training.get("batch_size"), "training.batch_size", allow_none=True)
    _positive_int(training.get("microbatch_size"), "training.microbatch_size", allow_none=True)
    _positive_int(training.get("world_size"), "training.world_size", allow_none=True)
    _positive_int(training.get("max_optimizer_steps"), "training.max_optimizer_steps", allow_none=True)
    _positive_float_or_none(training.get("learning_rate"), "training.learning_rate")
    device = training.get("device")
    if device is not None and (not isinstance(device, str) or not device.strip()):
        raise ValueError("training.device must be a non-empty device string or null")
    if device is not None and re.fullmatch(r"(?:cpu|cuda(?::[0-9]+)?)", device) is None:
        raise ValueError("training.device must be 'cpu', 'cuda', or 'cuda:<index>'")
    seed = training.get("seed")
    if isinstance(seed, bool) or not isinstance(seed, int):
        raise ValueError("training.seed must be an integer")

    global_batch_size = training.get("batch_size")
    world_size = training.get("world_size")
    microbatch_size = training.get("microbatch_size")
    if global_batch_size is not None and world_size is not None:
        if global_batch_size < world_size:
            raise ValueError("training.batch_size global value must be at least training.world_size")
        if global_batch_size % world_size:
            raise ValueError("training.batch_size global value must divide evenly across training.world_size")
        per_rank_batch_size = global_batch_size // world_size
        if microbatch_size is not None and microbatch_size > per_rank_batch_size:
            raise ValueError("training.microbatch_size cannot exceed the per-rank effective batch")

    if not isinstance(config["evaluation"].get("every_collection"), bool):
        raise ValueError("evaluation.every_collection must be a boolean")
    if not isinstance(config["wandb"].get("enabled"), bool):
        raise ValueError("wandb.enabled must be a boolean")

    for field in ("cpu_threads",):
        _positive_int(training.get(field, 1), f"training.{field}")
    _positive_float_or_none(training.get("gradient_clip_norm", 1.0), "training.gradient_clip_norm")
    if training.get("gradient_clip_norm", 1.0) is None:
        raise ValueError("training.gradient_clip_norm must be a positive number")
    weight_decay = training.get("weight_decay", 0.0)
    if isinstance(weight_decay, bool) or not isinstance(weight_decay, (int, float)) or not math.isfinite(weight_decay) or weight_decay < 0:
        raise ValueError("training.weight_decay must be finite and non-negative")
    for field in ("run_id", "resume_checkpoint"):
        value = training.get(field)
        if value is not None and (not isinstance(value, str) or not value.strip()):
            raise ValueError(f"training.{field} must be a non-empty string or null")
    run_id = training.get("run_id")
    if run_id is not None and (re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_.-]*", run_id) is None or run_id in {".", ".."}):
        raise ValueError("training.run_id must be a simple directory name")
    if not isinstance(training.get("output_root"), str) or not training["output_root"].strip():
        raise ValueError("training.output_root must be a non-empty path")
    for field in ("batch_size", "max_windows", "training_mean_windows"):
        _positive_int(config["evaluation"].get(field), f"evaluation.{field}")
    render_samples = config["evaluation"].get("render_samples")
    if isinstance(render_samples, bool) or not isinstance(render_samples, int) or render_samples < 0:
        raise ValueError("evaluation.render_samples must be a non-negative integer")

    unresolved = [path for path in _REQUIRED_CONFIG_PATHS if _nested(config, path) is None]
    if mode == "cached" and data.get("validation_manifest") is None:
        unresolved.append("data.validation_manifest")
    if require_resolved and unresolved:
        raise ValueError("Unresolved required observation probe settings: " + ", ".join(unresolved))
    return unresolved


def load_probe_config(
    path: str = DEFAULT_CONFIG, overrides: list[str] | None = None
) -> dict[str, Any]:
    """Load YAML and repeated dotted ``KEY=VALUE`` overrides through runtime.py."""

    config = load_config(path, overrides)
    validate_probe_config(config)
    return config
