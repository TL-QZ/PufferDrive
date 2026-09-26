"""Source-checkpoint metadata and one-round online probe collection."""
from __future__ import annotations

import copy
import hashlib
import os
from collections.abc import Mapping
from pathlib import Path
from typing import Any

from ..runtime import resolve_path


def _positive_int(value: Any, field: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 1:
        raise ValueError(f"{field} must be a positive integer")
    return value


def _mapping(value: Any, field: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise ValueError(f"{field} must be a mapping")
    return value


def _normalized_observation_layout(layout: Mapping[str, Any]) -> dict[str, Any]:
    """Treat the collector's omitted dtype metadata as its float32 storage default."""
    if not isinstance(layout, Mapping):
        raise ValueError("observation_layout must be a mapping")
    normalized = dict(layout)
    normalized.setdefault("dtype", "float32")
    return normalized


def inspect_online_source(checkpoint_path: str | Path) -> dict[str, Any]:
    """Read the frozen JEPA config and exported metadata without loading runtimes."""
    import torch

    path = resolve_path(checkpoint_path).resolve(strict=True)
    payload = torch.load(path, map_location="cpu", weights_only=False)
    if not isinstance(payload, Mapping) or payload.get("format") != "condition_b_v1":
        raise ValueError("online probe collection requires a Condition B checkpoint")
    model_metadata = _mapping(payload.get("model_config"), "checkpoint.model_config")
    source_config = _mapping(payload.get("config"), "checkpoint.config")
    model = _mapping(model_metadata.get("model"), "checkpoint.model_config.model")
    layout = _mapping(
        model_metadata.get("observation_layout"),
        "checkpoint.model_config.observation_layout",
    )
    teacher_config = _mapping(
        model_metadata.get("teacher_config"),
        "checkpoint.model_config.teacher_config",
    )
    teacher_env = _mapping(teacher_config.get("env"), "checkpoint teacher environment")
    teacher_section = _mapping(source_config.get("teacher"), "checkpoint.config.teacher")
    teacher_checkpoint = teacher_section.get("checkpoint")
    if not isinstance(teacher_checkpoint, str) or not teacher_checkpoint.strip():
        raise ValueError("frozen JEPA config is missing teacher.checkpoint")
    teacher_hash = source_config.get("teacher_checkpoint_sha256")
    if not isinstance(teacher_hash, str) or len(teacher_hash) != 64:
        raise ValueError("frozen JEPA config is missing teacher_checkpoint_sha256")
    try:
        int(teacher_hash, 16)
    except ValueError as exc:
        raise ValueError("frozen JEPA teacher_checkpoint_sha256 is not hexadecimal") from exc
    if not (resolve_path(teacher_checkpoint).is_file()):
        raise FileNotFoundError(
            f"source teacher checkpoint does not exist: {resolve_path(teacher_checkpoint)}"
        )

    observation_dim = _positive_int(layout.get("observation_dim"), "observation_layout.observation_dim")
    chunk_length = _positive_int(model.get("chunk_length"), "model.chunk_length")
    num_action_classes = _positive_int(model.get("num_action_classes"), "model.num_action_classes")
    action_table = model_metadata.get("action_table")
    if not isinstance(action_table, list) or len(action_table) != num_action_classes:
        raise ValueError("checkpoint action_table disagrees with model.num_action_classes")
    action_layout: dict[str, Any] = {
        "normalized_controls": copy.deepcopy(action_table),
        "control_order": ["longitudinal", "lateral"],
        "normalized_range": [-1.0, 1.0],
    }
    physical_action_table = model_metadata.get("action_table_physical")
    if physical_action_table is not None:
        action_layout["physical_controls"] = copy.deepcopy(physical_action_table)

    from .data import _teacher_observation_recipe

    recipe = _teacher_observation_recipe(teacher_config)
    if recipe is None:
        raise ValueError("frozen JEPA checkpoint is missing teacher observation metadata")
    compatibility_layout = _normalized_observation_layout(layout)
    compatibility = {
        "observation_dim": observation_dim,
        "observation_layout": compatibility_layout,
        "chunk_length": chunk_length,
        "num_action_classes": num_action_classes,
        "action_layout": action_layout,
        "teacher_observation_recipe": recipe,
    }
    return {
        "checkpoint_path": str(path),
        "source_config": copy.deepcopy(dict(source_config)),
        "model_metadata": copy.deepcopy(dict(model_metadata)),
        "teacher_checkpoint": str(resolve_path(teacher_checkpoint).resolve()),
        "teacher_checkpoint_sha256": teacher_hash.lower(),
        "compatibility": compatibility,
    }


def _validate_split_config(
    source_config: Mapping[str, Any],
    *,
    env: Mapping[str, Any],
    vec: Mapping[str, Any],
    transitions: int,
    split: str,
) -> int:
    agent_count = _positive_int(env.get("num_agents"), f"{split} environment num_agents")
    env_count = _positive_int(vec.get("num_envs"), f"{split} vec.num_envs")
    worker_count = _positive_int(vec.get("num_workers"), f"{split} vec.num_workers")
    batch_size = _positive_int(vec.get("batch_size"), f"{split} vec.batch_size")
    if env_count % worker_count:
        raise ValueError(f"{split} vec.num_envs must divide evenly across vec.num_workers")
    if batch_size != env_count:
        raise ValueError(f"{split} vec.batch_size must equal vec.num_envs")
    slot_count = agent_count * env_count
    if transitions % slot_count:
        raise ValueError(
            f"collection.{split}_transitions={transitions} must be a multiple of the "
            f"{split} vector slot count {slot_count}"
        )
    return slot_count


def online_collection_plan(config: Mapping[str, Any], source: Mapping[str, Any]) -> dict[str, Any]:
    """Resolve bounded budgets from the frozen JEPA recipe plus explicit overrides."""
    source_config = _mapping(source["source_config"], "source config")
    source_collection = _mapping(source_config.get("collection"), "source collection config")
    source_vec = _mapping(source_config.get("vec"), "source vec config")
    source_env = _mapping(
        _mapping(source["model_metadata"].get("teacher_config"), "teacher config").get("env"),
        "teacher environment",
    )
    collection = config["collection"]
    configured_world_size = config["training"].get("world_size")
    world_size = configured_world_size or int(os.environ.get("WORLD_SIZE", "1"))
    _positive_int(world_size, "training.world_size")
    overrides = {
        key: value
        for key, value in collection.get("env_overrides", {}).items()
    }
    train_env = dict(source_env)
    train_env.update(overrides)
    train_vec = dict(source_vec)
    train_vec.update(collection.get("vec_overrides", {}))

    from .data import _teacher_observation_recipe

    base_recipe = source["compatibility"]["teacher_observation_recipe"]
    adjusted_train_recipe = _teacher_observation_recipe({"env": train_env})
    if adjusted_train_recipe != base_recipe:
        raise ValueError(
            "collection.env_overrides changes the frozen teacher observation recipe "
            "(action type, dt, dynamics, reward conditioning, or normalization)"
        )
    if not isinstance(train_vec.get("backend"), str) or not train_vec["backend"]:
        raise ValueError("vec.backend must be a non-empty string")

    round_transitions = collection.get("transitions_per_round")
    if round_transitions is None:
        round_transitions = source_collection.get("transitions_per_round")
    round_transitions = _positive_int(round_transitions, "collection.transitions_per_round")
    max_transitions = collection.get("max_transitions")
    if max_transitions is None:
        max_transitions = source_collection.get("max_transitions")
    if max_transitions is not None:
        max_transitions = _positive_int(max_transitions, "collection.max_transitions")
    requested_rounds = collection.get("num_collections")
    if requested_rounds is None:
        requested_rounds = source_collection.get("num_collections")
    requested_rounds = _positive_int(requested_rounds, "collection.num_collections")
    train_slots = _validate_split_config(
        source_config,
        env=train_env,
        vec=train_vec,
        transitions=round_transitions,
        split="training",
    )
    requested_total = requested_rounds * round_transitions
    if max_transitions is not None and requested_total > max_transitions:
        raise ValueError(
            "collection.num_collections * collection.transitions_per_round exceeds "
            "collection.max_transitions; lower the round count or per-round budget"
        )
    usable_transitions = requested_total
    actual_rounds = requested_rounds

    max_disk_bytes = collection.get("max_disk_bytes")
    if max_disk_bytes is None:
        max_disk_bytes = source_collection.get("max_disk_bytes")
    if max_disk_bytes is not None:
        max_disk_bytes = _positive_int(max_disk_bytes, "collection.max_disk_bytes")
    action_selection = collection.get("action_selection") or source_collection.get("action_selection", "sample")
    if action_selection not in {"sample", "mean", "mode"}:
        raise ValueError("collection.action_selection must be 'sample', 'mean', or 'mode'")
    inference_batch_size = collection.get("inference_batch_size")
    if inference_batch_size is None:
        inference_batch_size = source_collection.get("inference_batch_size")
    if inference_batch_size is not None:
        inference_batch_size = _positive_int(inference_batch_size, "collection.inference_batch_size")

    source_validation = _mapping(source_config.get("validation", {}), "source validation config")
    validation_env = dict(source_env)
    validation_env.update(source_validation.get("env_overrides", {}))
    validation_env.update(collection.get("validation_env_overrides", {}))
    validation_vec = dict(source_validation.get("vec", train_vec))
    validation_vec.update(collection.get("validation_vec_overrides", {}))
    validation_transitions = collection.get("validation_transitions")
    if validation_transitions is None:
        validation_transitions = _mapping(source_config.get("training"), "source training config").get(
            "validation_transitions"
        )
    if validation_transitions is None:
        validation_transitions = round_transitions
    validation_transitions = _positive_int(validation_transitions, "collection.validation_transitions")
    validation_slots = _validate_split_config(
        source_config,
        env=validation_env,
        vec=validation_vec,
        transitions=validation_transitions,
        split="validation",
    )
    validation_recipe = _teacher_observation_recipe({"env": validation_env})
    if validation_recipe != base_recipe:
        raise ValueError(
            "validation environment changes the frozen teacher observation recipe"
        )
    validation_bytes = estimated_round_bytes(
        transitions=validation_transitions,
        slots=validation_slots,
        compatibility=source["compatibility"],
    )
    if max_disk_bytes is not None and validation_bytes >= max_disk_bytes:
        raise ValueError(
            "collection.max_disk_bytes cannot fit the retained validation collection"
        )
    rank_disk_budgets = [max_disk_bytes] * world_size
    if max_disk_bytes is not None:
        rank_disk_budgets[0] = max_disk_bytes - validation_bytes
        training_bytes = estimated_round_bytes(
            transitions=round_transitions,
            slots=train_slots,
            compatibility=source["compatibility"],
        )
        if any(training_bytes > budget for budget in rank_disk_budgets):
            raise ValueError(
                "collection.max_disk_bytes cannot fit one active training round "
                "plus the retained validation collection"
            )
    return {
        "requested_rounds": requested_rounds,
        "world_size": world_size,
        "round_count": actual_rounds,
        "transitions_per_round": round_transitions,
        "max_transitions": max_transitions,
        "usable_max_transitions": max_transitions,
        "round_transitions": [round_transitions for _ in range(actual_rounds)],
        "train_slots_per_rank": train_slots,
        "max_disk_bytes": max_disk_bytes,
        "rank_disk_budgets": rank_disk_budgets,
        "validation_estimated_bytes": validation_bytes,
        "action_selection": action_selection,
        "inference_batch_size": inference_batch_size,
        "train_env": train_env,
        "train_vec": train_vec,
        "validation_transitions": validation_transitions,
        "validation_slots": validation_slots,
        "validation_env": validation_env,
        "validation_vec": validation_vec,
    }


def estimated_round_bytes(
    *, transitions: int, slots: int, compatibility: Mapping[str, Any]
) -> int:
    """Return the collector's conservative uncompressed file-size estimate."""
    from ..streaming import _estimated_round_bytes

    return _estimated_round_bytes(
        transition_steps=transitions // slots,
        slots=slots,
        observation_dim=int(compatibility["observation_dim"]),
        classes=int(compatibility["num_action_classes"]),
        chunk_length=int(compatibility["chunk_length"]),
    )


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


class OnlineCollectionManager:
    """Collect, inspect, open, and safely remove one rank-owned training round."""

    def __init__(
        self,
        config: Mapping[str, Any],
        source: Mapping[str, Any],
        plan: Mapping[str, Any],
        *,
        rank: int,
        world_size: int,
        device: Any,
    ) -> None:
        self.config = config
        self.source = source
        self.plan = plan
        self.rank = rank
        self.world_size = world_size
        self.device = device
        from ..runtime import resolve_path

        run_root = resolve_path(config["data"]["collection_root"]).resolve() / config["training"]["run_id"]
        self.run_root = run_root
        self.rank_root = run_root / f"rank_{rank:03d}"
        self.training_root = self.rank_root / "train"
        self.validation_root = run_root / "rank_000" / "validation"
        self.teacher = None
        self.source_config = copy.deepcopy(dict(source["source_config"]))
        self.teacher_config = copy.deepcopy(
            dict(source["model_metadata"]["teacher_config"])
        )

    def load_teacher(self):
        if self.teacher is not None:
            return self.teacher
        from ..teacher import load_teacher

        self.teacher = load_teacher(
            self.source_config,
            resolved_config=self.teacher_config,
            device=str(self.device),
        )
        actual_hash = getattr(self.teacher, "condition_b_checkpoint_sha256", None)
        if actual_hash != self.source["teacher_checkpoint_sha256"]:
            raise ValueError(
                "loaded teacher checkpoint SHA-256 disagrees with the frozen JEPA source metadata"
            )
        return self.teacher

    def _expected_seed(self, *, split: str, round_idx: int, source_rank: int) -> int:
        seed = int(self.config["training"]["seed"])
        if split == "train":
            return seed + round_idx * self.world_size + source_rank
        return seed + int(self.plan["round_count"]) * self.world_size + self.world_size

    def _collection_config(
        self,
        *,
        split: str,
        round_idx: int,
        transitions: int,
        source_rank: int | None = None,
    ) -> tuple[dict[str, Any], dict[str, Any]]:
        collection_config = copy.deepcopy(self.source_config)
        metadata_teacher_config = copy.deepcopy(self.teacher_config)
        collection = collection_config["collection"]
        selected_rank = self.rank if source_rank is None else source_rank
        if split == "train":
            env = copy.deepcopy(dict(self.plan["train_env"]))
            vector = copy.deepcopy(dict(self.plan["train_vec"]))
            output_root = self.run_root / f"rank_{selected_rank:03d}" / "train"
        else:
            env = copy.deepcopy(dict(self.plan["validation_env"]))
            vector = copy.deepcopy(dict(self.plan["validation_vec"]))
            output_root = self.validation_root
        split_seed = self._expected_seed(
            split=split, round_idx=round_idx, source_rank=selected_rank
        )
        metadata_teacher_config["env"] = env
        collection_config["teacher_config"] = metadata_teacher_config
        collection_config["vec"] = vector
        collection["split"] = split
        collection["split_seeds"] = {split: split_seed}
        collection["transitions_per_round"] = transitions
        collection["max_transitions"] = transitions
        collection["max_disk_bytes"] = (
            self.plan["rank_disk_budgets"][selected_rank]
            if split == "train"
            else self.plan["max_disk_bytes"]
        )
        collection["action_selection"] = self.plan["action_selection"]
        collection["inference_batch_size"] = self.plan["inference_batch_size"]
        collection["num_collections"] = 1
        collection["output_root"] = str(output_root)
        collection["observation_layout"] = copy.deepcopy(
            self.source["model_metadata"]["observation_layout"]
        )
        return collection_config, metadata_teacher_config

    def _collect(self, *, split: str, round_idx: int, transitions: int) -> Path:
        from ..runtime import create_vecenv
        from ..streaming import collect_streaming_dataset

        collection_config, teacher_config = self._collection_config(
            split=split, round_idx=round_idx, transitions=transitions
        )
        teacher = self.load_teacher()
        env = create_vecenv(teacher_config, collection_config, split)
        try:
            result = collect_streaming_dataset(
                collection_config,
                self.training_root if split == "train" else self.validation_root,
                teacher=teacher,
                env=env,
                collection_round_idx=round_idx,
            )
        finally:
            close = getattr(env, "close", None)
            if callable(close):
                close()
        manifest_path = Path(result["manifest_path"])
        return manifest_path

    def _validate_manifest(
        self,
        manifest_path: Path,
        *,
        round_idx: int,
        split: str,
        source_rank: int | None = None,
        transitions: int | None = None,
    ) -> dict[str, Any]:
        import json

        from .data import _inspect_manifest

        info = _inspect_manifest(
            manifest_path.resolve(strict=True),
            expected_round_idx=round_idx,
            expected_split=split,
        )
        expected = self.source["compatibility"]
        for field in ("observation_dim", "chunk_length", "num_action_classes"):
            compatibility_field = "chunk_length" if field == "chunk_length" else field
            if info[field] != expected[compatibility_field]:
                raise ValueError(f"online {split} collection disagrees with frozen source at {field}")
        for field in ("observation_layout", "action_layout"):
            actual_layout = info[field]
            expected_layout = expected[field]
            if field == "observation_layout":
                actual_layout = _normalized_observation_layout(actual_layout)
                expected_layout = _normalized_observation_layout(expected_layout)
            if actual_layout != expected_layout:
                raise ValueError(f"online {split} collection disagrees with frozen source at {field}")
        if info["teacher_observation_recipe"] != expected["teacher_observation_recipe"]:
            raise ValueError(f"online {split} collection has incompatible teacher observation settings")
        if info.get("teacher_checkpoint_sha256") != self.source["teacher_checkpoint_sha256"]:
            raise ValueError(f"online {split} collection was produced by a different teacher checkpoint")
        selected_rank = self.rank if source_rank is None else source_rank
        expected_transitions = transitions
        if expected_transitions is None:
            expected_transitions = (
                int(self.plan["round_transitions"][round_idx])
                if split == "train"
                else int(self.plan["validation_transitions"])
            )
        if info.get("collection_transition_count") != expected_transitions:
            raise ValueError(f"online {split} collection has an unexpected transition budget")
        if info.get("split_seed") != self._expected_seed(
            split=split, round_idx=round_idx, source_rank=selected_rank
        ):
            raise ValueError(f"online {split} collection has an unexpected split seed")
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        effective_config = manifest.get("effective_config")
        expected_config, _ = self._collection_config(
            split=split,
            round_idx=round_idx,
            transitions=expected_transitions,
            source_rank=selected_rank,
        )
        if effective_config != expected_config:
            raise ValueError(f"online {split} collection config differs from the current run recipe")
        return info

    def ensure_training_manifests(self, round_idx: int, transitions: int) -> tuple[list[str], list[dict[str, Any]]]:
        import torch.distributed as dist
        import torch

        from ..collection_lifecycle import (
            expected_round_manifest_path,
            validate_round_manifest_path,
            validate_active_collection,
        )

        local_error: Exception | None = None
        try:
            validate_active_collection(self.training_root, round_idx)
            expected_manifest = expected_round_manifest_path(self.training_root, round_idx)
            if expected_manifest.exists():
                local_manifest = validate_round_manifest_path(
                    self.training_root, round_idx, expected_manifest
                )
            else:
                if expected_manifest.parent.exists():
                    raise RuntimeError(
                        f"incomplete online collection cannot be reused: {expected_manifest.parent}"
                    )
                local_manifest = self._collect(
                    split="train", round_idx=round_idx, transitions=transitions
                )
                local_manifest = validate_round_manifest_path(
                    self.training_root, round_idx, local_manifest
                )
            local_info = self._validate_manifest(
                local_manifest,
                round_idx=round_idx,
                split="train",
                source_rank=self.rank,
                transitions=transitions,
            )
            if local_info["valid_window_count"] < 1:
                raise ValueError(f"online training round {round_idx} contains no valid windows")
        except Exception as exc:
            local_error = exc

        if self.world_size > 1 and dist.is_initialized():
            failure = torch.tensor(
                int(local_error is not None), dtype=torch.int32, device=self.device
            )
            dist.all_reduce(failure, op=dist.ReduceOp.MAX)
            failed = bool(failure.item())
        else:
            failed = local_error is not None
        if failed:
            if local_error is not None:
                raise RuntimeError(
                    f"rank {self.rank} could not collect/validate round {round_idx}: {local_error}"
                ) from local_error
            raise RuntimeError(f"another rank could not collect/validate round {round_idx}")
        if self.world_size > 1 and dist.is_initialized():
            dist.barrier()

        manifest_paths: list[str] = []
        infos: list[dict[str, Any]] = []
        validation_error: Exception | None = None
        try:
            for source_rank in range(self.world_size):
                root = self.run_root / f"rank_{source_rank:03d}" / "train"
                expected_manifest = expected_round_manifest_path(root, round_idx)
                safe_manifest = validate_round_manifest_path(
                    root, round_idx, expected_manifest
                )
                info = self._validate_manifest(
                    safe_manifest,
                    round_idx=round_idx,
                    split="train",
                    source_rank=source_rank,
                    transitions=transitions,
                )
                manifest_paths.append(str(safe_manifest))
                infos.append(info)
        except Exception as exc:
            validation_error = exc
        if self.world_size > 1 and dist.is_initialized():
            failure = torch.tensor(
                int(validation_error is not None), dtype=torch.int32, device=self.device
            )
            dist.all_reduce(failure, op=dist.ReduceOp.MAX)
            failed = bool(failure.item())
        else:
            failed = validation_error is not None
        if failed:
            if validation_error is not None:
                raise RuntimeError(
                    f"rank {self.rank} could not validate pooled round {round_idx}: {validation_error}"
                ) from validation_error
            raise RuntimeError(f"another rank could not validate pooled round {round_idx}")

        dataset_error: Exception | None = None
        dataset = None
        try:
            from .data import ProbeCollectionDataset

            dataset = ProbeCollectionDataset(manifest_paths, expected_split="train")
            if len(dataset) != sum(info["valid_window_count"] for info in infos):
                raise ValueError(f"pooled window count changed for online round {round_idx}")
        except Exception as exc:
            dataset_error = exc
        finally:
            if dataset is not None:
                dataset.close()
        if self.world_size > 1 and dist.is_initialized():
            failure = torch.tensor(
                int(dataset_error is not None), dtype=torch.int32, device=self.device
            )
            dist.all_reduce(failure, op=dist.ReduceOp.MAX)
            failed = bool(failure.item())
        else:
            failed = dataset_error is not None
        if failed:
            if dataset_error is not None:
                raise RuntimeError(
                    f"rank {self.rank} could not open pooled round {round_idx}: {dataset_error}"
                ) from dataset_error
            raise RuntimeError(f"another rank could not open pooled round {round_idx}")
        return manifest_paths, infos

    def collect_heldout(self, *, split: str = "validation") -> str:
        """Rank 0 creates a persistent held-out split exactly once."""
        if split != "validation":
            raise ValueError("online collection only creates the validation split")
        from ..collection_lifecycle import (
            expected_round_manifest_path,
            validate_round_manifest_path,
            validate_active_collection,
        )

        validate_active_collection(self.validation_root, 0)
        manifest_path = expected_round_manifest_path(self.validation_root, 0)
        if manifest_path.exists():
            manifest_path = validate_round_manifest_path(
                self.validation_root, 0, manifest_path
            )
            self._validate_manifest(manifest_path, round_idx=0, split="validation")
            return str(manifest_path)
        if manifest_path.parent.exists():
            raise RuntimeError(f"incomplete online validation collection cannot be reused: {manifest_path.parent}")
        collected = self._collect(
            split="validation",
            round_idx=0,
            transitions=int(self.plan["validation_transitions"]),
        )
        collected = validate_round_manifest_path(self.validation_root, 0, collected)
        info = self._validate_manifest(collected, round_idx=0, split="validation")
        if info["valid_window_count"] < 1:
            raise ValueError("online validation collection contains no valid windows")
        return str(collected)

    def open_training_dataset(self, manifest_paths: list[str]):
        from .data import ProbeCollectionDataset

        return ProbeCollectionDataset(manifest_paths, expected_split="train")

    def remove_training_collection(self, round_idx: int) -> bool:
        from ..collection_lifecycle import remove_completed_collection

        return remove_completed_collection(self.training_root, round_idx)

    def validate_active(self, active_round_idx: int | None) -> None:
        from ..collection_lifecycle import validate_active_collection

        validate_active_collection(self.training_root, active_round_idx)

    def fingerprints(self, manifest_paths: list[str]) -> list[dict[str, str]]:
        results = []
        for manifest_text in manifest_paths:
            path = Path(manifest_text).resolve(strict=True)
            results.append({"path": str(path), "sha256": _file_sha256(path)})
        return results
