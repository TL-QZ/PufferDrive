"""Profile one real-data observation-probe DDP update under ``torchrun``.

Each invocation measures one batch/microbatch candidate. Run a fresh process for
each candidate so an OOM exits that candidate without contaminating the next.
The measured update calls the same ``train_update`` function as normal training;
it includes window reads, transfer, Hungarian matching, backward, and AdamW.
"""
from __future__ import annotations

import argparse
import copy
import json
import os
import statistics
import sys
import time
from pathlib import Path
from typing import Any, Sequence

import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel

from .config import DEFAULT_CONFIG, load_probe_config, validate_probe_config
from .data import ProbeCollectionDataset, selected_indices
from .model import ObservationDecoder, load_frozen_jepa
from .train import _epoch_batch_indices, _setup_runtime, train_update
from .train import preflight_probe
from ..runtime import resolve_path


def _positive_int(value: str) -> int:
    parsed = int(value)
    if parsed < 1:
        raise argparse.ArgumentTypeError("must be a positive integer")
    return parsed


def _atomic_write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path = path.with_suffix(path.suffix + ".tmp")
    temporary_path.write_text(
        json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    temporary_path.replace(path)


def _rank_batch_indices(
    order: Sequence[int], *, update_index: int, global_batch_size: int,
    local_batch_size: int, rank: int, repeat_batch: bool,
) -> list[int]:
    if repeat_batch:
        update_index = 0
    global_start = update_index * global_batch_size
    return _epoch_batch_indices(
        order,
        len(order),
        global_start + rank * local_batch_size,
        global_start + (rank + 1) * local_batch_size,
    )


def _gather_float(value: float, *, device: torch.device, world_size: int) -> list[float]:
    if world_size == 1:
        return [value]
    local = torch.tensor([value], dtype=torch.float64, device=device)
    gathered = [torch.empty_like(local) for _ in range(world_size)]
    dist.all_gather(gathered, local)
    return [float(item.item()) for item in gathered]


def _gather_rank_info(info: dict[str, Any], *, world_size: int) -> list[dict[str, Any]]:
    if world_size == 1:
        return [info]
    gathered: list[dict[str, Any] | None] = [None] * world_size
    dist.all_gather_object(gathered, info)
    return [item for item in gathered if item is not None]


def _build_config(arguments: argparse.Namespace) -> dict[str, Any]:
    config = load_probe_config(arguments.config, arguments.set)
    if config["data"].get("mode") != "cached":
        raise ValueError(
            "the throughput profiler reads an existing collection; set data.mode=cached "
            "and select its rounds/ranks explicitly"
        )
    if arguments.checkpoint is not None:
        config["checkpoint"] = arguments.checkpoint
    if not config.get("checkpoint"):
        raise ValueError("set checkpoint in the config or pass --checkpoint")

    selected_rounds = list(config["data"]["collection_rounds"])
    collection_round = (
        selected_rounds[0]
        if arguments.collection_round is None
        else arguments.collection_round
    )
    if collection_round not in selected_rounds:
        raise ValueError(
            f"collection round {collection_round} is not selected by data.collection_rounds"
        )
    config["data"]["collection_rounds"] = [collection_round]

    training = config["training"]
    training["batch_size"] = arguments.batch_size
    training["microbatch_size"] = arguments.microbatch_size
    training["world_size"] = arguments.world_size
    training["device"] = "cuda"
    if arguments.learning_rate is not None:
        training["learning_rate"] = arguments.learning_rate
    elif training.get("learning_rate") is None:
        # The decoder/optimizer has no selected learning-rate policy yet. This
        # bounded throughput probe uses a finite placeholder and changes no run config.
        training["learning_rate"] = 1e-3

    validate_probe_config(config)
    per_rank_batch_size = arguments.batch_size // arguments.world_size
    if arguments.batch_size % arguments.world_size:
        raise ValueError("--batch-size must divide evenly across --world-size")
    if arguments.microbatch_size > per_rank_batch_size:
        raise ValueError("--microbatch-size cannot exceed the per-rank batch size")
    return config


def _run(arguments: argparse.Namespace) -> dict[str, Any] | None:
    config = _build_config(arguments)
    torch, rank, world_size, local_rank, device, owns_process_group = _setup_runtime(config)
    dataset: ProbeCollectionDataset | None = None
    try:
        if device.type != "cuda":
            raise RuntimeError("the observation-probe profile requires CUDA")
        if world_size != arguments.world_size:
            raise ValueError(
                f"launched world size {world_size} does not match --world-size {arguments.world_size}"
            )
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA is unavailable")

        report = preflight_probe(config)
        data_report = report["data"]
        checkpoint_report = report["checkpoint"]
        if checkpoint_report.get("metadata_compatible") is not True:
            raise ValueError("selected checkpoint was not validated against the real collection")
        round_row = data_report["per_round"][0]
        manifest_paths = list(round_row["manifest_paths"])
        runtime_config = copy.deepcopy(config)
        runtime_config["_observation_layout"] = dict(
            data_report["compatibility"]["observation_layout"]
        )

        jepa = load_frozen_jepa(
            str(resolve_path(config["checkpoint"]).resolve(strict=True)), device=str(device)
        )
        decoder = ObservationDecoder(
            int(jepa.latent_dim),
            runtime_config["_observation_layout"],
            hidden_sizes=list(config["decoder"]["hidden_sizes"]),
        ).to(device)
        optimizer = torch.optim.AdamW(
            decoder.parameters(),
            lr=float(config["training"]["learning_rate"]),
            weight_decay=float(config["training"].get("weight_decay", 0.0)),
        )
        distributed_decoder = DistributedDataParallel(
            decoder,
            device_ids=[local_rank],
            broadcast_buffers=False,
        ) if world_size > 1 else None

        dataset = ProbeCollectionDataset(manifest_paths, expected_split="train")
        if len(dataset) < 1:
            raise ValueError("selected training collection contains no valid windows")
        index_order = selected_indices(
            len(dataset),
            seed=int(config["training"]["seed"]),
            epoch_index=0,
            collection_index=int(round_row["collection_round_idx"]),
            max_windows=config["data"].get("max_windows_per_collection"),
        )
        if len(dataset) != int(round_row["pooled_valid_window_count"]):
            raise ValueError("pooled window count changed after probe preflight")
        if len(index_order) < 1:
            raise ValueError("selected training collection contains no selected windows")
        global_batch_size = int(config["training"]["batch_size"])
        local_batch_size = global_batch_size // world_size
        effective_batch_mode = "repeat" if arguments.repeat_batch else "new"
        local_rank_info = {
            "rank": rank,
            "local_rank": local_rank,
            "device_name": torch.cuda.get_device_name(device),
            "device_total_memory_bytes": int(
                torch.cuda.get_device_properties(device).total_memory
            ),
        }
        rank_info = _gather_rank_info(local_rank_info, world_size=world_size)
        measured_updates: list[dict[str, Any]] = []
        warmup_losses: list[float] = []
        total_updates = arguments.warmup_steps + arguments.steps
        required_windows = global_batch_size * (1 if arguments.repeat_batch else total_updates)
        if required_windows > len(index_order):
            raise ValueError(
                f"profiling requires {required_windows} distinct selected windows, "
                f"but only {len(index_order)} are selected; remove the toy cap with "
                "--set data.max_windows_per_collection=null or explicitly use --repeat-batch "
                "with at least one full global batch available"
            )

        for update_index in range(total_updates):
            local_indices = _rank_batch_indices(
                index_order,
                update_index=update_index,
                global_batch_size=global_batch_size,
                local_batch_size=local_batch_size,
                rank=rank,
                repeat_batch=arguments.repeat_batch,
            )
            if world_size > 1:
                dist.barrier()
            torch.cuda.synchronize(device)
            torch.cuda.reset_peak_memory_stats(device)
            started = time.perf_counter()
            metrics = train_update(
                decoder,
                jepa,
                dataset,
                local_indices,
                optimizer,
                runtime_config,
                device,
                world_size=world_size,
                distributed_decoder=distributed_decoder,
            )
            torch.cuda.synchronize(device)
            elapsed_seconds = time.perf_counter() - started
            rank_elapsed_seconds = _gather_float(
                elapsed_seconds, device=device, world_size=world_size
            )
            peak_allocated_bytes = int(torch.cuda.max_memory_allocated(device))
            peak_reserved_bytes = int(torch.cuda.max_memory_reserved(device))
            rank_memory = _gather_rank_info(
                {
                    "rank": rank,
                    "peak_allocated_bytes": peak_allocated_bytes,
                    "peak_reserved_bytes": peak_reserved_bytes,
                },
                world_size=world_size,
            )
            if update_index < arguments.warmup_steps:
                warmup_losses.append(float(metrics["loss_total"]))
                continue

            elapsed_max_rank_seconds = max(rank_elapsed_seconds)
            measured_updates.append(
                {
                    "update": update_index - arguments.warmup_steps,
                    "max_rank_seconds": elapsed_max_rank_seconds,
                    "global_windows_per_second": global_batch_size / elapsed_max_rank_seconds,
                    "loss_total": float(metrics["loss_total"]),
                    "peak_allocated_bytes_by_rank": {
                        str(item["rank"]): item["peak_allocated_bytes"] for item in rank_memory
                    },
                    "peak_reserved_bytes_by_rank": {
                        str(item["rank"]): item["peak_reserved_bytes"] for item in rank_memory
                    },
                }
            )

        if world_size > 1:
            dist.barrier()
        if rank != 0:
            return None

        mean_seconds = sum(item["max_rank_seconds"] for item in measured_updates) / len(measured_updates)
        peak_allocated_by_rank = {
            str(item["rank"]): max(
                update["peak_allocated_bytes_by_rank"][str(item["rank"])]
                for update in measured_updates
            )
            for item in rank_info
        }
        peak_reserved_by_rank = {
            str(item["rank"]): max(
                update["peak_reserved_bytes_by_rank"][str(item["rank"])]
                for update in measured_updates
            )
            for item in rank_info
        }
        return {
            "status": "passed",
            "profile": "observation_probe.train.train_update",
            "checkpoint": checkpoint_report["path"],
            "collection_round_idx": int(round_row["collection_round_idx"]),
            "manifest_paths": manifest_paths,
            "dataset_windows": len(dataset),
            "selected_training_windows": len(index_order),
            "window_selection": {
                "mode": effective_batch_mode,
                "sampler": "deterministic shuffled windows from selected real collection",
                "rank_windows_disjoint_within_update": len(index_order) >= global_batch_size,
                "repeat_note": (
                    "the same exact per-rank indices are used for every warmup and measured update"
                    if arguments.repeat_batch
                    else "each update advances through new indices; the deterministic order wraps only if needed"
                ),
            },
            "candidate": {
                "global_batch_size": global_batch_size,
                "per_rank_batch_size": local_batch_size,
                "microbatch_size_per_rank": int(config["training"]["microbatch_size"]),
                "microbatches_per_rank": (local_batch_size + arguments.microbatch_size - 1)
                // arguments.microbatch_size,
                "world_size": world_size,
                "warmup_steps": arguments.warmup_steps,
                "timed_steps": arguments.steps,
                "learning_rate_used_for_profile_only": float(config["training"]["learning_rate"]),
            },
            "measurement_scope": (
                "Includes ProbeCollectionDataset.get_batch reads, CPU-to-GPU transfer, "
                "frozen JEPA target encoding, decoder forward, Hungarian matching, "
                "reconstruction loss, backward, gradient clipping, AdamW, and DDP reductions. "
                "Excludes process/checkpoint startup, evaluation, and checkpoint writing."
            ),
            "timing": {
                "synchronization": "CUDA synchronized before and after each update; report uses the slowest rank",
                "mean_max_rank_seconds_per_update": mean_seconds,
                "median_max_rank_seconds_per_update": statistics.median(
                    item["max_rank_seconds"] for item in measured_updates
                ),
                "mean_global_windows_per_second": global_batch_size / mean_seconds,
                "updates": measured_updates,
            },
            "gpu_by_rank": rank_info,
            "memory": {
                "peak_allocated_bytes_by_rank": peak_allocated_by_rank,
                "peak_reserved_bytes_by_rank": peak_reserved_by_rank,
            },
            "warmup_loss_total": warmup_losses,
        }
    except torch.cuda.OutOfMemoryError as error:
        raise RuntimeError(
            "OOM candidate: "
            f"global_batch_size={arguments.batch_size}, "
            f"microbatch_size_per_rank={arguments.microbatch_size}, "
            f"world_size={arguments.world_size}; run each candidate in a fresh torchrun process"
        ) from error
    finally:
        if dataset is not None:
            dataset.close()
        if owns_process_group and dist.is_initialized():
            dist.destroy_process_group()


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default=DEFAULT_CONFIG)
    parser.add_argument("--set", action="append", default=[], metavar="KEY=VALUE")
    parser.add_argument("--checkpoint", help="Condition B checkpoint; overrides config.checkpoint")
    parser.add_argument("--collection-round", type=int, help="Selected training collection round (default: first configured round)")
    parser.add_argument("--batch-size", required=True, type=_positive_int, help="Global windows per update")
    parser.add_argument("--microbatch-size", required=True, type=_positive_int, help="Windows per forward/backward pass on each GPU")
    parser.add_argument("--world-size", type=_positive_int, default=2)
    parser.add_argument("--learning-rate", type=float, help="Profile optimizer learning rate; defaults to config, then 1e-3")
    parser.add_argument("--warmup-steps", type=int, default=1)
    parser.add_argument("--steps", type=_positive_int, default=3, help="Timed optimizer updates")
    parser.add_argument("--repeat-batch", action="store_true", help="Reuse exactly the same indexed windows for all updates")
    parser.add_argument("--output", required=True, help="Destination JSON path; must be unique per candidate")
    arguments = parser.parse_args(argv)
    if arguments.warmup_steps < 0:
        parser.error("--warmup-steps must be non-negative")
    if arguments.learning_rate is not None and arguments.learning_rate <= 0:
        parser.error("--learning-rate must be positive")

    output_path = resolve_path(arguments.output).resolve()
    if output_path.exists():
        parser.error(f"output JSON already exists: {output_path}")
    try:
        result = _run(arguments)
        if result is not None:
            _atomic_write_json(output_path, result)
            print(json.dumps(result, sort_keys=True, allow_nan=False), flush=True)
    except (ValueError, OSError, RuntimeError, FloatingPointError, NotImplementedError) as error:
        print(
            f"Observation probe profile candidate failed "
            f"(global_batch_size={arguments.batch_size}, "
            f"microbatch_size_per_rank={arguments.microbatch_size}, "
            f"world_size={arguments.world_size}): {error}",
            file=sys.stderr,
            flush=True,
        )
        raise


if __name__ == "__main__":
    main()
