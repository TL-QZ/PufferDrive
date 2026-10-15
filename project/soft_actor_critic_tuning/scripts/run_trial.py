#!/usr/bin/env python3
"""Run one SAC training process for a torchrun-launched grid trial."""

import argparse
import json
import math
import numbers
import os
from pathlib import Path
import time

import yaml


def _json_value(value):
    if isinstance(value, dict):
        return {str(key): _json_value(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_value(item) for item in value]
    if isinstance(value, bool) or value is None or isinstance(value, (str, int)):
        return value
    if isinstance(value, numbers.Real):
        number = float(value)
        return number if math.isfinite(number) else str(number)
    if isinstance(value, float):
        return value
    if hasattr(value, "item"):
        return _json_value(value.item())
    return str(value)


def _write_json(path, data):
    temporary_path = path.with_name(path.name + ".tmp")
    with temporary_path.open("w", encoding="utf-8") as output:
        json.dump(_json_value(data), output, indent=2, sort_keys=True, allow_nan=False)
        output.write("\n")
    temporary_path.replace(path)


def _positive_finite_hours(value):
    if isinstance(value, bool):
        raise argparse.ArgumentTypeError("trial hours must be finite and positive")
    try:
        hours = float(value)
    except (TypeError, ValueError) as error:
        raise argparse.ArgumentTypeError("trial hours must be finite and positive") from error
    if not math.isfinite(hours) or hours <= 0 or not math.isfinite(hours * 3600):
        raise argparse.ArgumentTypeError("trial hours must be finite and positive")
    return hours


def _build_parser():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    parser.add_argument("--artifact-dir", required=True)
    parser.add_argument("--trial-hours", type=_positive_finite_hours)
    return parser


def _resolve_deadline_monotonic(trial_hours):
    if trial_hours is None:
        return None
    deadline_value = os.environ.get("SAC16_TRIAL_DEADLINE_MONOTONIC")
    if deadline_value is None:
        return time.monotonic() + trial_hours * 3600
    try:
        deadline_monotonic = float(deadline_value)
    except ValueError as error:
        raise ValueError("SAC16_TRIAL_DEADLINE_MONOTONIC must be finite") from error
    if not math.isfinite(deadline_monotonic):
        raise ValueError("SAC16_TRIAL_DEADLINE_MONOTONIC must be finite")
    return deadline_monotonic


class TrialMonitor:
    def __init__(self, metrics_path, stop_path, deadline_monotonic):
        self.metrics_path = metrics_path
        self.stop_path = stop_path
        self.deadline_monotonic = deadline_monotonic
        self.last_cycle_finished = time.monotonic()
        self.stop_reason = None
        self.metrics_path.touch(exist_ok=False)

    def __call__(self, logs):
        record = _json_value(logs)
        with self.metrics_path.open("a", encoding="utf-8") as output:
            output.write(json.dumps(record, sort_keys=True, allow_nan=False) + "\n")
            output.flush()
            os.fsync(output.fileno())

        for name, value in logs.items():
            if "loss" not in name.lower() or not isinstance(value, numbers.Real):
                continue
            if not math.isfinite(float(value)):
                self.stop_reason = "nonfinite_learner_loss"
                _write_json(self.stop_path, {"stop_reason": self.stop_reason, "metric": name})
                return True

        if self.deadline_monotonic is None:
            return False

        now = time.monotonic()
        cycle_seconds = now - self.last_cycle_finished
        self.last_cycle_finished = now
        if now >= self.deadline_monotonic or now + cycle_seconds >= self.deadline_monotonic:
            # A full replay pass cannot stop halfway; leave time for final artifacts.
            self.stop_reason = "deadline" if now >= self.deadline_monotonic else "budget_before_next_cycle"
            _write_json(self.stop_path, {
                "stop_reason": self.stop_reason,
                "deadline_monotonic": self.deadline_monotonic,
                "last_cycle_seconds": cycle_seconds,
            })
            return True
        return False


def _runtime_record(torch, rank, local_rank, device_name, elapsed_seconds, error=None):
    record = {
        "rank": rank,
        "local_rank": local_rank,
        "device_name": device_name,
        "elapsed_seconds": elapsed_seconds,
        "peak_allocated_bytes": None,
        "peak_reserved_bytes": None,
        "error": error,
    }
    if device_name is not None:
        device_index = torch.cuda.current_device()
        torch.cuda.synchronize(device_index)
        record["cuda_device_index"] = device_index
        record["cuda_visible_id"] = os.environ.get("CUDA_VISIBLE_DEVICES", "").split(",")[local_rank]
        record["peak_allocated_bytes"] = int(torch.cuda.max_memory_allocated(device_index))
        record["peak_reserved_bytes"] = int(torch.cuda.max_memory_reserved(device_index))
    return record


def main(argv=None):
    args = _build_parser().parse_args(argv)

    started = time.monotonic()
    artifact_dir = Path(args.artifact_dir)
    artifact_dir.mkdir(parents=True, exist_ok=True)
    rank = int(os.environ.get("RANK", "0"))
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    deadline_monotonic = _resolve_deadline_monotonic(args.trial_hours)
    with Path(args.config).open(encoding="utf-8") as source:
        config = yaml.safe_load(source)

    import torch
    from pufferlib import pufferl_sac

    device_name = None
    if str(config["train"].get("device", "cpu")).startswith("cuda"):
        torch.cuda.set_device(local_rank)
        torch.cuda.reset_peak_memory_stats(local_rank)
        device_name = torch.cuda.get_device_name(local_rank)

    metrics_path = artifact_dir / "metrics.jsonl"
    stop_path = artifact_dir / "stop_reason.json"
    monitor = TrialMonitor(metrics_path, stop_path, deadline_monotonic) if rank == 0 else None
    error_text = None
    try:
        pufferl_sac.train(
            env_name=config["env_name"],
            args=config,
            early_stop_fn=monitor,
        )
    except BaseException as error:
        error_text = f"{type(error).__name__}: {error}"
        raise
    finally:
        runtime = _runtime_record(
            torch,
            rank,
            local_rank,
            device_name,
            time.monotonic() - started,
            error=error_text,
        )
        runtime_path = artifact_dir / f"rank_{rank}_runtime.json"
        _write_json(runtime_path, runtime)

    if rank == 0 and monitor.stop_reason == "nonfinite_learner_loss":
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
