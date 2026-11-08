"""Isolate replay gathering/transfers; synthetic data, no network or simulator."""
import argparse
import json
import os
import socket
import statistics
import subprocess
import time
from pathlib import Path

import torch

from pufferlib.sac_replay import TensorReplay


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--capacity", type=int, default=2048000)
    parser.add_argument("--observation-width", type=int, default=923)
    parser.add_argument("--microbatch", type=int, default=32000)
    parser.add_argument("--repeats", type=int, default=12)
    parser.add_argument("--threads", type=int, default=4)
    parser.add_argument("--output", type=Path, required=True)
    options = parser.parse_args()
    if options.output.exists():
        raise FileExistsError(options.output)
    torch.set_num_threads(options.threads)
    torch.manual_seed(42)
    replay = TensorReplay(options.capacity, (options.observation_width,), pin_memory=True)
    # Touch every page before timing; the contents are deliberately synthetic.
    for storage in (replay.observations, replay.next_observations, replay.actions,
                    replay.rewards, replay.discounts):
        storage.fill_(0)
    indices = torch.randperm(options.capacity)[:options.microbatch]
    results = {}
    # Warm the CUDA context and kernels before collecting timings.
    torch.zeros(1, device="cuda")
    for variant in ("stock_actor_all_fields", "actor_observations_only", "critic_all_fields"):
        measurements = []
        for repeat_idx in range(options.repeats + 3):
            torch.cuda.synchronize()
            started = time.perf_counter()
            if variant == "actor_observations_only":
                batch = torch.empty((options.microbatch, options.observation_width), pin_memory=True)
                torch.index_select(replay.observations, 0, indices, out=batch)
                fields = (batch,)
            else:
                fields = replay.get_batch(indices)
            gathered = time.perf_counter()
            selected = fields if variant == "critic_all_fields" else fields[:1]
            device_fields = tuple(field.to("cuda", non_blocking=True) for field in selected)
            torch.cuda.synchronize()
            finished = time.perf_counter()
            if repeat_idx >= 3:
                measurements.append({"gather_seconds": gathered - started,
                                     "transfer_seconds": finished - gathered,
                                     "total_seconds": finished - started})
            del device_fields, fields
        results[variant] = {
            "samples": measurements,
            "median": {key: statistics.median(row[key] for row in measurements)
                       for key in measurements[0]},
        }
    report = {
        "kind": "synthetic_replay_microbenchmark_not_end_to_end",
        "options": {key: str(value) if isinstance(value, Path) else value
                    for key, value in vars(options).items()},
        "torch_version": torch.__version__, "gpu": torch.cuda.get_device_name(),
        "host": socket.gethostname(), "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
        "git_commit": subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip(),
        "results": results,
        "limitations": "Fixed random indices reused; CPU cache and allocation reuse may help. "
                        "Variant order is fixed; repeat in independent processes. No network overlap.",
    }
    options.output.parent.mkdir(parents=True, exist_ok=True)
    options.output.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps({key: value["median"] for key, value in results.items()}, indent=2))


if __name__ == "__main__":
    main()
