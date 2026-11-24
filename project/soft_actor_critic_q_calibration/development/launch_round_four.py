"""Collect a round-four checkpoint, expanding to all workers after validating a pilot state."""
import argparse
import json
import os
from pathlib import Path
import subprocess
import sys
import time

from .run_layout import PROJECT_DIR, initialize_model_directory, model_directory

PACKAGE = "project.soft_actor_critic_q_calibration.development"
REPOSITORY = PROJECT_DIR.parents[1]


def launch_worker(model, source, records, worker, count, gpu):
    module = "collect_ppo_probes" if model.name == "ppo" else "collect_binned_probes"
    mode_args = ["--round-four"] if model.name == "ppo" else [
        "--repeats", "4", "--record-horizons", "8", "32", "128", "1280", "2560",
    ]
    command = [sys.executable, "-u", "-m", f"{PACKAGE}.{module}", *mode_args,
               "--model-dir", str(model), "--source-model-dir", str(source), "--device", "cuda:0",
               "--workers", str(count), "--worker-index", str(worker)]
    environment = {**os.environ, "CUDA_VISIBLE_DEVICES": gpu, "OMP_NUM_THREADS": "1",
                   "MKL_NUM_THREADS": "1", "OPENBLAS_NUM_THREADS": "1", "NUMPY_MADVISE_HUGEPAGE": "0"}
    with (records / "logs" / f"worker_{worker:02}.log").open("a") as log:
        return subprocess.Popen(command, cwd=REPOSITORY, env=environment, stdout=log, stderr=subprocess.STDOUT)


def collect(model, count, gpus):
    records = model / "records"
    selection = json.loads((records / "selection.json").read_text())
    source = (PROJECT_DIR / selection["source_records_dir"]).resolve().parent
    logs = records / "logs"
    logs.mkdir(exist_ok=True)
    verifier = [sys.executable, "-m", f"{PACKAGE}.verify_round_four", str(model)]
    procs = {0: launch_worker(model, source, records, 0, count, gpus[0])}
    try:
        for _ in range(7200):
            if any((records / "probes").glob("*.json")):
                break
            if procs[0].poll() is not None:
                raise RuntimeError(f"Pilot ended without a saved state. See {logs}/worker_00.log")
            time.sleep(2)
        else:
            raise TimeoutError("No saved pilot state within four hours")
        with (logs / "pilot_verification.log").open("w") as log:
            subprocess.run(verifier + ["--one"], check=True, cwd=REPOSITORY, stdout=log, stderr=subprocess.STDOUT)
        print("Pilot verified; starting all workers.", flush=True)
        for worker in range(1, count):
            procs[worker] = launch_worker(model, source, records, worker, count, gpus[worker % len(gpus)])
        for worker, process in procs.items():
            if process.wait() != 0:
                raise RuntimeError(f"Worker {worker} failed. See {logs}/worker_{worker:02}.log")
        subprocess.run(verifier, check=True, cwd=REPOSITORY)
    finally:
        for process in procs.values():
            if process.poll() is None:
                process.terminate()
        for process in procs.values():
            process.wait()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("model_dir")
    parser.add_argument("--workers", type=int, required=True)
    parser.add_argument("--gpus", nargs="+", required=True)
    args = parser.parse_args()
    if args.workers < 1:
        parser.error("Expected at least one worker")
    collect(initialize_model_directory(model_directory(args.model_dir)), args.workers, args.gpus)


if __name__ == "__main__":
    main()
