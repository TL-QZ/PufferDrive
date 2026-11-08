"""Run a bounded sequential benchmark matrix; invoke with CUDA_VISIBLE_DEVICES."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import time


ROOT = Path(__file__).resolve().parent
CASES = {
    "ppo_filter": ["--algorithm", "ppo"],
    "ppo_all": ["--algorithm", "ppo", "--ppo-filter", "off"],
    "sac_stock": ["--algorithm", "sac"],
    "sac_obs_only": ["--algorithm", "sac", "--variant", "obs_only"],
    "sac_cached": ["--algorithm", "sac", "--variant", "cached"],
    "sac_threads4": ["--algorithm", "sac", "--threads", "4"],
    "sac_cached_threads4": ["--algorithm", "sac", "--variant", "cached", "--threads", "4"],
    "sac_gpu": ["--algorithm", "sac", "--replay-location", "gpu"],
    "sac_obs_only_gpu": ["--algorithm", "sac", "--variant", "obs_only", "--replay-location", "gpu"],
    "sac_profile": ["--algorithm", "sac", "--stock-profiler", "on"],
    "sac_fixed": ["--algorithm", "sac", "--variant", "fixed_targets", "--target-audit-rows", "512"],
    "sac_fixed_cached": ["--algorithm", "sac", "--variant", "fixed_targets_cached", "--target-audit-rows", "512"],
    "sac_fixed_gpu": ["--algorithm", "sac", "--variant", "fixed_targets", "--replay-location", "gpu", "--target-audit-rows", "512"],
    "sac_fixed_both_gpu": ["--algorithm", "sac", "--variant", "fixed_targets_actor_q", "--replay-location", "gpu", "--target-audit-rows", "512"],
}


def stop_owned_group(process):
    """Stop the session created for this case, including surviving vector workers."""
    try:
        os.killpg(process.pid, signal.SIGTERM)
    except ProcessLookupError:
        process.wait()
        return
    try:
        process.wait(timeout=10)
    except subprocess.TimeoutExpired:
        pass
    # The leader may exit before workers; signal the owned group either way.
    try:
        os.killpg(process.pid, signal.SIGKILL)
    except ProcessLookupError:
        pass
    process.wait()


def interrupt_matrix(_signum, _frame):
    raise KeyboardInterrupt("matrix interrupted")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--cases", nargs="+", choices=CASES,
                        default=[case for case in CASES if not case.startswith("sac_fixed")])
    parser.add_argument("--timeout-seconds", type=int, default=1800)
    parser.add_argument("--production", action="store_true")
    parser.add_argument("--replay-capacity", type=int,
                        help="Override SAC replay rows without changing simulator geometry")
    parser.add_argument("--ranks", type=int, choices=(1, 2), default=1)
    parser.add_argument("--compile", choices=("on", "off"), default="on")
    parser.add_argument("--warmup", type=int, default=2)
    parser.add_argument("--cycles", type=int, default=3)
    parser.add_argument("--ppo-weights", type=Path)
    parser.add_argument("--sac-weights", type=Path)
    parser.add_argument("--trace-cycle", type=int)
    options = parser.parse_args()
    signal.signal(signal.SIGTERM, interrupt_matrix)
    output = options.output_dir.resolve()
    output.mkdir(parents=True, exist_ok=False)
    if options.ranks > 1 and "ppo_all" in options.cases:
        raise ValueError("Unfiltered PPO DDP is excluded: masked ranks may have unequal backward counts")
    replay_capacity = options.replay_capacity
    if replay_capacity is None:
        replay_capacity = 2048000 if options.production else 512000
    common = ["--num-envs", "20" if options.production else "4",
              "--agents", "3200" if options.production else "1000",
              "--horizon", "128", "--replay-capacity", str(replay_capacity),
              "--logical-batch", "128000", "--microbatch", "32000",
              "--update-epochs", "3", "--threads", "1", "--compile", options.compile,
              "--warmup", str(options.warmup), "--cycles", str(options.cycles)]
    if options.trace_cycle is not None:
        common += ["--trace-cycle", str(options.trace_cycle)]
    records = []
    for case in options.cases:
        prefix = [sys.executable]
        if options.ranks > 1:
            prefix += ["-m", "torch.distributed.run", "--standalone", "--nproc-per-node=2", "--max-restarts=0"]
        command = prefix + [str(ROOT / "profile_training.py"), "--output-dir", str(output / case)] + common + CASES[case]
        weights = options.ppo_weights if case.startswith("ppo") else options.sac_weights
        if weights is not None:
            command += ["--weights-from", str(weights.resolve())]
        started = time.time()
        print(f"Starting {case}: {json.dumps(command)}", flush=True)
        with (output / f"{case}.log").open("w") as log:
            process = subprocess.Popen(command, stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
            state = {"case": case, "pid": process.pid, "command": command,
                     "started_unix_seconds": started, "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
                     "source_sha256": {name: hashlib.sha256((ROOT / name).read_bytes()).hexdigest()
                                       for name in ("profile_training.py", "sac_variants.py", "run_matrix.py")}}
            (output / "active.json").write_text(json.dumps(state, indent=2) + "\n")
            try:
                returncode = process.wait(timeout=options.timeout_seconds)
            except subprocess.TimeoutExpired:
                stop_owned_group(process)
                returncode = 124
            except BaseException:
                stop_owned_group(process)
                raise
            if returncode:
                stop_owned_group(process)
        state.update(returncode=returncode, wall_seconds=time.time() - started)
        records.append(state)
        (output / "matrix.json").write_text(json.dumps(records, indent=2) + "\n")
        print(f"Finished {case}: exit={returncode}, seconds={state['wall_seconds']:.1f}", flush=True)
        if returncode:
            raise SystemExit(returncode)
    (output / "active.json").unlink()


if __name__ == "__main__":
    main()
