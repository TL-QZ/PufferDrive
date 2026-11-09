#!/usr/bin/env python3
"""Run manifest-defined CARLA evaluations with durable per-job status."""

from __future__ import annotations

import argparse
import atexit
import fcntl
from functools import partial
import hashlib
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import time
from datetime import datetime, timezone


PROJECT_DIR = Path(__file__).resolve().parent
REPO_ROOT = PROJECT_DIR.parents[1]
GIB = 1024**3
GPU_IDLE_LIMIT_MIB = 100
LAUNCH_MEMORY_MIN_BYTES = 64 * GIB
STOP_MEMORY_MIN_BYTES = 32 * GIB
MAX_RUNTIME_SECONDS = 7 * 24 * 60 * 60
RESOURCE_SAMPLE_SECONDS = 60
POLL_SECONDS = 5
RETRYABLE_STATES = {"failed", "memory_guard_stopped", "duration_limit_stopped", "interrupted"}


def utc_now():
    return datetime.now(timezone.utc).isoformat()


def print_status(message):
    print(f"[{utc_now()}] {message}", flush=True)


def atomic_json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    temp_path = path.with_name(path.name + f".tmp.{os.getpid()}")
    with temp_path.open("w", encoding="utf-8") as output:
        json.dump(value, output, indent=2, sort_keys=True)
        output.write("\n")
        output.flush()
        os.fsync(output.fileno())
    os.replace(temp_path, path)


def available_memory_bytes():
    with Path("/proc/meminfo").open(encoding="utf-8") as meminfo:
        for line in meminfo:
            if line.startswith("MemAvailable:"):
                return int(line.split()[1]) * 1024
    raise RuntimeError("MemAvailable is missing from /proc/meminfo")


def query_gpu(gpu):
    result = subprocess.run(
        [
            "nvidia-smi",
            "--id",
            str(gpu),
            "--query-gpu=index,memory.used,memory.total,utilization.gpu",
            "--format=csv,noheader,nounits",
        ],
        check=True,
        capture_output=True,
        text=True,
        timeout=15,
    )
    fields = [field.strip() for field in result.stdout.strip().split(",")]
    if len(fields) != 4:
        raise RuntimeError(f"Unexpected nvidia-smi output for GPU {gpu}: {result.stdout.strip()!r}")
    return {
        "gpu": str(gpu),
        "index": fields[0],
        "memory_used_mib": int(fields[1]),
        "memory_total_mib": int(fields[2]),
        "utilization_percent": int(fields[3]),
    }


def process_group_exists(process_group_id):
    if not process_group_id:
        return False
    try:
        os.killpg(int(process_group_id), 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        return True
    return True


def group_memory(process_group_id):
    rss_bytes = 0
    process_count = 0
    for stat_path in Path("/proc").glob("[0-9]*/stat"):
        try:
            stat_text = stat_path.read_text(encoding="utf-8")
            fields_after_name = stat_text[stat_text.rfind(")") + 2 :].split()
            group_id = int(fields_after_name[2])
            if group_id != int(process_group_id):
                continue
            status_text = stat_path.with_name("status").read_text(encoding="utf-8")
            for line in status_text.splitlines():
                if line.startswith("VmRSS:"):
                    rss_bytes += int(line.split()[1]) * 1024
                    process_count += 1
                    break
        except (OSError, ValueError, IndexError):
            continue
    return rss_bytes, process_count


def verify_sha256(path, expected, label):
    if not expected:
        return
    digest = hashlib.sha256()
    with Path(path).open("rb") as source:
        for block in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(block)
    actual = digest.hexdigest()
    if actual.lower() != str(expected).lower():
        raise ValueError(f"{label} SHA-256 mismatch: expected {expected}, got {actual} ({path})")


def read_manifest(path):
    with path.open(encoding="utf-8") as manifest_file:
        manifest = json.load(manifest_file)
    jobs = manifest.get("jobs")
    if not isinstance(jobs, list) or not jobs:
        raise ValueError(f"{path} must contain a non-empty jobs list")

    seen_cases = set()
    for job in jobs:
        missing = {"case", "algorithm", "checkpoint", "catalog", "gpu"} - job.keys()
        if missing:
            raise ValueError(f"Manifest job is missing fields: {', '.join(sorted(missing))}")
        case = str(job["case"])
        if not case or any(character not in "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789_.-" for character in case):
            raise ValueError(f"Invalid case name: {case!r}")
        if case in seen_cases:
            raise ValueError(f"Duplicate case in manifest: {case}")
        seen_cases.add(case)
        algorithm = str(job["algorithm"]).upper()
        if algorithm not in {"SAC", "PPO"}:
            raise ValueError(f"Unsupported algorithm for {case}: {job['algorithm']!r}")
        job["algorithm"] = algorithm
        for field in ("checkpoint", "catalog"):
            value = Path(str(job[field]))
            if not value.is_absolute():
                value = REPO_ROOT / value
            value = value.resolve()
            if not value.is_file():
                raise ValueError(f"{case}: {field} file does not exist: {value}")
            job[field] = str(value)
        verify_sha256(job["checkpoint"], job.get("checkpoint_sha256"), f"{case} checkpoint")
        verify_sha256(job["catalog"], job.get("catalog_sha256"), f"{case} catalog")
        if job.get('failure_replay_csv'):
            replay_csv = Path(job['failure_replay_csv'])
            if not replay_csv.is_file():
                raise ValueError(f'{case}: missing failure replay CSV: {replay_csv}')
            verify_sha256(replay_csv, job.get('failure_replay_csv_sha256'), f'{case} failure CSV')
            verify_sha256(replay_csv.with_name('resolved_benchmark.yaml'),
                          job.get('resolved_benchmark_sha256'), f'{case} resolved benchmark')
        config_path = Path(str(job.get("config") or job.get("config_path") or Path(job["checkpoint"]).with_name("config.yaml")))
        if not config_path.is_absolute():
            config_path = REPO_ROOT / config_path
        if job.get("config_sha256"):
            if not config_path.is_file():
                raise ValueError(f"{case}: config file for SHA-256 check does not exist: {config_path}")
            verify_sha256(config_path, job["config_sha256"], f"{case} config")
        job["gpu"] = str(job["gpu"])
        if not job["gpu"]:
            raise ValueError(f"{case}: gpu must be a non-empty index or UUID")
    catalog_sha256 = manifest.get("catalog_sha256")
    if catalog_sha256:
        for catalog_path in {job["catalog"] for job in jobs}:
            verify_sha256(catalog_path, catalog_sha256, "catalog")
    return jobs


def build_command(job):
    module = "pufferlib.pufferl_sac" if job["algorithm"] == "SAC" else "pufferlib.pufferl"
    output_directory = (PROJECT_DIR / "results" / job["case"]).resolve()
    return [
        sys.executable,
        "-B",
        "-m",
        module,
        "eval",
        "puffer_drive",
        "carla",
        f"load_model_path={job['checkpoint']}",
        f"eval.benchmark_config={job['catalog']}",
        f"eval.output_dir_name={output_directory}",
        "eval.num_agents=300",
        "eval.action_selection=mean",
        "eval.render_scenarios=false",
        f"eval.render_filter={job.get('render_filter', 'null')}",
        f"eval.failure_replay_csv={job.get('failure_replay_csv', 'null')}",
        f"eval.capture_observations={str(job.get('capture_observations', False)).lower()}",
        f"eval.max_rendered_failures={job.get('max_rendered_failures', 'null')}",
        "vec.num_envs=20",
        "vec.num_workers=auto",
        "train.device=cuda",
        "train.amp=true",
        "train.precision=bfloat16",
        "train.compile=false",
        "wandb=false",
    ]


def status_path(case):
    return PROJECT_DIR / f"{case}.status.json"


def load_prior_status(case):
    path = status_path(case)
    if not path.is_file():
        return None
    with path.open(encoding="utf-8") as status_file:
        return json.load(status_file)


def make_base_status(job, command):
    return {
        "case": job["case"],
        "algorithm": job["algorithm"],
        "gpu": job["gpu"],
        "checkpoint": job["checkpoint"],
        "catalog": job["catalog"],
        "command": command,
        "pid": None,
        "process_group_id": None,
        "started_at": None,
        "ended_at": None,
        "exitcode": None,
        "state": "pending",
        "error": None,
        "log": None,
    }


def record_resources(active_jobs, resource_log, monitored_gpus):
    available = available_memory_bytes()
    gpu_stats = []
    gpu_error = None
    for gpu in sorted(set(monitored_gpus)):
        try:
            gpu_stats.append(query_gpu(gpu))
        except Exception as error:  # Resource sampling must continue if telemetry briefly fails.
            gpu_error = str(error)
            break
    jobs = []
    for entry in active_jobs.values():
        rss_bytes, process_count = group_memory(entry["process"].pid)
        jobs.append(
            {
                "case": entry["job"]["case"],
                "pid": entry["process"].pid,
                "process_group_id": entry["process"].pid,
                "rss_bytes": rss_bytes,
                "process_count": process_count,
            }
        )
    sample = {
        "timestamp": utc_now(),
        "host_mem_available_bytes": available,
        "host_mem_available_gib": round(available / GIB, 3),
        "gpus": gpu_stats,
        "gpu_query_error": gpu_error,
        "jobs": jobs,
    }
    resource_log.parent.mkdir(parents=True, exist_ok=True)
    with resource_log.open("a", encoding="utf-8") as output:
        output.write(json.dumps(sample, sort_keys=True) + "\n")
        output.flush()
        os.fsync(output.fileno())
    return available


def terminate_owned_group(entry, final_state, error):
    process = entry["process"]
    try:
        os.killpg(process.pid, signal.SIGTERM)
    except ProcessLookupError:
        pass
    deadline = time.monotonic() + 10
    while process_group_exists(process.pid) and time.monotonic() < deadline:
        time.sleep(0.2)
    if process_group_exists(process.pid):
        try:
            os.killpg(process.pid, signal.SIGKILL)
        except ProcessLookupError:
            pass
    try:
        exitcode = process.wait(timeout=2)
    except subprocess.TimeoutExpired:
        exitcode = process.poll()
    entry["status"].update(
        state=final_state,
        ended_at=utc_now(),
        exitcode=exitcode,
        error=error,
    )
    atomic_json(entry["status_path"], entry["status"])


def cleanup_owned_jobs(active_jobs):
    for case, entry in list(active_jobs.items()):
        try:
            terminate_owned_group(entry, "interrupted", "runner exited before evaluation completed")
            del active_jobs[case]
        except Exception as error:
            print_status(f"{case}: cleanup failed: {error}")


def handle_termination_signal(signum, frame):
    raise KeyboardInterrupt(f"received signal {signum}")


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, default=PROJECT_DIR / "manifest.json")
    parser.add_argument("--max-concurrent", type=int, default=2)
    parser.add_argument("--cases", nargs="+", help="Optional case names to run")
    parser.add_argument("--retry-failed", action="store_true", help="Explicitly rerun jobs with a prior failure")
    args = parser.parse_args()
    if args.max_concurrent < 1:
        parser.error("--max-concurrent must be at least 1")
    if args.cases:
        args.cases = {case for value in args.cases for case in value.split(",") if case}
    args.manifest = args.manifest.resolve()
    return args


def main():
    args = parse_args()
    jobs = read_manifest(args.manifest)
    manifest_cases = {job["case"] for job in jobs}
    if args.cases and not args.cases <= manifest_cases:
        raise ValueError(f"Unknown cases: {', '.join(sorted(args.cases - manifest_cases))}")
    selected_jobs = [job for job in jobs if not args.cases or job["case"] in args.cases]

    case_lock_files = []
    for job in selected_jobs:
        lock_directory = PROJECT_DIR / ".case_locks"
        lock_directory.mkdir(parents=True, exist_ok=True)
        lock_file = (lock_directory / f"{job['case']}.lock").open("a+")
        try:
            fcntl.flock(lock_file.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as error:
            lock_file.close()
            raise RuntimeError(f"{job['case']}: another runner already owns this case") from error
        case_lock_files.append(lock_file)

    statuses = {}
    pending = []
    for job in selected_jobs:
        command = build_command(job)
        prior = load_prior_status(job["case"])
        if prior and prior.get("state") == "finished" and prior.get("exitcode") == 0:
            print_status(f"{job['case']}: already completed; preserving result")
            continue
        if prior and prior.get("state") == "running":
            if process_group_exists(prior.get("process_group_id") or prior.get("pid")):
                raise RuntimeError(f"{job['case']}: refusing duplicate; recorded process group is still running")
            prior.update(state="failed", ended_at=prior.get("ended_at") or utc_now(), error="runner exited while job status was running")
            atomic_json(status_path(job["case"]), prior)
        if prior and prior.get("state") not in {"pending", None}:
            if prior.get("state") not in RETRYABLE_STATES or not args.retry_failed:
                raise RuntimeError(
                    f"{job['case']}: prior state is {prior.get('state')!r}; use --retry-failed to rerun it"
                )
        status = make_base_status(job, command)
        status["attempt"] = (int(prior.get("attempt", 0)) + 1) if prior else 1
        status["created_at"] = utc_now()
        statuses[job["case"]] = (status, status_path(job["case"]))
        pending.append(job)
        atomic_json(status_path(job["case"]), status)

    if not pending:
        print_status("No pending evaluations selected")
        for lock_file in case_lock_files:
            lock_file.close()
        return 0

    log_directory = PROJECT_DIR / "logs"
    resource_log = PROJECT_DIR / "resource_usage.jsonl"
    active = {}
    atexit.register(cleanup_owned_jobs, active)
    signal.signal(signal.SIGTERM, handle_termination_signal)
    started_monotonic = time.monotonic()
    last_resource_sample = 0.0
    last_hourly_report = started_monotonic
    stop_launching = False
    exit_status = 0
    print_status(f"Starting {len(pending)} evaluation(s), max concurrent={args.max_concurrent}")

    while pending or active:
        elapsed = time.monotonic() - started_monotonic
        if elapsed >= MAX_RUNTIME_SECONDS:
            stop_launching = True
            exit_status = 1
            for case, entry in list(active.items()):
                terminate_owned_group(entry, "duration_limit_stopped", "runner reached its seven-day duration limit")
                del active[case]

        for case, entry in list(active.items()):
            exitcode = entry["process"].poll()
            if exitcode is None:
                continue
            status = entry["status"]
            status.update(
                state="finished" if exitcode == 0 else "failed",
                ended_at=utc_now(),
                exitcode=exitcode,
            )
            if exitcode != 0:
                status["error"] = f"evaluation exited with code {exitcode}"
                exit_status = 1
            atomic_json(entry["status_path"], status)
            print_status(f"{case}: {status['state']} (exitcode={exitcode})")
            del active[case]

        available = available_memory_bytes()
        if active and available < STOP_MEMORY_MIN_BYTES:
            stop_launching = True
            exit_status = 1
            error = f"MemAvailable fell below 32 GiB ({available / GIB:.2f} GiB)"
            print_status(f"Memory guard: {error}; stopping owned evaluation process groups")
            for case, entry in list(active.items()):
                terminate_owned_group(entry, "memory_guard_stopped", error)
                del active[case]

        if not stop_launching:
            while len(active) < args.max_concurrent and pending and available >= LAUNCH_MEMORY_MIN_BYTES:
                occupied_gpus = {entry["job"]["gpu"] for entry in active.values()}
                next_job_index = next(
                    (index for index, job in enumerate(pending) if job["gpu"] not in occupied_gpus),
                    None,
                )
                if next_job_index is None:
                    break
                job = pending.pop(next_job_index)
                status, status_file = statuses[job["case"]]
                command = status["command"]
                try:
                    gpu_status = query_gpu(job["gpu"])
                    if gpu_status["memory_used_mib"] > GPU_IDLE_LIMIT_MIB:
                        raise RuntimeError(
                            f"GPU {job['gpu']} has {gpu_status['memory_used_mib']} MiB in use "
                            f"(limit {GPU_IDLE_LIMIT_MIB} MiB)"
                        )
                    log_directory.mkdir(parents=True, exist_ok=True)
                    log_path = log_directory / f"{job['case']}.{datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S')}.log"
                    log_file = log_path.open("ab", buffering=0)
                    environment = os.environ.copy()
                    environment["CUDA_VISIBLE_DEVICES"] = job["gpu"]
                    environment["OMP_NUM_THREADS"] = "1"
                    environment["MKL_NUM_THREADS"] = "1"
                    environment["OPENBLAS_NUM_THREADS"] = "1"
                    environment["NUMEXPR_NUM_THREADS"] = "1"
                    environment["PYTHONUNBUFFERED"] = "1"
                    if job.get("capture_observations"):
                        # Large replay histories can stall in kernel huge-page compaction.
                        environment["NUMPY_MADVISE_HUGEPAGE"] = "0"
                    mpl_config_directory = PROJECT_DIR / "mplconfig" / job["case"]
                    mpl_config_directory.mkdir(parents=True, exist_ok=True)
                    environment["MPLCONFIGDIR"] = str(mpl_config_directory)
                    previous_signal_mask = signal.pthread_sigmask(signal.SIG_BLOCK, {signal.SIGINT, signal.SIGTERM})
                    try:
                        process = subprocess.Popen(
                            command,
                            cwd=REPO_ROOT,
                            env=environment,
                            stdin=subprocess.DEVNULL,
                            stdout=log_file,
                            stderr=subprocess.STDOUT,
                            start_new_session=True,
                            # The single-threaded runner blocks signals only while
                            # registering ownership; evaluators must not inherit it.
                            preexec_fn=partial(signal.pthread_sigmask, signal.SIG_SETMASK, previous_signal_mask),
                        )
                        status.update(
                            state="running",
                            pid=process.pid,
                            process_group_id=process.pid,
                            started_at=utc_now(),
                            log=str(log_path),
                            gpu_preflight=gpu_status,
                        )
                        active[job["case"]] = {
                            "job": job,
                            "process": process,
                            "status": status,
                            "status_path": status_file,
                        }
                        atomic_json(status_file, status)
                    finally:
                        signal.pthread_sigmask(signal.SIG_SETMASK, previous_signal_mask)
                except Exception as error:
                    status.update(state="failed", ended_at=utc_now(), error=str(error))
                    atomic_json(status_file, status)
                    print_status(f"{job['case']}: preflight/launch failed: {error}")
                    stop_launching = True
                    exit_status = 1
                    break
                log_file.close()
                print_status(f"{job['case']}: started pid={process.pid} gpu={job['gpu']} log={log_path}")
                available = available_memory_bytes()

        now = time.monotonic()
        if now - last_resource_sample >= RESOURCE_SAMPLE_SECONDS:
            available = record_resources(active, resource_log, (job["gpu"] for job in selected_jobs))
            last_resource_sample = now
            if active and available < STOP_MEMORY_MIN_BYTES:
                continue
        if now - last_hourly_report >= 60 * 60:
            completed_count = sum(
                1 for status, _ in statuses.values() if status.get("state") == "finished"
            )
            print_status(
                f"Hourly progress: {completed_count} completed, {len(active)} running, "
                f"{len(pending)} pending; elapsed={elapsed / 3600:.1f}h"
            )
            last_hourly_report = now
        if stop_launching and not active:
            break
        time.sleep(min(POLL_SECONDS, RESOURCE_SAMPLE_SECONDS - (now - last_resource_sample)))

    if pending:
        print_status(f"Stopped with {len(pending)} evaluation(s) still pending")
    for lock_file in case_lock_files:
        lock_file.close()
    return exit_status


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except KeyboardInterrupt:
        print_status("Interrupted; stopping owned evaluation process groups and recording status")
        raise SystemExit(130)
