#!/usr/bin/env python3
"""Launch named full-CARLA evaluations for the saved SAC16 checkpoints."""

import argparse
import csv
from datetime import datetime
import hashlib
import json
import math
import os
from pathlib import Path
import re
import shlex
import signal
import subprocess
import sys
import time

import yaml


SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_DIR = SCRIPT_DIR.parent
REPO_ROOT = PROJECT_DIR.parents[1]
DEFAULT_SWEEP_DIR = PROJECT_DIR / "runs" / "sac16_grid_20261012T032408Z"
BENCHMARK_CONFIG = (
    REPO_ROOT / "project" / "baseline_run_sync_2026-08-24"
    / "override_config" / "evaluation_benchmarks.yaml"
)
PARAMETER_KEYS = (
    "train.update_epochs",
    "train.replay_update_ratio",
    "train.initial_alpha",
    "train.tau",
)
TRIAL_COUNT = 16

if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))
from project.soft_actor_critic_tuning.scripts.run_sac16_grid import (  # noqa: E402
    _terminate_owned_group,
    _track_owned_descendants,
)


def _json(path):
    with path.open(encoding="utf-8") as source:
        return json.load(source)


def _yaml(path):
    with path.open(encoding="utf-8") as source:
        return yaml.safe_load(source)


def _write_json(path, data):
    temporary = path.with_name(path.name + ".tmp")
    with temporary.open("w", encoding="utf-8") as output:
        json.dump(data, output, indent=2, sort_keys=True, allow_nan=False)
        output.write("\n")
    temporary.replace(path)


def _sha256(path):
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _absolute(raw_path, field):
    path = Path(raw_path).expanduser()
    if not path.is_absolute():
        raise ValueError(f"{field} must be absolute: {raw_path!r}")
    return path.resolve()


def _trial_ids(raw):
    if raw is None:
        return list(range(1, TRIAL_COUNT + 1))
    try:
        ids = [int(value.strip()) for value in raw.split(",") if value.strip()]
    except ValueError as error:
        raise ValueError("--trial-ids must be comma-separated 1-based integers") from error
    if not ids or len(ids) != len(set(ids)) or any(value < 1 or value > TRIAL_COUNT for value in ids):
        raise ValueError(f"--trial-ids must contain unique IDs in [1, {TRIAL_COUNT}]")
    return ids


def _gpu_ids(raw):
    try:
        ids = [int(value.strip()) for value in raw.split(",") if value.strip()]
    except ValueError as error:
        raise ValueError("--gpus must be comma-separated nonnegative IDs, such as '0,1'") from error
    if not ids or any(value < 0 for value in ids) or len(ids) != len(set(ids)):
        raise ValueError("--gpus must contain unique nonnegative IDs")
    return ids


def _evaluation_name(name):
    if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9._-]{0,63}", name) or name in {".", ".."}:
        raise ValueError("--evaluation-name must be a safe 1-64 character path component")
    return name


def _load_trials(sweep_dir, manifest_path):
    source = _json(manifest_path)
    if source.get("experiment") != "SAC16-grid" or _absolute(source.get("output_dir", ""), "output_dir") != sweep_dir:
        raise ValueError(f"Not the requested SAC16 sweep: {manifest_path}")
    trials = source.get("trials")
    if not isinstance(trials, list) or len(trials) != TRIAL_COUNT:
        raise ValueError(f"Expected {TRIAL_COUNT} source trials in {manifest_path}")
    by_id = {record.get("trial_id"): record for record in trials if isinstance(record, dict)}
    if set(by_id) != set(range(1, TRIAL_COUNT + 1)):
        raise ValueError("Source manifest must contain unique trial IDs 1 through 16")

    records = []
    for trial_id in range(1, TRIAL_COUNT + 1):
        source_trial = by_id[trial_id]
        if source_trial.get("status") != "completed":
            raise ValueError(f"Source trial {trial_id:02d} is not completed")
        parameters = source_trial.get("parameters")
        if not isinstance(parameters, dict) or set(parameters) != set(PARAMETER_KEYS):
            raise ValueError(f"Source trial {trial_id:02d} has unexpected parameter values")
        source_config_path = _absolute(source_trial.get("config_path", ""), f"trial {trial_id} config_path")
        summary_path = _absolute(source_trial.get("summary_path", ""), f"trial {trial_id} summary_path")
        trial_dir = _absolute(source_trial.get("trial_dir", ""), f"trial {trial_id} trial_dir")
        source_config, summary = _yaml(source_config_path), _json(summary_path)
        source_train = source_config.get("train", {})
        if not source_config_path.is_file() or not summary_path.is_file():
            raise FileNotFoundError(f"Source config or summary missing for trial {trial_id:02d}")
        if summary.get("trial_id") != trial_id or summary.get("status") != "completed" or summary.get("parameters") != parameters:
            raise ValueError(f"Source trial {trial_id:02d} manifest and summary disagree")
        if any(source_train.get(key.split(".", 1)[1]) != parameters[key] for key in PARAMETER_KEYS):
            raise ValueError(f"Source trial {trial_id:02d} config and manifest parameters disagree")
        duration = summary.get("full_time_seconds")
        steps, updates = summary.get("agent_steps"), summary.get("learner_updates")
        seed = source_train.get("seed")
        if (
            isinstance(seed, bool) or not isinstance(seed, int)
            or isinstance(duration, bool) or not isinstance(duration, (int, float)) or not math.isfinite(duration) or duration <= 0
            or isinstance(steps, bool) or not isinstance(steps, int) or steps <= 0
            or isinstance(updates, bool) or not isinstance(updates, int) or updates <= 0
        ):
            raise ValueError(f"Source trial {trial_id:02d} has invalid training metadata")
        if source_trial.get("full_time_seconds") != duration or Path(source_train.get("data_dir", "")).resolve() != trial_dir / "learner":
            raise ValueError(f"Source trial {trial_id:02d} manifest and saved config disagree")
        checkpoint = _absolute(summary.get("artifacts", {}).get("final_model", ""), f"trial {trial_id} checkpoint")
        expected_checkpoint = Path(source_train.get("data_dir", "")) / source_train.get("final_model_name", "final_model.pt")
        if not expected_checkpoint.is_absolute() or expected_checkpoint.resolve() != checkpoint or not checkpoint.is_file():
            raise ValueError(f"Source trial {trial_id:02d} checkpoint disagrees with its saved config")
        config_path = checkpoint.parent / "config.yaml"
        learner_config = _yaml(config_path)
        learner_train = learner_config.get("train", {})
        if any(learner_train.get(key.split(".", 1)[1]) != parameters[key] for key in PARAMETER_KEYS):
            raise ValueError(f"Trial {trial_id:02d} checkpoint config and source parameters disagree")
        if Path(learner_train.get("data_dir", "")).resolve() != checkpoint.parent.resolve():
            raise ValueError(f"Trial {trial_id:02d} checkpoint config points to a different learner directory")
        records.append({
            "trial_id": trial_id,
            "parameters": {key: parameters[key] for key in PARAMETER_KEYS},
            "checkpoint_path": str(checkpoint),
            "config_path": str(config_path),
            "training_summary_path": str(summary_path),
            "training_seed": seed,
            "training_duration_seconds": float(duration),
            "agent_steps": steps,
            "learner_updates": updates,
            "status": "planned",
            "gpu": None,
            "command": [],
            "evaluation_summary_path": None,
            "episode_metrics_path": None,
            "resolved_benchmark_path": None,
        })
    return records


def _check_benchmark(path):
    config = _yaml(path)
    benchmarks = config.get("benchmarks", [])
    matches = [item for item in benchmarks if isinstance(item, dict) and item.get("name") == "carla"]
    if len(matches) != 1:
        raise ValueError(f"Expected exactly one carla benchmark in {path}")
    benchmark = matches[0]
    expected = {"seed": 42, "num_scenarios": 1000}
    expected_env = {
        "simulation_mode": "gigaflow", "num_maps": 8, "max_agents_per_env": 50,
        "scenario_length": 6000, "control_mode": "control_vehicles",
        "map_dir": "pufferlib/resources/drive/binaries/carla",
    }
    if config.get("env", {}).get("dt") != 0.1 or any(benchmark.get(key) != value for key, value in expected.items()):
        raise ValueError("Baseline benchmark must use dt=0.1, seed=42 and 1,000 CARLA scenarios")
    if any(benchmark.get("env", {}).get(key) != value for key, value in expected_env.items()):
        raise ValueError("Baseline CARLA benchmark settings do not match the agreed protocol")


def _command(record, benchmark_path, evaluation_name, no_wandb):
    return [
        sys.executable, "-m", "pufferlib.pufferl_sac", "eval", "puffer_drive", "carla",
        f"load_model_path={record['checkpoint_path']}",
        f"eval.benchmark_config={benchmark_path}",
        f"eval.output_name=final_model_mean_metrics_{evaluation_name}",
        "eval.output_dir_name=eval", "eval.num_agents=300", "eval.action_selection=mean",
        "eval.render_scenarios=false", "eval.render_filter=null", "eval.capture_observations=false",
        "vec.num_envs=20", "vec.num_workers=auto", "train.device=cuda", "train.amp=true",
        "train.precision=bfloat16", "train.compile=false", f"wandb={'false' if no_wandb else 'true'}",
    ]


def _output_root(record, name):
    return Path(record["checkpoint_path"]).parent / "eval" / f"carla_final_model_mean_metrics_{name}"


def _directories(path):
    return {item.resolve() for item in path.iterdir() if item.is_dir()} if path.is_dir() else set()


def _summaries(path):
    return sorted(item.resolve() for item in path.glob("*/evaluation_summary.json")) if path.is_dir() else []


def _validate_output(summary_path, csv_path, resolved_path, benchmark_path, name, wandb_enabled, record):
    summary = _json(summary_path)
    metrics = summary.get("metrics_mean")
    if summary.get("num_scenarios") != 1000 or summary.get("num_episodes") != 1000 or not isinstance(metrics, dict) or not metrics:
        raise ValueError("evaluation summary must contain 1,000 scenarios, 1,000 episodes and non-empty metrics_mean")
    if any(isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value) for value in metrics.values()):
        raise ValueError("evaluation summary contains non-finite metrics")
    with csv_path.open(encoding="utf-8", newline="") as source:
        if sum(1 for _ in csv.DictReader(source)) != 1000:
            raise ValueError("episode_metrics.csv must contain exactly 1,000 rows")

    resolved = _yaml(resolved_path)
    benchmark, args = resolved.get("benchmark", {}), resolved.get("args", {})
    env, evaluation = args.get("env", {}), args.get("eval", {})
    vector, train = args.get("vec", {}), args.get("train", {})
    expected_benchmark = {"name": "carla", "seed": 42, "num_scenarios": 1000}
    expected_env = {
        "simulation_mode": "gigaflow", "num_maps": 8, "max_agents_per_env": 50,
        "scenario_length": 6000, "control_mode": "control_vehicles",
        "sdc_controller": "policy", "non_sdc_controller": "policy", "non_vehicle_controller": "auto",
    }
    if any(benchmark.get(key) != value for key, value in expected_benchmark.items()):
        raise ValueError("resolved benchmark identity does not match CARLA seed 42 / 1,000 scenarios")
    if any(benchmark.get("env", {}).get(key) != value for key, value in expected_env.items()):
        raise ValueError("resolved CARLA benchmark settings do not match the baseline")
    map_dir = Path(benchmark.get("env", {}).get("map_dir", ""))
    if not map_dir.is_absolute() or map_dir.resolve() != (REPO_ROOT / "pufferlib/resources/drive/binaries/carla").resolve() or env.get("dt") != 0.1:
        raise ValueError("resolved CARLA map or dt does not match the baseline")
    expected_eval = {
        "num_agents": 300, "action_selection": "mean", "output_name": f"final_model_mean_metrics_{name}",
        "output_dir_name": "eval", "render_scenarios": False, "render_filter": None, "capture_observations": False,
    }
    if (
        any(evaluation.get(key) != value for key, value in expected_eval.items())
        or vector.get("num_envs") != 20 or vector.get("num_workers") != "auto"
        or not str(train.get("device", "")).startswith("cuda") or train.get("amp") is not True
        or train.get("precision") != "bfloat16" or train.get("compile") is not False
        or args.get("wandb") is not wandb_enabled
        or Path(resolved.get("benchmark_config", "")).resolve() != benchmark_path
        or Path(args.get("load_model_path", "")).resolve() != Path(record["checkpoint_path"]).resolve()
        or Path(resolved.get("checkpoint_config", "")).resolve() != Path(record["config_path"]).resolve()
    ):
        raise ValueError("resolved evaluation settings or checkpoint provenance do not match the requested protocol")


def _resume_records(manifest, expected, records):
    for key in (
        "schema_version", "source_sweep_manifest", "source_sweep_manifest_sha256",
        "benchmark_config", "benchmark_config_sha256", "evaluation_name", "protocol",
    ):
        if manifest.get(key) != expected[key]:
            raise ValueError(f"--resume session identity mismatch in {key}")
    saved = manifest.get("trials", [])
    by_id = {item.get("trial_id"): item for item in saved if isinstance(item, dict)}
    if len(saved) != TRIAL_COUNT or set(by_id) != set(range(1, TRIAL_COUNT + 1)):
        raise ValueError("--resume manifest must contain unique trial IDs 1 through 16")
    for current in records:
        old = by_id[current["trial_id"]]
        for key in ("parameters", "checkpoint_path", "config_path", "training_summary_path", "training_seed", "training_duration_seconds", "agent_steps", "learner_updates"):
            if old.get(key) != current[key]:
                raise ValueError(f"--resume source identity mismatch for trial {current['trial_id']:02d} field {key}")
    return [by_id[trial_id] for trial_id in range(1, TRIAL_COUNT + 1)]


def _parser():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sweep-dir", default=str(DEFAULT_SWEEP_DIR))
    parser.add_argument("--gpus", default="0,1", help="One GPU per evaluation, e.g. 0,1")
    parser.add_argument("--max-concurrent", type=int, default=2)
    parser.add_argument("--trial-ids", help="Comma-separated IDs; defaults to all 16")
    parser.add_argument("--evaluation-name", help="Safe session name; required for launch")
    parser.add_argument("--resume", action="store_true", help="Resume the named evaluation session")
    parser.add_argument("--no-wandb", action="store_true")
    parser.add_argument("--dry-run", action="store_true", help="Print commands without creating a session or launching workers")
    return parser


def main(argv=None):
    args = _parser().parse_args(argv)
    sweep_dir = Path(args.sweep_dir).expanduser().resolve()
    source_path = sweep_dir / "manifest.json"
    benchmark_path = BENCHMARK_CONFIG.resolve()
    if not source_path.is_file() or not benchmark_path.is_file():
        raise FileNotFoundError(f"Missing sweep manifest or baseline benchmark YAML: {source_path}, {benchmark_path}")
    _check_benchmark(benchmark_path)
    source_records = _load_trials(sweep_dir, source_path)
    selected_ids, gpus = _trial_ids(args.trial_ids), _gpu_ids(args.gpus)
    if args.max_concurrent < 1 or args.max_concurrent > len(gpus):
        raise ValueError("--max-concurrent must be between 1 and the number of --gpus")
    if not args.dry_run and not args.evaluation_name:
        raise ValueError("An actual evaluation requires --evaluation-name")
    name = _evaluation_name(args.evaluation_name or "dry_run")
    wandb_enabled = not args.no_wandb
    protocol = {
        "benchmark": "carla", "seed": 42, "num_scenarios": 1000, "scenario_length": 6000, "dt": 0.1,
        "num_agents": 300, "num_envs": 20, "num_workers": "auto", "device": "cuda",
        "amp": True, "precision": "bfloat16", "compile": False, "action_selection": "mean",
        "render_scenarios": False, "capture_observations": False, "wandb": wandb_enabled,
    }
    expected = {
        "schema_version": 1,
        "source_sweep_manifest": str(source_path.resolve()),
        "source_sweep_manifest_sha256": _sha256(source_path),
        "benchmark_config": str(benchmark_path),
        "benchmark_config_sha256": _sha256(benchmark_path),
        "evaluation_name": name,
        "protocol": protocol,
        "trials": source_records,
    }
    session_dir = sweep_dir / "evaluations" / name
    manifest_path = session_dir / "manifest.json"
    if args.resume:
        if not manifest_path.is_file():
            raise FileNotFoundError(f"No resumable session manifest at {manifest_path}")
        manifest = _json(manifest_path)
        records = _resume_records(manifest, expected, source_records)
    else:
        if session_dir.exists():
            raise FileExistsError(f"Session exists: {session_dir}; use --resume or a new --evaluation-name")
        manifest, records = expected, source_records

    pending = []
    for trial_id in selected_ids:
        record = records[trial_id - 1]
        output_root = _output_root(record, name)
        if args.resume and record.get("status") == "completed":
            saved_paths = (
                record.get("evaluation_summary_path"), record.get("episode_metrics_path"),
                record.get("resolved_benchmark_path"),
            )
            if not all(isinstance(path, str) and Path(path).is_absolute() for path in saved_paths):
                raise ValueError(f"Completed trial {trial_id:02d} lacks explicit outputs; use a new --evaluation-name")
            summary = Path(record["evaluation_summary_path"])
            csv_path = Path(record["episode_metrics_path"])
            resolved_path = Path(record["resolved_benchmark_path"])
            summaries = _summaries(output_root)
            if len(summaries) != 1 or summaries[0] != summary.resolve():
                raise ValueError(f"Completed trial {trial_id:02d} has missing or ambiguous summaries; use a new --evaluation-name")
            if summary.parent.parent != output_root.resolve() or csv_path.parent != summary.parent or resolved_path.parent != summary.parent:
                raise ValueError(f"Completed trial {trial_id:02d} has output paths outside its named output")
            _validate_output(summary, csv_path, resolved_path, benchmark_path, name, wandb_enabled, record)
            continue
        if _summaries(output_root):
            raise ValueError(
                f"Trial {trial_id:02d} has an old report but is not recorded completed; "
                "use a new --evaluation-name to avoid ambiguous outputs."
            )
        if not args.resume and output_root.exists():
            raise FileExistsError(f"Output exists for trial {trial_id:02d}: {output_root}; use a new --evaluation-name")
        pending.append(record)

    if args.dry_run:
        if args.resume:
            print(f"Resume dry run: {len(pending)} selected trial(s) remain")
        for index, record in enumerate(pending):
            gpu = gpus[index % args.max_concurrent]
            print(f"trial {record['trial_id']:02d} GPU {gpu}: {shlex.join([f'CUDA_VISIBLE_DEVICES={gpu}', *_command(record, benchmark_path, name, args.no_wandb)])}")
        if not args.evaluation_name and not args.resume:
            print("Dry-run only; pass --evaluation-name before launching.")
        return 0

    session_dir.mkdir(parents=True, exist_ok=True)
    logs_dir = session_dir / "logs"
    logs_dir.mkdir(exist_ok=True)
    if not args.resume:
        _write_json(manifest_path, manifest)

    available_gpus, active, failures = list(gpus[:args.max_concurrent]), {}, 0
    cancelled_signal, previous_handlers = None, {}

    def request_cancel(signum, _frame):
        nonlocal cancelled_signal
        cancelled_signal = signum

    def save_manifest():
        _write_json(manifest_path, manifest)

    def finish(process_info, cancelled=False):
        record, process = process_info["record"], process_info["process"]
        _track_owned_descendants(process, process_info["descendants"])
        exit_code = _terminate_owned_group(process, process_info["descendants"])
        process_info["log"].close()
        new_dirs = _directories(process_info["output_root"]) - process_info["before"]
        if len(new_dirs) != 1:
            record["status"] = "cancelled" if cancelled else "failed"
            record["error"] = f"Expected one new output directory, found {len(new_dirs)}; use a new --evaluation-name if reports are ambiguous."
        else:
            output_dir = next(iter(new_dirs))
            record["evaluation_summary_path"] = str((output_dir / "evaluation_summary.json").resolve())
            record["episode_metrics_path"] = str((output_dir / "episode_metrics.csv").resolve())
            record["resolved_benchmark_path"] = str((output_dir / "resolved_benchmark.yaml").resolve())
            try:
                if exit_code != 0:
                    raise ValueError(f"evaluation exited with code {exit_code}")
                _validate_output(
                    Path(record["evaluation_summary_path"]), Path(record["episode_metrics_path"]),
                    Path(record["resolved_benchmark_path"]), benchmark_path, name, wandb_enabled, record,
                )
                record["status"] = "completed"
                record.pop("error", None)
            except (OSError, ValueError, yaml.YAMLError, json.JSONDecodeError) as error:
                record["status"], record["error"] = "failed", str(error)
        if cancelled:
            record["status"] = "cancelled"
        return exit_code

    for signum in (signal.SIGINT, signal.SIGTERM):
        previous_handlers[signum] = signal.signal(signum, request_cancel)
    try:
        while pending or active:
            if cancelled_signal is not None:
                for record in pending:
                    record["status"], record["error"] = "cancelled", "Cancelled before start"
                pending.clear()
                for process_id, info in list(active.items()):
                    finish(info, cancelled=True)
                    info["record"]["error"] = f"Cancelled by signal {cancelled_signal}"
                    available_gpus.append(info["gpu"])
                    del active[process_id]
                save_manifest()
                return 128 + cancelled_signal

            while pending and len(active) < args.max_concurrent:
                record, gpu = pending.pop(0), available_gpus.pop(0)
                output_root = _output_root(record, name)
                output_root.mkdir(parents=True, exist_ok=True)
                before = _directories(output_root)
                for _ in range(12):
                    if not (output_root / datetime.now().strftime("%Y%m%d-%H%M%S")).exists():
                        break
                    time.sleep(0.25)
                else:
                    raise RuntimeError(f"Timestamp collision under {output_root}; use a new --evaluation-name")
                command = _command(record, benchmark_path, name, args.no_wandb)
                record.update(status="running", gpu=gpu, command=command)
                record.update(evaluation_summary_path=None, episode_metrics_path=None, resolved_benchmark_path=None)
                record.pop("error", None)
                save_manifest()
                log = (logs_dir / f"trial_{record['trial_id']:02d}.log").open("a", encoding="utf-8")
                log.write("\nLAUNCH: " + shlex.join(command) + f"\nCUDA_VISIBLE_DEVICES={gpu}\n")
                log.flush()
                environment = os.environ.copy()
                environment.update(CUDA_VISIBLE_DEVICES=str(gpu), OMP_NUM_THREADS="1", MKL_NUM_THREADS="1")
                if args.no_wandb:
                    environment["WANDB_MODE"] = "disabled"
                try:
                    process = subprocess.Popen(
                        command, cwd=REPO_ROOT, env=environment, stdout=log,
                        stderr=subprocess.STDOUT, start_new_session=True,
                    )
                except OSError as error:
                    log.close()
                    record.update(status="failed", error=f"Could not start evaluation: {error}")
                    failures += 1
                    available_gpus.append(gpu)
                    save_manifest()
                    continue
                descendants = {}
                _track_owned_descendants(process, descendants)
                active[process.pid] = {
                    "process": process, "record": record, "gpu": gpu, "log": log,
                    "output_root": output_root, "before": before,
                    "descendants": descendants,
                }

            for process_id, info in list(active.items()):
                process = info["process"]
                _track_owned_descendants(process, info["descendants"])
                if process.poll() is not None:
                    finish(info)
                    if info["record"]["status"] != "completed":
                        failures += 1
                    available_gpus.append(info["gpu"])
                    del active[process_id]
                    save_manifest()
            if active:
                time.sleep(0.25)
    except BaseException as error:
        for process_id, info in list(active.items()):
            finish(info, cancelled=True)
            info["record"].update(status="cancelled", error=f"Launcher stopped: {error}")
            del active[process_id]
        for record in pending:
            record.update(status="cancelled", error=f"Launcher stopped: {error}")
        save_manifest()
        raise
    finally:
        for signum, handler in previous_handlers.items():
            signal.signal(signum, handler)

    save_manifest()
    if failures:
        print(f"{failures} evaluation(s) failed; see {logs_dir} and {manifest_path}", file=sys.stderr)
        return 1
    print(f"Completed {len(selected_ids)} selected evaluation(s); manifest: {manifest_path}")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (OSError, ValueError, RuntimeError, yaml.YAMLError, json.JSONDecodeError) as error:
        print(f"Evaluation launcher error: {error}", file=sys.stderr)
        raise SystemExit(2)
