#!/usr/bin/env python3
"""Validate and compare final CARLA results from the 16-trial SAC grid."""

from __future__ import annotations

import argparse
import csv
import itertools
import json
import math
import os
from pathlib import Path
import re
import sys
import tempfile

os.environ.setdefault("MPLCONFIGDIR", str(Path(tempfile.gettempdir()) / "pufferdrive-matplotlib"))

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import yaml

REPO_ROOT = Path(__file__).resolve().parents[3]
PROJECT_DIR = REPO_ROOT / "project/soft_actor_critic_tuning"
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))
from project.metric_analysis.plot_benchmark_comparison import (  # noqa: E402
    GROUP_TITLES,
    METRIC_GROUPS,
    METRIC_UNITS,
)

PARAMETERS = (
    "train.update_epochs",
    "train.replay_update_ratio",
    "train.initial_alpha",
    "train.tau",
)
GRID = {
    "train.update_epochs": (3, 10),
    "train.replay_update_ratio": (0.05, 0.1),
    "train.initial_alpha": (0.1, 0.2),
    "train.tau": (0.005, 0.01),
}
METRICS = tuple(metric for group in METRIC_GROUPS.values() for metric in group)
REQUIRED_METRICS = set(METRICS) - {"infractions_per_scenario"}
REQUIRED_METRICS.add("total_infractions")
EXPECTED_IDS = set(range(1, 17))
SCENARIOS = 1_000
MAPS = 8
BENCHMARK = {"name": "carla", "seed": 42, "num_scenarios": SCENARIOS}
ENVIRONMENT = {
    "dt": 0.1,
    "scenario_length": 6_000,
    "simulation_mode": "gigaflow",
    "num_maps": MAPS,
    "max_agents_per_env": 50,
}
PARAMETER_LABELS = {
    "train.update_epochs": "epochs",
    "train.replay_update_ratio": "refresh",
    "train.initial_alpha": "alpha",
    "train.tau": "tau",
}
METRIC_DIRECTIONS = {
    "offroad_rate": "lower is better",
    "collision_rate": "lower is better",
    "at_fault_collision_rate": "lower is better",
    "red_light_violation_rate": "lower is better",
    "infractions_per_scenario": "lower is better",
    "avg_distance_per_infraction": "higher is better",
    "num_goals_reached": "higher is better",
    "score": "higher is better",
    "dnf_rate": "lower is better",
    "velocity_progress_sum": "higher is better",
    "lane_center_rate": "higher is better",
    "comfort_violation_count": "lower is better",
    "progress_ratio": "higher is better",
    "making_progress_rate": "higher is better",
    "driving_direction_score": "higher is better",
    "speed_limit_compliance": "higher is better",
    "multi_lane_score": "higher is better",
    "puffer_score": "higher is better",
}
CONTEXT_METRICS = {"n", "avg_speed_per_agent", "multi_lane_time"}
PARAMETER_DIRS = {
    "train.update_epochs": "update_epochs",
    "train.replay_update_ratio": "replay_update_ratio",
    "train.initial_alpha": "initial_alpha",
    "train.tau": "tau",
}


def read_json(path: Path, label: str) -> dict:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as error:
        raise ValueError(f"Cannot read {label} {path}: {error}") from error
    if not isinstance(value, dict):
        raise ValueError(f"Expected a JSON object for {label}: {path}")
    return value


def read_yaml(path: Path, label: str) -> dict:
    try:
        value = yaml.safe_load(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, yaml.YAMLError) as error:
        raise ValueError(f"Cannot read {label} {path}: {error}") from error
    if not isinstance(value, dict):
        raise ValueError(f"Expected a YAML mapping for {label}: {path}")
    return value


def required_file(value, label: str) -> Path:
    if not isinstance(value, str) or not Path(value).is_absolute():
        raise ValueError(f"{label} must be an absolute path, found {value!r}")
    path = Path(value).resolve()
    if not path.is_file():
        raise FileNotFoundError(f"Missing {label}: {path}")
    return path


def finite(value, label: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value):
        raise ValueError(f"{label} must be finite numeric data, found {value!r}")
    return float(value)


def same_path(left: Path | str, right: Path | str) -> bool:
    return Path(left).resolve() == Path(right).resolve()


def parameter_values(record: dict, label: str) -> dict:
    values = record.get("parameters")
    if not isinstance(values, dict) or set(values) != set(PARAMETERS):
        raise ValueError(f"{label} must contain exactly the four SAC grid parameters")
    normalized = {}
    for name in PARAMETERS:
        value = values[name]
        if name == "train.update_epochs":
            if type(value) is not int:
                raise ValueError(f"{label} {name} must be an integer")
            normalized[name] = value
        else:
            normalized[name] = finite(value, f"{label} {name}")
        if normalized[name] not in GRID[name]:
            raise ValueError(f"{label} {name} is outside the configured grid: {value!r}")
    return normalized


def validate_protocol(
    benchmark_path: Path,
    resolved_path: Path,
    trial_id: int,
    checkpoint_path: Path,
    params: dict,
) -> None:
    raw = read_yaml(benchmark_path, "benchmark config")
    if not math.isclose(finite(raw.get("env", {}).get("dt"), "benchmark env.dt"), 0.1, abs_tol=1e-12):
        raise ValueError("Benchmark config env.dt must be 0.1")
    carla = [item for item in raw.get("benchmarks", []) if item.get("name") == "carla"]
    if len(carla) != 1:
        raise ValueError("Benchmark config must define one CARLA benchmark")
    raw_benchmark = carla[0]
    for key, expected in BENCHMARK.items():
        if raw_benchmark.get(key) != expected:
            raise ValueError(f"Benchmark config carla.{key} must be {expected!r}")
    raw_env = raw_benchmark.get("env", {})
    for key, expected in ENVIRONMENT.items():
        if key == "dt":
            continue
        actual = raw_env.get(key)
        if actual != expected:
            raise ValueError(f"Benchmark config carla.env.{key} must be {expected!r}")

    resolved = read_yaml(resolved_path, f"resolved benchmark trial {trial_id:02d}")
    if not same_path(resolved.get("benchmark_config", ""), benchmark_path):
        raise ValueError(f"Trial {trial_id:02d} resolved benchmark references a different config")
    checkpoint_config_path = required_file(
        resolved.get("checkpoint_config"), f"trial {trial_id:02d} checkpoint_config"
    )
    expected_checkpoint_config = checkpoint_path.parent / "config.yaml"
    if not same_path(checkpoint_config_path, expected_checkpoint_config):
        raise ValueError(f"Trial {trial_id:02d} resolved checkpoint config does not belong to its checkpoint")
    benchmark, args = resolved.get("benchmark", {}), resolved.get("args", {})
    if any(benchmark.get(key) != value for key, value in BENCHMARK.items()):
        raise ValueError(f"Trial {trial_id:02d} resolved benchmark does not match CARLA seed 42 / 1000")
    benchmark_env = benchmark.get("env", {})
    args_env = args.get("env", {})
    protocol_envs = (
        (benchmark_env, "resolved benchmark env", {key: value for key, value in ENVIRONMENT.items() if key != "dt"}),
        (args_env, "resolved args.env", ENVIRONMENT),
    )
    for env, label, requirements in protocol_envs:
        for key, expected in requirements.items():
            actual = env.get(key)
            if isinstance(expected, float):
                if not math.isclose(finite(actual, f"trial {trial_id:02d} {label}.{key}"), expected, abs_tol=1e-12):
                    raise ValueError(f"Trial {trial_id:02d} {label}.{key} must be {expected!r}")
            elif actual != expected:
                raise ValueError(f"Trial {trial_id:02d} {label}.{key} must be {expected!r}")
    eval_args, vec_args, train_args = args.get("eval", {}), args.get("vec", {}), args.get("train", {})
    model_path = args.get("load_model_path")
    if not isinstance(model_path, str) or not Path(model_path).is_absolute() or not same_path(model_path, checkpoint_path):
        raise ValueError(f"Trial {trial_id:02d} resolved model path does not match its checkpoint")
    checkpoint_config = read_yaml(checkpoint_config_path, f"checkpoint config trial {trial_id:02d}")
    checkpoint_train = checkpoint_config.get("train", {})
    if checkpoint_train.get("seed") != 0:
        raise ValueError(f"Trial {trial_id:02d} checkpoint config training seed must be 0")
    for name, expected in params.items():
        actual = finite(checkpoint_train.get(name.split(".", 1)[1]), f"trial {trial_id:02d} checkpoint {name}")
        if not math.isclose(actual, expected, abs_tol=1e-12):
            raise ValueError(f"Trial {trial_id:02d} checkpoint config {name} disagrees with its manifest")
    if args.get("num_scenarios") != SCENARIOS or args_env.get("num_agents") != 300:
        raise ValueError(f"Trial {trial_id:02d} must evaluate 1000 scenarios with 300 agents")
    if eval_args.get("num_agents") != 300 or eval_args.get("action_selection") != "mean":
        raise ValueError(f"Trial {trial_id:02d} must use 300 agents and mean action selection")
    selected_benchmarks = eval_args.get("benchmarks")
    if selected_benchmarks != "carla" and selected_benchmarks != ["carla"]:
        raise ValueError(f"Trial {trial_id:02d} must select only carla")
    if vec_args.get("num_envs") != 20 or vec_args.get("backend") != "Multiprocessing":
        raise ValueError(f"Trial {trial_id:02d} must use 20 multiprocessing environments")
    if train_args.get("seed") != 42 or vec_args.get("seed") != 42:
        raise ValueError(f"Trial {trial_id:02d} resolved benchmark seeds must be 42")


def scenario_identities(path: Path, trial_id: int) -> set[tuple[str, str, str]]:
    identities = set()
    with path.open(newline="", encoding="utf-8") as source:
        reader = csv.DictReader(source)
        if not {"map_name", "scenario_id", "seed"}.issubset(reader.fieldnames or []):
            raise ValueError(f"Trial {trial_id:02d} episode CSV needs map_name, scenario_id and seed columns")
        for row in reader:
            map_name = str(row["map_name"]).replace("\\", "/").rsplit("/", 1)[-1].split(".", 1)[0].strip()
            identity = (map_name, str(row["scenario_id"]).strip(), str(row["seed"]).strip())
            if not all(identity) or identity in identities:
                raise ValueError(f"Trial {trial_id:02d} episode CSV has an empty or duplicate scenario identity")
            identities.add(identity)
    if len(identities) != SCENARIOS or len({item[0] for item in identities}) != MAPS:
        raise ValueError(f"Trial {trial_id:02d} episode CSV must cover 1000 scenarios across eight maps")
    return identities


def load_inputs(manifest_path: Path, output_name: str) -> tuple[list[dict], dict]:
    manifest_path = manifest_path.expanduser().resolve()
    manifest = read_json(manifest_path, "evaluation manifest")
    if type(manifest.get("schema_version")) is not int or manifest["schema_version"] != 1:
        raise ValueError("Evaluation manifest schema_version must be 1")
    source_path = required_file(manifest.get("source_sweep_manifest"), "source_sweep_manifest")
    benchmark_path = required_file(manifest.get("benchmark_config"), "benchmark_config")
    if not isinstance(manifest.get("evaluation_name"), str) or not manifest["evaluation_name"]:
        raise ValueError("Evaluation manifest needs a non-empty evaluation_name")
    records = manifest.get("trials")
    if not isinstance(records, list) or len(records) != 16 or any(not isinstance(item, dict) for item in records):
        raise ValueError("Evaluation manifest must contain all 16 trial records")
    if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9._-]*", output_name) or output_name in {".", ".."}:
        raise ValueError("--output-name must be one safe path component")

    sweep = read_json(source_path, "source sweep manifest")
    if sweep.get("overall_status") != "completed":
        raise ValueError("Source sweep manifest is not completed")
    sweep_trials = {item.get("trial_id"): item for item in sweep.get("trials", [])}
    if set(sweep_trials) != EXPECTED_IDS:
        raise ValueError("Source sweep manifest must contain trial IDs 1 through 16")
    if {item.get("trial_id") for item in records} != EXPECTED_IDS:
        raise ValueError("Evaluation manifest must contain trial IDs 1 through 16 exactly once")

    expected_grid = set(itertools.product(*(GRID[name] for name in PARAMETERS)))
    trials, identities_by_trial, seen_grid = [], [], set()
    for record in records:
        trial_id = record["trial_id"]
        if type(trial_id) is not int or record.get("status") != "completed":
            raise ValueError(f"Evaluation trial {trial_id!r} must be completed with an integer ID")
        params = parameter_values(record, f"Trial {trial_id:02d}")
        combination = tuple(params[name] for name in PARAMETERS)
        if combination in seen_grid:
            raise ValueError(f"Duplicate parameter combination in trial {trial_id:02d}")
        seen_grid.add(combination)

        source = sweep_trials[trial_id]
        if source.get("status") != "completed" or parameter_values(source, f"Source trial {trial_id:02d}") != params:
            raise ValueError(f"Trial {trial_id:02d} does not match its completed source sweep record")
        for source_key in ("config_path", "summary_path"):
            if not isinstance(source.get(source_key), str) or not Path(source[source_key]).is_absolute():
                raise ValueError(f"Source trial {trial_id:02d} {source_key} must be absolute")
        paths = {
            key: required_file(record.get(key), f"trial {trial_id:02d} {key}")
            for key in (
                "checkpoint_path", "config_path", "training_summary_path",
                "evaluation_summary_path", "episode_metrics_path", "resolved_benchmark_path",
            )
        }
        if not same_path(paths["checkpoint_path"].parent / "config.yaml", paths["config_path"]):
            raise ValueError(f"Trial {trial_id:02d} config must be beside its final checkpoint")
        if not same_path(source.get("summary_path", ""), paths["training_summary_path"]):
            raise ValueError(f"Trial {trial_id:02d} training summary path is not from its source sweep record")
        if len({paths[key].parent for key in ("evaluation_summary_path", "episode_metrics_path", "resolved_benchmark_path")}) != 1:
            raise ValueError(f"Trial {trial_id:02d} evaluation files must come from one report directory")
        paths["source_sweep_manifest"] = source_path
        paths["benchmark_config"] = benchmark_path

        config = read_yaml(paths["config_path"], f"saved config trial {trial_id:02d}")
        train = config.get("train", {})
        for name, expected in params.items():
            actual = finite(train.get(name.split(".", 1)[1]), f"trial {trial_id:02d} saved {name}")
            if not math.isclose(actual, expected, abs_tol=1e-12):
                raise ValueError(f"Trial {trial_id:02d} saved config {name} disagrees with its manifest")
        if train.get("seed") != record.get("training_seed"):
            raise ValueError(f"Trial {trial_id:02d} training seed disagrees with saved config")
        if type(train.get("seed")) is not int or type(record.get("training_seed")) is not int:
            raise ValueError(f"Trial {trial_id:02d} training seed must be an integer")

        training = read_json(paths["training_summary_path"], f"training summary trial {trial_id:02d}")
        if training.get("trial_id") != trial_id or training.get("status") != "completed":
            raise ValueError(f"Trial {trial_id:02d} training summary has wrong identity/status")
        if parameter_values(training, f"Training summary trial {trial_id:02d}") != params:
            raise ValueError(f"Trial {trial_id:02d} training summary parameters disagree")
        duration = finite(record.get("training_duration_seconds"), "training duration")
        if duration <= 0 or not math.isclose(duration, finite(training.get("full_time_seconds"), "training summary duration"), abs_tol=1e-9):
            raise ValueError(f"Trial {trial_id:02d} training duration disagrees with its summary")
        for key in ("agent_steps", "learner_updates"):
            if type(record.get(key)) is not int or record[key] <= 0 or record[key] != training.get(key):
                raise ValueError(f"Trial {trial_id:02d} {key} disagrees with its training summary")
        if train.get("seed") != 0:
            raise ValueError("The SAC grid is expected to use training seed 0")
        checkpoint = required_file(training.get("artifacts", {}).get("final_model"), f"trial {trial_id:02d} source checkpoint")
        if not same_path(paths["checkpoint_path"], checkpoint):
            raise ValueError(f"Trial {trial_id:02d} checkpoint path is not the source final model")

        validate_protocol(
            benchmark_path, paths["resolved_benchmark_path"], trial_id,
            paths["checkpoint_path"], params,
        )
        summary = read_json(paths["evaluation_summary_path"], f"evaluation summary trial {trial_id:02d}")
        if any(type(summary.get(key)) is not int or summary[key] != SCENARIOS for key in ("num_scenarios", "num_episodes")):
            raise ValueError(f"Trial {trial_id:02d} must have 1000 scenarios and 1000 episodes")
        metrics = summary.get("metrics_mean", {})
        if not isinstance(metrics, dict):
            raise ValueError(f"Trial {trial_id:02d} summary metrics_mean must be an object")
        missing = REQUIRED_METRICS - set(metrics)
        if missing:
            raise ValueError(f"Trial {trial_id:02d} summary is missing metrics: {sorted(missing)}")
        values = {metric: finite(metrics[metric], f"trial {trial_id:02d} {metric}") for metric in REQUIRED_METRICS}
        values["infractions_per_scenario"] = values["total_infractions"] / SCENARIOS
        ids = scenario_identities(paths["episode_metrics_path"], trial_id)
        identities_by_trial.append(ids)
        trials.append({
            "trial_id": trial_id, "parameters": params, "training_seed": train["seed"],
            "training_duration_seconds": duration,
            "agent_steps": record["agent_steps"], "learner_updates": record["learner_updates"],
            "num_scenarios": SCENARIOS, "num_episodes": SCENARIOS,
            "metrics": values, "paths": paths,
        })
    if seen_grid != expected_grid:
        raise ValueError("Evaluation manifest must contain the complete 2x2x2x2 SAC grid")
    if any(ids != identities_by_trial[0] for ids in identities_by_trial[1:]):
        raise ValueError("The 16 evaluations do not cover matching (map_name, scenario_id, seed) identities")

    output_dir = PROJECT_DIR / "plots" / output_name
    if output_dir.exists():
        raise FileExistsError(f"Refusing to overwrite existing output directory: {output_dir}")
    return sorted(trials, key=lambda item: item["trial_id"]), {
        "output_dir": output_dir,
        "evaluation_name": manifest["evaluation_name"],
    }


def effect_rows(trials: list[dict]) -> list[dict]:
    result = []
    for parameter in PARAMETERS:
        others = [name for name in PARAMETERS if name != parameter]
        contexts = {}
        for trial in trials:
            context = tuple(trial["parameters"][name] for name in others)
            contexts.setdefault(context, {})[trial["parameters"][parameter]] = trial
        if len(contexts) != 8:
            raise ValueError(f"Expected eight matched pairs for {parameter}")
        low, high = GRID[parameter]
        for context, pair in sorted(contexts.items()):
            low_trial, high_trial = pair[low], pair[high]
            row = {
                "parameter": parameter, "low_value": low, "high_value": high,
                "low_trial_id": low_trial["trial_id"], "high_trial_id": high_trial["trial_id"],
            }
            row.update(dict(zip(others, context, strict=True)))
            for metric in METRICS:
                row[f"{metric}_low"] = low_trial["metrics"][metric]
                row[f"{metric}_high"] = high_trial["metrics"][metric]
                row[f"{metric}_high_minus_low"] = high_trial["metrics"][metric] - low_trial["metrics"][metric]
            result.append(row)
    return result


def write_csv(path: Path, rows: list[dict]) -> None:
    with path.open("w", newline="", encoding="utf-8") as output:
        fieldnames = list(dict.fromkeys(key for row in rows for key in row))
        writer = csv.DictWriter(output, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def metric_label(metric: str) -> str:
    if metric == "comfort_score":
        direction = "known broken; not interpretable"
    elif metric in CONTEXT_METRICS:
        direction = "context only; no single better direction"
    else:
        direction = METRIC_DIRECTIONS[metric]
    return f"{metric}\n{METRIC_UNITS[metric]}; {direction}"


def overview_plot(trials: list[dict], group: str, metrics: list[str], path: Path) -> None:
    row_settings = [(e, a) for e in GRID[PARAMETERS[0]] for a in GRID[PARAMETERS[2]]]
    col_settings = [(r, t) for r in GRID[PARAMETERS[1]] for t in GRID[PARAMETERS[3]]]
    lookup = {tuple(item["parameters"][name] for name in PARAMETERS): item for item in trials}
    rows, cols = math.ceil(len(metrics) / 2), 2
    fig, axes = plt.subplots(rows, cols, figsize=(16, 5.2 * rows), squeeze=False)
    for ax, metric in zip(axes.flat, metrics, strict=False):
        values = np.zeros((4, 4))
        labels = []
        for row_idx, (epochs, alpha) in enumerate(row_settings):
            label_row = []
            for col_idx, (refresh, tau) in enumerate(col_settings):
                item = lookup[(epochs, refresh, alpha, tau)]
                value = item["metrics"][metric]
                values[row_idx, col_idx] = value
                label_row.append(f"T{item['trial_id']:02d}\n{value:.3g}" + ("\nknown broken" if metric == "comfort_score" else ""))
            labels.append(label_row)
        image = ax.imshow(values, cmap="viridis", aspect="auto")
        for row_idx in range(4):
            for col_idx in range(4):
                ax.text(col_idx, row_idx, labels[row_idx][col_idx], ha="center", va="center", fontsize=8,
                        bbox={"facecolor": "white", "alpha": 0.72, "edgecolor": "none", "pad": 1})
        ax.set_xticks(range(4), [f"refresh={r:g}\ntau={t:g}" for r, t in col_settings])
        ax.set_yticks(range(4), [f"epochs={e}, alpha={a:g}" for e, a in row_settings])
        ax.set_xlabel("Replay refresh and tau")
        ax.set_ylabel("Update epochs and initial alpha")
        ax.set_title(metric_label(metric), fontsize=9)
        fig.colorbar(image, ax=ax, fraction=0.045, pad=0.03)
    for ax in list(axes.flat)[len(metrics):]:
        ax.set_visible(False)
    fig.suptitle(GROUP_TITLES[group].replace("Benchmark Comparison:", "SAC16 CARLA:"), fontsize=15)
    fig.text(0.5, 0.006, "All training seeds are 0; single-seed endpoint comparisons. See trial_index.csv for duration and collected agent steps.", ha="center", fontsize=9)
    fig.tight_layout(rect=(0, 0.025, 1, 0.97))
    fig.savefig(path, dpi=170, bbox_inches="tight")
    plt.close(fig)


def context_label(parameter: str, row: dict) -> str:
    parts = []
    symbols = {
        "train.update_epochs": "e",
        "train.replay_update_ratio": "r",
        "train.initial_alpha": "a",
        "train.tau": "t",
    }
    for name in PARAMETERS:
        if name == parameter:
            continue
        value = row[name]
        value = f"{value * 100:g}%" if name == "train.replay_update_ratio" else f"{value:g}"
        parts.append(f"{symbols[name]}{value}")
    return ", ".join(parts)


def parameter_plot(rows: list[dict], parameter: str, group: str, metrics: list[str], path: Path) -> None:
    pairs = [row for row in rows if row["parameter"] == parameter]
    nrows, ncols = math.ceil(len(metrics) / 2), 2
    fig, axes = plt.subplots(nrows, ncols, figsize=(18, 4.5 * nrows), squeeze=False)
    for ax, metric in zip(axes.flat, metrics, strict=False):
        differences = [row[f"{metric}_high_minus_low"] for row in pairs]
        x = np.arange(len(pairs))
        ax.bar(x, differences, color="#4C78A8", alpha=0.8)
        ax.axhline(0, color="#333333", linewidth=1, label="zero")
        mean = float(np.mean(differences))
        ax.axhline(mean, color="#E45756", linestyle="--", linewidth=1.4, label=f"mean={mean:.3g}")
        ax.set_xticks(
            x,
            [
                f"{row['low_trial_id']:02d}–{row['high_trial_id']:02d}\n{context_label(parameter, row)}"
                for row in pairs
            ],
            fontsize=6,
        )
        ax.set_ylabel(f"high − low ({METRIC_UNITS[metric]})")
        ax.set_title(metric_label(metric), fontsize=9)
        ax.grid(axis="y", alpha=0.2)
        ax.set_axisbelow(True)
        ax.legend(loc="best", fontsize=7)
    for ax in list(axes.flat)[len(metrics):]:
        ax.set_visible(False)
    fig.suptitle(f"{GROUP_TITLES[group].replace('Benchmark Comparison:', 'SAC16 CARLA:')} — {parameter}", fontsize=14)
    fig.text(
        0.5, 0.006,
        "Pair labels are low–high trial IDs; context codes e=epochs, r=refresh %, a=initial alpha, t=tau. "
        "Eight matched pairs; training seed 0 only.",
        ha="center", fontsize=9,
    )
    fig.tight_layout(rect=(0, 0.025, 1, 0.96))
    fig.savefig(path, dpi=170, bbox_inches="tight")
    plt.close(fig)


def interaction_plot(trials: list[dict], group: str, metrics: list[str], path: Path) -> None:
    lookup = {tuple(item["parameters"][name] for name in PARAMETERS): item for item in trials}
    colors = {0.1: "#4C78A8", 0.2: "#F58518"}
    styles = {(3, 0.05): "-", (3, 0.1): "--", (10, 0.05): ":", (10, 0.1): "-."}
    nrows, ncols = math.ceil(len(metrics) / 2), 2
    fig, axes = plt.subplots(nrows, ncols, figsize=(16, 4.8 * nrows), squeeze=False)
    legend_handles, legend_labels = [], []
    for metric_idx, (ax, metric) in enumerate(zip(axes.flat, metrics, strict=False)):
        for epochs in GRID[PARAMETERS[0]]:
            for refresh in GRID[PARAMETERS[1]]:
                for alpha in GRID[PARAMETERS[2]]:
                    points = [lookup[(epochs, refresh, alpha, tau)] for tau in GRID[PARAMETERS[3]]]
                    values = [item["metrics"][metric] for item in points]
                    (line,) = ax.plot(GRID[PARAMETERS[3]], values, marker="o", color=colors[alpha],
                                      linestyle=styles[(epochs, refresh)], linewidth=1.5, markersize=4)
                    line_number = len(legend_labels)
                    for point_idx, (tau, value, item) in enumerate(zip(GRID[PARAMETERS[3]], values, points, strict=True)):
                        ax.annotate(f"T{item['trial_id']:02d}", (tau, value),
                                    xytext=(4 if point_idx == 0 else -22, 5 + (line_number % 4) * 4),
                                    textcoords="offset points", fontsize=6)
                    if metric_idx == 0:
                        legend_handles.append(line)
                        legend_labels.append(f"alpha={alpha:g}, epochs={epochs}, refresh={refresh * 100:g}%")
        ax.set_xticks(GRID[PARAMETERS[3]], [f"{value:g}" for value in GRID[PARAMETERS[3]]])
        ax.set_xlabel("Target-network tau")
        ax.set_ylabel(METRIC_UNITS[metric])
        ax.set_title(metric_label(metric), fontsize=9)
        ax.margins(x=0.18)
        ax.grid(alpha=0.22)
        ax.set_axisbelow(True)
    for ax in list(axes.flat)[len(metrics):]:
        ax.set_visible(False)
    fig.legend(legend_handles, legend_labels, loc="upper center", bbox_to_anchor=(0.5, 0.935),
               ncol=4, fontsize=8)
    fig.suptitle(
        GROUP_TITLES[group].replace("Benchmark Comparison:", "SAC16 CARLA:")
        + " — alpha–tau interactions",
        fontsize=14,
    )
    fig.text(0.5, 0.006, "Lines compare alpha across tau; labels show trial ID and raw metric. Training seed 0 only.", ha="center", fontsize=9)
    fig.tight_layout(rect=(0, 0.025, 1, 0.89))
    fig.savefig(path, dpi=170, bbox_inches="tight")
    plt.close(fig)


def planned_outputs(output_dir: Path) -> list[Path]:
    outputs = [output_dir / name for name in ("trial_index.csv", "aggregate_metrics.csv", "parameter_effects.csv")]
    outputs += [output_dir / "overview" / f"{group}.png" for group in METRIC_GROUPS]
    outputs += [
        output_dir / "parameter_effects" / PARAMETER_DIRS[param] / f"{group}.png"
        for param in PARAMETERS for group in METRIC_GROUPS
    ]
    outputs += [output_dir / "interactions" / f"{group}.png" for group in METRIC_GROUPS]
    return outputs


def write_outputs(trials: list[dict], output_dir: Path) -> list[Path]:
    output_dir.mkdir(parents=True, exist_ok=False)
    for folder in ("overview", "parameter_effects", "interactions"):
        (output_dir / folder).mkdir()
    index_rows, aggregate_rows = [], []
    for item in trials:
        common = {
            "trial_id": item["trial_id"], **item["parameters"],
            "training_seed": item["training_seed"],
            "training_duration_seconds": item["training_duration_seconds"],
            "agent_steps": item["agent_steps"], "learner_updates": item["learner_updates"],
        }
        index_rows.append({**common, **{key: str(value) for key, value in item["paths"].items()}})
        aggregate_rows.append({
            **common, "num_scenarios": item["num_scenarios"], "num_episodes": item["num_episodes"],
            **item["metrics"], **{f"{key}_path": str(value) for key, value in item["paths"].items()},
        })
    matched = effect_rows(trials)
    write_csv(output_dir / "trial_index.csv", index_rows)
    write_csv(output_dir / "aggregate_metrics.csv", aggregate_rows)
    write_csv(output_dir / "parameter_effects.csv", matched)
    for group, metrics in METRIC_GROUPS.items():
        overview_plot(trials, group, metrics, output_dir / "overview" / f"{group}.png")
        interaction_plot(trials, group, metrics, output_dir / "interactions" / f"{group}.png")
    for parameter in PARAMETERS:
        folder = output_dir / "parameter_effects" / PARAMETER_DIRS[parameter]
        folder.mkdir()
        for group, metrics in METRIC_GROUPS.items():
            parameter_plot(matched, parameter, group, metrics, folder / f"{group}.png")
    return planned_outputs(output_dir)


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--evaluation-manifest", type=Path, required=True)
    parser.add_argument("--output-name", help="Safe folder name under the SAC project plots directory")
    parser.add_argument("--dry-run", action="store_true", help="Validate inputs and print planned outputs only")
    args = parser.parse_args(argv)
    try:
        manifest_path = args.evaluation_manifest.expanduser().resolve()
        output_name = args.output_name
        if output_name is None:
            manifest = read_json(manifest_path, "evaluation manifest")
            source_path = manifest.get("source_sweep_manifest")
            if not isinstance(source_path, str) or not Path(source_path).is_absolute():
                raise ValueError("source_sweep_manifest must be an absolute path")
            output_name = f"{Path(source_path).parent.name}_carla_final"
        trials, metadata = load_inputs(manifest_path, output_name)
        outputs = planned_outputs(metadata["output_dir"])
        print(f"Validated {len(trials)} completed trials for {metadata['evaluation_name']}.")
        print(f"Output directory: {metadata['output_dir']}")
        if args.dry_run:
            for path in outputs:
                print(f"Would write: {path}")
            return 0
        for path in write_outputs(trials, metadata["output_dir"]):
            print(path)
        return 0
    except (FileNotFoundError, FileExistsError, ValueError, OSError) as error:
        print(f"Error: {error}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
