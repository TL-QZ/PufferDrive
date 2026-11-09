#!/usr/bin/env python3
"""Create reproducible PPO-anchor-matched SAC learning curves from saved evals."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import os
import re
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import yaml


ROOT = Path(__file__).resolve().parent
REPO_ROOT = ROOT.parents[1]
CURVES_DIR = ROOT / "curves"
RAW_CURVES_PATH = CURVES_DIR / "all_curves_raw.csv"
DEDUPED_CURVES_PATH = CURVES_DIR / "deduplicated_curves.csv"
MATCHED_POINTS_PATH = CURVES_DIR / "matched_points.csv"
REPORT_PATH = ROOT / "learning_curve_report.md"
PLOT_PATH = CURVES_DIR / "learning_curves"
STEP_PATTERN = re.compile(r"epoch_(\d+)_step_(\d+)$")
METRICS = (
    "score",
    "puffer_score",
    "progress_ratio",
    "making_progress_rate",
    "num_goals_reached",
    "collision_rate",
    "at_fault_collision_rate",
    "offroad_rate",
    "red_light_violation_rate",
    "avg_distance_per_infraction",
    "total_distance_travelled",
    "avg_speed_per_agent",
)


@dataclass(frozen=True)
class SourceSpec:
    label: str
    path: str
    priority: int


@dataclass(frozen=True)
class RunSpec:
    run_id: str
    label: str
    sources: tuple[SourceSpec, ...]


SAC_RUNS = (
    RunSpec(
        "round2_tes_sac_20261029T044847.014369240_539543_trial_98",
        "TES R2",
        (SourceSpec("original", "project/soft_actor_critic_target_entropy_scheduling/runs/round2_tes_sac_20261029T044847.014369240_539543/trial_98", 0),),
    ),
    RunSpec(
        "sac108_grid_20261031T194034Z658157854_trial_98",
        "n=4 R2",
        (SourceSpec("original", "project/soft_actor_critic_multi_step_return/runs/round2_nstep4/sac108_grid_20261031T194034Z658157854/trial_98", 0),),
    ),
    RunSpec(
        "sac108_grid_20261102T220540Z788100439_trial_98",
        "n=2 TES R4.1",
        (
            SourceSpec("original", "project/soft_actor_critic_multi_step_return/runs/round4_1_nstep2_tes_sac/sac108_grid_20261102T220540Z788100439/trial_98", 0),
            SourceSpec("recovery 2026-11-04", "project/run_recovery/2026-11-04_seqamgpu01/nstep2_tes", 1),
            SourceSpec("recovery 2026-11-05", "project/run_recovery/2026-11-05_seqamgpu01/nstep2_tes", 2),
        ),
    ),
    RunSpec(
        "sac108_grid_20261104T051212Z257259744_trial_98",
        "PQL R4.1",
        (
            SourceSpec("original", "project/soft_actor_critic_multi_step_return_PQL/runs/round4_1_pql/sac108_grid_20261104T051212Z257259744/trial_98", 0),
            SourceSpec("recovery 2026-11-04", "project/run_recovery/2026-11-04_seqamgpu01/pql_round4_1", 1),
            SourceSpec("recovery 2026-11-05", "project/run_recovery/2026-11-05_seqamgpu01/pql_round4_1", 2),
        ),
    ),
    RunSpec(
        "run_98_20261104T225607Z355122223_trial_98",
        "PQL official 1",
        (SourceSpec("original", "project/soft_actor_critic_multi_step_return_PQL/runs/PQL_official_round1/run_98_20261104T225607Z355122223/trial_98", 0),),
    ),
    RunSpec(
        "sac1_grid_20261104T230325Z530796082_trial_01",
        "low-LR TES R2",
        (SourceSpec("original", "project/soft_actor_critic_multi_step_return/runs/round2_enhance_low_lr_slow_tes/sac1_grid_20261104T230325Z530796082/trial_01", 0),),
    ),
    RunSpec(
        "run_98_20261104T232616Z717021155_trial_98",
        "PQL official 2",
        (
            SourceSpec("original", "project/soft_actor_critic_multi_step_return_PQL/runs/PQL_official_round2/run_98_20261104T232616Z717021155/trial_98", 0),
            SourceSpec("recovery 2026-11-04", "project/run_recovery/2026-11-04_seqamgpu01/pql_round2", 1),
            SourceSpec("recovery 2026-11-05", "project/run_recovery/2026-11-05_seqamgpu01/pql_round2", 2),
        ),
    ),
)

PPO_RUN_ID = "baseline_ppo_4b_carla_2026-11-06_13-58-32_seed0"
PPO_RUN = RunSpec(
    PPO_RUN_ID,
    "PPO 4B",
    (SourceSpec("PPO baseline", "experiments/baseline_ppo_4b_carla/baseline_ppo_4b_carla_2026-11-06_13-58-32_seed0", 0),),
)
ALL_RUNS = SAC_RUNS + (PPO_RUN,)


def relative_path(path: Path) -> str:
    try:
        return path.resolve().relative_to(REPO_ROOT).as_posix()
    except ValueError:
        return path.as_posix()


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as input_file:
        for chunk in iter(lambda: input_file.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def sha256_bytes(contents: bytes) -> str:
    return hashlib.sha256(contents).hexdigest()


def numeric(value: Any) -> float | None:
    if isinstance(value, bool):
        return float(value)
    if isinstance(value, (int, float)) and math.isfinite(float(value)):
        return float(value)
    return None


def optional_number(value: Any) -> str:
    number = numeric(value)
    return "" if number is None else format(number, ".12g")


def source_files(source_root: Path, is_ppo: bool) -> list[Path]:
    relative_eval_root = "eval/training/carla_fast" if is_ppo else "learner/eval/training/carla_fast"
    eval_root = source_root / relative_eval_root
    return sorted(eval_root.glob("epoch_*/evaluation_summary.json"))


def resolved_metadata(summary_path: Path, source_root: Path) -> dict[str, Any]:
    resolved_path = summary_path.with_name("resolved_benchmark.yaml")
    resolved: dict[str, Any] = {}
    if resolved_path.is_file():
        loaded = yaml.safe_load(resolved_path.read_text(encoding="utf-8"))
        if isinstance(loaded, dict):
            resolved = loaded

    args = resolved.get("args", {}) if isinstance(resolved.get("args", {}), dict) else {}
    eval_env = args.get("env", {}) if isinstance(args.get("env", {}), dict) else {}
    benchmark = resolved.get("benchmark", {}) if isinstance(resolved.get("benchmark", {}), dict) else {}

    config_path = source_root / "learner/config.yaml"
    if not config_path.is_file():
        config_path = source_root / "config.yaml"
    if not config_path.is_file():
        trial_id = "trial_01" if source_root.name.endswith("trial_01") else "trial_98"
        config_path = source_root.parent / "configs" / f"{trial_id}.yaml"
    saved_config: dict[str, Any] = {}
    if config_path.is_file():
        loaded_config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
        if isinstance(loaded_config, dict):
            saved_config = loaded_config
    saved_env = saved_config.get("env", {}) if isinstance(saved_config.get("env", {}), dict) else {}
    saved_eval = saved_config.get("eval", {}) if isinstance(saved_config.get("eval", {}), dict) else {}

    return {
        "resolved_benchmark_path": relative_path(resolved_path) if resolved_path.is_file() else "",
        "resolved_benchmark_sha256": sha256(resolved_path) if resolved_path.is_file() else "",
        "benchmark_config_path": resolved.get("benchmark_config", ""),
        "benchmark_name": benchmark.get("name", ""),
        "benchmark_seed": optional_number(benchmark.get("seed")),
        "num_scenarios": optional_number(benchmark.get("num_scenarios")),
        "eval_dt_seconds": optional_number(eval_env.get("dt")),
        "eval_min_goal_spacing_m": optional_number(eval_env.get("min_goal_spacing")),
        "eval_max_goal_spacing_m": optional_number(eval_env.get("max_goal_spacing")),
        "train_dt_seconds": optional_number(saved_env.get("dt")),
        "train_min_goal_spacing_m": optional_number(saved_env.get("min_goal_spacing")),
        "train_max_goal_spacing_m": optional_number(saved_env.get("max_goal_spacing")),
        "train_benchmark_config_path": saved_eval.get("benchmark_config", ""),
        "resolved_run_name": args.get("run_name", ""),
        "saved_config_path": relative_path(config_path) if config_path.is_file() else "",
        "saved_config_sha256": sha256(config_path) if config_path.is_file() else "",
    }


def gather_raw_points(snapshot_utc: str) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for run in ALL_RUNS:
        for source in run.sources:
            source_root = REPO_ROOT / source.path
            files = source_files(source_root, run.run_id == PPO_RUN_ID)
            if not files:
                raise FileNotFoundError(f"No saved carla_fast evaluations found for {run.run_id}: {source_root}")
            metadata = resolved_metadata(files[0], source_root)
            if metadata["resolved_run_name"] and metadata["resolved_run_name"] != run.run_id:
                raise ValueError(
                    f"Saved run identity does not match configured continuation mapping: "
                    f"{source_root} resolved as {metadata['resolved_run_name']}, expected {run.run_id}"
                )
            if len(files) > 1:
                last_metadata = resolved_metadata(files[-1], source_root)
                compare_fields = (
                    "benchmark_config_path",
                    "benchmark_name",
                    "benchmark_seed",
                    "eval_dt_seconds",
                    "eval_min_goal_spacing_m",
                    "eval_max_goal_spacing_m",
                    "resolved_run_name",
                )
                if any(metadata[field] != last_metadata[field] for field in compare_fields):
                    raise ValueError(f"Resolved benchmark settings change within source directory: {source_root}")
            for summary_path in files:
                match = STEP_PATTERN.fullmatch(summary_path.parent.name)
                if match is None:
                    raise ValueError(f"Unexpected evaluation directory name: {summary_path.parent}")
                summary_bytes = summary_path.read_bytes()
                summary = json.loads(summary_bytes)
                if not isinstance(summary, dict) or not isinstance(summary.get("metrics_mean"), dict):
                    raise ValueError(f"Invalid evaluation summary: {summary_path}")
                resolved_path = summary_path.with_name("resolved_benchmark.yaml")
                resolved_hash = sha256(resolved_path) if resolved_path.is_file() else ""
                row_metadata = dict(metadata)
                row_metadata["resolved_benchmark_path"] = relative_path(resolved_path) if resolved_path.is_file() else ""
                row_metadata["resolved_benchmark_sha256"] = resolved_hash
                if resolved_hash and resolved_hash != metadata["resolved_benchmark_sha256"]:
                    row_metadata = resolved_metadata(summary_path, source_root)
                    if row_metadata["resolved_run_name"] and row_metadata["resolved_run_name"] != run.run_id:
                        raise ValueError(
                            f"Saved run identity changed within source directory: {summary_path} "
                            f"resolved as {row_metadata['resolved_run_name']}, expected {run.run_id}"
                        )
                row: dict[str, Any] = {
                    "snapshot_created_utc": snapshot_utc,
                    "run_id": run.run_id,
                    "run_label": run.label,
                    "run_type": "PPO" if run.run_id == PPO_RUN_ID else "SAC",
                    "source_label": source.label,
                    "source_priority": source.priority,
                    "agent_steps": int(match.group(2)),
                    "epoch": int(match.group(1)),
                    "source_root": relative_path(source_root),
                    "summary_path": relative_path(summary_path),
                    "summary_sha256": sha256_bytes(summary_bytes),
                    "num_scenarios": summary.get("num_scenarios", metadata["num_scenarios"]),
                    "num_episodes": summary.get("num_episodes", ""),
                }
                row.update(row_metadata)
                for metric in METRICS:
                    row[metric] = optional_number(summary["metrics_mean"].get(metric))
                rows.append(row)
    return rows


def read_raw_snapshot(path: Path) -> list[dict[str, Any]]:
    with path.open(newline="", encoding="utf-8") as input_file:
        rows = list(csv.DictReader(input_file))
    if not rows:
        raise ValueError(f"Snapshot has no rows: {path}")
    for row in rows:
        row["agent_steps"] = int(row["agent_steps"])
        row["epoch"] = int(row["epoch"])
        row["source_priority"] = int(row["source_priority"])
        row["run_type"] = "PPO" if row["run_id"] == PPO_RUN_ID else "SAC"
        if "avg_speed_per_agent" not in row:
            # Enrich older snapshots only from the exact recorded source bytes.
            summary_path = REPO_ROOT / row["summary_path"]
            summary_bytes = summary_path.read_bytes()
            if sha256_bytes(summary_bytes) != row["summary_sha256"]:
                raise ValueError(f"Snapshot source changed: {summary_path}")
            speed = numeric(json.loads(summary_bytes)["metrics_mean"].get("avg_speed_per_agent"))
            if speed is None:
                raise ValueError(f"Missing or invalid average speed: {summary_path}")
            row["avg_speed_per_agent"] = optional_number(speed)
    return rows


def write_csv(path: Path, rows: list[dict[str, Any]], fields: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as output_file:
        writer = csv.DictWriter(output_file, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def deduplicate_points(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    # A recovered continuation wins at duplicate global steps; ties use epoch then path.
    chosen: dict[tuple[str, int], dict[str, Any]] = {}
    for row in rows:
        key = (row["run_id"], row["agent_steps"])
        current = chosen.get(key)
        priority = (row["source_priority"], row["epoch"], row["summary_path"])
        if current is None or priority > (current["source_priority"], current["epoch"], current["summary_path"]):
            chosen[key] = row
    return sorted(chosen.values(), key=lambda row: (row["run_id"], row["agent_steps"], row["epoch"]))


def evenly_spaced_indices(length: int, count: int) -> list[int]:
    if count <= 0 or count > length:
        raise ValueError(f"Invalid requested subset size {count} for {length} items")
    if count == length:
        return list(range(length))
    if count == 1:
        return [length // 2]
    indices = [round(index * (length - 1) / (count - 1)) for index in range(count)]
    if len(set(indices)) != count:
        raise AssertionError("Evenly spaced anchor selection produced duplicate indices")
    return indices


def minimum_cost_monotone_match(
    anchors: list[int], observation_steps: list[int]
) -> list[tuple[int, int]]:
    """Match the smaller sequence to unique points in the larger, minimizing step error."""
    if not anchors or not observation_steps:
        return []
    anchors = sorted(set(anchors))
    observation_steps = sorted(set(observation_steps))
    if len(anchors) > len(observation_steps):
        anchors = [anchors[index] for index in evenly_spaced_indices(len(anchors), len(observation_steps))]

    match_count = len(anchors)
    point_count = len(observation_steps)
    previous_costs = [abs(observation_steps[index] - anchors[0]) for index in range(point_count)]
    parents = [[-1] * point_count for _ in range(match_count)]

    for anchor_index in range(1, match_count):
        current_costs = [math.inf] * point_count
        prefix_cost = math.inf
        prefix_index = -1
        for point_index in range(point_count):
            prior_index = point_index - 1
            if prior_index >= 0 and previous_costs[prior_index] < prefix_cost:
                prefix_cost = previous_costs[prior_index]
                prefix_index = prior_index
            if point_index < anchor_index or prefix_index < 0:
                continue
            current_costs[point_index] = prefix_cost + abs(observation_steps[point_index] - anchors[anchor_index])
            parents[anchor_index][point_index] = prefix_index
        previous_costs = current_costs

    final_index = min(range(match_count - 1, point_count), key=lambda index: (previous_costs[index], index))
    chosen_indices = [final_index]
    for anchor_index in range(match_count - 1, 0, -1):
        previous_index = parents[anchor_index][chosen_indices[-1]]
        if previous_index < 0:
            raise AssertionError("Could not backtrack step matching")
        chosen_indices.append(previous_index)
    chosen_indices.reverse()
    return list(zip(anchors, (observation_steps[index] for index in chosen_indices)))


def select_matched_points(rows: list[dict[str, Any]]) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    ppo_steps = sorted({row["agent_steps"] for row in rows if row["run_id"] == PPO_RUN_ID})
    if not ppo_steps:
        raise ValueError("PPO anchor evaluations were not found in the input snapshot")
    deduped = deduplicate_points(rows)
    sac_by_run: dict[str, list[dict[str, Any]]] = {}
    for row in deduped:
        if row["run_type"] == "SAC":
            sac_by_run.setdefault(row["run_id"], []).append(row)

    matches: list[dict[str, Any]] = []
    coverage: dict[str, Any] = {"ppo_anchor_count": len(ppo_steps), "runs": {}}
    for run in SAC_RUNS:
        points = sorted(sac_by_run[run.run_id], key=lambda row: row["agent_steps"])
        low_step = points[0]["agent_steps"]
        high_step = points[-1]["agent_steps"]
        covered_anchors = [step for step in ppo_steps if low_step <= step <= high_step]
        pairs = minimum_cost_monotone_match(covered_anchors, [row["agent_steps"] for row in points])
        by_step = {row["agent_steps"]: row for row in points}
        for anchor_step, sac_step in pairs:
            row = dict(by_step[sac_step])
            row["ppo_anchor_steps"] = anchor_step
            row["signed_offset_steps"] = sac_step - anchor_step
            row["absolute_offset_steps"] = abs(sac_step - anchor_step)
            matches.append(row)
        coverage["runs"][run.run_id] = {
            "run_label": run.label,
            "raw_point_count": sum(1 for row in rows if row["run_id"] == run.run_id),
            "unique_observation_count": len(points),
            "coverage_min_step": low_step,
            "coverage_max_step": high_step,
            "covered_ppo_anchor_count": len(covered_anchors),
            "selected_point_count": len(pairs),
            "selected_step_min": min((pair[1] for pair in pairs), default=""),
            "selected_step_max": max((pair[1] for pair in pairs), default=""),
            "signed_offset_min": min((pair[1] - pair[0] for pair in pairs), default=""),
            "signed_offset_max": max((pair[1] - pair[0] for pair in pairs), default=""),
            "max_abs_offset_steps": max((abs(pair[1] - pair[0]) for pair in pairs), default=""),
        }
        assert len({row["agent_steps"] for row in matches if row["run_id"] == run.run_id}) == len(pairs)
        assert all(low_step <= row["agent_steps"] <= high_step for row in matches if row["run_id"] == run.run_id)
    matches.sort(key=lambda row: (row["run_id"], row["ppo_anchor_steps"]))
    return matches, coverage


def summarize_metadata(rows: list[dict[str, Any]], run_id: str) -> dict[str, list[str]]:
    run_rows = [row for row in rows if row["run_id"] == run_id]
    fields = (
        "eval_dt_seconds",
        "eval_min_goal_spacing_m",
        "eval_max_goal_spacing_m",
        "train_dt_seconds",
        "train_min_goal_spacing_m",
        "train_max_goal_spacing_m",
        "benchmark_name",
        "benchmark_seed",
        "num_scenarios",
    )
    return {
        field: sorted({str(row.get(field, "")) for row in run_rows if str(row.get(field, "")) != ""})
        for field in fields
    }


def display_values(values: list[str], suffix: str = "") -> str:
    if not values:
        return "not recorded"
    rendered = []
    for value in values:
        try:
            number = float(value)
            rendered.append(f"{number:g}{suffix}")
        except ValueError:
            rendered.append(value)
    return ", ".join(rendered)


def write_run_data(rows: list[dict[str, Any]], matches: list[dict[str, Any]]) -> None:
    CURVES_DIR.mkdir(parents=True, exist_ok=True)
    raw_fields = list(rows[0])
    write_csv(RAW_CURVES_PATH, rows, raw_fields)

    deduped = deduplicate_points(rows)
    match_by_key = {(row["run_id"], row["agent_steps"]): row for row in matches}
    for row in deduped:
        match = match_by_key.get((row["run_id"], row["agent_steps"]))
        row["is_matched"] = 1 if match else 0
        row["ppo_anchor_steps"] = match["ppo_anchor_steps"] if match else ""
        row["signed_offset_steps"] = match["signed_offset_steps"] if match else ""
    deduped_fields = list(deduped[0])
    write_csv(DEDUPED_CURVES_PATH, deduped, deduped_fields)
    write_csv(MATCHED_POINTS_PATH, matches, raw_fields + ["ppo_anchor_steps", "signed_offset_steps", "absolute_offset_steps"])


def plot_curves(
    rows: list[dict[str, Any]], matches: list[dict[str, Any]],
    runs: tuple[RunSpec, ...] = ALL_RUNS, plot_path: Path = PLOT_PATH,
) -> None:
    os.environ.setdefault("MPLCONFIGDIR", str(CURVES_DIR / ".matplotlib-cache"))
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    deduped = deduplicate_points(rows)
    matched_keys = {(row["run_id"], row["agent_steps"]) for row in matches}
    run_colors = {
        SAC_RUNS[0].run_id: "#3769a5",
        SAC_RUNS[1].run_id: "#d17b18",
        SAC_RUNS[2].run_id: "#27916b",
        SAC_RUNS[3].run_id: "#9655a0",
        SAC_RUNS[4].run_id: "#b64a53",
        SAC_RUNS[5].run_id: "#36a4b7",
        SAC_RUNS[6].run_id: "#7a7a7a",
        PPO_RUN_ID: "#111111",
    }
    metric_specs = (
        ("puffer_score", "Puffer score", 1.0),
        ("score", "Raw score", 1.0),
        ("progress_ratio", "Progress ratio", 1.0),
        ("making_progress_rate", "Making progress (%)", 100.0),
        ("num_goals_reached", "Goals reached (reported mean)", 1.0),
        ("collision_rate", "Collision rate (%)", 100.0),
        ("at_fault_collision_rate", "At-fault collision rate (%)", 100.0),
        ("offroad_rate", "Off-road rate (%)", 100.0),
        ("red_light_violation_rate", "Red-light violation rate (%)", 100.0),
        ("avg_distance_per_infraction", "Pooled distance per infraction (m)", 1.0),
        ("total_distance_travelled", "Summed distance (km)", 1.0 / 1000.0),
        ("avg_speed_per_agent", "Average speed (m/s)", 1.0),
    )
    labels = {run.run_id: run.label for run in runs}
    for run in runs:
        spacing = summarize_metadata(rows, run.run_id)["eval_min_goal_spacing_m"]
        if spacing:
            labels[run.run_id] += f" · {display_values(spacing)}m goals"
    fig, axes = plt.subplots(4, 3, figsize=(16, 16), sharex=True)
    fig.text(
        0.5,
        0.99,
        "CARLA fast learning curves aligned to PPO evaluation anchors",
        ha="center",
        va="top",
        fontsize=15,
        fontweight="bold",
    )
    for axis, (metric, ylabel, scale) in zip(axes.flat, metric_specs):
        for run in runs:
            run_rows = [row for row in deduped if row["run_id"] == run.run_id]
            run_rows.sort(key=lambda row: row["agent_steps"])
            x_values = [row["agent_steps"] / 1e9 for row in run_rows]
            y_values = [float(row[metric]) * scale if row.get(metric, "") != "" else math.nan for row in run_rows]
            color = run_colors[run.run_id]
            if run.run_id == PPO_RUN_ID:
                axis.plot(x_values, y_values, color=color, linewidth=1.6, alpha=0.9, label=labels[run.run_id], zorder=3)
                continue
            axis.plot(x_values, y_values, color=color, linewidth=0.8, alpha=0.18, zorder=1)
            selected_rows = [row for row in run_rows if (run.run_id, row["agent_steps"]) in matched_keys]
            selected_rows.sort(key=lambda row: row["agent_steps"])
            selected_x = [row["agent_steps"] / 1e9 for row in selected_rows]
            selected_y = [float(row[metric]) * scale if row.get(metric, "") != "" else math.nan for row in selected_rows]
            axis.plot(
                selected_x,
                selected_y,
                color=color,
                linewidth=1.45,
                marker="o",
                markersize=2.1,
                alpha=0.95,
                label=labels[run.run_id],
                zorder=2,
            )
        axis.set_ylabel(ylabel)
        axis.grid(True, alpha=0.22)
        axis.spines["top"].set_visible(False)
        axis.spines["right"].set_visible(False)
    for axis in axes[-1, :]:
        axis.set_xlabel("Received agent steps (billions)")
    handles, legend_labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(handles, legend_labels, loc="lower center", ncol=4, fontsize=8, frameon=False, bbox_to_anchor=(0.5, 0.005))
    fig.text(
        0.5,
        0.96,
        ("Historical protocols differ (10 m / 30 m goals); cross-run curves are descriptive."
         if len({row["eval_min_goal_spacing_m"] for row in rows}) > 1
         else "Saved CARLA fast evaluations · 30 m goals · evaluation dt = 0.1 s · one seed per run"),
        ha="center",
        fontsize=9,
        color="#555555",
    )
    fig.text(
        0.5,
        0.94,
        "Faint lines show all saved SAC evaluations; bold markers show distinct PPO-anchor matches at actual SAC steps.",
        ha="center",
        fontsize=8,
    )
    fig.text(
        0.5,
        0.92,
        "Distance panels: total distance sums evaluated agent episodes (km); per-infraction distance is pooled across episodes (m).",
        ha="center",
        fontsize=8,
        color="#555555",
    )
    for axis in axes[-1, :2]:
        axis.set_xlabel("Received agent steps (billions)")
    fig.tight_layout(rect=(0.02, 0.085, 0.99, 0.89))
    CURVES_DIR.mkdir(parents=True, exist_ok=True)
    fig.savefig(plot_path.with_suffix(".png"), dpi=220, bbox_inches="tight")
    fig.savefig(plot_path.with_suffix(".pdf"), bbox_inches="tight")
    plt.close(fig)


def render_offset(value: Any) -> str:
    return f"{int(value):+,}" if value != "" else "—"


def write_report(
    rows: list[dict[str, Any]], matches: list[dict[str, Any]], coverage: dict[str, Any]
) -> None:
    snapshot_utc = str(rows[0].get("snapshot_created_utc", "unknown"))
    ppo_rows = [row for row in rows if row["run_id"] == PPO_RUN_ID]
    ppo_steps = sorted({row["agent_steps"] for row in ppo_rows})
    lines = [
        "# Reproducible learning-curve analysis",
        "",
        f"Snapshot captured at **{snapshot_utc} UTC** from saved `carla_fast` evaluation summaries. No new evaluations or training were run.",
        "",
        "## Selection method",
        "",
        f"The PPO run contributes {len(ppo_steps)} saved global received-agent-step evaluation anchors ({ppo_steps[0]:,} to {ppo_steps[-1]:,}). For each SAC logical run, I retain every unique saved `carla_fast` observation in the full curve and select points only for PPO anchors inside that run's observed minimum–maximum step range. No PPO anchor outside the run's coverage is used, and no SAC observation is selected more than once.",
        "",
        "When multiple feasible assignments exist, the script uses a deterministic monotone assignment that minimizes the total absolute step distance between SAC observations and PPO anchors. If a run has fewer unique observations than covered anchors, it selects evenly spaced anchors including the coverage endpoints before matching. The assignment uses only step coordinates and coverage; it never reads score or other metric values. Signed offset is `SAC agent_steps - PPO anchor steps`, so positive values mean the selected SAC evaluation occurred later.",
        "",
        "The raw CSV preserves every source summary, including duplicate steps and recovery provenance. The deduplicated full curve keeps the highest-priority recovery source at repeated global steps, then the greater epoch and path as deterministic tie-breakers. The matched CSV preserves the actual SAC evaluation step and PPO anchor for each selected point. The plot's faint lines show full SAC observations; bold lines and markers show selected observations at their actual SAC x coordinates. PPO is shown at its saved evaluation coordinates.",
        "",
        "Reproduce the saved snapshot and outputs without rereading changing run directories with `source .venv/bin/activate` followed by `python project/soft_actor_critic_4b_runs_analysis/analyze_learning_curves.py --from-snapshot project/soft_actor_critic_4b_runs_analysis/curves/all_curves_raw.csv`. To capture a newer point-in-time source snapshot, rerun the script without `--from-snapshot`.",
        "",
        "## Run coverage and selected counts",
        "",
        "All step values below are global received-agent steps. `Raw` counts source summaries before duplicate-step resolution; `unique` counts the full curve after resolution. Each run's selected count equals its covered PPO anchors where enough unique SAC points exist.",
        "",
        "| SAC run | Raw | Unique | Full step range | PPO anchors in coverage | Selected | Selected SAC step range | Signed offset range (SAC − PPO) | Max absolute offset |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    all_by_run: dict[str, list[dict[str, Any]]] = {}
    for row in rows:
        all_by_run.setdefault(row["run_id"], []).append(row)
    for run in SAC_RUNS:
        stats = coverage["runs"][run.run_id]
        lines.append(
            f"| {run.label} (`{run.run_id}`) | {stats['raw_point_count']} | {stats['unique_observation_count']} | {stats['coverage_min_step']:,}–{stats['coverage_max_step']:,} | {stats['covered_ppo_anchor_count']} | {stats['selected_point_count']} | {stats['selected_step_min']:,}–{stats['selected_step_max']:,} | {render_offset(stats['signed_offset_min'])} to {render_offset(stats['signed_offset_max'])} | {stats['max_abs_offset_steps']:,} |"
        )

    lines.extend(["", "## Native evaluation settings and provenance", ""])
    lines.append(
        "These are historical native `carla_fast` evaluations. Their resolved evaluation `dt` is 0.1 s, while saved learner configs use a training `dt` of 0.3 s. Goal spacing differs across the saved evaluation catalogs, so the plotted cross-run curves are not a controlled benchmark comparison. In particular, the 10 m and 30 m goal catalogs can change goal counts and score behavior. The table reports the settings embedded in each run's resolved benchmark files; it does not normalize or rewrite those evaluations."
    )
    lines.extend(
        [
            "",
            "| Run | Evaluation dt (s) | Evaluation goal spacing (m) | Training dt (s) | Training goal spacing (m) | Saved sources |",
            "|---|---:|---:|---:|---:|---|",
        ]
    )
    for run in SAC_RUNS + (PPO_RUN,):
        metadata = summarize_metadata(rows, run.run_id)
        spacing = f"{display_values(metadata['eval_min_goal_spacing_m'])}–{display_values(metadata['eval_max_goal_spacing_m'])}"
        train_spacing = f"{display_values(metadata['train_min_goal_spacing_m'])}–{display_values(metadata['train_max_goal_spacing_m'])}"
        sources = sorted({row["source_label"] for row in all_by_run.get(run.run_id, [])})
        lines.append(
            f"| {run.label} | {display_values(metadata['eval_dt_seconds'])} | {spacing} | {display_values(metadata['train_dt_seconds'])} | {train_spacing} | {', '.join(sources)} |"
        )
    lines.extend(
        [
            "",
            "PQL official round 2 trial 98 is included through its completed 4B training history. The low-LR TES trial 01 remains in these curves while excluded from the current full evaluation set. Recovered continuation points are merged into the original logical run using their saved `run_name`; their source path and file hash remain in `all_curves_raw.csv`.",
            "",
            "## Distance metric aggregation",
            "",
            "The environment adds each completed agent episode's `distance_since_spawn` to traveled distance and counts at most one infraction for that agent episode when any off-road, collision, or red-light violation occurred ([drive.h](../../pufferlib/ocean/drive/drive.h), [binding.c](../../pufferlib/ocean/drive/binding.c)). The evaluation reducer sums distance and infraction counts across episode records, then computes `avg_distance_per_infraction` as total distance divided by `max(total_infraction_count, 1)` ([utils.py](../../pufferlib/utils.py), [evaluation_utils.py](../../pufferlib/ocean/evaluation_utils/evaluation_utils.py)). Thus the plotted total distance is a genuine sum over the evaluated agent episodes, converted from meters to kilometers; it is not a per-scenario mean. The plotted distance per infraction is a pooled distance-to-affected-agent-episode ratio in meters, not an average of episode ratios or literal spacing between individual infraction events. The raw CSV retains the source values in meters.",
            "",
            "Recovered portions came from learner-only restarts: replay and environment state were restarted, so these are resumed training histories rather than exact uninterrupted rollouts. Each named run is one training seed; these curves do not provide independent-seed replication.",
            "",
            "## Artifacts",
            "",
            "- `curves/learning_curves.png` and `curves/learning_curves.pdf`: twelve score, progress, goal, safety, distance, and average-speed metrics.",
            "- `curves/all_curves_raw.csv`: every saved evaluation summary and source provenance, before duplicate-step resolution.",
            "- `curves/deduplicated_curves.csv`: the full unique-step series, with a `is_matched` flag and matched anchor/offset fields.",
            "- `curves/matched_points.csv`: selected SAC points with actual step, PPO anchor, and signed/absolute offsets.",
            "",
            "`all_curves_raw.csv` is the frozen input snapshot for the `--from-snapshot` reproduction command above. Its summary SHA-256 values identify the exact source JSON files used.",
            "",
        ]
    )
    REPORT_PATH.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--from-snapshot",
        type=Path,
        help="reproduce outputs from an existing all_curves_raw.csv snapshot",
    )
    args = parser.parse_args()

    if args.from_snapshot:
        snapshot_path = args.from_snapshot
        if not snapshot_path.is_absolute():
            snapshot_path = REPO_ROOT / snapshot_path
        rows = read_raw_snapshot(snapshot_path)
    else:
        snapshot_utc = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S")
        rows = gather_raw_points(snapshot_utc)

    matches, coverage = select_matched_points(rows)
    write_run_data(rows, matches)
    plot_curves(rows, matches)
    write_report(rows, matches, coverage)

    print(f"Wrote {RAW_CURVES_PATH}")
    print(f"Wrote {DEDUPED_CURVES_PATH}")
    print(f"Wrote {MATCHED_POINTS_PATH}")
    print(f"Wrote {PLOT_PATH.with_suffix('.png')} and {PLOT_PATH.with_suffix('.pdf')}")
    print(f"Wrote {REPORT_PATH}")
    for run in SAC_RUNS:
        stats = coverage["runs"][run.run_id]
        print(
            f"{run.label}: raw={stats['raw_point_count']} unique={stats['unique_observation_count']} "
            f"matched={stats['selected_point_count']} steps={stats['coverage_min_step']}..{stats['coverage_max_step']}"
        )


if __name__ == "__main__":
    main()
