"""Compare a CARLA-trained model before and after nuPlan fine-tuning.

The selected training seed must have both an original run and a fine-tuned run
listed in the two dictionaries near the top of this file.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import tempfile
from pathlib import Path

# Matplotlib needs a writable cache directory on shared machines.
os.environ.setdefault(
    "MPLCONFIGDIR",
    str(Path(tempfile.gettempdir()) / "pufferdrive-matplotlib"),
)

import matplotlib.pyplot as plt
from matplotlib.patches import Patch


REPO_ROOT = Path(__file__).resolve().parents[2]
EXPECTED_SCENARIO_COUNT = 1_000

# Add future seeds here after both evaluations have completed.
ORIGINAL_RUN_BY_SEED = {
    0: Path(
        "experiments/second_run_nightly_best_config/"
        "nightly_best_local_2gpu_2026-08-14_15-33-43_seed0"
    ),
    1: Path(
        "experiments/second_run_nightly_best_config/"
        "nightly_best_local_2gpu_2026-08-14_15-40-12_seed1"
    ),
    2: Path(
        "experiments/second_run_nightly_best_config/"
        "nightly_best_local_2gpu_2026-08-14_16-05-25_seed2"
    ),
}

FINETUNED_RUN_BY_SEED = {
    0: Path(
        "experiments/second_run_nightly_best_config/nuplan_sdc_finetune/"
        "nuplan_sdc_finetune_2026-08-20_18-21-53_seed0"
    ),
}

BENCHMARKS = {
    "carla": "CARLA",
    "nuplan_single": "nuPlan single",
    "nuplan_multi": "nuPlan multi",
}

MODEL_STAGES = {
    "before": "Before fine-tuning",
    "after": "After fine-tuning",
}

STAGE_COLORS = {
    "before": "#4C78A8",
    "after": "#F58518",
}

METRIC_GROUPS = {
    "infraction_metrics": [
        "offroad_rate",
        "collision_rate",
        "at_fault_collision_rate",
        "red_light_violation_rate",
        "infractions_per_scenario",
        "avg_distance_per_infraction",
    ],
    "goal_completion_metrics": [
        "num_goals_reached",
        "score",
        "dnf_rate",
        "n",
    ],
    "motion_lane_comfort_metrics": [
        "avg_speed_per_agent",
        "velocity_progress_sum",
        "lane_center_rate",
        "comfort_violation_count",
    ],
    "puffer_score_metrics": [
        "progress_ratio",
        "making_progress_rate",
        "driving_direction_score",
        "speed_limit_compliance",
        "multi_lane_time",
        "multi_lane_score",
        "comfort_score",
        "puffer_score",
    ],
}

GROUP_TITLES = {
    "infraction_metrics": "Fine-Tuning Comparison: Infraction Metrics",
    "goal_completion_metrics": "Fine-Tuning Comparison: Goal and Completion Metrics",
    "motion_lane_comfort_metrics": "Fine-Tuning Comparison: Motion, Lane, and Comfort Metrics",
    "puffer_score_metrics": "Fine-Tuning Comparison: Puffer-Score Metrics",
}

METRIC_UNITS = {
    "offroad_rate": "fraction of agents",
    "collision_rate": "fraction of agents",
    "at_fault_collision_rate": "fraction of agents",
    "red_light_violation_rate": "fraction of agents",
    "infractions_per_scenario": "affected-agent infractions per scenario",
    "avg_distance_per_infraction": "meters",
    "num_goals_reached": "goals per agent",
    "score": "fraction of agents",
    "dnf_rate": "fraction of agents",
    "n": "controlled agents per scenario",
    "avg_speed_per_agent": "m/s",
    "velocity_progress_sum": "normalized score",
    "lane_center_rate": "fraction of agent-timesteps",
    "comfort_violation_count": "violations per agent-step",
    "progress_ratio": "ratio",
    "making_progress_rate": "fraction of agents",
    "driving_direction_score": "score",
    "speed_limit_compliance": "score",
    "multi_lane_time": "seconds per agent",
    "multi_lane_score": "score",
    "comfort_score": "score (known broken metric)",
    "puffer_score": "score",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--seed",
        type=int,
        default=0,
        help="Training seed whose original and fine-tuned runs should be compared (default: 0)",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Directory for generated PNG files",
    )
    return parser.parse_args()


def resolve_run_pair(seed: int) -> dict[str, Path]:
    """Return the two run directories configured for one training seed."""
    paired_seeds = sorted(set(ORIGINAL_RUN_BY_SEED) & set(FINETUNED_RUN_BY_SEED))
    if seed not in paired_seeds:
        available = ", ".join(str(value) for value in paired_seeds) or "none"
        raise ValueError(
            f"Seed {seed} does not have both runs configured. "
            f"Available paired seeds: {available}."
        )

    return {
        "before": REPO_ROOT / ORIGINAL_RUN_BY_SEED[seed],
        "after": REPO_ROOT / FINETUNED_RUN_BY_SEED[seed],
    }


def find_summary(run_dir: Path, benchmark: str) -> Path:
    """Find the single final evaluation summary for one benchmark."""
    summary_paths = sorted(
        run_dir.glob(f"eval/{benchmark}_final_model_mean_metrics/*/evaluation_summary.json")
    )
    if len(summary_paths) != 1:
        raise ValueError(
            f"Expected one final {benchmark!r} evaluation summary under {run_dir}, "
            f"found {len(summary_paths)}: {summary_paths}"
        )
    return summary_paths[0]


def load_summary_record(
    summary_path: Path,
    benchmark: str,
    model_stage: str,
) -> dict[str, str | float]:
    """Load one summary and return only the values needed for plotting."""
    try:
        with summary_path.open(encoding="utf-8") as file:
            summary = json.load(file)
    except json.JSONDecodeError as error:
        raise ValueError(f"Invalid JSON in {summary_path}: {error}") from error

    if not isinstance(summary, dict):
        raise ValueError(f"Expected a JSON object in {summary_path}")

    scenario_count = summary.get("num_scenarios")
    episode_count = summary.get("num_episodes")
    if scenario_count != EXPECTED_SCENARIO_COUNT or episode_count != EXPECTED_SCENARIO_COUNT:
        raise ValueError(
            f"Expected {EXPECTED_SCENARIO_COUNT:,} scenarios and episodes in {summary_path}; "
            f"found scenarios={scenario_count!r}, episodes={episode_count!r}"
        )

    metrics_mean = summary.get("metrics_mean")
    if not isinstance(metrics_mean, dict):
        raise ValueError(f"Missing JSON object 'metrics_mean' in {summary_path}")

    plotted_metrics = {metric for metrics in METRIC_GROUPS.values() for metric in metrics}
    required_metrics = (plotted_metrics - {"infractions_per_scenario"}) | {"total_infractions"}
    missing_metrics = sorted(required_metrics - set(metrics_mean))
    if missing_metrics:
        raise ValueError(f"Missing metrics {missing_metrics} in {summary_path}")

    record: dict[str, str | float] = {
        "benchmark": benchmark,
        "model_stage": model_stage,
    }
    for metric in required_metrics:
        value = metrics_mean[metric]
        if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value):
            raise ValueError(f"Metric {metric!r} must be a finite number in {summary_path}")
        record[metric] = float(value)

    record["infractions_per_scenario"] = (
        float(metrics_mean["total_infractions"]) / scenario_count
    )
    return record


def load_comparison_metrics(seed: int) -> list[dict[str, str | float]]:
    """Load the three benchmark summaries for both model stages."""
    run_pair = resolve_run_pair(seed)
    records = []

    for model_stage, run_dir in run_pair.items():
        if not run_dir.is_dir():
            raise FileNotFoundError(f"Missing {MODEL_STAGES[model_stage]} run: {run_dir}")

        for benchmark in BENCHMARKS:
            summary_path = find_summary(run_dir, benchmark)
            records.append(load_summary_record(summary_path, benchmark, model_stage))

    if len(records) != 6:
        raise ValueError(f"Expected 6 aggregate records, found {len(records)}")
    return records


def add_metric_bars(
    axis: plt.Axes,
    records: list[dict[str, str | float]],
    metric: str,
) -> None:
    """Draw before-and-after bars for each evaluation benchmark."""
    benchmark_order = list(BENCHMARKS)
    benchmark_positions = list(range(len(benchmark_order)))
    stage_offsets = {"before": -0.14, "after": 0.14}
    maximum_value = 0.0

    for model_stage in MODEL_STAGES:
        values = []
        for benchmark in benchmark_order:
            matches = [
                record
                for record in records
                if record["benchmark"] == benchmark and record["model_stage"] == model_stage
            ]
            if len(matches) != 1:
                raise ValueError(
                    f"Expected one record for {benchmark!r} and {model_stage!r}, "
                    f"found {len(matches)}"
                )
            values.append(float(matches[0][metric]))

        positions = [position + stage_offsets[model_stage] for position in benchmark_positions]
        color = STAGE_COLORS[model_stage]
        bars = axis.bar(
            positions,
            values,
            width=0.24,
            color=color,
            edgecolor=color,
            alpha=0.75,
        )
        axis.bar_label(
            bars,
            labels=[f"{value:.3g}" for value in values],
            padding=3,
            fontsize=8,
        )
        maximum_value = max(maximum_value, *values)

    title_padding = 22 if metric == "comfort_score" else None
    axis.set_title(metric, fontsize=11, pad=title_padding)
    axis.set_ylabel(METRIC_UNITS[metric])
    axis.set_xticks(benchmark_positions, [BENCHMARKS[name] for name in benchmark_order])
    axis.grid(axis="y", alpha=0.25)
    axis.set_axisbelow(True)
    axis.set_ylim(0, maximum_value * 1.18 if maximum_value > 0 else 1.0)

    if metric == "comfort_score":
        axis.text(
            0.5,
            1.01,
            "Known broken metric: currently always 1",
            transform=axis.transAxes,
            ha="center",
            va="bottom",
            color="#B22222",
            fontsize=9,
        )


def plot_metric_group(
    records: list[dict[str, str | float]],
    seed: int,
    group_name: str,
    metrics: list[str],
    output_dir: Path,
) -> Path:
    columns = 2
    rows = math.ceil(len(metrics) / columns)
    figure, axes = plt.subplots(rows, columns, figsize=(14, 4.6 * rows), squeeze=False)

    for axis, metric in zip(axes.flat, metrics, strict=False):
        add_metric_bars(axis, records, metric)

    for unused_axis in list(axes.flat)[len(metrics) :]:
        unused_axis.set_visible(False)

    legend_handles = [
        Patch(
            facecolor=STAGE_COLORS[stage],
            edgecolor=STAGE_COLORS[stage],
            alpha=0.75,
            label=MODEL_STAGES[stage],
        )
        for stage in MODEL_STAGES
    ]
    figure.suptitle(f"{GROUP_TITLES[group_name]} — Seed {seed}", fontsize=16, y=0.995)
    figure.tight_layout(rect=(0, 0, 1, 0.89))
    legend = figure.legend(
        handles=legend_handles,
        loc="upper center",
        bbox_to_anchor=(0.5, 0.955),
        ncol=2,
    )
    legend.set_zorder(1000)

    output_path = output_dir / f"{group_name}.png"
    figure.savefig(output_path, dpi=200, bbox_inches="tight")
    plt.close(figure)
    return output_path


def main() -> None:
    args = parse_args()

    # Validate and load everything before creating output files.
    try:
        records = load_comparison_metrics(args.seed)
    except (FileNotFoundError, ValueError) as error:
        raise SystemExit(f"Error: {error}") from error
    output_dir = (
        args.output_dir.resolve()
        if args.output_dir is not None
        else REPO_ROOT / f"project/metric_analysis/output/finetune_comparison/seed{args.seed}"
    )
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"Loaded 6 aggregate records for training seed {args.seed}.")
    for group_name, metrics in METRIC_GROUPS.items():
        output_path = plot_metric_group(records, args.seed, group_name, metrics, output_dir)
        print(f"Wrote {output_path}")


if __name__ == "__main__":
    main()
