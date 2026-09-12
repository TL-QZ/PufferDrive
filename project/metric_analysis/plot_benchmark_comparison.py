"""Compare aggregate CARLA, nuPlan single, and nuPlan multi metrics.

Run this file from anywhere. By default, figures are written to
``project/metric_analysis/output/benchmark_comparison``.
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
EXPECTED_SCENARIOS_PER_SUMMARY = 1_000
EXPECTED_EPISODES_PER_SUMMARY = 1_000

BENCHMARK_LABELS = {
    "carla": "CARLA",
    "nuplan_single": "nuPlan single",
    "nuplan_multi": "nuPlan multi",
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
    "infraction_metrics": "Benchmark Comparison: Infraction Metrics",
    "goal_completion_metrics": "Benchmark Comparison: Goal and Completion Metrics",
    "motion_lane_comfort_metrics": "Benchmark Comparison: Motion, Lane, and Comfort Metrics",
    "puffer_score_metrics": "Benchmark Comparison: Puffer-Score Metrics",
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

SEED_COLORS = {
    0: "#4C78A8",
    1: "#F58518",
    2: "#54A24B",
}


def parse_args() -> argparse.Namespace:
    default_output_dir = REPO_ROOT / "project/metric_analysis/output/benchmark_comparison"
    parser = argparse.ArgumentParser(description=__doc__)
    for benchmark in BENCHMARK_LABELS:
        parser.add_argument(
            f"--{benchmark.replace('_', '-')}-json",
            type=Path,
            nargs="+",
            required=True,
            help=f"{BENCHMARK_LABELS[benchmark]} summaries in model-seed order",
        )
    parser.add_argument(
        "--model-seed",
        type=int,
        choices=SEED_COLORS,
        nargs="+",
        help="Seed label for each set of benchmark JSONs (default: 0, 1, ...)",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=default_output_dir,
        help=f"Directory for generated PNG files (default: {default_output_dir})",
    )
    return parser.parse_args()


def resolve_input_jsons(args: argparse.Namespace) -> dict[str, dict[int, Path]]:
    """Pair each benchmark JSON with its model-seed label."""
    supplied = {
        benchmark: getattr(args, f"{benchmark}_json")
        for benchmark in BENCHMARK_LABELS
    }
    path_counts = {len(paths) for paths in supplied.values()}
    if len(path_counts) != 1 or not 1 <= next(iter(path_counts)) <= len(SEED_COLORS):
        raise ValueError("Supply the same number of JSON paths (1-3) for each benchmark.")
    path_count = next(iter(path_counts))
    model_seeds = args.model_seed or list(range(path_count))
    if len(model_seeds) != path_count or len(set(model_seeds)) != path_count:
        raise ValueError("Supply one unique --model-seed for each set of benchmark JSONs.")
    return {
        benchmark: dict(zip(model_seeds, paths, strict=True))
        for benchmark, paths in supplied.items()
    }


def load_benchmark_metrics(
    json_by_benchmark_and_seed: dict[str, dict[int, Path]],
) -> list[dict[str, str | int | float]]:
    """Load and validate one aggregate record per benchmark and model seed."""
    plotted_metrics = {metric for metrics in METRIC_GROUPS.values() for metric in metrics}
    required_json_metrics = plotted_metrics - {"infractions_per_scenario"}
    required_json_metrics.add("total_infractions")
    records = []

    for benchmark, json_by_model_seed in json_by_benchmark_and_seed.items():
        for model_seed, json_path in json_by_model_seed.items():
            json_path = json_path.resolve()
            if not json_path.is_file():
                raise FileNotFoundError(
                    f"Missing metrics JSON for benchmark {benchmark!r}, model seed "
                    f"{model_seed}: {json_path}"
                )

            try:
                with json_path.open(encoding="utf-8") as file:
                    summary = json.load(file)
            except json.JSONDecodeError as error:
                raise ValueError(
                    f"Invalid metrics JSON for benchmark {benchmark!r}, model seed "
                    f"{model_seed}: {json_path}: {error}"
                ) from error

            if not isinstance(summary, dict):
                raise ValueError(f"Expected a JSON object in {json_path}")

            num_scenarios = summary.get("num_scenarios")
            num_episodes = summary.get("num_episodes")
            if type(num_scenarios) is not int or num_scenarios != EXPECTED_SCENARIOS_PER_SUMMARY:
                raise ValueError(
                    f"Expected {EXPECTED_SCENARIOS_PER_SUMMARY:,} scenarios for benchmark "
                    f"{benchmark!r}, model seed {model_seed}, found {num_scenarios!r}: {json_path}"
                )
            if type(num_episodes) is not int or num_episodes != EXPECTED_EPISODES_PER_SUMMARY:
                raise ValueError(
                    f"Expected {EXPECTED_EPISODES_PER_SUMMARY:,} episodes for benchmark "
                    f"{benchmark!r}, model seed {model_seed}, found {num_episodes!r}: {json_path}"
                )

            metrics_mean = summary.get("metrics_mean")
            if not isinstance(metrics_mean, dict):
                raise ValueError(f"Missing JSON object 'metrics_mean' in {json_path}")

            missing_metrics = sorted(required_json_metrics - set(metrics_mean))
            if missing_metrics:
                raise ValueError(
                    f"Metrics JSON for benchmark {benchmark!r}, model seed {model_seed} "
                    f"is missing metrics {missing_metrics}: {json_path}"
                )

            record: dict[str, str | int | float] = {
                "benchmark": benchmark,
                "model_seed": model_seed,
            }
            for metric in required_json_metrics:
                value = metrics_mean[metric]
                if isinstance(value, bool) or not isinstance(value, (int, float)):
                    raise ValueError(
                        f"Metric {metric!r} must be numeric for benchmark {benchmark!r}, "
                        f"model seed {model_seed}, found {value!r}: {json_path}"
                    )
                if not math.isfinite(value):
                    raise ValueError(
                        f"Metric {metric!r} must be finite for benchmark {benchmark!r}, "
                        f"model seed {model_seed}, found {value!r}: {json_path}"
                    )
                record[metric] = float(value)

            record["infractions_per_scenario"] = (
                float(metrics_mean["total_infractions"]) / num_scenarios
            )
            records.append(record)

    expected_record_count = sum(len(paths) for paths in json_by_benchmark_and_seed.values())
    if len(records) != expected_record_count:
        raise ValueError(f"Expected {expected_record_count} aggregate records, found {len(records)}")
    return records


def add_metric_bars(
    axis: plt.Axes,
    data: list[dict[str, str | int | float]],
    metric: str,
    model_seeds: list[int],
) -> None:
    """Draw three side-by-side model-seed aggregate bars for each benchmark."""
    benchmark_order = list(BENCHMARK_LABELS)
    benchmark_positions = list(range(len(benchmark_order)))
    seed_offsets = {
        seed: (index - (len(model_seeds) - 1) / 2) * 0.24
        for index, seed in enumerate(model_seeds)
    }
    maximum_value = 0.0

    for model_seed in model_seeds:
        values_by_benchmark = []
        for benchmark in benchmark_order:
            matching_records = [
                record
                for record in data
                if record["model_seed"] == model_seed and record["benchmark"] == benchmark
            ]
            if len(matching_records) != 1:
                raise ValueError(
                    f"Expected one aggregate record for benchmark {benchmark!r}, "
                    f"model seed {model_seed}, found {len(matching_records)}"
                )
            values_by_benchmark.append(float(matching_records[0][metric]))

        positions = [position + seed_offsets[model_seed] for position in benchmark_positions]
        color = SEED_COLORS[model_seed]
        bars = axis.bar(
            positions,
            values_by_benchmark,
            width=0.20,
            color=color,
            edgecolor=color,
            alpha=0.72,
        )
        axis.bar_label(
            bars,
            labels=[f"{value:.3g}" for value in values_by_benchmark],
            padding=3,
            fontsize=8,
        )
        maximum_value = max(maximum_value, *values_by_benchmark)

    title_padding = 22 if metric == "comfort_score" else None
    axis.set_title(metric, fontsize=11, pad=title_padding)
    axis.set_ylabel(METRIC_UNITS[metric])
    axis.set_xticks(
        benchmark_positions,
        [BENCHMARK_LABELS[benchmark] for benchmark in benchmark_order],
    )
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
    data: list[dict[str, str | int | float]],
    group_name: str,
    metrics: list[str],
    output_dir: Path,
    model_seeds: list[int],
) -> Path:
    columns = 2
    rows = math.ceil(len(metrics) / columns)
    figure, axes = plt.subplots(rows, columns, figsize=(14, 4.6 * rows), squeeze=False)

    for axis, metric in zip(axes.flat, metrics, strict=False):
        add_metric_bars(axis, data, metric, model_seeds)

    for unused_axis in list(axes.flat)[len(metrics) :]:
        unused_axis.set_visible(False)

    legend_handles = [
        Patch(
            facecolor=SEED_COLORS[seed],
            edgecolor=SEED_COLORS[seed],
            alpha=0.72,
            label=f"Model seed {seed}",
        )
        for seed in model_seeds
    ]
    figure.suptitle(GROUP_TITLES[group_name], fontsize=16, y=0.995)
    figure.legend(
        handles=legend_handles,
        loc="upper center",
        bbox_to_anchor=(0.5, 0.955),
        ncol=len(model_seeds),
    )
    figure.tight_layout(rect=(0, 0, 1, 0.89))

    output_path = output_dir / f"{group_name}.png"
    figure.savefig(output_path, dpi=200, bbox_inches="tight")
    plt.close(figure)
    return output_path


def main() -> None:
    args = parse_args()
    try:
        input_jsons = resolve_input_jsons(args)
        data = load_benchmark_metrics(input_jsons)
    except (FileNotFoundError, ValueError) as error:
        raise SystemExit(f"Error: {error}") from error

    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    model_seeds = sorted(input_jsons["carla"])
    print(f"Loaded {len(data)} aggregate JSON records for {len(model_seeds)} model seed(s).")
    for group_name, metrics in METRIC_GROUPS.items():
        output_path = plot_metric_group(data, group_name, metrics, output_dir, model_seeds)
        print(f"Wrote {output_path}")


if __name__ == "__main__":
    main()
