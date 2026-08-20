"""Plot CARLA evaluation metrics by town and training seed.

Run this file from anywhere. By default, figures are written to
``project/metric_analysis/output/carla``.
"""

from __future__ import annotations

import argparse
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
import pandas as pd
from matplotlib.patches import Patch


REPO_ROOT = Path(__file__).resolve().parents[2]

CARLA_CSV_BY_MODEL_SEED = {
    0: Path(
        "experiments/second_run_nightly_best_config/"
        "nightly_best_local_2gpu_2026-08-14_15-33-43_seed0/"
        "eval/carla_final_model_mean_metrics/20260815-200339/episode_metrics.csv"
    ),
    1: Path(
        "experiments/second_run_nightly_best_config/"
        "nightly_best_local_2gpu_2026-08-14_15-40-12_seed1/"
        "eval/carla_final_model_mean_metrics/20260815-200501/episode_metrics.csv"
    ),
    2: Path(
        "experiments/second_run_nightly_best_config/"
        "nightly_best_local_2gpu_2026-08-14_16-05-25_seed2/"
        "eval/carla_final_model_mean_metrics/20260815-200548/episode_metrics.csv"
    ),
}

METRIC_GROUPS = {
    "infraction_metrics": [
        "offroad_rate",
        "collision_rate",
        "at_fault_collision_rate",
        "red_light_violation_rate",
        "total_infraction_count",
        "avg_distance_per_infraction",
    ],
    "goal_completion_metrics": [
        "num_goals_reached",
        "score",
        "dnf_rate",
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
    "infraction_metrics": "CARLA Infraction Metrics",
    "goal_completion_metrics": "CARLA Goal and Completion Metrics",
    "motion_lane_comfort_metrics": "CARLA Motion, Lane, and Comfort Metrics",
    "puffer_score_metrics": "CARLA Puffer-Score Metrics",
}

METRIC_UNITS = {
    "offroad_rate": "fraction of agents",
    "collision_rate": "fraction of agents",
    "at_fault_collision_rate": "fraction of agents",
    "red_light_violation_rate": "fraction of agents",
    "total_infraction_count": "agents",
    "avg_distance_per_infraction": "meters",
    "num_goals_reached": "goals per agent",
    "score": "fraction of agents",
    "dnf_rate": "fraction of agents",
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
    default_output_dir = REPO_ROOT / "project/metric_analysis/output/carla"
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=default_output_dir,
        help=f"Directory for generated PNG files (default: {default_output_dir})",
    )
    return parser.parse_args()


def load_carla_metrics() -> tuple[pd.DataFrame, list[str]]:
    """Load the three model runs and verify that their town sets match."""
    required_metrics = [metric for metrics in METRIC_GROUPS.values() for metric in metrics]
    required_columns = {"map_name", *required_metrics}
    frames = []
    maps_by_model_seed = {}

    for model_seed, relative_csv_path in CARLA_CSV_BY_MODEL_SEED.items():
        csv_path = REPO_ROOT / relative_csv_path
        if not csv_path.is_file():
            raise FileNotFoundError(f"Missing CARLA metrics CSV: {csv_path}")

        frame = pd.read_csv(csv_path)
        missing_columns = sorted(required_columns - set(frame.columns))
        if missing_columns:
            raise ValueError(f"{csv_path} is missing columns: {missing_columns}")

        frame = frame[["map_name", *required_metrics]].copy()
        frame["model_seed"] = model_seed
        frames.append(frame)
        maps_by_model_seed[model_seed] = set(frame["map_name"].unique())

    first_map_set = maps_by_model_seed[min(maps_by_model_seed)]
    for model_seed, map_set in maps_by_model_seed.items():
        if map_set != first_map_set:
            raise ValueError(
                f"Model seed {model_seed} has a different map set: "
                f"{sorted(map_set)} instead of {sorted(first_map_set)}"
            )

    if len(first_map_set) != 8:
        raise ValueError(f"Expected 8 CARLA maps, found {len(first_map_set)}: {sorted(first_map_set)}")

    map_order = sorted(first_map_set)
    return pd.concat(frames, ignore_index=True), map_order


def short_map_name(map_name: str) -> str:
    return map_name.removeprefix("opendrive__")


def add_metric_boxplots(
    axis: plt.Axes,
    data: pd.DataFrame,
    metric: str,
    map_order: list[str],
) -> None:
    """Draw three side-by-side model-seed boxes for each CARLA town."""
    map_positions = list(range(len(map_order)))
    seed_offsets = {0: -0.24, 1: 0.0, 2: 0.24}

    for model_seed in CARLA_CSV_BY_MODEL_SEED:
        values_by_map = [
            data.loc[
                (data["model_seed"] == model_seed) & (data["map_name"] == map_name),
                metric,
            ].dropna()
            for map_name in map_order
        ]
        positions = [position + seed_offsets[model_seed] for position in map_positions]
        color = SEED_COLORS[model_seed]

        axis.boxplot(
            values_by_map,
            positions=positions,
            widths=0.20,
            patch_artist=True,
            manage_ticks=False,
            boxprops={"facecolor": color, "edgecolor": color, "alpha": 0.72},
            medianprops={"color": "black", "linewidth": 1.3},
            whiskerprops={"color": color, "linewidth": 1.0},
            capprops={"color": color, "linewidth": 1.0},
            flierprops={
                "marker": "o",
                "markersize": 2.2,
                "markerfacecolor": color,
                "markeredgecolor": "none",
                "alpha": 0.25,
            },
        )

    axis.set_title(metric, fontsize=11)
    axis.set_ylabel(METRIC_UNITS[metric])
    axis.set_xticks(map_positions, [short_map_name(name) for name in map_order], rotation=30, ha="right")
    axis.grid(axis="y", alpha=0.25)
    axis.set_axisbelow(True)

    if metric == "comfort_score":
        axis.text(
            0.02,
            0.95,
            "Known broken metric: currently always 1",
            transform=axis.transAxes,
            ha="left",
            va="top",
            color="#B22222",
            fontsize=9,
        )


def plot_metric_group(
    data: pd.DataFrame,
    map_order: list[str],
    group_name: str,
    metrics: list[str],
    output_dir: Path,
) -> Path:
    columns = 2
    rows = math.ceil(len(metrics) / columns)
    figure, axes = plt.subplots(rows, columns, figsize=(16, 4.6 * rows), squeeze=False)

    for axis, metric in zip(axes.flat, metrics, strict=False):
        add_metric_boxplots(axis, data, metric, map_order)

    for unused_axis in list(axes.flat)[len(metrics) :]:
        unused_axis.set_visible(False)

    legend_handles = [
        Patch(facecolor=SEED_COLORS[seed], edgecolor=SEED_COLORS[seed], alpha=0.72, label=f"Model seed {seed}")
        for seed in CARLA_CSV_BY_MODEL_SEED
    ]
    figure.suptitle(GROUP_TITLES[group_name], fontsize=16, y=0.995)
    figure.legend(handles=legend_handles, loc="upper center", bbox_to_anchor=(0.5, 0.955), ncol=3)
    figure.tight_layout(rect=(0, 0, 1, 0.89))

    output_path = output_dir / f"{group_name}.png"
    figure.savefig(output_path, dpi=200, bbox_inches="tight")
    plt.close(figure)
    return output_path


def main() -> None:
    args = parse_args()
    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    data, map_order = load_carla_metrics()
    print(f"Loaded {len(data):,} rows across {len(map_order)} CARLA maps and 3 model seeds.")

    for group_name, metrics in METRIC_GROUPS.items():
        output_path = plot_metric_group(data, map_order, group_name, metrics, output_dir)
        print(f"Wrote {output_path}")


if __name__ == "__main__":
    main()
