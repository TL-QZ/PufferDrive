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
        "--carla-csv",
        type=Path,
        nargs="+",
        required=True,
        help="CARLA episode_metrics.csv files in model-seed order",
    )
    parser.add_argument(
        "--model-seed",
        type=int,
        choices=SEED_COLORS,
        nargs="+",
        help="Seed label for each CSV (default: 0, 1, ...)",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=default_output_dir,
        help=f"Directory for generated PNG files (default: {default_output_dir})",
    )
    return parser.parse_args()


def resolve_input_csvs(args: argparse.Namespace) -> dict[int, Path]:
    model_seeds = args.model_seed or list(range(len(args.carla_csv)))
    if not 1 <= len(args.carla_csv) <= len(SEED_COLORS):
        raise ValueError("Supply 1-3 CARLA CSV files.")
    if len(model_seeds) != len(args.carla_csv) or len(set(model_seeds)) != len(model_seeds):
        raise ValueError("Supply one unique --model-seed for each CARLA CSV.")
    return dict(zip(model_seeds, args.carla_csv, strict=True))


def load_carla_metrics(csv_by_model_seed: dict[int, Path]) -> tuple[pd.DataFrame, list[str]]:
    """Load model runs and verify that their town sets match."""
    required_metrics = [metric for metrics in METRIC_GROUPS.values() for metric in metrics]
    required_columns = {"map_name", *required_metrics}
    frames = []
    maps_by_model_seed = {}

    for model_seed, csv_path in csv_by_model_seed.items():
        csv_path = csv_path.resolve()
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
    model_seeds: list[int],
) -> None:
    """Draw side-by-side model-seed boxes for each CARLA town."""
    map_positions = list(range(len(map_order)))
    seed_offsets = {
        seed: (index - (len(model_seeds) - 1) / 2) * 0.24
        for index, seed in enumerate(model_seeds)
    }

    for model_seed in model_seeds:
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
    model_seeds: list[int],
) -> Path:
    columns = 2
    rows = math.ceil(len(metrics) / columns)
    figure, axes = plt.subplots(rows, columns, figsize=(16, 4.6 * rows), squeeze=False)

    for axis, metric in zip(axes.flat, metrics, strict=False):
        add_metric_boxplots(axis, data, metric, map_order, model_seeds)

    for unused_axis in list(axes.flat)[len(metrics) :]:
        unused_axis.set_visible(False)

    legend_handles = [
        Patch(facecolor=SEED_COLORS[seed], edgecolor=SEED_COLORS[seed], alpha=0.72, label=f"Model seed {seed}")
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
        input_csvs = resolve_input_csvs(args)
        data, map_order = load_carla_metrics(input_csvs)
    except (FileNotFoundError, ValueError) as error:
        raise SystemExit(f"Error: {error}") from error

    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    model_seeds = sorted(input_csvs)
    print(
        f"Loaded {len(data):,} rows across {len(map_order)} CARLA maps "
        f"and {len(model_seeds)} model seed(s)."
    )

    for group_name, metrics in METRIC_GROUPS.items():
        output_path = plot_metric_group(
            data,
            map_order,
            group_name,
            metrics,
            output_dir,
            model_seeds,
        )
        print(f"Wrote {output_path}")


if __name__ == "__main__":
    main()
