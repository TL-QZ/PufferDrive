"""Plot CARLA metrics for the synced baseline and JEPA chunk evaluations."""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
import sys
import tempfile
from pathlib import Path

os.environ.setdefault(
    "MPLCONFIGDIR", str(Path(tempfile.gettempdir()) / "pufferdrive-matplotlib")
)
import matplotlib
import yaml

REPO_ROOT = Path(__file__).resolve().parents[3]
EXPECTED_COUNT = 1_000
EXPECTED_KS = (1, 2, 3, 4)
SERIES = ("Baseline", "JEPA (2026-09-26)", "JEPA (no warm start)")
COLORS = {
    SERIES[0]: "#4C78A8",
    SERIES[1]: "#F58518",
    SERIES[2]: "#54A24B",
}
OVERVIEW_METRICS = (
    "puffer_score", "score", "num_goals_reached", "collision_rate",
    "offroad_rate", "infractions_per_scenario",
)
BASELINE_JSON = Path(
    "experiments/baseline_run_sync_2026-08-24/"
    "baseline_run_sync_2026-08-24_2026-08-26_16-23-18_seed0/eval/"
    "carla_final_model_mean_metrics/20260911-191210/evaluation_summary.json"
)

sys.path.insert(0, str(REPO_ROOT / "project/metric_analysis"))
matplotlib.use("Agg")
from plot_finetune_comparison import METRIC_GROUPS, METRIC_UNITS  # noqa: E402

import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.lines import Line2D  # noqa: E402
from matplotlib.patches import Patch  # noqa: E402

METRICS = list(dict.fromkeys(metric for group in METRIC_GROUPS.values() for metric in group))
DERIVED_CSV_METRICS = {"infractions_per_scenario", "avg_distance_per_infraction"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--baseline-json", type=Path, default=REPO_ROOT / BASELINE_JSON)
    parser.add_argument(
        "--chunk-eval-root", type=Path,
        default=REPO_ROOT / "experiments/jepa_distill/carla_chunk_eval_20260926",
    )
    parser.add_argument(
        "--no-warm-start-root", type=Path,
        default=REPO_ROOT / "experiments/jepa_distill/carla_renders_no_warm_start",
    )
    parser.add_argument(
        "--output-dir", type=Path,
        default=REPO_ROOT / "project/jepa_distill/analysis_output/carla_comparison",
    )
    return parser.parse_args()


def load_json(path: Path) -> dict:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise ValueError(f"Cannot read JSON {path}: {error}") from error
    if not isinstance(value, dict):
        raise ValueError(f"Expected a JSON object in {path}")
    return value


def finite_number(value: object, name: str, path: Path) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value):
        raise ValueError(f"{name} must be a finite number in {path}; found {value!r}")
    return float(value)


def finite_csv_number(value: str | None, name: str, path: Path) -> float:
    try:
        number = float(value) if value is not None else math.nan
    except ValueError as error:
        raise ValueError(f"{name} must be numeric in {path}; found {value!r}") from error
    if not math.isfinite(number):
        raise ValueError(f"{name} must be finite in {path}; found {value!r}")
    return number


def validate_counts(scenarios: object, episodes: object, path: Path) -> None:
    if scenarios != EXPECTED_COUNT or episodes != EXPECTED_COUNT:
        raise ValueError(
            f"Expected {EXPECTED_COUNT} scenarios and episodes in {path}; "
            f"found {scenarios!r} and {episodes!r}"
        )


def validate_jepa_protocol(protocol: dict, k: int, path: Path) -> None:
    expected_replan = 1 if k == 1 else 3 * k
    expected_hold = 1 if k == 1 else 3
    expected = {
        "action_selection": "mean", "action_type": "continuous", "benchmark": "carla",
        "evaluation_dt": 0.1, "execution_k": k, "hold_steps_per_slot": expected_hold,
        "num_scenarios": EXPECTED_COUNT, "replan_interval_steps": expected_replan,
        "scenario_length_steps": 6_000, "seed": 42, "smoke": False,
    }
    for key, value in expected.items():
        if protocol.get(key) != value:
            raise ValueError(f"Expected {key}={value!r} in {path}; found {protocol.get(key)!r}")
    if not protocol.get("checkpoint"):
        raise ValueError(f"Missing checkpoint path in {path}")


def read_episode_csv(
    path: Path, required_metrics: list[str], compute_infraction_metrics: bool = False,
) -> tuple[int, set[tuple[str, str]], dict[str, float]]:
    try:
        with path.open(newline="", encoding="utf-8") as file:
            reader = csv.DictReader(file)
            fields = set(reader.fieldnames or [])
            required = {"map_name", "seed", *required_metrics}
            if compute_infraction_metrics:
                required.update({"total_infraction_count", "total_distance_travelled_sum"})
            missing = sorted(required - fields)
            if missing:
                raise ValueError(f"Missing columns {missing} in {path}")
            keys: set[tuple[str, str]] = set()
            values: dict[str, list[float]] = {metric: [] for metric in required_metrics}
            infraction_counts: list[float] = []
            traveled_distances: list[float] = []
            row_count = 0
            for row in reader:
                row_count += 1
                scenario_key = (row["map_name"], row["seed"])
                if not all(scenario_key) or scenario_key in keys:
                    raise ValueError(f"Missing or duplicate (map_name, seed) row in {path}: {scenario_key}")
                keys.add(scenario_key)
                for metric in required_metrics:
                    values[metric].append(finite_csv_number(row[metric], metric, path))
                if compute_infraction_metrics:
                    infraction_counts.append(finite_csv_number(row["total_infraction_count"], "total_infraction_count", path))
                    traveled_distances.append(finite_csv_number(row["total_distance_travelled_sum"], "total_distance_travelled_sum", path))
    except OSError as error:
        raise ValueError(f"Cannot read episode CSV {path}: {error}") from error

    if row_count != EXPECTED_COUNT:
        raise ValueError(f"Expected {EXPECTED_COUNT} CSV rows in {path}; found {row_count}")
    means = {metric: math.fsum(samples) / row_count for metric, samples in values.items()}
    if infraction_counts:
        infraction_total = math.fsum(infraction_counts)
        means["infractions_per_scenario"] = infraction_total / row_count
        means["avg_distance_per_infraction"] = math.fsum(traveled_distances) / max(infraction_total, 1.0)
    return row_count, keys, means


def summary_metrics(path: Path) -> tuple[dict[str, float], int, int]:
    summary = load_json(path)
    scenarios, episodes = summary.get("num_scenarios"), summary.get("num_episodes")
    validate_counts(scenarios, episodes, path)
    metrics_mean = summary.get("metrics_mean")
    if not isinstance(metrics_mean, dict):
        raise ValueError(f"Missing metrics_mean object in {path}")
    metrics = {
        metric: finite_number(metrics_mean.get(metric), metric, path)
        for metric in METRICS if metric != "infractions_per_scenario"
    }
    total_infractions = finite_number(metrics_mean.get("total_infractions"), "total_infractions", path)
    metrics["infractions_per_scenario"] = total_infractions / int(scenarios)
    return metrics, int(scenarios), int(episodes)


def load_records(args: argparse.Namespace) -> tuple[list[dict], dict]:
    records: list[dict] = []
    protocols: list[dict] = []
    baseline_path = args.baseline_json.resolve()
    baseline_csv = baseline_path.with_name("episode_metrics.csv")
    baseline_yaml = baseline_path.with_name("resolved_benchmark.yaml")
    try:
        baseline_config = yaml.safe_load(baseline_yaml.read_text(encoding="utf-8"))
    except (OSError, ImportError, yaml.YAMLError) as error:
        raise ValueError(f"Cannot read baseline protocol {baseline_yaml}: {error}") from error
    baseline_benchmark = baseline_config.get("benchmark", {})
    baseline_args = baseline_config.get("args", {})
    baseline_env = baseline_args.get("env", {})
    baseline_eval = baseline_args.get("eval", {})
    baseline_protocol = {
        "benchmark": baseline_benchmark.get("name"), "seed": baseline_benchmark.get("seed"),
        "num_scenarios": baseline_benchmark.get("num_scenarios"),
        "dt": baseline_env.get("dt"), "scenario_length": baseline_env.get("scenario_length"),
        "action_selection": baseline_eval.get("action_selection"),
        "checkpoint": baseline_args.get("load_model_path"),
    }
    if baseline_protocol != {
        "benchmark": "carla", "seed": 42, "num_scenarios": EXPECTED_COUNT,
        "dt": 0.1, "scenario_length": 6_000, "action_selection": "mean",
        "checkpoint": baseline_protocol["checkpoint"],
    } or not baseline_protocol["checkpoint"]:
        raise ValueError(f"Baseline protocol mismatch in {baseline_yaml}: {baseline_protocol}")
    base_metrics, scenario_count, episode_count = summary_metrics(baseline_path)
    base_csv_count, baseline_keys, unused_means = read_episode_csv(baseline_csv, [])
    records.append({
        "series": SERIES[0], "k": "", "metrics": base_metrics,
        "source_path": str(baseline_path), "episode_csv": str(baseline_csv),
        "protocol_path": str(baseline_yaml), "checkpoint": baseline_protocol["checkpoint"],
        "scenario_count": scenario_count, "episode_count": episode_count,
        "csv_row_count": base_csv_count,
    })
    protocols.append({"series": SERIES[0], "protocol_path": str(baseline_yaml), **baseline_protocol})

    all_keys = [baseline_keys]
    for k in EXPECTED_KS:
        chunk_dir = args.chunk_eval_root.resolve() / f"k{k}"
        summary_path = chunk_dir / "eval" / f"carla_act_k{k}" / "standalone" / "evaluation_summary.json"
        # The evaluator places the output name inside the benchmark directory.
        summary_path = (chunk_dir / "eval" / f"carla_act_k{k}" / "standalone" / "evaluation_summary.json").resolve()
        csv_path = summary_path.with_name("episode_metrics.csv")
        protocol_path = (chunk_dir / "protocol.json").resolve()
        protocol = load_json(protocol_path)
        validate_jepa_protocol(protocol, k, protocol_path)
        metrics, scenarios, episodes = summary_metrics(summary_path)
        csv_count, scenario_keys, unused_means = read_episode_csv(csv_path, [])
        all_keys.append(scenario_keys)
        records.append({
            "series": SERIES[1], "k": k, "metrics": metrics,
            "source_path": str(summary_path), "episode_csv": str(csv_path),
            "protocol_path": str(protocol_path), "checkpoint": protocol["checkpoint"],
            "scenario_count": scenarios, "episode_count": episodes, "csv_row_count": csv_count,
        })
        protocols.append({"series": SERIES[1], "k": k, "protocol_path": str(protocol_path), **protocol})

        run_dir = args.no_warm_start_root.resolve() / f"run_k{k}" / f"k{k}"
        csv_path = (run_dir / "source_episode_metrics.csv").resolve()
        protocol_path = (run_dir / "source_protocol.json").resolve()
        protocol = load_json(protocol_path)
        validate_jepa_protocol(protocol, k, protocol_path)
        csv_count, scenario_keys, metrics = read_episode_csv(
            csv_path,
            [metric for metric in METRICS if metric not in DERIVED_CSV_METRICS],
            compute_infraction_metrics=True,
        )
        all_keys.append(scenario_keys)
        records.append({
            "series": SERIES[2], "k": k, "metrics": metrics,
            "source_path": str(csv_path), "episode_csv": str(csv_path),
            "protocol_path": str(protocol_path), "checkpoint": protocol["checkpoint"],
            "scenario_count": csv_count, "episode_count": csv_count, "csv_row_count": csv_count,
        })
        protocols.append({"series": SERIES[2], "k": k, "protocol_path": str(protocol_path), **protocol})

    if any(keys != baseline_keys for keys in all_keys[1:]):
        raise ValueError("CARLA episode CSVs do not contain the same map/seed scenarios")
    return records, {"protocols": protocols, "scenario_set_matches": True}


def draw_metric(axis: plt.Axes, records: list[dict], metric: str, mode: str) -> None:
    baseline = next(record for record in records if record["series"] == SERIES[0])
    axis.set_title(metric, fontsize=10, pad=20 if metric == "comfort_score" else None)
    if mode == "k1":
        selected = [baseline] + [
            record for record in records if record["k"] == 1 and record["series"] != SERIES[0]
        ]
        labels = [record["series"] for record in selected]
        values = [record["metrics"][metric] for record in selected]
        bars = axis.barh(labels, values, color=[COLORS[label] for label in labels], alpha=0.82)
        axis.bar_label(bars, fmt="%.3g", padding=3, fontsize=8)
        axis.margins(x=0.18)
        axis.invert_yaxis()
        axis.set_xlabel(METRIC_UNITS[metric])
        axis.grid(axis="x", alpha=0.25)
        axis.set_axisbelow(True)
    else:
        for series in SERIES[1:]:
            points = sorted(
                (record for record in records if record["series"] == series),
                key=lambda record: record["k"],
            )
            axis.plot(
                [record["k"] for record in points],
                [record["metrics"][metric] for record in points],
                marker="o", color=COLORS[series], label=series,
            )
        axis.axhline(
            baseline["metrics"][metric], color=COLORS[SERIES[0]],
            linestyle="--", linewidth=1.5, label=SERIES[0],
        )
        axis.set_xticks(EXPECTED_KS)
        axis.set_xlabel("Chunk schedule k")
        axis.set_ylabel(METRIC_UNITS[metric])
        axis.grid(axis="y", alpha=0.25)
        axis.set_axisbelow(True)
    if metric == "comfort_score":
        axis.text(
            0.5, 1.02, "Known broken metric: currently always 1",
            transform=axis.transAxes, ha="center", va="bottom", color="#B22222", fontsize=8,
        )


def save_figure(records: list[dict], metrics: tuple | list, base_path: Path, title: str, mode: str) -> None:
    rows = math.ceil(len(metrics) / 2)
    figure, axes = plt.subplots(rows, 2, figsize=(14, 4.4 * rows), squeeze=False)
    for axis, metric in zip(axes.flat, metrics):
        draw_metric(axis, records, metric, mode)
    for axis in list(axes.flat)[len(metrics):]:
        axis.set_visible(False)
    if mode == "k1":
        handles = [Patch(facecolor=COLORS[label], label=label) for label in SERIES]
    else:
        handles = [
            Line2D([0], [0], color=COLORS[label], marker="o", label=label)
            for label in SERIES[1:]
        ] + [Line2D([0], [0], color=COLORS[SERIES[0]], linestyle="--", label=SERIES[0])]
    figure.suptitle(title, fontsize=15, y=0.995)
    figure.legend(handles=handles, loc="upper center", bbox_to_anchor=(0.5, 0.965), ncol=3)
    figure.tight_layout(rect=(0, 0, 1, 0.91))
    base_path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(base_path.with_suffix(".png"), dpi=180, bbox_inches="tight")
    figure.savefig(base_path.with_suffix(".pdf"), bbox_inches="tight")
    plt.close(figure)


def write_reports(records: list[dict], provenance: dict, output_dir: Path) -> None:
    metric_columns = [
        "series", "k", "checkpoint", "source_path", "episode_csv", "protocol_path",
        "scenario_count", "episode_count", "csv_row_count", *METRICS,
    ]
    with (output_dir / "aggregate_metrics.csv").open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=metric_columns)
        writer.writeheader()
        for record in records:
            writer.writerow({**{key: record.get(key, "") for key in metric_columns}, **record["metrics"]})
    provenance["records"] = [
        {key: record[key] for key in (
            "series", "k", "checkpoint", "source_path", "episode_csv", "protocol_path",
            "scenario_count", "episode_count", "csv_row_count",
        )}
        for record in records
    ]
    (output_dir / "provenance.json").write_text(
        json.dumps({"expected_scenarios": EXPECTED_COUNT, **provenance}, indent=2) + "\n",
        encoding="utf-8",
    )
    baseline = records[0]["metrics"]["puffer_score"]
    chunk = [next(r for r in records if r["series"] == SERIES[1] and r["k"] == k) for k in EXPECTED_KS]
    no_warm = [next(r for r in records if r["series"] == SERIES[2] and r["k"] == k) for k in EXPECTED_KS]
    lines = [
        "# CARLA metric comparison", "",
        "All nine episode CSVs contain the same 1,000 map/seed pairs. The baseline and both JEPA series use CARLA at `dt=0.1`, 6,000 steps, seed 42, and mean action selection.", "",
        "## Puffer score", "",
        "| Evaluation | Puffer score |", "|---|---:|",
        f"| Baseline | {baseline:.5f} |",
        f"| JEPA (2026-09-26), k=1 | {chunk[0]['metrics']['puffer_score']:.5f} |",
        f"| JEPA (no warm start), k=1 | {no_warm[0]['metrics']['puffer_score']:.5f} |", "",
        "| k | JEPA (2026-09-26) | JEPA (no warm start) |", "|---:|---:|---:|",
    ]
    lines.extend(
        f"| {k} | {chunk[k - 1]['metrics']['puffer_score']:.5f} | {no_warm[k - 1]['metrics']['puffer_score']:.5f} |"
        for k in EXPECTED_KS
    )
    lines += [
        "", "## Aggregation and interpretation", "",
        "Aggregate JSON values are used for the baseline and chunk series. No-warm-start ordinary metrics are CSV means; `infractions_per_scenario = sum(total_infraction_count) / row_count`, and `avg_distance_per_infraction = sum(total_distance_travelled_sum) / max(sum(total_infraction_count), 1)`, matching the evaluator reducer.", "",
        "At evaluation `dt=0.1`, k=1 replans each step; k=2, 3, and 4 hold each selected slot for three steps and replan every 6, 9, and 12 steps. One checkpoint is available per series, so plots have no error bars. `comfort_score` is a known broken metric.", "",
        "The 2026-09-26 chunk series uses `checkpoint.pt`; the no-warm-start series uses a separately trained `final_model.pt`. This comparison does not isolate warm-start effects.", "",
    ]
    (output_dir / "summary.md").write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    args = parse_args()
    try:
        records, provenance = load_records(args)
    except (OSError, ValueError) as error:
        raise SystemExit(f"Error: {error}") from error

    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    save_figure(records, OVERVIEW_METRICS, output_dir / "k1_vs_baseline", "CARLA comparison at k=1", "k1")
    save_figure(records, OVERVIEW_METRICS, output_dir / "performance_vs_k", "CARLA performance by chunk schedule", "curve")
    for comparison, mode, title in (
        ("k1_vs_baseline", "k1", "k=1 vs baseline"),
        ("performance_vs_k", "curve", "Performance by k"),
    ):
        for group_name, metrics in METRIC_GROUPS.items():
            save_figure(
                records, metrics, output_dir / "details" / comparison / group_name,
                f"CARLA {title}: {group_name.replace('_', ' ')}", mode,
            )
    write_reports(records, provenance, output_dir)
    print(f"Wrote CARLA comparison figures and reports to {output_dir}")


if __name__ == "__main__":
    main()
