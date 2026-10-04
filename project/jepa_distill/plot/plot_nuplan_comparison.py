"""Compare nuPlan chunk evaluations with the seed-0 baseline."""

from __future__ import annotations

import argparse
import csv
import json
import os
import sys
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parents[2]
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

import yaml

import plot_carla_comparison as shared  # noqa: E402

BENCHMARKS = ("nuplan_single", "nuplan_multi")
BASELINE_EVAL_ROOT = (
    REPO_ROOT
    / "experiments/baseline_run_sync_2026-08-24/"
    "baseline_run_sync_2026-08-24_2026-08-26_16-23-18_seed0/eval"
)
DEFAULT_OUTPUT_DIR = REPO_ROOT / "project/jepa_distill/analysis_output/nuplan_comparison"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--jepa-root", type=Path, required=True)
    parser.add_argument("--no-warm-start-root", type=Path, required=True)
    parser.add_argument("--baseline-eval-root", type=Path, default=BASELINE_EVAL_ROOT)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    return parser.parse_args()


def load_yaml(path: Path) -> dict:
    try:
        value = yaml.safe_load(path.read_text(encoding="utf-8"))
    except (OSError, yaml.YAMLError) as error:
        raise ValueError(f"Cannot read baseline protocol {path}: {error}") from error
    if not isinstance(value, dict):
        raise ValueError(f"Expected a YAML object in {path}")
    return value


def require_equal_fields(actual: dict, expected: dict, path: Path) -> None:
    for key, expected_value in expected.items():
        actual_value = actual.get(key)
        if isinstance(expected_value, bool):
            matches = type(actual_value) is bool and actual_value is expected_value
        else:
            matches = actual_value == expected_value
        if not matches:
            raise ValueError(
                f"Expected {key}={expected_value!r} in {path}; found {actual_value!r}"
            )


def baseline_summary_path(baseline_eval_root: Path, benchmark: str) -> Path:
    metrics_dir = baseline_eval_root / f"{benchmark}_final_model_mean_metrics"
    candidates = sorted(
        path for path in metrics_dir.glob("*/evaluation_summary.json") if path.is_file()
    )
    if len(candidates) != 1:
        raise ValueError(
            f"Expected exactly one immediate evaluation summary under {metrics_dir}; "
            f"found {len(candidates)}"
        )
    return candidates[0].resolve()


def load_baseline_protocol(path: Path, benchmark: str) -> dict:
    config = load_yaml(path)
    benchmark_config = config.get("benchmark")
    arguments = config.get("args")
    if not isinstance(benchmark_config, dict) or not isinstance(arguments, dict):
        raise ValueError(f"Missing benchmark or args object in baseline protocol {path}")
    benchmark_env = benchmark_config.get("env")
    runtime_env = arguments.get("env")
    runtime_eval = arguments.get("eval")
    vector_config = arguments.get("vec")
    if not all(isinstance(value, dict) for value in (benchmark_env, runtime_env, runtime_eval, vector_config)):
        raise ValueError(f"Missing nested baseline protocol object in {path}")

    expected = {
        "benchmark": benchmark,
        "seed": 42,
        "num_scenarios": shared.EXPECTED_COUNT,
        "benchmark_scenario_length": 200,
        "evaluation_dt": 0.1,
        "scenario_length_steps": 200,
        "runtime_seed": 42,
        "runtime_num_scenarios": shared.EXPECTED_COUNT,
        "action_selection": "mean",
    }
    actual = {
        "benchmark": benchmark_config.get("name"),
        "seed": benchmark_config.get("seed"),
        "num_scenarios": benchmark_config.get("num_scenarios"),
        "benchmark_scenario_length": benchmark_env.get("scenario_length"),
        "evaluation_dt": runtime_env.get("dt"),
        "scenario_length_steps": runtime_env.get("scenario_length"),
        "runtime_seed": vector_config.get("seed"),
        "runtime_num_scenarios": arguments.get("num_scenarios"),
        "action_selection": runtime_eval.get("action_selection"),
    }
    require_equal_fields(actual, expected, path)
    checkpoint = arguments.get("load_model_path")
    if not isinstance(checkpoint, str) or not checkpoint:
        raise ValueError(f"Missing checkpoint path in baseline protocol {path}")
    return {
        "benchmark": benchmark,
        "seed": 42,
        "num_scenarios": shared.EXPECTED_COUNT,
        "evaluation_dt": 0.1,
        "scenario_length_steps": 200,
        "action_selection": "mean",
        "checkpoint": checkpoint,
    }


def validate_chunk_protocol(protocol: dict, benchmark: str, k: int, path: Path) -> None:
    expected = {
        "benchmark": benchmark,
        "evaluation_dt": 0.1,
        "execution_k": k,
        "hold_steps_per_slot": 1 if k == 1 else 3,
        "num_scenarios": shared.EXPECTED_COUNT,
        "replan_interval_steps": 1 if k == 1 else 3 * k,
        "scenario_length_steps": 200,
        "seed": 42,
        "action_selection": "mean",
        "action_type": "continuous",
        "smoke": False,
    }
    require_equal_fields(protocol, expected, path)
    if not isinstance(protocol.get("checkpoint"), str) or not protocol["checkpoint"]:
        raise ValueError(f"Missing checkpoint path in {path}")


def load_records(args: argparse.Namespace) -> tuple[dict[str, list[dict]], dict[str, dict]]:
    records_by_benchmark: dict[str, list[dict]] = {}
    provenance_by_benchmark: dict[str, dict] = {}
    checkpoints_by_series: dict[str, set[str]] = {series: set() for series in shared.SERIES}

    for benchmark in BENCHMARKS:
        records: list[dict] = []
        protocols: list[dict] = []
        scenario_key_sets: list[set[tuple[str, str]]] = []
        baseline_path = baseline_summary_path(args.baseline_eval_root.resolve(), benchmark)
        baseline_protocol_path = baseline_path.with_name("resolved_benchmark.yaml")
        baseline_protocol = load_baseline_protocol(baseline_protocol_path, benchmark)
        baseline_csv = baseline_path.with_name("episode_metrics.csv")
        baseline_metrics, scenario_count, episode_count = shared.summary_metrics(baseline_path)
        csv_count, baseline_keys, unused_means = shared.read_episode_csv(baseline_csv, [])
        scenario_key_sets.append(baseline_keys)
        checkpoints_by_series[shared.SERIES[0]].add(baseline_protocol["checkpoint"])
        records.append({
            "benchmark": benchmark,
            "series": shared.SERIES[0],
            "k": "",
            "metrics": baseline_metrics,
            "source_path": str(baseline_path),
            "episode_csv": str(baseline_csv.resolve()),
            "protocol_path": str(baseline_protocol_path.resolve()),
            "checkpoint": baseline_protocol["checkpoint"],
            "scenario_count": scenario_count,
            "episode_count": episode_count,
            "csv_row_count": csv_count,
        })
        protocols.append({
            "benchmark": benchmark,
            "series": shared.SERIES[0],
            "protocol_path": str(baseline_protocol_path.resolve()),
            **baseline_protocol,
        })

        for jepa_root, series in (
            (args.jepa_root.resolve(), shared.SERIES[1]),
            (args.no_warm_start_root.resolve(), shared.SERIES[2]),
        ):
            for k in shared.EXPECTED_KS:
                run_dir = jepa_root / benchmark / f"run_k{k}" / f"k{k}"
                summary_path = (
                    run_dir / "eval" / f"{benchmark}_act_k{k}" / "standalone"
                    / "evaluation_summary.json"
                ).resolve()
                protocol_path = (run_dir / "protocol.json").resolve()
                protocol = shared.load_json(protocol_path)
                validate_chunk_protocol(protocol, benchmark, k, protocol_path)
                metrics, scenarios, episodes = shared.summary_metrics(summary_path)
                csv_path = summary_path.with_name("episode_metrics.csv")
                csv_count, scenario_keys, unused_means = shared.read_episode_csv(csv_path, [])
                scenario_key_sets.append(scenario_keys)
                checkpoint = protocol["checkpoint"]
                checkpoints_by_series[series].add(checkpoint)
                records.append({
                    "benchmark": benchmark,
                    "series": series,
                    "k": k,
                    "metrics": metrics,
                    "source_path": str(summary_path),
                    "episode_csv": str(csv_path),
                    "protocol_path": str(protocol_path),
                    "checkpoint": checkpoint,
                    "scenario_count": scenarios,
                    "episode_count": episodes,
                    "csv_row_count": csv_count,
                })
                protocols.append({
                    "benchmark": benchmark,
                    "series": series,
                    "k": k,
                    "protocol_path": str(protocol_path),
                    **protocol,
                })

        if any(keys != baseline_keys for keys in scenario_key_sets[1:]):
            raise ValueError(f"{benchmark} episode CSVs do not contain the same map/seed scenarios")
        records_by_benchmark[benchmark] = records
        provenance_by_benchmark[benchmark] = {
            "protocols": protocols,
            "scenario_set_matches": True,
        }

    for series, checkpoints in checkpoints_by_series.items():
        if len(checkpoints) != 1:
            raise ValueError(
                f"Expected one checkpoint across both benchmarks for {series}; "
                f"found {sorted(checkpoints)}"
            )
    return records_by_benchmark, provenance_by_benchmark


def write_reports(records: list[dict], provenance: dict, output_dir: Path) -> None:
    metric_columns = [
        "benchmark", "series", "k", "checkpoint", "source_path", "episode_csv",
        "protocol_path", "scenario_count", "episode_count", "csv_row_count", *shared.METRICS,
    ]
    with (output_dir / "aggregate_metrics.csv").open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=metric_columns)
        writer.writeheader()
        for record in records:
            writer.writerow({**{key: record.get(key, "") for key in metric_columns}, **record["metrics"]})

    provenance["records"] = [
        {key: record[key] for key in (
            "benchmark", "series", "k", "checkpoint", "source_path", "episode_csv",
            "protocol_path", "scenario_count", "episode_count", "csv_row_count",
        )}
        for record in records
    ]
    (output_dir / "provenance.json").write_text(
        json.dumps({"expected_scenarios": shared.EXPECTED_COUNT, **provenance}, indent=2) + "\n",
        encoding="utf-8",
    )

    baseline = records[0]["metrics"]["puffer_score"]
    at_k1 = [
        next(record for record in records if record["series"] == series and record["k"] == 1)
        for series in shared.SERIES[1:]
    ]
    lines = [
        f"# {records[0]['benchmark']} metric comparison", "",
        "All nine episode CSVs contain the same 1,000 map/seed pairs. Summary JSON supplies the 22 plotted metrics; episode CSVs validate scenario identity and counts.", "",
        "## Puffer score at k=1", "",
        "| Evaluation | Puffer score |", "|---|---:|",
        f"| Baseline | {baseline:.5f} |",
    ]
    lines.extend(
        f"| {record['series']} | {record['metrics']['puffer_score']:.5f} |"
        for record in at_k1
    )
    lines += [
        "", "## Puffer score by chunk schedule", "",
        f"| k | {shared.SERIES[1]} | {shared.SERIES[2]} |", "|---:|---:|---:|",
    ]
    for k in shared.EXPECTED_KS:
        values = [
            next(record for record in records if record["series"] == series and record["k"] == k)
            for series in shared.SERIES[1:]
        ]
        lines.append(f"| {k} | {values[0]['metrics']['puffer_score']:.5f} | {values[1]['metrics']['puffer_score']:.5f} |")
    lines += [
        "", "## Protocol and interpretation", "",
        "Both JEPA series and the baseline use seed 42, mean actions, `dt=0.1`, and 200-step episodes. At this rate, k=1 replans every step; k=2, 3, and 4 hold each selected slot for three steps and replan every 6, 9, and 12 steps. The `comfort_score` metric is known to be broken.", "",
        f"The shared series label `{shared.SERIES[1]}` is retained for consistency with the CARLA plots. Exact checkpoint paths for every record are in `provenance.json`.", "",
    ]
    (output_dir / "summary.md").write_text("\n".join(lines), encoding="utf-8")


def generate_comparisons(args: argparse.Namespace) -> None:
    # Validate both benchmarks before the first directory or plot is written.
    records_by_benchmark, provenance_by_benchmark = load_records(args)
    output_root = args.output_dir.resolve()
    for benchmark in BENCHMARKS:
        records = records_by_benchmark[benchmark]
        output_dir = output_root / benchmark
        output_dir.mkdir(parents=True, exist_ok=True)
        shared.save_figure(
            records, shared.OVERVIEW_METRICS, output_dir / "k1_vs_baseline",
            f"{benchmark}: comparison at k=1", "k1",
        )
        shared.save_figure(
            records, shared.OVERVIEW_METRICS, output_dir / "performance_vs_k",
            f"{benchmark}: performance by chunk schedule", "curve",
        )
        for comparison, mode, title in (
            ("k1_vs_baseline", "k1", "k=1 vs baseline"),
            ("performance_vs_k", "curve", "performance by k"),
        ):
            for group_name, metrics in shared.METRIC_GROUPS.items():
                shared.save_figure(
                    records,
                    metrics,
                    output_dir / "details" / comparison / group_name,
                    f"{benchmark}: {title}: {group_name.replace('_', ' ')}",
                    mode,
                )
        write_reports(records, provenance_by_benchmark[benchmark], output_dir)
    print(f"Wrote nuPlan comparison figures and reports to {output_root}")


def main() -> None:
    args = parse_args()
    try:
        generate_comparisons(args)
    except (OSError, ValueError) as error:
        raise SystemExit(f"Error: {error}") from error


if __name__ == "__main__":
    main()
