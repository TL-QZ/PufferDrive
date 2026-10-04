"""Loader and output contract checks for the nuPlan comparison plotter."""

import csv
import json
import tempfile
from pathlib import Path
from types import SimpleNamespace

import pytest

from project.jepa_distill.plot import plot_nuplan_comparison as plotter


def _write_csv(path: Path, *, changed_seed_row: bool = False) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as output_file:
        writer = csv.DictWriter(output_file, fieldnames=("map_name", "seed"))
        writer.writeheader()
        for row_idx in range(plotter.shared.EXPECTED_COUNT):
            seed = "43" if changed_seed_row and row_idx == 999 else "42"
            writer.writerow({"map_name": f"map_{row_idx}", "seed": seed})


def _write_summary(path: Path, score: float) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    metrics = {
        metric: score
        for metric in plotter.shared.METRICS
        if metric != "infractions_per_scenario"
    }
    metrics["total_infractions"] = 100.0
    path.write_text(
        json.dumps({
            "num_scenarios": plotter.shared.EXPECTED_COUNT,
            "num_episodes": plotter.shared.EXPECTED_COUNT,
            "metrics_mean": metrics,
        }),
        encoding="utf-8",
    )


def _write_baseline(root: Path, benchmark: str) -> None:
    output_dir = root / f"{benchmark}_final_model_mean_metrics" / "synthetic"
    output_dir.mkdir(parents=True)
    (output_dir / "resolved_benchmark.yaml").write_text(
        """benchmark:
  name: {benchmark}
  seed: 42
  num_scenarios: 1000
  env:
    scenario_length: 200
args:
  load_model_path: /synthetic/checkpoints/baseline.pt
  num_scenarios: 1000
  vec:
    seed: 42
  env:
    dt: 0.1
    scenario_length: 200
  eval:
    action_selection: mean
""".format(benchmark=benchmark),
        encoding="utf-8",
    )
    _write_summary(output_dir / "evaluation_summary.json", 0.25)
    _write_csv(output_dir / "episode_metrics.csv")


def _write_chunk(
    root: Path,
    benchmark: str,
    k: int,
    *,
    checkpoint: str,
    wrong_benchmark: bool = False,
    changed_seed_row: bool = False,
) -> None:
    run_dir = root / benchmark / f"run_k{k}" / f"k{k}"
    summary_path = (
        run_dir / "eval" / f"{benchmark}_act_k{k}" / "standalone"
        / "evaluation_summary.json"
    )
    _write_summary(summary_path, 0.5 + 0.01 * k)
    _write_csv(summary_path.with_name("episode_metrics.csv"), changed_seed_row=changed_seed_row)
    protocol = {
        "benchmark": "nuplan_single" if wrong_benchmark else benchmark,
        "evaluation_dt": 0.1,
        "execution_k": k,
        "hold_steps_per_slot": 1 if k == 1 else 3,
        "num_scenarios": 1000,
        "replan_interval_steps": 1 if k == 1 else 3 * k,
        "scenario_length_steps": 200,
        "seed": 42,
        "action_selection": "mean",
        "action_type": "continuous",
        "smoke": False,
        "checkpoint": checkpoint,
    }
    run_dir.mkdir(parents=True, exist_ok=True)
    (run_dir / "protocol.json").write_text(json.dumps(protocol), encoding="utf-8")


def _make_fixture(root: Path) -> SimpleNamespace:
    baseline_root = root / "baseline"
    jepa_root = root / "jepa"
    no_warm_root = root / "no_warm"
    for benchmark in plotter.BENCHMARKS:
        _write_baseline(baseline_root, benchmark)
        for k in plotter.shared.EXPECTED_KS:
            _write_chunk(jepa_root, benchmark, k, checkpoint="/synthetic/checkpoints/jepa.pt")
            _write_chunk(
                no_warm_root, benchmark, k,
                checkpoint="/synthetic/checkpoints/no_warm.pt",
            )
    return SimpleNamespace(
        baseline_eval_root=baseline_root,
        jepa_root=jepa_root,
        no_warm_start_root=no_warm_root,
        output_dir=root / "output",
    )


def test_nuplan_loader_validates_both_benchmarks_before_writing_outputs():
    with tempfile.TemporaryDirectory(prefix="nuplan-plot-test-", dir="/tmp") as temp_dir:
        root = Path(temp_dir)
        args = _make_fixture(root)
        bad_protocol = (
            args.no_warm_start_root / "nuplan_multi/run_k4/k4/protocol.json"
        )
        protocol = json.loads(bad_protocol.read_text(encoding="utf-8"))
        protocol["benchmark"] = "wrong_benchmark"
        bad_protocol.write_text(json.dumps(protocol), encoding="utf-8")

        with pytest.raises(ValueError, match="Expected benchmark='nuplan_multi'"):
            plotter.generate_comparisons(args)
        assert not args.output_dir.exists()

        protocol["benchmark"] = "nuplan_multi"
        bad_protocol.write_text(json.dumps(protocol), encoding="utf-8")
        plotter.generate_comparisons(args)

        for benchmark in plotter.BENCHMARKS:
            output_dir = args.output_dir / benchmark
            assert (output_dir / "aggregate_metrics.csv").is_file()
            assert (output_dir / "provenance.json").is_file()
            assert (output_dir / "summary.md").is_file()
            assert len(list(output_dir.glob("*.png"))) == 2
            assert len(list(output_dir.glob("*.pdf"))) == 2
            assert len(list((output_dir / "details").rglob("*.png"))) == 8
            assert len(list((output_dir / "details").rglob("*.pdf"))) == 8


def test_nuplan_loader_requires_matching_episode_scenarios_per_benchmark():
    with tempfile.TemporaryDirectory(prefix="nuplan-plot-test-", dir="/tmp") as temp_dir:
        root = Path(temp_dir)
        args = _make_fixture(root)
        mismatched_csv = (
            args.jepa_root / "nuplan_single/run_k3/k3/eval/"
            "nuplan_single_act_k3/standalone/episode_metrics.csv"
        )
        _write_csv(mismatched_csv, changed_seed_row=True)

        with pytest.raises(ValueError, match="nuplan_single episode CSVs do not contain the same"):
            plotter.load_records(args)
        assert not args.output_dir.exists()
