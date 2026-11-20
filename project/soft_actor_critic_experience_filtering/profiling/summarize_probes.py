"""Validate row counters and summarize bounded TES profiling artifacts."""

import argparse
import json
import statistics
from pathlib import Path


def summarize_case(directory):
    summary = json.loads((directory / "summary.json").read_text())
    if not summary["all_parameters_finite"]:
        raise ValueError(f"Nonfinite model parameters in {directory}")
    cycles = []
    for path in sorted(directory.glob("cycle_*.json")):
        record = json.loads(path.read_text())
        cycle_index = record["cycle_index"]
        totals = record["global"]
        sidecars = [json.loads(path.read_text()) for path in directory.glob(
            f"tes_filter_cycle_{cycle_index:03d}_rank_*.json"
        )]
        if len(sidecars) != summary["world_size"]:
            raise ValueError("Missing per-rank sidecars")
        critic_rows = sum(row["row_exposures_local"]["unfiltered_critic_loss_rows"]
                          + row["row_exposures_local"]["filtered_critic_loss_sum_rows"]
                          for row in sidecars)
        actor_rows = sum(row["row_exposures_local"]["policy_loss_rows"] for row in sidecars)
        retained_rows = totals["filter_retained_rows"]
        if critic_rows != retained_rows or actor_rows != retained_rows:
            raise ValueError(f"Retained/actor/critic counter mismatch in {path}")
        if totals["loss_row_exposures"] != actor_rows + critic_rows:
            raise ValueError(f"Harness loss counter mismatch in {path}")
        cycle = {
            "cycle_index": cycle_index, "measured": record["measurement_included"],
            "collection_seconds": totals["collection_seconds"],
            "learner_seconds": totals["optimization_seconds"],
            "cycle_seconds": totals["cycle_seconds"],
            "received_rows": totals["received_rows"],
            "candidate_rows": totals["filter_candidate_rows"],
            "target_forward_rows": sum(row["target_forward_rows_local"] for row in record["ranks"]),
            "retained_rows": retained_rows,
            "retained_fraction": retained_rows / totals["filter_candidate_rows"],
            "optimizer_steps_summed_across_ranks": totals["optimizer_steps"],
        }
        rank_zero = next(row for row in sidecars if row["rank"] == 0)
        if rank_zero["detailed_timing"]:
            cycle["rank0_method_seconds_inclusive"] = {
                key: value["inclusive_total_seconds"]
                for key, value in rank_zero["detailed_timing"]["methods"].items()
            }
        cycles.append(cycle)
    measured = [row for row in cycles if row["measured"]]
    return {
        "directory": str(directory), "all_parameters_finite": True,
        "all_row_counters_agree": True, "cycles": cycles,
        "mean_learner_seconds": statistics.mean(row["learner_seconds"] for row in measured),
        "mean_collection_seconds": statistics.mean(row["collection_seconds"] for row in measured),
        "mean_cycle_seconds": statistics.mean(row["cycle_seconds"] for row in measured),
        "retained_fraction": sum(row["retained_rows"] for row in measured) / sum(row["candidate_rows"] for row in measured),
        "initialization": summary["weight_initialization"],
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("root", type=Path)
    args = parser.parse_args()
    cases = {name: summarize_case(args.root / name) for name in (
        "baseline_restored", "filter_restored", "all_keep_restored", "filter_detailed",
    )}
    (args.root / "probe_comparison.json").write_text(json.dumps(cases, indent=2) + "\n")
    for name, result in cases.items():
        print(name, {key: result[key] for key in (
            "mean_learner_seconds", "mean_collection_seconds", "retained_fraction",
        )})


if __name__ == "__main__":
    main()
