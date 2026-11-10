"""Compare completed cycles while the 1B experiment is still running."""
import argparse
import json
from pathlib import Path


def summarize_cycles(case_directory, limit=None):
    cycles = [json.loads(path.read_text()) for path in sorted(case_directory.glob("cycle_*.json"))]
    measured = [cycle for cycle in cycles if cycle["measurement_included"]
                and not cycle.get("torch_profiler_intrusive", False)]
    if limit is not None:
        measured = measured[:limit]
    if not measured:
        return None
    records = [cycle["global"] for cycle in measured]
    received = sum(record["received_rows"] for record in records)
    collection = sum(record["collection_seconds"] for record in records)
    learning = sum(record["optimization_seconds"] for record in records)
    candidates = sum(record["filter_candidate_rows"] for record in records)
    retained = sum(record["filter_retained_rows"] for record in records)
    return {
        "cycles": len(records), "received_rows": received,
        "learner_seconds_per_cycle": learning / len(records),
        "collection_seconds_per_cycle": collection / len(records),
        "received_rows_per_second": received / (collection + learning),
        "retained_fraction": retained / candidates if candidates else None,
        "peak_gpu_allocated_gib": max(record["peak_gpu_allocated_bytes"] or 0
                                      for record in records) / 2**30,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run_directory", type=Path)
    args = parser.parse_args()
    stock = summarize_cycles(args.run_directory / "stock_baseline", 3)
    optimized = summarize_cycles(args.run_directory / "optimized", 3)
    report = {"stock_first_three_measured": stock,
              "optimized_first_three_measured": optimized,
              "optimized_all_measured": summarize_cycles(args.run_directory / "optimized")}
    if stock and optimized and stock["cycles"] == optimized["cycles"] == 3:
        report["early_learner_speedup"] = (stock["learner_seconds_per_cycle"]
                                            / optimized["learner_seconds_per_cycle"])
        report["early_collection_plus_learning_throughput_speedup"] = (
            optimized["received_rows_per_second"] / stock["received_rows_per_second"])
    report["scope"] = ("Warmup excluded; first three measured cycles are an early-training "
                       "comparison, not a matched-policy or full-1B stock comparison. "
                       "Snapshot and startup costs are excluded from phase throughput.")
    print(json.dumps(report, indent=2, allow_nan=False))


if __name__ == "__main__":
    main()
