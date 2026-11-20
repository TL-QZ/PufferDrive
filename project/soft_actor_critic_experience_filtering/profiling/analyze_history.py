"""Summarize completed TES filtering runs without conflating timer domains."""

import argparse
import json
import statistics
from pathlib import Path


def analyze_trial(trial):
    rows = [json.loads(line) for line in (trial / "metrics.jsonl").read_text().splitlines()]
    summary = json.loads((trial / "summary.json").read_text())
    final = rows[-1]
    if any(current["epoch"] != previous["epoch"] + 1 for previous, current in zip(rows, rows[1:])):
        raise ValueError(f"Expected contiguous per-cycle metrics: {trial}")
    import yaml
    config = yaml.safe_load((trial / "learner/config.yaml").read_text())
    if config["train"]["evaluation_interval_epochs"] != 2:
        raise ValueError("Alternating-cycle estimate requires evaluation every two epochs")
    durations = {"after_benchmark_epoch": [], "without_benchmark": []}
    for previous, current in zip(rows, rows[1:]):
        if current["epoch"] <= 10:
            continue
        key = "after_benchmark_epoch" if previous["epoch"] % 2 == 0 else "without_benchmark"
        durations[key].append(current["uptime"] - previous["uptime"])
    timing = {
        key: {"count": len(values), "mean_seconds": statistics.mean(values),
              "median_seconds": statistics.median(values)}
        for key, values in durations.items()
    }
    windows = []
    previous = None
    for boundary in (1_000_000_000, 2_000_000_000, 3_000_000_000, 4_000_000_000):
        current = next(row for row in rows if row["agent_steps"] >= boundary)
        previous_steps = previous["agent_steps"] if previous else 0
        previous_uptime = previous["uptime"] if previous else 0
        window = {"boundary": boundary, "actual_end_steps": current["agent_steps"],
                  "received_steps": current["agent_steps"] - previous_steps,
                  "uptime_delta_seconds": current["uptime"] - previous_uptime}
        window["seconds_per_million_received"] = window["uptime_delta_seconds"] / window["received_steps"] * 1e6
        for key in ("performance/train", "performance/eval",
                    "filter/cumulative_retained_count", "filter/cumulative_candidate_count"):
            if key in current:
                window[key + "_delta"] = current[key] - (previous[key] if previous else 0)
        windows.append(window)
        previous = current
    estimated_train = final["performance/train"]
    estimated_collection = final["performance/eval"]
    return {
        "trial": str(trial), "status": summary["status"],
        "wall_seconds": summary["full_time_seconds"], "logged_uptime_seconds": final["uptime"],
        "received_steps": summary["agent_steps"], "epochs": summary["epoch"],
        "optimizer_updates": summary["learner_updates"],
        "benchmark_evaluations": len(list((trial / "learner/eval/training/carla_fast").glob("*/evaluation_summary.json"))),
        "candidate_exposures": summary["sampled_transitions"],
        "retained_exposures": final.get("filter/cumulative_retained_count", summary["sampled_transitions"]),
        "retained_fraction": final.get("filter/cumulative_retained_fraction", 1.0),
        "global_empty_batches": final.get("filter/cumulative_empty_batches", 0),
        "estimated_learner_seconds": estimated_train,
        "estimated_collection_seconds": estimated_collection,
        "unattributed_wall_seconds": summary["full_time_seconds"] - estimated_train - estimated_collection,
        "estimated_phase_seconds": {key: value for key, value in final.items() if key.startswith("performance/")},
        "alternating_cycle_durations": timing,
        "estimated_benchmark_increment_seconds": timing["after_benchmark_epoch"]["mean_seconds"] - timing["without_benchmark"]["mean_seconds"],
        "step_windows": windows,
        "actor_retained_counts_agree_all_cycles": all(row.get("filter/actor_rows") == row.get("filter/retained_count") for row in rows),
        "notes": [
            "performance timers sample every fifth cycle and multiply duration by five; they are estimates.",
            "performance/eval is simulation collection, not scheduled CARLA benchmark evaluation.",
            "Unattributed wall time includes benchmarks, checkpointing, logging, setup and timer estimation error.",
            "Alternating-cycle difference estimates benchmark overhead; it is not direct benchmark instrumentation.",
            "Step windows use first logged crossing; actual endpoints are recorded, not exactly equal across runs.",
        ],
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--baseline", type=Path, required=True)
    parser.add_argument("--filtered", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    baseline = analyze_trial(args.baseline)
    filtered = analyze_trial(args.filtered)
    result = {"baseline": baseline, "filtered": filtered,
              "wall_speedup": baseline["wall_seconds"] / filtered["wall_seconds"],
              "estimated_learner_speedup": baseline["estimated_learner_seconds"] / filtered["estimated_learner_seconds"]}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps({key: value for key, value in result.items() if isinstance(value, float)}, indent=2))


if __name__ == "__main__":
    main()
