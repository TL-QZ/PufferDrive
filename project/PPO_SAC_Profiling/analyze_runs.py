"""Summarize completed bounded runs using actual sampled rows, excluding traces."""
import argparse
import json
from pathlib import Path
import statistics


def summarize_run(path):
    summary = json.loads(path.read_text())
    cycles = [json.loads((path.parent / name).read_text()) for name in summary["cycle_files"]]
    measured = [cycle for cycle in cycles if cycle["measurement_included"]
                and not cycle.get("torch_profiler_intrusive", False)]
    if not measured:
        return None
    globals_ = [cycle["global"] for cycle in measured]
    collection = sum(row["collection_seconds"] for row in globals_)
    training = sum(row["optimization_seconds"] for row in globals_)
    received = sum(row["received_rows"] for row in globals_)
    # SAC joint actor+critic loss-call rows count each replay sample twice.
    sampled = sum(row.get("sampled_transitions", sum(
        rank["critic_loss_rows_local"] if summary["algorithm"] == "sac"
        else rank["loss_row_exposures_local"] for rank in cycle["ranks"]
    )) for row, cycle in zip(globals_, measured))
    return {
        "run": str(path.parent), "algorithm": summary["algorithm"],
        "variant": summary["variant"], "settings": summary["profile_settings"],
        "world_size": summary["world_size"], "measured_cycles": len(measured),
        "received_rows": received, "sampled_transitions": sampled,
        "training_exposures_per_received_row": sampled / received,
        "collection_seconds": collection, "training_seconds": training,
        "mean_collection_seconds": collection / len(measured),
        "mean_training_seconds": training / len(measured),
        "median_training_seconds": statistics.median(row["optimization_seconds"] for row in globals_),
        "min_training_seconds": min(row["optimization_seconds"] for row in globals_),
        "max_training_seconds": max(row["optimization_seconds"] for row in globals_),
        "received_rows_per_second": received / (collection + training),
        "learner_microseconds_per_sample": training / sampled * 1e6,
        "peak_gpu_allocated_gib": max(row["peak_gpu_allocated_bytes"] or 0 for row in globals_) / 2**30,
        "cycle_files": [str(path.parent / f"cycle_{cycle['cycle_index']:03d}.json") for cycle in measured],
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("inputs", nargs="+", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    paths = sorted({path for directory in args.inputs for path in directory.rglob("summary.json")})
    rows = [row for path in paths if (row := summarize_run(path)) is not None]
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.with_suffix(".json").write_text(json.dumps(rows, indent=2) + "\n")
    lines = ["# Bounded training measurements", "",
             "Warmup and intrusive trace cycles excluded. Sampled rows count a SAC replay row once, "
             "even though it contributes to both critic and actor losses. GPU times are synchronized "
             "phase wall times; full raw per-rank data are retained.", "",
             "| Run | Cycles | Collect s/cycle | Train s/cycle | Train range s | Samples/received row | Learner us/sample | Received rows/s | Peak GPU GiB |",
             "|---|---:|---:|---:|---:|---:|---:|---:|---:|"]
    for row in rows:
        lines.append(f"| {row['run']} | {row['measured_cycles']} | "
                     f"{row['mean_collection_seconds']:.3f} | {row['mean_training_seconds']:.3f} | "
                     f"{row['min_training_seconds']:.3f}–{row['max_training_seconds']:.3f} | "
                     f"{row['training_exposures_per_received_row']:.3f} | "
                     f"{row['learner_microseconds_per_sample']:.3f} | "
                     f"{row['received_rows_per_second']:.0f} | {row['peak_gpu_allocated_gib']:.2f} |")
    args.output.with_suffix(".md").write_text("\n".join(lines) + "\n")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
