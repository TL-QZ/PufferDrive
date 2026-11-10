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
    loss_exposures = sum(row.get("loss_row_exposures", sum(
        rank["loss_row_exposures_local"] for rank in cycle["ranks"]
    )) for row, cycle in zip(globals_, measured))
    sac_training = summary.get("sac_training") or {}
    candidate_rows = sum(row.get("filter_candidate_rows", 0) for row in globals_)
    retained_rows = sum(row.get("filter_retained_rows", 0) for row in globals_)
    score_means = [
        (row.get("filter_score_mean"), row.get("filter_candidate_rows", 0))
        for row in globals_ if row.get("filter_score_mean") is not None
    ]
    score_mean = (
        sum(mean * count for mean, count in score_means) / sum(count for _mean, count in score_means)
        if score_means and sum(count for _mean, count in score_means) else None
    )
    cycle_filtering = [{
        "cycle_index": cycle["cycle_index"],
        "candidate_rows": row.get("filter_candidate_rows", 0),
        "retained_rows": row.get("filter_retained_rows", 0),
        "score_min": row.get("filter_score_min"),
        "score_max": row.get("filter_score_max"),
        "score_mean": row.get("filter_score_mean"),
        "filter_current_max": row.get("filter_current_max"),
        "filter_ema_max": row.get("filter_ema_max"),
        "filter_threshold": row.get("filter_threshold"),
        "filter_retained_before_sync": row.get("filter_retained_before_sync"),
    } for row, cycle in zip(globals_, measured)]
    return {
        "run": str(path.parent), "algorithm": summary["algorithm"],
        "variant": summary["variant"], "settings": summary["profile_settings"],
        "sac_training": sac_training,
        "return_estimator": (
            sac_training.get("return_estimator", "n_step")
            if summary["algorithm"] == "sac" else None
        ),
        "pql_horizon": sac_training.get("pql_horizon"),
        "pql_lambda": sac_training.get("pql_lambda"),
        "profile_filter_score": sac_training.get(
            "profile_filter_score", summary["profile_settings"].get("sac_filter_score", "none"),
        ),
        "profile_keep_fraction": sac_training.get(
            "profile_keep_fraction", summary["profile_settings"].get("sac_keep_fraction", 1.0),
        ),
        "profile_filter_mode": sac_training.get(
            "profile_filter_mode", summary["profile_settings"].get("sac_filter_mode", "fraction"),
        ),
        "profile_filter_ewma_beta": sac_training.get(
            "profile_filter_ewma_beta", summary["profile_settings"].get("sac_filter_ewma_beta", 0.25),
        ),
        "profile_filter_threshold_scale": sac_training.get(
            "profile_filter_threshold_scale",
            summary["profile_settings"].get("sac_filter_threshold_scale", 0.01),
        ),
        "world_size": summary["world_size"], "measured_cycles": len(measured),
        "received_rows": received, "sampled_transitions": sampled,
        "loss_row_exposures": loss_exposures,
        "training_exposures_per_received_row": sampled / received if received else None,
        "loss_row_exposures_per_received_row": loss_exposures / received if received else None,
        "filter_candidate_rows": candidate_rows,
        "filter_retained_rows": retained_rows,
        "filter_retention_fraction": retained_rows / candidate_rows if candidate_rows else None,
        "filter_score_min": min(
            (row["score_min"] for row in cycle_filtering
             if row["score_min"] is not None), default=None,
        ),
        "filter_score_max": max(
            (row["score_max"] for row in cycle_filtering
             if row["score_max"] is not None), default=None,
        ),
        "filter_score_mean": score_mean,
        "cycle_filtering": cycle_filtering,
        "collection_seconds": collection, "training_seconds": training,
        "mean_collection_seconds": collection / len(measured),
        "mean_training_seconds": training / len(measured),
        "median_training_seconds": statistics.median(row["optimization_seconds"] for row in globals_),
        "min_training_seconds": min(row["optimization_seconds"] for row in globals_),
        "max_training_seconds": max(row["optimization_seconds"] for row in globals_),
        "received_rows_per_second": received / (collection + training)
        if collection + training else None,
        "learner_microseconds_per_sample": training / sampled * 1e6 if sampled else None,
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
             "Warmup and intrusive trace cycles excluded. For SAC, `sampled_transitions` counts critic "
             "replay-row exposures; PPO counts combined loss-row exposures. `loss_row_exposures` also "
             "includes actor loss rows for SAC. GPU times are synchronized "
             "phase wall times; full raw per-rank data are retained.", "",
             "| Run | Estimator | Filter (keep) | Cycles | Collect s/cycle | Train s/cycle | Retained/candidate rows | Training samples/received row | Loss rows/received row | Learner us/sample | Peak GPU GiB |",
             "|---|---|---|---:|---:|---:|---:|---:|---:|---:|---:|"]
    for row in rows:
        filter_name = row["profile_filter_score"]
        filter_setting = "none" if filter_name == "none" else (
            f"{filter_name} ewma_threshold"
            if row["profile_filter_mode"] == "ewma_threshold"
            else f"{filter_name} ({row['profile_keep_fraction']:.0%})"
        )
        retained = (
            f"{row['filter_retained_rows']:,}/{row['filter_candidate_rows']:,}"
            if row["filter_candidate_rows"] else "—"
        )
        estimator = row["return_estimator"] or "—"
        training_samples_per_received = row["training_exposures_per_received_row"]
        training_samples_per_received_text = (
            "—" if training_samples_per_received is None else f"{training_samples_per_received:.3f}"
        )
        loss_rows_per_received = row["loss_row_exposures_per_received_row"]
        loss_rows_per_received_text = "—" if loss_rows_per_received is None else f"{loss_rows_per_received:.3f}"
        learner_us = row["learner_microseconds_per_sample"]
        learner_us_text = "—" if learner_us is None else f"{learner_us:.3f}"
        lines.append(f"| {row['run']} | {estimator} | {filter_setting} | "
                     f"{row['measured_cycles']} | "
                     f"{row['mean_collection_seconds']:.3f} | {row['mean_training_seconds']:.3f} | "
                     f"{retained} | {training_samples_per_received_text} | "
                     f"{loss_rows_per_received_text} | "
                     f"{learner_us_text} | {row['peak_gpu_allocated_gib']:.2f} |")
    args.output.with_suffix(".md").write_text("\n".join(lines) + "\n")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
