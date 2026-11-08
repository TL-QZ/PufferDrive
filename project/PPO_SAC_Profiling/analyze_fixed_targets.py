"""Summarize fixed-target timing, actual bootstrap work and final target drift."""
import argparse
import json
from pathlib import Path

from analyze_runs import summarize_run


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("inputs", nargs="+", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    records = []
    paths = sorted({path for root in args.inputs for path in root.rglob("summary.json")})
    for path in paths:
        row = summarize_run(path)
        if row is None:
            continue
        summary = json.loads(path.read_text())
        cycles = [json.loads(Path(name).read_text()) for name in row["cycle_files"]]
        row["target_forward_rows"] = sum(
            rank["target_forward_rows_local"] for cycle in cycles for rank in cycle["ranks"])
        row["target_precompute_rows"] = sum(
            rank["fixed_target_precompute_rows_local"] for cycle in cycles for rank in cycle["ranks"])
        row["actor_q_precompute_rows"] = sum(
            rank.get("fixed_actor_q_precompute_rows_local", 0)
            for cycle in cycles for rank in cycle["ranks"])
        row["actor_q_cache_bytes_per_rank"] = max(
            rank.get("fixed_actor_q_cache_bytes_local", 0)
            for cycle in cycles for rank in cycle["ranks"])
        row["actor_training_rows"] = sum(
            rank["policy_loss_rows_local"] for cycle in cycles for rank in cycle["ranks"])
        row["target_rows_per_learner_sample"] = row["target_forward_rows"] / row["sampled_transitions"]
        row["final_states"] = summary["final_model_states"]
        row["all_parameters_finite"] = summary["all_parameters_finite"]
        row["rank_hashes_match"] = len({state["sha256"] for state in row["final_states"]}) == 1
        row["target_audits"] = [state["fixed_target_audit"] for state in row["final_states"]
                                if state.get("fixed_target_audit") is not None]
        records.append(row)
    args.output.with_suffix(".json").write_text(json.dumps(records, indent=2) + "\n")
    lines = ["# Fixed-target measurements", "",
             "Precomputation is inside learner timing. Target drift is measured after all timed cycles. "
             "Fixed targets preserve the cycle-start Bellman formula, not equality to the fresh-target update sequence.", "",
             "| Run | GPUs | Learner s/cycle | Collect s/cycle | Learner µs/sample | Received rows/s | Target rows/sample | Worst audit RMS |",
             "|---|---:|---:|---:|---:|---:|---:|---:|"]
    for row in records:
        worst_rms = max((audit["root_mean_square_difference"] for audit in row["target_audits"]), default=None)
        rms_text = "—" if worst_rms is None else f"{worst_rms:.6g}"
        lines.append(f"| {row['run']} | {row['world_size']} | {row['mean_training_seconds']:.3f} | "
                     f"{row['mean_collection_seconds']:.3f} | {row['learner_microseconds_per_sample']:.3f} | "
                     f"{row['received_rows_per_second']:.0f} | {row['target_rows_per_learner_sample']:.3f} | {rms_text} |")
    args.output.with_suffix(".md").write_text("\n".join(lines) + "\n")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
