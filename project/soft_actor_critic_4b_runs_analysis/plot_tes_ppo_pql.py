#!/usr/bin/env python3
"""Render the existing learning-curve format with only TES R2, PQL R2, and PPO."""

import argparse

from analyze_learning_curves import (
    CURVES_DIR, PPO_RUN, RAW_CURVES_PATH, SAC_RUNS,
    deduplicate_points, plot_curves, read_raw_snapshot,
    select_matched_points, write_csv,
)

RUNS = (SAC_RUNS[0], SAC_RUNS[-1], PPO_RUN)
OUTPUT = CURVES_DIR / "tes_round2_ppo_pql_official_round2"


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--small-goals", action="store_true",
                        help="Plot the four completed 10 m goal runs against PPO")
    args = parser.parse_args()
    runs = SAC_RUNS[1:5] + (PPO_RUN,) if args.small_goals else RUNS
    output = (CURVES_DIR / "small_goal_spacing_runs_vs_ppo") if args.small_goals else OUTPUT
    snapshot = read_raw_snapshot(RAW_CURVES_PATH)
    matches, _ = select_matched_points(snapshot)
    run_ids = {run.run_id for run in runs}
    rows = [row for row in snapshot if row["run_id"] in run_ids]
    matches = [row for row in matches if row["run_id"] in run_ids]
    plot_curves(rows, matches, runs=runs, plot_path=output)
    match_by_key = {(row["run_id"], row["agent_steps"]): row for row in matches}
    deduped = deduplicate_points(rows)
    for row in deduped:
        match = match_by_key.get((row["run_id"], row["agent_steps"]))
        is_ppo = row["run_id"] == PPO_RUN.run_id
        row["is_plotted_match"] = int(is_ppo or match is not None)
        row["ppo_anchor_steps"] = row["agent_steps"] if is_ppo else (
            match["ppo_anchor_steps"] if match else ""
        )
    write_csv(output.with_suffix(".csv"), deduped, list(deduped[0]))
    print(f"Saved {output}.png, .pdf, .csv")
    print("Runs: " + ", ".join(run.label for run in runs))


if __name__ == "__main__":
    main()
