"""Summarize third-run target/outcome agreement for PPO, PQL and TES."""
import argparse
import csv
import json
from pathlib import Path

import numpy as np

from .plot_binned_driving_quality import branch_outcome, outcome_metrics, state_scores
from .run_layout import records_directory


def summarize_model(model_dir):
    records = records_directory(model_dir)
    states = json.loads((records / "selection.json").read_text())["states"]
    is_ppo = model_dir.name == "ppo"
    target_key = "ppo_value_target" if is_ppo else "actual_critic_target"
    labels = outcome_metrics(model_dir.name, include_common_return=True)
    outcomes = {key: [] for key in labels}
    scores = {key: [] for key in labels}
    for state in states:
        stem = state["state_id"].replace("/", "__")
        probe = json.loads((records / "probes" / f"{stem}.json").read_text())
        branches = probe["branches"]
        if probe["state_id"] != state["state_id"] or sorted(b["action_idx"] for b in branches) != list(range(12)):
            raise ValueError(f"Incomplete action group for {stem}")
        targets = np.array([b[target_key] for b in branches], dtype=float)
        for key in labels:
            values = np.array([branch_outcome(b, key) for b in branches], dtype=float)
            if not np.isfinite(values).all() or not np.isfinite(targets).all():
                raise ValueError(f"Nonfinite input for {stem}, {key}")
            outcomes[key].extend(values)
            scores[key].append(state_scores(targets, values, lower_is_better=(key == "total_infractions")))

    rows = {}
    for key, label in labels.items():
        correlations, rankings = np.array(scores[key]).T
        defined_r = correlations[np.isfinite(correlations)]
        defined_rank = rankings[np.isfinite(rankings)]
        rows[key] = dict(outcome=label, raw_mean=float(np.mean(outcomes[key])),
                         mean_r=defined_r.mean() if defined_r.size else None,
                         mean_accuracy=defined_rank.mean() if defined_rank.size else None,
                         states_with_r=defined_r.size, states_with_accuracy=defined_rank.size,
                         state_count=len(states), rollout_count=len(outcomes[key]))
    return rows


def format_number(value, spec):
    return format(value, spec) if value is not None else "—"


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run_dir", type=Path)
    run = parser.parse_args().run_dir.resolve()
    models = ("pql", "tes", "ppo")
    results = {name: summarize_model(run / name) for name in models}
    columns = " | ".join(f"{name.upper()} mean | {name.upper()} r | {name.upper()} rank (n)" for name in models)
    table = [f"| Outcome | {columns} |", "|" + "---|" * 10]
    for key, entry in results["pql"].items():
        cells = [entry["outcome"]]
        for name in models:
            row = results[name].get(key)
            if row is None:
                cells.extend(["—"] * 3)
                continue
            cells.extend([format_number(row["raw_mean"], ".4f"), format_number(row["mean_r"], ".3f"),
                          f'{format_number(row["mean_accuracy"], ".1%")} ({row["states_with_accuracy"]})'])
        table.append("| " + " | ".join(cells) + " |")
    for name, rows in results.items():
        output = records_directory(run / name) / "analysis" / "driving_quality_plots"
        output.mkdir(parents=True, exist_ok=True)
        with (output / "pooled_target_outcomes.csv").open("w", newline="") as stream:
            writer = csv.DictWriter(stream, fieldnames=["metric", *next(iter(rows.values())).keys()])
            writer.writeheader()
            for key, row in rows.items():
                writer.writerow(dict(metric=key, **row))
    report = [
        "# Round-three target and outcome comparison", "",
        "Each checkpoint contributes 240 start states and 2,880 forced-action rollouts. Each policy supplies its own self-play states. Rows are pooled across the same 16 distance bins. The raw mean includes every rollout; mean Pearson r and pairwise accuracy weight each informative starting state equally. Parentheses give the number of states scored for ranking. The CSV files also report separate Pearson counts.", "",
        *table, "",
        "PPO uses its frozen, on-policy value-training target V(s) + GAE. The first forced action starts a 128-row learner segment. The baseline V(s) is shared by all 12 branches, so ranking by the target gives the same ordering as ranking by the sampled advantage. The target is computed after simulating each action and includes rewards also used to measure outcomes. Its agreement with return is a target/outcome comparison. It does not measure the ability to rank the 12 choices from the initial observation alone.", "",
        "PQL and TES use their respective saved SAC critic targets. The sampled soft-return row uses SAC’s explicit temperature. PPO’s entropy term enters the actor loss; PPO’s value target has no SAC temperature. Discounted reward return is available for all three checkpoints.", "",
        "Ranking favors smaller total infractions, signed penalties closer to zero, and larger values for the remaining outcomes. For speed it measures preference for faster driving. A negative raw infraction correlation means larger targets accompany fewer infractions.", "",
        "[PQL plots](pql/plots/driving_quality_plots/README.md) · [TES plots](tes/plots/driving_quality_plots/README.md) · [PPO plots](ppo/plots/driving_quality_plots/README.md) · [PPO replays](ppo/renders/README.md)",
    ]
    (run / "TARGET_COMPARISON.md").write_text("\n".join(report) + "\n")


if __name__ == "__main__":
    main()
