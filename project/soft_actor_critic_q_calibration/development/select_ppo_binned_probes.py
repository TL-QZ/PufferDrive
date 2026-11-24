"""Select 240 PPO rollout states across the retained calibration bins."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from . import select_binned_probes as selector
from .run_layout import initialize_model_directory, model_directory, records_directory


DEFAULT_SOURCE = Path("project/soft_actor_critic_q_calibration/runs/second_20261122/ppo")
DEFAULT_OUTPUT = Path("project/soft_actor_critic_q_calibration/runs/third_20261122/ppo")
GOAL_AXIS = "nearest_visible_goal_distance_m"
EXCLUDED_GOAL_BINS = (0, 1, 2)
ALGORITHM = "balanced-map-episode-agent-no-adjacent-frame-exclude-goal-under-12m-v1"


def build_argument_parser():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-model-dir", type=Path, default=DEFAULT_SOURCE)
    parser.add_argument("--model-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--seed", type=int, default=selector.DEFAULT_SEED)
    parser.add_argument("--per-bin", type=int, default=selector.REQUESTED_PER_BIN)
    return parser


def _canonical_sha256(value):
    return selector._canonical_sha256(value)


def build_ppo_selection(source_records_dir: Path, *, seed: int, per_bin: int):
    if seed < 0:
        raise ValueError("seed must be non-negative")
    if per_bin <= 0:
        raise ValueError("per-bin must be positive")

    pool = selector._load_candidate_pool(source_records_dir)
    original_goal_bins = pool["bin_indices"][GOAL_AXIS]
    pool["bin_indices"] = dict(pool["bin_indices"])
    retained_goal_bins = original_goal_bins.copy()
    retained_goal_bins[np.isin(retained_goal_bins, EXCLUDED_GOAL_BINS)] = -1
    pool["bin_indices"][GOAL_AXIS] = retained_goal_bins

    source_records_relative = source_records_dir.resolve().relative_to(
        selector.PROJECT_DIR.resolve()
    ).as_posix()
    selection = selector.build_selection(
        pool,
        model="ppo",
        source_records_dir=source_records_relative,
        seed=seed,
        per_bin=per_bin,
    )
    edges = selector.PRIMARY_AXES[GOAL_AXIS]
    excluded = []
    for bin_index in EXCLUDED_GOAL_BINS:
        tag = selector._interval(GOAL_AXIS, bin_index)
        excluded.append({
            **tag,
            "candidate_count": int(np.count_nonzero(original_goal_bins == bin_index)),
            "selected_count": 0,
            "reason": "visible goal distance below 12 m excluded from goal-assigned quota",
        })
    selection["bin_summary"][GOAL_AXIS] = [
        record for record in selection["bin_summary"][GOAL_AXIS]
        if record["bin_index"] not in EXCLUDED_GOAL_BINS
    ]
    selection["shortages"] = [
        shortage for shortage in selection["shortages"]
        if not (shortage["axis"] == GOAL_AXIS and shortage["bin_index"] in EXCLUDED_GOAL_BINS)
    ]
    retained_bin_count = sum(len(axis_edges) - 1 for axis_edges in selector.PRIMARY_AXES.values()) - len(
        EXCLUDED_GOAL_BINS
    )
    selection["requested_state_count"] = per_bin * retained_bin_count
    selection["selected_state_count"] = len(selection["states"])
    selection["algorithm"] = ALGORITHM
    selection["excluded_assigned_bins"] = excluded
    selection["selection_filter"] = {
        "axis": GOAL_AXIS,
        "excluded_bin_indices": list(EXCLUDED_GOAL_BINS),
        "excluded_upper_bound_m": float(edges[len(EXCLUDED_GOAL_BINS)]),
        "meaning": "only goal-assigned bins 03 through 06 are selected; states may still be selected under agent or road-boundary bins",
    }
    signature_inputs = {
        "base_selection_signature": selection["signature"],
        "algorithm": ALGORITHM,
        "excluded_assigned_bins": excluded,
        "requested_per_bin": per_bin,
        "source_geometry_manifest_signature": pool["source_geometry_manifest_signature"],
    }
    selection["signature"] = _canonical_sha256(signature_inputs)
    return selection


def run_selection(args):
    source_model_dir = model_directory(args.source_model_dir)
    output_model_dir = model_directory(args.model_dir)
    if source_model_dir == output_model_dir:
        raise ValueError("PPO source geometry and selected probes must use separate model directories")
    if source_model_dir.name != "ppo" or output_model_dir.name != "ppo":
        raise ValueError("PPO geometry selection paths must both end in /ppo")
    source_records_dir = records_directory(source_model_dir)
    selection = build_ppo_selection(source_records_dir, seed=args.seed, per_bin=args.per_bin)
    if selection["selected_state_count"] != selection["requested_state_count"]:
        raise RuntimeError(
            f"PPO geometry selection found {selection['selected_state_count']} of "
            f"{selection['requested_state_count']} requested states"
        )
    if selection["shortages"]:
        raise RuntimeError(f"PPO geometry selection has bin shortages: {selection['shortages']}")
    initialize_model_directory(output_model_dir)
    output_path = records_directory(output_model_dir) / "selection.json"
    selector._write_selection(output_path, selection)
    print(json.dumps({
        "model": "ppo",
        "selection": str(output_path),
        "selected_state_count": selection["selected_state_count"],
        "requested_state_count": selection["requested_state_count"],
        "bin_counts": {
            axis: len(bins) for axis, bins in selection["bin_summary"].items()
        },
        "signature": selection["signature"],
    }, sort_keys=True))
    return selection


def main(argv=None):
    selection = run_selection(build_argument_parser().parse_args(argv))
    if selection["selected_state_count"] != 240:
        raise RuntimeError("Expected exactly 240 states: 16 retained bins x 15 states")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
