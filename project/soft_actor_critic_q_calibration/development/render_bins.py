"""Render one saved 12-action probe for every assigned distance bin."""
import argparse
import json
from collections import defaultdict
from pathlib import Path
from statistics import median

from .run_layout import model_directory

AXIS_DIRS = {
    "nearest_observed_partner_center_distance_m": "nearest_agent",
    "nearest_observed_road_boundary_distance_m": "nearest_road_boundary",
    "nearest_visible_goal_distance_m": "nearest_visible_goal",
}


def choose_examples(model_dir):
    records = model_dir / "records"
    states = json.loads((records / "selection.json").read_text())["states"]
    groups = defaultdict(list)
    for state in states:
        groups[state["assigned_bin"]["bin_id"]].append(state)

    for bin_id, candidates in sorted(groups.items()):
        bounds = candidates[0]["assigned_bin"]
        axis = bounds["axis"]
        middle = median(s["distances_m"][axis] for s in candidates)
        # Prefer compact scenes for browser playback. Avoid a single-frame
        # example when the bin offers at least ten frames of visible driving.
        usable = []
        for state in candidates:
            stem = state["state_id"].replace("/", "__")
            metadata = json.loads((records / "probes" / f"{stem}.json").read_text())
            duration = max(branch["observed_horizon"] for branch in metadata["branches"])
            usable.append((duration >= 10, state))
        moving = [state for long_enough, state in usable if long_enough] or candidates
        smallest = min(state["agent_count"] for state in moving)
        small_scenes = [state for state in moving if state["agent_count"] <= 1.5 * smallest]
        example = min(small_scenes, key=lambda s: (abs(s["distances_m"][axis] - middle), s["state_id"]))
        low, high = bounds["lower_m"], bounds["upper_m"]
        interval = f"{low:g}-{high:g}m" if high is not None else f"{low:g}m-plus"
        name = example["state_id"].replace("/", "__") + ".html"
        yield bin_id, example, model_dir / "renders" / "binnings" / AXIS_DIRS[axis] / interval / name


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("model_dir", type=Path)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--worker-index", type=int, default=0)
    parser.add_argument("--workers", type=int, default=1)
    args = parser.parse_args()
    if not 0 <= args.worker_index < args.workers:
        parser.error("expected 0 <= worker-index < workers")
    from .render_capture import capture
    from .rollout_viewer import write_viewer

    model_dir = model_directory(args.model_dir)
    for idx, (bin_id, example, path) in enumerate(choose_examples(model_dir)):
        if idx % args.workers != args.worker_index or path.is_file():
            continue
        print(bin_id, example["state_id"], flush=True)
        payload = capture(model_dir, example["state_id"], 0, device=args.device)
        staging = path.with_suffix(".partial")
        write_viewer(payload, staging)
        staging.replace(path)
        print(path, path.stat().st_size, flush=True)


if __name__ == "__main__":
    main()
