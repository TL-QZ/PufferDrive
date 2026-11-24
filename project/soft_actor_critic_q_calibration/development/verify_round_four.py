"""Audit round-four branch identities and prefix measurements against their trajectories."""
import argparse
import collections
import hashlib
import json
from pathlib import Path

import numpy as np

from .run_layout import model_directory

HORIZONS = (8, 32, 128, 1280, 2560)


def check_state(path, selected, signature, audit):
    gamma, alpha = audit["gamma"], audit.get("alpha")
    record = json.loads(path.read_text())
    if (record["state_id"] != selected["state_id"] or record["assigned_bin"] != selected["assigned_bin"]
            or record["run_signature"] != signature or record["status"] != "complete"):
        raise ValueError("record identity or manifest differs")
    branches = record["branches"]
    ids = [(b["action_idx"], b["repeat_idx"]) for b in branches]
    if sorted(ids) != [(action, repeat) for action in range(12) for repeat in range(4)]:
        raise ValueError("expected four branches for each of twelve actions")
    seeds = {repeat: {b["continuation_seed"] for b in branches if b["repeat_idx"] == repeat}
             for repeat in range(4)}
    if any(len(s) != 1 for s in seeds.values()) or len(set.union(*seeds.values())) != 4:
        raise ValueError("continuation seeds differ from the paired-repeat design")

    data_path = path.with_name(record["data_file"])
    with data_path.open("rb") as stream:
        if hashlib.sha256(stream.read()).hexdigest() != record["data_sha256"]:
            raise ValueError("NPZ hash differs from JSON")
    with np.load(data_path, allow_pickle=False) as arrays:
        array_ids = list(zip(arrays["action_idx"].tolist(), arrays["repeat_idx"].tolist()))
        if array_ids != ids:
            raise ValueError("JSON and NPZ action and repeat columns differ")
        valid = arrays["valid"].astype(bool)
        rewards = arrays["rewards"]
        log_probabilities = arrays["log_probs"]
        goals = arrays["goal_events"]
        if alpha is None:
            import torch
            import pufferlib._C  # Register the trainer's native GAE kernel.
            inputs = [torch.tensor(arrays[k].T.copy(), dtype=torch.float32)
                      for k in ("ppo_values", "ppo_training_rewards", "ppo_training_dones", "ppo_importance")]
            advantages = torch.zeros_like(inputs[0])
            torch.ops.pufferlib.compute_puff_advantage(*inputs, advantages, gamma, audit["gae_lambda"],
                                                        audit["vtrace_rho_clip"], audit["vtrace_c_clip"])
            targets = (inputs[0] + advantages)[:, 0].numpy()
            target_key = "ppo_value_target"
        else:
            from .probe_returns import compute_probe_measurements
            target_key = "actual_critic_target"
            targets = compute_probe_measurements(
                rewards, log_probabilities, valid, arrays["terminals"], arrays["truncations"],
                arrays["next_soft_values"], gamma, alpha, audit["return_estimator"],
                audit["pql_horizon"], audit["pql_lambda"],
            )[target_key]
        np.testing.assert_allclose(targets, [b[target_key] for b in branches], atol=1e-6, rtol=1e-6)
    coverage = collections.Counter()
    endings = collections.Counter()
    for column, branch in enumerate(branches):
        end = branch["observed_horizon"]
        if end < 1 or end > 2560 or not np.array_equal(valid[:, column], np.arange(len(valid)) < end):
            raise ValueError(f"noncontiguous or incorrect horizon in branch {column}")
        expected = {str(h) for h in HORIZONS if h <= end}
        if set(branch["horizon_metrics"]) != expected:
            raise ValueError(f"missing or extra exact horizon in branch {column}")
        discounted = np.power(gamma, np.arange(end)) * rewards[:end, column]
        bonus = np.zeros(end)
        if alpha is not None:
            bonus[1:] = -alpha * np.power(gamma, np.arange(1, end)) * log_probabilities[1:end, column]
        measurements = [(end, branch)] + [(int(h), v) for h, v in branch["horizon_metrics"].items()]
        full_metrics = branch["metrics_mean"]
        if len(full_metrics) != 40:
            raise ValueError(f"unexpected metric schema in branch {column}")
        for length, point in measurements:
            metrics = point["metrics_mean"]
            if metrics.keys() != full_metrics.keys() or not np.isfinite(list(metrics.values())).all():
                raise ValueError(f"incomplete metrics at {length} in branch {column}")
            np.testing.assert_allclose(point["discounted_reward_return"], discounted[:length].sum(), atol=2e-6, rtol=2e-6)
            if alpha is not None:
                np.testing.assert_allclose(point["standard_soft_return"], (discounted[:length] + bonus[:length]).sum(), atol=2e-6, rtol=2e-6)
            np.testing.assert_allclose(metrics["episode_length"], length)
            np.testing.assert_allclose(metrics["num_goals_reached"], goals[:length, column].sum())
            # Native interval logs subtract cumulative float32 counters.
            np.testing.assert_allclose(metrics["episode_return"], rewards[:length, column].sum(dtype=np.float64), atol=1e-3, rtol=1e-4)
            if length == end:
                np.testing.assert_allclose([metrics[k] for k in full_metrics], list(full_metrics.values()))
        coverage.update(expected)
        endings[branch["ending_reason"]] += 1
    return coverage, endings


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("model_dir", type=Path)
    parser.add_argument("--one", action="store_true", help="Check one saved state during the pilot")
    args = parser.parse_args()
    records = model_directory(args.model_dir) / "records"
    manifest = json.loads((records / "manifest.json").read_text())
    audit = manifest["checkpoint"]["source_audit"]
    states = {s["state_id"]: s for s in json.loads((records / "selection.json").read_text())["states"]}
    files = sorted((records / "probes").glob("*.json"))
    if not files or (not args.one and len(files) != len(states)):
        raise ValueError(f"Found {len(files)} of {len(states)} expected states")
    coverage, endings = collections.Counter(), collections.Counter()
    for path in files[:1] if args.one else files:
        selected = states[path.stem.replace("__", "/")]
        try:
            prefixes, reasons = check_state(path, selected, manifest["signature"], audit)
        except (AssertionError, ValueError, KeyError) as exc:
            raise ValueError(f"Verification failed for {path}: {exc}") from exc
        coverage.update(prefixes)
        endings.update(reasons)
    report = dict(state_count=sum(endings.values()) // 48, branch_count=sum(endings.values()),
                  exact_horizon_counts={h: coverage[str(h)] for h in HORIZONS}, ending_reasons=dict(endings))
    print(json.dumps(report, indent=2))
    if not args.one:
        (records / "verification.json").write_text(json.dumps(report, indent=2) + "\n")


if __name__ == "__main__":
    main()
