"""Resolve the isolated self-play configuration and reject incompatible resumes."""

import argparse
import math
import os
from pathlib import Path
import sys
from unittest.mock import patch

import yaml

from finetune_config import PROJECT, cli_values

REPO = PROJECT.parents[1]
CONFIG = PROJECT / "override_config/nuplan_selfplay.yaml"
WORLD_SIZE = 2
ROLLOUT_HORIZON = 128
EVALUATION_TRANSITIONS = 100_000_000
CHECKPOINT_TRANSITIONS = 25_000_000


def selfplay_overrides():
    return yaml.safe_load(CONFIG.read_text())


def resource_overrides(agents, workers, microbatch):
    """Keep inference batches at half the concurrent population on each rank."""
    global_rollout = WORLD_SIZE * agents * workers * ROLLOUT_HORIZON
    return {
        "env.num_agents": agents,
        "vec.num_envs": workers,
        "vec.num_workers": workers,
        "vec.batch_size": workers // 2,
        "train.max_minibatch_size": microbatch,
        "train.evaluation_interval_epochs": math.ceil(EVALUATION_TRANSITIONS / global_rollout),
        "train.checkpoint_interval": math.ceil(CHECKPOINT_TRANSITIONS / global_rollout),
    }


def resolve_config(overrides):
    from pufferlib.pufferl import load_config

    with patch.object(sys, "argv", ["selfplay_config", *cli_values(overrides)]):
        return load_config("puffer_drive")


def validate_config(resolved):
    env, vec, train = (resolved[section] for section in ("env", "vec", "train"))
    if resolved["rnn_name"] is not None or train["bptt_horizon"] != ROLLOUT_HORIZON:
        raise ValueError("This self-play baseline requires a feed-forward policy and horizon 128")
    map_dir = Path(env["map_dir"])
    if not map_dir.is_absolute():
        map_dir = REPO / map_dir
    with os.scandir(map_dir) as entries:
        map_count = sum(entry.name.endswith(".bin") and entry.is_file() for entry in entries)
    if map_count != env["num_maps"]:
        raise ValueError(f"Full training pool requires env.num_maps={map_count}; got {env['num_maps']}")
    if any(isinstance(value, bool) or not isinstance(value, int) or value <= 0 for value in (
        env["num_agents"], vec["num_envs"], vec["num_workers"], vec["batch_size"],
        train["bptt_horizon"], train["minibatch_size"], train["max_minibatch_size"],
    )):
        raise ValueError("Agent, worker, rollout and batch counts must be positive integers")
    if vec["num_envs"] != vec["num_workers"] or vec["num_envs"] != 2 * vec["batch_size"]:
        raise ValueError("Self-play requires one environment per worker and two inference batches")
    rollout = env["num_agents"] * vec["num_envs"] * train["bptt_horizon"]
    if rollout < train["minibatch_size"]:
        raise ValueError("Rollout must contain at least one logical minibatch")
    if train["minibatch_size"] % train["max_minibatch_size"] or train["max_minibatch_size"] % train["bptt_horizon"]:
        raise ValueError("Logical batch must divide into microbatches divisible by the horizon")
    if resolved["load_model_path"] is not None or resolved["load_id"] is not None:
        raise ValueError("Self-play starts from scratch; pretrained model loading is forbidden")
    expected_resources = resource_overrides(env["num_agents"], vec["num_workers"], train["max_minibatch_size"])
    for field in ("evaluation_interval_epochs", "checkpoint_interval"):
        if train[field] != expected_resources[f"train.{field}"]:
            raise ValueError(f"Update train.{field} to {expected_resources[f'train.{field}']} for this rollout size")


def validate_resume(run_dir, resolved):
    """Compare resolved defaults too, so later architecture/default changes fail early."""
    run_dir = Path(run_dir)
    saved = yaml.safe_load((run_dir / "config.yaml").read_text())
    for field in ("policy_name", "rnn_name", "load_model_path", "load_id", "run_name"):
        if saved.get(field) != resolved[field]:
            raise ValueError(f"Incompatible resume: {field}")
    for section in ("policy", "rnn", "env", "vec", "train", "eval"):
        for field, expected in resolved[section].items():
            if section == "train" and field == "resume_state_path":
                continue
            if section == "train" and field == "total_timesteps":
                expected //= WORLD_SIZE
            actual = saved.get(section, {}).get(field)
            if actual != expected:
                raise ValueError(f"Incompatible resume: {section}.{field}={actual!r}; expected {expected!r}")
    if Path(saved["train"]["data_dir"]).resolve() != run_dir.resolve():
        raise ValueError("Resume configuration belongs to another run directory")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seed", type=int, choices=(0, 1, 2), required=True)
    parser.add_argument("--run-dir", type=Path, required=True)
    options = parser.parse_args()
    run_dir = options.run_dir.resolve()
    overrides = selfplay_overrides()
    overrides.update({"train.seed": options.seed, "train.data_dir": str(run_dir), "run_name": run_dir.name})
    resolved = resolve_config(overrides)
    validate_config(resolved)
    if (run_dir / "trainer_state.pt").is_file():
        validate_resume(run_dir, resolved)
        overrides["train.resume_state_path"] = str(run_dir / "trainer_state.pt")
    elif run_dir.exists() and any(run_dir.iterdir()):
        raise ValueError(f"Existing nonempty run has no trainer_state.pt: {run_dir}; use a new RUN_NAME")
    print("\n".join(cli_values(overrides)))
