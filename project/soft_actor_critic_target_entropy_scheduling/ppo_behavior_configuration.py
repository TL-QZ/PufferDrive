"""Resolve the mirrored Round 2 grid for frozen PPO-only collection."""

import argparse
from copy import deepcopy
import hashlib
import itertools
import json
from pathlib import Path

import yaml

from project.soft_actor_critic_tuning.scripts import run_sac16_grid


EXPERIMENT_DIR = Path(__file__).resolve().parent
REPO_ROOT = EXPERIMENT_DIR.parents[1]
CONFIG_DIR = EXPERIMENT_DIR / "configs/round2_tes_sac_ppo_behavior"
TRIAL_IDS = (97, 98)


def build_trial_config(trial_id):
    if isinstance(trial_id, bool) or trial_id not in TRIAL_IDS:
        raise ValueError("trial must be 97 or 98")
    _, base_config, launcher = run_sac16_grid._load_configs(CONFIG_DIR)
    axes = run_sac16_grid._grid_axes(base_config)
    combinations = list(itertools.product(*(values for _, values in axes)))
    config = deepcopy(base_config)
    for (path, _), value in zip(axes, combinations[trial_id - 1]):
        run_sac16_grid._set_dotted(config, path, value)
    train = config["train"]
    train["evaluation_interval_epochs"] = launcher["trial_evaluation_interval_epochs"][trial_id]
    train["total_timesteps"] = launcher["trial_total_timesteps"][trial_id]
    config["behavior_policy"]["total_steps"] = train["total_timesteps"]
    for key in ("checkpoint", "teacher_config"):
        config["behavior_policy"][key] = str((REPO_ROOT / config["behavior_policy"][key]).resolve())
    from pufferlib.pufferl_sac_ppo_init_behavior_policy import validate_config

    validate_config(config)
    return config


def prepare_run(trial_id, run_dir, *, no_wandb=False):
    config = build_trial_config(trial_id)
    run_dir = Path(run_dir).expanduser()
    if not run_dir.is_absolute():
        run_dir = REPO_ROOT / run_dir
    run_dir = run_dir.resolve()
    run_path_id = hashlib.sha256(str(run_dir).encode()).hexdigest()[:8]
    config["run_name"] = f"{run_dir.name}_trial_{trial_id}_{run_path_id}"
    config["train"]["data_dir"] = str(run_dir / "learner")
    if no_wandb:
        config.update(wandb=False, neptune=False, tb=False, no_model_upload=True)
    run_dir.mkdir(parents=True, exist_ok=False)
    config_path = run_dir / "config.yaml"
    with config_path.open("x") as output:
        yaml.safe_dump(config, output, sort_keys=False)
    return config_path, config


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("validate", "prepare"))
    parser.add_argument("--trial", type=int, choices=TRIAL_IDS, required=True)
    parser.add_argument("--run-dir")
    parser.add_argument("--no-wandb", action="store_true")
    args = parser.parse_args()
    if args.command == "validate":
        build_trial_config(args.trial)
        print(json.dumps({"valid_trial": args.trial}))
        return
    if not args.run_dir:
        parser.error("prepare requires --run-dir")
    path, config = prepare_run(args.trial, args.run_dir, no_wandb=args.no_wandb)
    print(json.dumps({"config": str(path), "run_name": config["run_name"]}, indent=2))


if __name__ == "__main__":
    main()
