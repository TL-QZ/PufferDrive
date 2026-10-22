"""Build and validate dedicated PPO-behavior SAC trial configurations."""

import argparse
from copy import deepcopy
import hashlib
import itertools
import json
from pathlib import Path
import sys

import yaml


EXPERIMENT_DIR = Path(__file__).resolve().parent
REPO_ROOT = EXPERIMENT_DIR.parents[1]
CONFIG_DIR = EXPERIMENT_DIR / "config"
SOURCE_CONFIG_DIR = REPO_ROOT / "project/soft_actor_critic_tuning/configs/round4_1"
TRIAL_IDS = (97, 98)
TOTAL_STEPS = 1_000_000_000
WANDB_GROUP = "sac_ppo_init_behavior_round4_1"
BEHAVIOR_POLICY_SETTINGS = {
    "checkpoint": "experiments/ppo_sac_seed0_20261019_222638/ppo/best_models/best_trainer_state_000123.pt",
    "teacher_config": "experiments/ppo_sac_seed0_20261019_222638/ppo/config.yaml",
    "ppo_only_steps": 250_000_000,
    "decay_steps": 250_000_000,
    "total_steps": TOTAL_STEPS,
    "seed": 0,
    "source": "project/soft_actor_critic_tuning/configs/round4_1",
}
EXPECTED_TRIAL_SETTINGS = {
    97: {"replay_update_ratio": 0.1, "evaluation_interval_epochs": 10},
    98: {"replay_update_ratio": 0.5, "evaluation_interval_epochs": 2},
}

if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from project.soft_actor_critic_tuning.scripts import run_sac16_grid


def _require_trial_id(trial_id):
    if isinstance(trial_id, bool) or not isinstance(trial_id, int) or trial_id not in TRIAL_IDS:
        raise ValueError(f"trial must be one of {', '.join(map(str, TRIAL_IDS))}")
    return trial_id


def _build_trial_config(trial_id):
    trial_id = _require_trial_id(trial_id)
    _, base_config, launcher = run_sac16_grid._load_configs(SOURCE_CONFIG_DIR)
    axes = run_sac16_grid._grid_axes(base_config)
    combinations = list(itertools.product(*(values for _, values in axes)))
    if trial_id > len(combinations):
        raise ValueError(f"Source grid has no trial {trial_id}")

    resolved = deepcopy(base_config)
    for (path, _), value in zip(axes, combinations[trial_id - 1]):
        run_sac16_grid._set_dotted(resolved, path, value)

    evaluation_intervals = launcher.get("trial_evaluation_interval_epochs")
    if not isinstance(evaluation_intervals, dict) or trial_id not in evaluation_intervals:
        raise ValueError(f"Source launcher has no evaluation cadence for trial {trial_id}")
    resolved["train"]["evaluation_interval_epochs"] = evaluation_intervals[trial_id]

    # The source schema defaults this field to true; make the intended native
    # automatic-temperature behavior explicit in the immutable snapshot.
    resolved["train"]["learn_alpha"] = True
    resolved["train"]["total_timesteps"] = TOTAL_STEPS
    resolved["wandb_group"] = WANDB_GROUP
    resolved["behavior_policy"] = {
        **deepcopy(BEHAVIOR_POLICY_SETTINGS),
        "source_trial": trial_id,
    }
    _validate_snapshot(resolved, trial_id)
    return resolved


def _validate_snapshot(config, trial_id):
    trial_id = _require_trial_id(trial_id)
    if not isinstance(config, dict):
        raise ValueError("Trial config must be a YAML mapping")
    train = config.get("train")
    behavior = config.get("behavior_policy")
    if not isinstance(train, dict) or not isinstance(behavior, dict):
        raise ValueError("Trial config must contain train and behavior_policy mappings")
    expected = EXPECTED_TRIAL_SETTINGS[trial_id]
    if train.get("replay_update_ratio") != expected["replay_update_ratio"]:
        raise ValueError(f"Trial {trial_id} has the wrong replay refresh ratio")
    if train.get("evaluation_interval_epochs") != expected["evaluation_interval_epochs"]:
        raise ValueError(f"Trial {trial_id} has the wrong evaluation interval")
    if train.get("total_timesteps") != TOTAL_STEPS:
        raise ValueError("train.total_timesteps must be 1000000000 received rows")
    if train.get("learn_alpha") is not True:
        raise ValueError("train.learn_alpha must remain true")
    if config.get("wandb_group") != WANDB_GROUP:
        raise ValueError(f"wandb_group must be {WANDB_GROUP}")
    expected_behavior = {
        **BEHAVIOR_POLICY_SETTINGS,
        "source_trial": trial_id,
    }
    for key, value in expected_behavior.items():
        if key in ("checkpoint", "teacher_config"):
            source_path = behavior.get(key)
            if not isinstance(source_path, str) or not source_path.strip():
                raise ValueError(f"behavior_policy.{key} must be a non-empty path")
            continue
        if behavior.get(key) != value:
            raise ValueError(f"behavior_policy.{key} must equal {value!r}")
    for key in ("load_model_path", "load_id"):
        if config.get(key) is not None:
            raise ValueError(f"{key} must stay null; PPO is a separate frozen behavior policy")
    if train.get("resume_state_path") is not None:
        raise ValueError("train.resume_state_path must stay null")


def _load_trial_snapshot(trial_id):
    trial_id = _require_trial_id(trial_id)
    path = CONFIG_DIR / f"trial_{trial_id}.yaml"
    try:
        with path.open(encoding="utf-8") as source:
            config = yaml.safe_load(source)
    except FileNotFoundError as error:
        raise FileNotFoundError(
            f"Missing config snapshot {path}; generate snapshots with "
            "python -m project.soft_actor_critic_PPO_init_behavior_policy.configuration generate"
        ) from error
    _validate_snapshot(config, trial_id)
    return config


def _resolve_source_artifacts(config):
    behavior = config["behavior_policy"]
    for key in ("checkpoint", "teacher_config"):
        source_path = Path(behavior[key])
        if not source_path.is_absolute():
            source_path = REPO_ROOT / source_path
        source_path = source_path.resolve()
        if not source_path.is_file():
            raise FileNotFoundError(f"Teacher {key} not found: {source_path}")
        behavior[key] = str(source_path)


def _write_yaml(path, config, *, exclusive=False):
    path.parent.mkdir(parents=True, exist_ok=True)
    mode = "x" if exclusive else "w"
    with path.open(mode, encoding="utf-8") as output:
        yaml.safe_dump(config, output, sort_keys=False, allow_unicode=True)


def generate_trial_snapshots():
    CONFIG_DIR.mkdir(parents=True, exist_ok=True)
    paths = []
    for trial_id in TRIAL_IDS:
        path = CONFIG_DIR / f"trial_{trial_id}.yaml"
        temporary_path = path.with_name(path.name + ".tmp")
        with temporary_path.open("w", encoding="utf-8") as output:
            yaml.safe_dump(_build_trial_config(trial_id), output, sort_keys=False, allow_unicode=True)
        temporary_path.replace(path)
        paths.append(path)
    return paths


def prepare_run(trial_id, run_dir, *, no_wandb=False):
    trial_id = _require_trial_id(trial_id)
    run_dir = Path(run_dir).expanduser()
    if not run_dir.is_absolute():
        run_dir = REPO_ROOT / run_dir
    run_dir = run_dir.resolve()
    if run_dir.exists():
        raise FileExistsError(f"Refusing existing run directory: {run_dir}")

    config = deepcopy(_load_trial_snapshot(trial_id))
    _resolve_source_artifacts(config)
    run_path_id = hashlib.sha256(str(run_dir).encode("utf-8")).hexdigest()[:8]
    config["run_name"] = f"{run_dir.name}_trial_{trial_id}_{run_path_id}"
    config["train"]["data_dir"] = str((run_dir / "learner").resolve())
    config["train"]["fresh_run"] = True
    if no_wandb:
        config.update(wandb=False, neptune=False, tb=False, no_model_upload=True)

    # Use the trainer's own checks before any run artifacts are created.
    from pufferlib.pufferl_sac_ppo_init_behavior_policy import validate_config

    validate_config(config)
    run_dir.parent.mkdir(parents=True, exist_ok=True)
    run_dir.mkdir(exist_ok=False)
    config_path = run_dir / "config.yaml"
    _write_yaml(config_path, config, exclusive=True)
    return config_path, config


def _build_parser():
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)
    subparsers.add_parser("generate", help="Regenerate the two trial snapshots from round4_1")

    validate_parser = subparsers.add_parser("validate", help="Validate snapshots and teacher files")
    validate_parser.add_argument("--trial", type=int, choices=TRIAL_IDS, action="append")

    prepare_parser = subparsers.add_parser("prepare", help="Write a fresh run config from one snapshot")
    prepare_parser.add_argument("--trial", required=True, type=int, choices=TRIAL_IDS)
    prepare_parser.add_argument("--run-dir", required=True)
    prepare_parser.add_argument("--no-wandb", action="store_true")
    return parser


def main(argv=None):
    args = _build_parser().parse_args(argv)
    if args.command == "generate":
        paths = generate_trial_snapshots()
        print(json.dumps({"configs": [str(path.resolve()) for path in paths]}, indent=2))
        return 0
    if args.command == "validate":
        trial_ids = args.trial or list(TRIAL_IDS)
        for trial_id in trial_ids:
            config = _load_trial_snapshot(trial_id)
            _resolve_source_artifacts(config)
        print(json.dumps({"valid_trials": trial_ids}, indent=2))
        return 0
    config_path, config = prepare_run(args.trial, args.run_dir, no_wandb=args.no_wandb)
    print(json.dumps({"config": str(config_path), "run_name": config["run_name"]}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
