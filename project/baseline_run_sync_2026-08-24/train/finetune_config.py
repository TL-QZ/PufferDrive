"""Experiment-local resource overrides and trainer-state compatibility checks."""

import argparse
import json
from pathlib import Path

import yaml

PROJECT = Path(__file__).resolve().parents[1]
RESOURCE_KEYS = {"env.num_agents", "train.max_minibatch_size",
                 "train.evaluation_interval_epochs", "train.checkpoint_interval"}


def finetune_overrides(resources=None):
    config = yaml.safe_load((PROJECT / "override_config/nuplan_sdc_finetune.yaml").read_text())
    if resources is not None:
        selected = yaml.safe_load(Path(resources).read_text())
        if not isinstance(selected, dict) or set(selected) != RESOURCE_KEYS:
            raise ValueError(f"Resource file must contain exactly {sorted(RESOURCE_KEYS)}")
        config.update(selected)
    concurrency = config["env.num_agents"]
    microbatch = config["train.max_minibatch_size"]
    if concurrency not in (16, 32, 64) or microbatch not in (16000, 32000, 64000):
        raise ValueError("Unsupported concurrency or microbatch")
    if (config["train.evaluation_interval_epochs"], config["train.checkpoint_interval"]) != (
        640 // concurrency, 128 // concurrency
    ):
        raise ValueError("Resource cadence must preserve the 25.6M / 5.12M global-transition intervals")
    return config


def validate_resume(run_dir, overrides):
    run_dir = Path(run_dir)
    saved = yaml.safe_load((run_dir / "config.yaml").read_text())
    if saved.get("rnn_name") is not None:
        raise ValueError("This fine-tune requires rnn_name: null")
    # Saved train.total_timesteps is per rank after load_config's DDP division.
    for key, expected in overrides.items():
        if not key.startswith(("env.", "vec.", "train.")):
            continue
        section, field = key.split(".", 1)
        if key == "train.total_timesteps":
            expected //= 2
        actual = saved.get(section, {}).get(field)
        if actual != expected:
            raise ValueError(f"Incompatible resume {run_dir}: {key}={actual!r}; expected {expected!r}")
    if Path(saved["train"]["data_dir"]).resolve() != run_dir.resolve():
        raise ValueError("Resume configuration belongs to another run directory")


def cli_values(config):
    for key, value in config.items():
        if isinstance(value, bool):
            value = str(value).lower()
        elif value is None:
            value = "null"
        elif isinstance(value, str) and "," in value:
            value = json.dumps(value)
        yield f"{key}={value}"


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--resources", type=Path)
    parser.add_argument("--resume-run", type=Path)
    args = parser.parse_args()
    config = finetune_overrides(args.resources)
    if args.resume_run is not None:
        validate_resume(args.resume_run, config)
    print("\n".join(cli_values(config)))
