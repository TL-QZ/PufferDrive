"""Write reviewable resolved inputs without creating a simulator or training."""

import argparse
import json
from pathlib import Path

from omegaconf import OmegaConf

from project.soft_actor_critic_warm_start.configuration import (
    memory_estimate,
    resolve_config,
    smoke_config,
)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--config", help="Local warm-start YAML (defaults to config/warm_start.yaml)")
    parser.add_argument("--seed", type=int, help="Override the seed stored in the YAML")
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--no-wandb", action="store_true")
    args = parser.parse_args()
    if args.seed is not None and args.seed < 0:
        parser.error("--seed must be nonnegative")
    config = resolve_config(args.run_dir, seed=args.seed, config_path=args.config)
    if args.smoke:
        config = smoke_config(config)
    if args.no_wandb:
        for phase in ("ppo", "sac"):
            config[phase].update(wandb=False, neptune=False, tb=False)
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("x") as stream:
        stream.write(OmegaConf.to_yaml(OmegaConf.create(config)))
    print(json.dumps({"config": str(output), **memory_estimate(config, 1 if args.smoke else 2)}, indent=2))


if __name__ == "__main__":
    main()
