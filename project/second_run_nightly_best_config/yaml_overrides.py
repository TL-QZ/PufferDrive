"""Print a flat YAML mapping as Hydra command-line overrides."""

import sys

import yaml


def hydra_value(value):
    if isinstance(value, bool):
        return str(value).lower()
    if value is None:
        return "null"
    if isinstance(value, (dict, list)):
        raise TypeError("Only scalar values are supported")
    return str(value)


config_path = sys.argv[1]
with open(config_path, encoding="utf-8") as config_file:
    config = yaml.safe_load(config_file)

if not isinstance(config, dict):
    raise TypeError(f"Expected a YAML mapping in {config_path}")

for key, value in config.items():
    print(f"{key}={hydra_value(value)}")
