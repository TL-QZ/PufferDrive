"""Model and config logging helpers for the discrete SAC learner."""

import os
from pathlib import Path
import shutil
import signal
import tempfile

import yaml


SAC_MODEL_FIELD = "sac/model"
SAC_CONFIG_FIELD = "sac/config"


# pufferl installs an exit-on-SIGINT handler at import time. Keep the importing
# process's handler intact while reusing its Neptune logger implementation.
_previous_sigint_handler = signal.getsignal(signal.SIGINT)
try:
    from pufferlib import pufferl as _pufferl
finally:
    if signal.getsignal(signal.SIGINT) != _previous_sigint_handler:
        signal.signal(signal.SIGINT, _previous_sigint_handler)


class SACNeptuneLogger(_pufferl.NeptuneLogger):
    """Neptune logger that stores SAC actor and config as portable File fields."""

    def upload_model(self, model_path):
        model_path = os.fspath(model_path)
        if not os.path.isfile(model_path):
            raise ValueError(f"SAC actor checkpoint not found: {model_path}")

        config_path = os.path.join(
            _pufferl.drive_benchmark.resolve_run_dir(model_path), "config.yaml",
        )
        if not os.path.isfile(config_path):
            raise ValueError(
                f"SAC checkpoint config not found: {config_path}. "
                "Place config.yaml beside the run checkpoint before uploading the actor."
            )

        self.neptune[SAC_MODEL_FIELD].upload(model_path)
        self.neptune[SAC_CONFIG_FIELD].upload(config_path)


def _source_model_name(config):
    if "train" not in config:
        return None
    train_config = config["train"]
    if not isinstance(train_config, dict):
        raise ValueError("Source config train must be a mapping")

    model_name = train_config.get("final_model_name")
    if model_name is None:
        return None
    if not isinstance(model_name, str) or not model_name.strip():
        raise ValueError("Source config train.final_model_name must be a non-empty path")

    relative_path = Path(model_name)
    if relative_path.is_absolute() or ".." in relative_path.parts:
        raise ValueError("Source config train.final_model_name must stay inside the downloaded artifact")
    return relative_path


def _select_model(download_directory, config):
    download_directory = Path(download_directory)
    model_name = _source_model_name(config)
    if model_name is not None:
        candidates = (download_directory / model_name, download_directory / model_name.name)
        for candidate in candidates:
            if candidate.is_file():
                return candidate
        raise ValueError(
            "The downloaded model artifact does not contain the file named by "
            f"source config train.final_model_name ({model_name})."
        )

    model_paths = sorted(path for path in download_directory.rglob("*.pt") if path.is_file())
    if len(model_paths) != 1:
        names = ", ".join(str(path.relative_to(download_directory)) for path in model_paths)
        raise ValueError(
            "Source config has no train.final_model_name and the downloaded artifact must contain "
            f"exactly one .pt model; found {len(model_paths)}: {names or 'none'}."
        )
    return model_paths[0]


def _read_source_config(config_path):
    try:
        with open(config_path, "r") as config_file:
            config = yaml.safe_load(config_file)
    except yaml.YAMLError as exc:
        raise ValueError(f"Downloaded source config is invalid YAML: {config_path}") from exc
    if not isinstance(config, dict):
        raise ValueError(f"Downloaded source config must contain a mapping: {config_path}")
    return config


def download_model(args):
    """Download a remote SAC actor and its source config into this run's artifacts directory.

    ``args`` may be a full Hydra config (``train.data_dir``) or the flat training
    config (``data_dir``). The returned actor path has config.yaml beside it.
    """
    load_id = args.get("load_id")
    if not isinstance(load_id, str) or not load_id.strip():
        raise ValueError("Remote SAC model download requires a non-empty load_id")

    train_args = args.get("train", args)
    data_dir = train_args.get("data_dir") if isinstance(train_args, dict) else None
    if not isinstance(data_dir, (str, os.PathLike)) or not os.fspath(data_dir):
        raise ValueError("Remote SAC model download requires data_dir or train.data_dir")
    artifact_root = Path(data_dir) / "artifacts"
    artifact_root.mkdir(parents=True, exist_ok=True)
    download_directory = Path(tempfile.mkdtemp(prefix="remote-model-", dir=str(artifact_root)))
    config_path = download_directory / "config.yaml"

    providers = [provider for provider in ("wandb", "neptune") if args.get(provider)]
    if len(providers) != 1:
        raise ValueError("Remote SAC model download requires exactly one of wandb or neptune")

    if providers[0] == "wandb":
        import wandb

        project = args.get("wandb_project")
        if not isinstance(project, str) or not project.strip():
            raise ValueError("Remote W&B model download requires wandb_project")
        api = wandb.Api()
        artifact = api.artifact(f"{project}/{load_id}:latest", type="model")
        artifact_directory = Path(artifact.download(root=str(download_directory)))
        config_path = artifact_directory / "config.yaml"
        if not config_path.is_file():
            try:
                source_config = api.run(f"{project}/{load_id}").config
            except Exception as exc:
                raise ValueError(
                    "W&B model artifact has no config.yaml and the source run config could not be "
                    "retrieved. Supply a local actor path with its matching config.yaml."
                ) from exc
            if not isinstance(source_config, dict) or not source_config:
                raise ValueError(
                    "W&B model artifact has no config.yaml and the source run config is empty. "
                    "Supply a local actor path with its matching config.yaml."
                )
            try:
                with open(config_path, "w") as config_file:
                    yaml.safe_dump(source_config, config_file, sort_keys=False)
            except yaml.YAMLError as exc:
                raise ValueError("W&B source run config cannot be saved as config.yaml") from exc

        source_config = _read_source_config(config_path)
        model_path = _select_model(artifact_directory, source_config)
        adjacent_config_path = model_path.parent / "config.yaml"
        if config_path.resolve() != adjacent_config_path.resolve():
            shutil.copyfile(config_path, adjacent_config_path)
        return str(model_path)

    import neptune

    neptune_name = args.get("neptune_name")
    neptune_project = args.get("neptune_project")
    if not isinstance(neptune_name, str) or not neptune_name.strip():
        raise ValueError("Remote Neptune model download requires neptune_name")
    if not isinstance(neptune_project, str) or not neptune_project.strip():
        raise ValueError("Remote Neptune model download requires neptune_project")

    run = neptune.init_run(
        project=f"{neptune_name}/{neptune_project}",
        with_id=load_id,
        mode="read-only",
        capture_hardware_metrics=False,
        capture_stdout=False,
        capture_stderr=False,
        capture_traceback=False,
    )
    try:
        try:
            run[SAC_CONFIG_FIELD].download(destination=str(config_path))
        except Exception as exc:
            raise ValueError(
                "Neptune source is missing the portable sac/config field or it could not be "
                "downloaded. Supply a local actor path with its matching config.yaml."
            ) from exc
        try:
            source_config = _read_source_config(config_path)
        except FileNotFoundError as exc:
            raise ValueError(
                "Neptune sac/config download did not create config.yaml. "
                "Supply a local actor path with its matching config.yaml."
            ) from exc
        model_name = _source_model_name(source_config)
        model_filename = model_name.name if model_name is not None else "model.pt"
        model_path = download_directory / model_filename
        try:
            run[SAC_MODEL_FIELD].download(destination=str(model_path))
        except Exception as exc:
            raise ValueError(
                "Neptune source is missing the portable sac/model field or it could not be "
                "downloaded. Supply a local actor path with its matching config.yaml."
            ) from exc
        return str(model_path)
    finally:
        run.stop()
