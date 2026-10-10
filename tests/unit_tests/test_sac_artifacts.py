import sys
import types
from pathlib import Path

import pytest
import yaml

from pufferlib.sac_logging import SACNeptuneLogger, download_model


def _write_config(path, model_name="actor.pt"):
    path.write_text(yaml.safe_dump({"train": {"final_model_name": model_name}}))


class _UploadField:
    def __init__(self, field_name, uploads):
        self.field_name = field_name
        self.uploads = uploads

    def upload(self, path):
        self.uploads.append((self.field_name, path))


class _UploadRun:
    def __init__(self):
        self.uploads = []

    def __getitem__(self, field_name):
        return _UploadField(field_name, self.uploads)


def test_neptune_upload_uses_portable_sac_file_fields(tmp_path):
    actor_path = tmp_path / "actor.pt"
    actor_path.write_bytes(b"actor weights")
    config_path = tmp_path / "config.yaml"
    _write_config(config_path)

    logger = object.__new__(SACNeptuneLogger)
    logger.neptune = _UploadRun()
    logger.upload_model(actor_path)

    assert logger.neptune.uploads == [
        ("sac/model", str(actor_path)),
        ("sac/config", str(config_path)),
    ]


def test_wandb_download_uses_source_model_name_and_keeps_config_beside_actor(monkeypatch, tmp_path):
    calls = {}

    class _Artifact:
        def download(self, root):
            calls["root"] = root
            artifact_dir = Path(root)
            artifact_dir.mkdir(parents=True, exist_ok=True)
            (artifact_dir / "actor.pt").write_bytes(b"source actor")
            (artifact_dir / "other.pt").write_bytes(b"another model")
            _write_config(artifact_dir / "config.yaml", "actor.pt")
            return str(artifact_dir)

    class _Api:
        def artifact(self, name, type):
            calls["artifact"] = (name, type)
            return _Artifact()

        def run(self, _name):
            raise AssertionError("run config fallback is unnecessary when artifact has config.yaml")

    wandb_stub = types.ModuleType("wandb")
    wandb_stub.Api = _Api
    monkeypatch.setitem(sys.modules, "wandb", wandb_stub)

    args = {
        "wandb": True,
        "load_id": "source-run",
        "wandb_project": "drive-sac",
        "train": {"data_dir": str(tmp_path / "run")},
    }
    actor_path = download_model(args)

    actor_file = Path(actor_path)
    assert actor_file.name == "actor.pt"
    assert actor_file.parent.parent == tmp_path / "run" / "artifacts"
    assert (actor_file.parent / "config.yaml").is_file()
    assert calls["artifact"] == ("drive-sac/source-run:latest", "model")
    assert Path(calls["root"]).parent == tmp_path / "run" / "artifacts"


def test_wandb_download_uses_run_config_when_artifact_has_no_config(monkeypatch, tmp_path):
    calls = {}

    class _Artifact:
        def download(self, root):
            artifact_dir = Path(root)
            artifact_dir.mkdir(parents=True, exist_ok=True)
            (artifact_dir / "only_model.pt").write_bytes(b"source actor")
            return str(artifact_dir)

    class _Api:
        def artifact(self, _name, type):
            assert type == "model"
            return _Artifact()

        def run(self, name):
            calls["run"] = name
            return types.SimpleNamespace(config={"train": {"final_model_name": "only_model.pt"}})

    wandb_stub = types.ModuleType("wandb")
    wandb_stub.Api = _Api
    monkeypatch.setitem(sys.modules, "wandb", wandb_stub)

    stale_artifacts_dir = tmp_path / "flat-run" / "artifacts"
    stale_artifacts_dir.mkdir(parents=True)
    _write_config(stale_artifacts_dir / "config.yaml", "stale.pt")

    args = {
        "wandb": True,
        "load_id": "legacy-run",
        "wandb_project": "drive-sac",
        "data_dir": str(tmp_path / "flat-run"),
    }
    actor_path = download_model(args)

    actor_file = Path(actor_path)
    assert actor_file.name == "only_model.pt"
    assert actor_file.parent.parent == stale_artifacts_dir
    assert yaml.safe_load((actor_file.parent / "config.yaml").read_text()) == {
        "train": {"final_model_name": "only_model.pt"},
    }
    assert calls["run"] == "drive-sac/legacy-run"


def test_neptune_download_is_read_only_and_requires_portable_config(monkeypatch, tmp_path):
    calls = {"init": [], "stopped": 0}

    class _Field:
        def __init__(self, content):
            self.content = content

        def download(self, destination):
            destination_path = Path(destination)
            destination_path.parent.mkdir(parents=True, exist_ok=True)
            destination_path.write_bytes(self.content)

    class _Run:
        def __init__(self, config_content):
            self.fields = {
                "sac/config": _Field(config_content),
                "sac/model": _Field(b"neptune actor"),
            }

        def __getitem__(self, field_name):
            return self.fields[field_name]

        def stop(self):
            calls["stopped"] += 1

    config_content = yaml.safe_dump({"train": {"final_model_name": "actor.pt"}}).encode()
    run = _Run(config_content)

    def init_run(**kwargs):
        calls["init"].append(kwargs)
        return run

    neptune_stub = types.ModuleType("neptune")
    neptune_stub.init_run = init_run
    monkeypatch.setitem(sys.modules, "neptune", neptune_stub)

    args = {
        "neptune": True,
        "load_id": "N-123",
        "neptune_name": "pufferai",
        "neptune_project": "drive-sac",
        "train": {"data_dir": str(tmp_path / "neptune-run")},
    }
    actor_path = download_model(args)

    actor_file = Path(actor_path)
    assert actor_file.name == "actor.pt"
    assert actor_file.parent.parent == tmp_path / "neptune-run" / "artifacts"
    assert (actor_file.parent / "config.yaml").read_bytes() == config_content
    assert actor_file.read_bytes() == b"neptune actor"
    assert calls["init"] == [{
        "project": "pufferai/drive-sac",
        "with_id": "N-123",
        "mode": "read-only",
        "capture_hardware_metrics": False,
        "capture_stdout": False,
        "capture_stderr": False,
        "capture_traceback": False,
    }]
    assert calls["stopped"] == 1

    class _MissingConfigRun(_Run):
        def __getitem__(self, field_name):
            if field_name == "sac/config":
                raise KeyError(field_name)
            return self.fields[field_name]

    missing_config_run = _MissingConfigRun(config_content)
    neptune_stub.init_run = lambda **_kwargs: missing_config_run
    with pytest.raises(ValueError, match="local actor path with its matching config.yaml"):
        download_model(args)
    assert calls["stopped"] == 2
