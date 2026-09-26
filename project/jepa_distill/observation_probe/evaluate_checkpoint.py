"""Evaluate a saved observation-probe decoder on its fixed heldout split."""
from __future__ import annotations

import argparse
import copy
import json
import math
import os
import tempfile
from pathlib import Path
from typing import Any, Mapping, Sequence

from .config import validate_probe_config
from .data import ProbeCollectionDataset
from .evaluate import evaluate_probe
from .train import _make_identity, _validate_heldout_compatibility, preflight_probe
from ..runtime import resolve_path


def _positive_int(value: str) -> int:
    try:
        number = int(value)
    except ValueError as exc:
        raise argparse.ArgumentTypeError("must be a positive integer") from exc
    if number < 1:
        raise argparse.ArgumentTypeError("must be a positive integer")
    return number


def _nonnegative_int(value: str) -> int:
    try:
        number = int(value)
    except ValueError as exc:
        raise argparse.ArgumentTypeError("must be a non-negative integer") from exc
    if number < 0:
        raise argparse.ArgumentTypeError("must be a non-negative integer")
    return number


def _prepare_output_dir(path_text: str) -> Path:
    output_dir = Path(path_text).expanduser().resolve()
    if output_dir.exists():
        if not output_dir.is_dir():
            raise NotADirectoryError(f"evaluation output path is not a directory: {output_dir}")
        if any(output_dir.iterdir()):
            raise FileExistsError(f"evaluation output directory is not empty: {output_dir}")
    return output_dir


def _evaluation_batches(dataset: Any, batch_size: int):
    for start in range(0, len(dataset), batch_size):
        indices = list(range(start, min(start + batch_size, len(dataset))))
        yield dataset.get_batch(indices)


def _atomic_json_save(payload: Mapping[str, Any], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_path = tempfile.mkstemp(
        prefix=f".{path.name}.", suffix=".tmp", dir=path.parent
    )
    os.close(descriptor)
    try:
        path_text = json.dumps(dict(payload), indent=2, sort_keys=True, allow_nan=False)
        Path(temporary_path).write_text(path_text + "\n", encoding="utf-8")
        os.replace(temporary_path, path)
    finally:
        if os.path.exists(temporary_path):
            os.unlink(temporary_path)


def evaluate_checkpoint(
    probe_checkpoint: str | Path,
    *,
    split: str = "validation",
    device: str = "cpu",
    output_dir: str | Path,
    max_windows: int | None = None,
    render_samples: int | None = None,
) -> dict[str, Any]:
    """Validate provenance, restore frozen JEPA/decoder and write heldout results."""
    import torch

    if split not in {"validation", "test"}:
        raise ValueError("split must be 'validation' or 'test'")
    if max_windows is not None and (
        isinstance(max_windows, bool) or not isinstance(max_windows, int) or max_windows < 1
    ):
        raise ValueError("max_windows must be a positive integer or null")
    if render_samples is not None and (
        isinstance(render_samples, bool)
        or not isinstance(render_samples, int)
        or render_samples < 0
    ):
        raise ValueError("render_samples must be a non-negative integer or null")
    output_path = _prepare_output_dir(str(output_dir))
    torch.set_num_threads(1)

    probe_path = Path(probe_checkpoint).expanduser().resolve(strict=True)
    if not probe_path.is_file():
        raise FileNotFoundError(f"probe checkpoint is not a file: {probe_path}")
    payload = torch.load(probe_path, map_location="cpu", weights_only=False)
    if not isinstance(payload, Mapping) or payload.get("format") != "observation_probe_v1":
        raise ValueError("probe checkpoint must use format observation_probe_v1")
    identity = payload.get("identity")
    if not isinstance(identity, Mapping) or not isinstance(identity.get("config"), Mapping):
        raise ValueError("probe checkpoint is missing its training identity/config")
    saved_config = copy.deepcopy(dict(identity["config"]))
    validate_probe_config(saved_config, require_resolved=True)
    if split == "test" and saved_config["data"].get("test_manifest") is None:
        raise ValueError("the saved probe config has no data.test_manifest")
    if not isinstance(payload.get("decoder_state"), Mapping):
        raise ValueError("probe checkpoint is missing decoder parameters")

    frozen_path = resolve_path(saved_config["checkpoint"]).resolve(strict=True)
    online = saved_config["data"].get("mode") == "online"
    if online:
        from .online import inspect_online_source

        if payload.get("training_mean_observation") is None:
            raise ValueError("online probe checkpoint is missing its saved training mean")
        data_report = {"compatibility": inspect_online_source(frozen_path)["compatibility"]}
        manifest_paths = []
    else:
        data_report = preflight_probe(saved_config)["data"]
        manifest_paths = [
            path for row in data_report["per_round"] for path in row["manifest_paths"]
        ]
    current_identity = _make_identity(saved_config, frozen_path, manifest_paths)
    if dict(identity) != current_identity:
        raise ValueError(
            "probe checkpoint identity no longer matches its frozen JEPA checkpoint, manifests, or config"
        )

    manifest_path = resolve_path(saved_config["data"][f"{split}_manifest"]).resolve(strict=True)
    dataset = ProbeCollectionDataset([str(manifest_path)], expected_split=split)
    try:
        compatibility = data_report["compatibility"]
        heldout_compatibility = _validate_heldout_compatibility(
            dataset, compatibility, split=split
        )
        if len(dataset) < 1:
            raise ValueError(f"{split} manifest contains no eligible windows")

        from .model import ObservationDecoder, load_frozen_jepa

        jepa = load_frozen_jepa(str(frozen_path), device=device)
        decoder = ObservationDecoder(
            int(jepa.latent_dim),
            compatibility["observation_layout"],
            hidden_sizes=list(saved_config["decoder"]["hidden_sizes"]),
        )
        decoder.load_state_dict(payload["decoder_state"], strict=True)
        decoder.to(device)
        decoder.eval()
        jepa.eval()

        evaluation_config = copy.deepcopy(saved_config)
        if payload.get("training_mean_observation") is not None:
            evaluation_config["_training_mean_observation"] = payload["training_mean_observation"]
        evaluation_config["training"]["device"] = device
        if max_windows is not None:
            evaluation_config["evaluation"]["max_windows"] = max_windows
        if render_samples is not None:
            evaluation_config["evaluation"]["render_samples"] = render_samples
        evaluation_config["_render_dir"] = str(output_path / "renders")
        batch_size = int(evaluation_config["evaluation"]["batch_size"])
        metrics = {
            str(name): float(value)
            for name, value in evaluate_probe(
                jepa,
                decoder,
                _evaluation_batches(dataset, batch_size),
                config=evaluation_config,
            ).items()
        }
        if any(not math.isfinite(value) for value in metrics.values()):
            raise FloatingPointError("probe evaluation returned a non-finite metric")
        result = {
            "format": "observation_probe_evaluation_v1",
            "probe_checkpoint": str(probe_path),
            "frozen_checkpoint": str(frozen_path),
            "split": split,
            "heldout_compatibility": heldout_compatibility,
            "metrics": metrics,
        }
        _atomic_json_save(result, output_path / "evaluation.json")
        return result
    finally:
        dataset.close()


def main(argv: Sequence[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--probe-checkpoint", required=True)
    parser.add_argument("--split", choices=("validation", "test"), default="validation")
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--max-windows", type=_positive_int)
    parser.add_argument("--render-samples", type=_nonnegative_int)
    args = parser.parse_args(argv)
    try:
        result = evaluate_checkpoint(
            args.probe_checkpoint,
            split=args.split,
            device=args.device,
            output_dir=args.output_dir,
            max_windows=args.max_windows,
            render_samples=args.render_samples,
        )
        print(json.dumps(result, indent=2, sort_keys=True, allow_nan=False))
    except (ValueError, OSError, RuntimeError, FloatingPointError, TypeError) as error:
        parser.exit(2, f"Observation probe evaluation: {error}\n")


if __name__ == "__main__":
    main()
