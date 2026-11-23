import csv
import hashlib
import json
from pathlib import Path

import numpy as np
import pytest

from project.soft_actor_critic_q_calibration.development import plot_geometry


def _write_shard(path, *, frame_idx, agent_id, values, eligible=None, frame_count=None,
                 joint_actions=None):
    path.parent.mkdir(parents=True, exist_ok=True)
    arrays = {
        "frame_idx": np.asarray(frame_idx, dtype=np.int32),
        "agent_id": np.asarray(agent_id, dtype=np.int32),
        "slot": np.zeros(len(frame_idx), dtype=np.int32),
        "eligible": np.ones(len(frame_idx), dtype=np.bool_) if eligible is None
        else np.asarray(eligible, dtype=np.bool_),
        **{name: np.asarray(column, dtype=np.float32) for name, column in values.items()},
    }
    if frame_count is not None:
        arrays["frame_count"] = np.asarray(frame_count, dtype=np.int32)
    if joint_actions is not None:
        arrays["joint_actions"] = np.zeros(joint_actions, dtype=np.int32)
    np.savez(path, **arrays)


def _write_manifested_collection(model_dir, *, incomplete_map=None, add_orphan=False):
    references_dir = model_dir / "records" / "references"
    map_records = [
        {"map_index": index, "map_name": f"map_{index}", "path": f"/maps/map_{index}.bin",
         "sha256": f"map-hash-{index}"}
        for index in range(8)
    ]
    settings = {"maps": map_records, "steps_per_map": 2}
    encoded_settings = json.dumps(settings, sort_keys=True, separators=(",", ":"), allow_nan=False)
    signature = hashlib.sha256(encoded_settings.encode("utf-8")).hexdigest()
    records_dir = model_dir / "records"
    records_dir.mkdir(parents=True, exist_ok=True)
    (records_dir / "geometry_manifest.json").write_text(json.dumps({
        "format_version": 1,
        "signature": signature,
        "settings": settings,
    }))

    for map_record in map_records:
        map_index = map_record["map_index"]
        map_name = map_record["map_name"]
        map_dir = references_dir / map_name
        frame_count = 1 if map_name == incomplete_map else 2
        shard_path = map_dir / "episode_00000.npz"
        columns = plot_geometry.DISTANCE_COLUMNS
        _write_shard(
            shard_path,
            frame_idx=[0], agent_id=[map_index], frame_count=frame_count,
            values={column: [float(map_index)] for column in columns},
        )
        start = 0
        sidecar = {
            "format_version": 1,
            "run_signature": signature,
            "map_index": map_index,
            "map_name": map_name,
            "map_path": map_record["path"],
            "map_sha256": map_record["sha256"],
            "episode_index": 0,
            "frame_count": frame_count,
            "cumulative_frame_start": start,
            "cumulative_frame_end": start + frame_count,
            "eligible_row_count": 1,
            "data_file": shard_path.name,
            "data_sha256": hashlib.sha256(shard_path.read_bytes()).hexdigest(),
        }
        (map_dir / "episode_00000.json").write_text(json.dumps(sidecar))
        progress_frames = frame_count
        complete = frame_count >= settings["steps_per_map"]
        (map_dir / "progress.json").write_text(json.dumps({
            "run_signature": signature,
            "map_index": map_index,
            "map_name": map_name,
            "target_joint_frames": settings["steps_per_map"],
            "completed_joint_frames": progress_frames,
            "completed_episodes": 1,
            "complete": complete,
        }))

    if add_orphan:
        orphan_path = references_dir / "map_0" / "episode_00001.npz"
        orphan_path.write_bytes((references_dir / "map_0" / "episode_00000.npz").read_bytes())


def test_plot_geometry_writes_ecdf_and_map_pooled_coverage(tmp_path, monkeypatch):
    model_dir = tmp_path / "runs" / "geometry_test" / "pql"
    references_dir = model_dir / "records" / "references"
    columns = plot_geometry.DISTANCE_COLUMNS
    _write_shard(
        references_dir / "map_a" / "episode_000.npz",
        frame_idx=[0, 0, 1, 2], agent_id=[10, 11, 10, -1],
        eligible=[True, True, True, False], frame_count=3,
        values={
            columns[0]: [1.0, 2.0, np.nan, 999.0],
            columns[1]: [5.0, np.nan, 7.0, 999.0],
            columns[2]: [10.0, 20.0, 30.0, 999.0],
            columns[3]: [4.0, 5.0, 6.0, 999.0],
        },
    )
    _write_shard(
        references_dir / "map_a" / "episode_001.npz",
        frame_idx=[0, 1], agent_id=[10, 10], joint_actions=(4, 2),
        values={
            columns[0]: [3.0, 4.0],
            columns[1]: [8.0, 9.0],
            columns[2]: [11.0, 12.0],
            columns[3]: [8.0, np.nan],
        },
    )
    _write_shard(
        references_dir / "map_b" / "episode_000.npz",
        frame_idx=[0], agent_id=[21],
        values={
            columns[0]: [np.nan],
            columns[1]: [np.nan],
            columns[2]: [2.0],
            columns[3]: [0.0],
        },
    )

    plot_dir = model_dir / "plots" / "geometry_plots"
    analysis_dir = model_dir / "records" / "analysis" / "geometry_plots"

    def local_model_directory(path):
        return Path(path)

    def local_records_directory(path):
        return Path(path) / "records"

    def local_plot_output_dir(path, family, *, artifact_type="plots"):
        base = Path(path) / ("plots" if artifact_type == "plots" else "records/analysis")
        output = base / family
        output.mkdir(parents=True, exist_ok=True)
        return output

    monkeypatch.setattr(plot_geometry, "model_directory", local_model_directory)
    monkeypatch.setattr(plot_geometry, "records_directory", local_records_directory)
    monkeypatch.setattr(plot_geometry, "plot_output_dir", local_plot_output_dir)

    plot_geometry.plot_geometry(model_dir)

    assert (plot_dir / "geometry_ecdf.png").read_bytes().startswith(b"\x89PNG\r\n\x1a\n")
    assert (plot_dir / "README.md").is_file()
    assert {path.name for path in plot_dir.iterdir()} == {"README.md", "geometry_ecdf.png"}
    assert (analysis_dir / "README.md").is_file()
    with (analysis_dir / "coverage.csv").open(newline="") as stream:
        coverage = list(csv.DictReader(stream))

    pooled_partner = next(row for row in coverage
                          if row["map"] == "__pooled__" and row["metric"] == columns[0])
    assert pooled_partner["episode_count"] == "3"
    assert pooled_partner["agent_count"] == "4"
    assert pooled_partner["frame_count"] == "8"
    assert pooled_partner["total_count"] == "6"
    assert pooled_partner["finite_count"] == "4"
    assert pooled_partner["missing_count"] == "2"
    assert float(pooled_partner["missing_percent"]) == 100.0 / 3.0
    assert float(pooled_partner["minimum_m"]) == 1.0
    assert float(pooled_partner["maximum_m"]) == 4.0
    assert float(pooled_partner["p50_m"]) == 2.5

    map_a_road = next(row for row in coverage
                      if row["map"] == "map_a" and row["metric"] == columns[1])
    assert map_a_road["episode_count"] == "2"
    assert map_a_road["agent_count"] == "3"
    assert map_a_road["frame_count"] == "7"
    assert map_a_road["finite_count"] == "4"
    assert any(row["metric"] == columns[3] for row in coverage)


def test_load_reference_shards_rejects_infinite_distance(tmp_path, monkeypatch):
    model_dir = tmp_path / "model"
    shard_path = model_dir / "records" / "references" / "map_a" / "episode_000.npz"
    columns = plot_geometry.DISTANCE_COLUMNS
    _write_shard(
        shard_path,
        frame_idx=[0], agent_id=[1],
        values={
            columns[0]: [np.inf],
            columns[1]: [1.0],
            columns[2]: [2.0],
            columns[3]: [3.0],
        },
    )
    monkeypatch.setattr(plot_geometry, "records_directory", lambda path: Path(path) / "records")

    try:
        plot_geometry.load_reference_shards(model_dir)
    except ValueError as error:
        assert "nonnegative finite distances or NaN" in str(error)
    else:
        raise AssertionError("Infinite geometry distance should be rejected")


def test_manifested_collection_requires_all_eight_complete_budgets(tmp_path, monkeypatch):
    model_dir = tmp_path / "model"
    _write_manifested_collection(model_dir, incomplete_map="map_7")
    monkeypatch.setattr(plot_geometry, "records_directory", lambda path: Path(path) / "records")

    with pytest.raises(ValueError, match="Incomplete geometry map budget for map_7"):
        plot_geometry.load_reference_shards(model_dir)


def test_manifested_collection_rejects_orphan_npz_shard(tmp_path, monkeypatch):
    model_dir = tmp_path / "model"
    _write_manifested_collection(model_dir, add_orphan=True)
    monkeypatch.setattr(plot_geometry, "records_directory", lambda path: Path(path) / "records")

    with pytest.raises(ValueError, match="Uncommitted/orphan geometry episodes"):
        plot_geometry.load_reference_shards(model_dir)


def test_manifested_collection_rejects_sidecar_signature_mismatch(tmp_path, monkeypatch):
    model_dir = tmp_path / "model"
    _write_manifested_collection(model_dir)
    sidecar_path = model_dir / "records" / "references" / "map_0" / "episode_00000.json"
    sidecar = json.loads(sidecar_path.read_text())
    sidecar["run_signature"] = "different-run"
    sidecar_path.write_text(json.dumps(sidecar))
    monkeypatch.setattr(plot_geometry, "records_directory", lambda path: Path(path) / "records")

    with pytest.raises(ValueError, match="sidecar does not commit"):
        plot_geometry.load_reference_shards(model_dir)


def test_complete_manifest_validates_all_eight_maps(tmp_path, monkeypatch):
    model_dir = tmp_path / "model"
    _write_manifested_collection(model_dir)
    monkeypatch.setattr(plot_geometry, "records_directory", lambda path: Path(path) / "records")

    episodes_by_map = plot_geometry.load_reference_shards(model_dir)

    assert sorted(episodes_by_map) == [f"map_{index}" for index in range(8)]
    assert all(len(episodes) == 1 for episodes in episodes_by_map.values())
