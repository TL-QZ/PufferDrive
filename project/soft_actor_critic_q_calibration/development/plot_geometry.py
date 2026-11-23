"""Plot reference-episode geometry distributions and write measurement coverage."""

import argparse
import csv
import hashlib
import json
from pathlib import Path
import re

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter, MaxNLocator, PercentFormatter
import numpy as np

if __package__:
    from .plot_paths import model_directory, plot_output_dir, records_directory
else:
    from plot_paths import model_directory, plot_output_dir, records_directory


DISTANCE_COLUMNS = (
    "nearest_observed_partner_center_distance_m",
    "nearest_observed_road_boundary_distance_m",
    "nearest_visible_goal_distance_m",
    "current_goal_distance_m",
)
PLOTTED_COLUMNS = DISTANCE_COLUMNS[:3]
PLOT_LABELS = {
    "nearest_observed_partner_center_distance_m": "Nearest observed partner center",
    "nearest_observed_road_boundary_distance_m": "Nearest observed road boundary",
    "nearest_visible_goal_distance_m": "Nearest visible goal",
}
IDENTIFIER_COLUMNS = ("frame_idx", "agent_id", "slot")
PERCENTILES = (1, 5, 25, 50, 75, 95, 99)
COVERAGE_COLUMNS = (
    "map",
    "metric",
    "episode_count",
    "agent_count",
    "frame_count",
    "total_count",
    "finite_count",
    "missing_count",
    "missing_percent",
    "minimum_m",
    "maximum_m",
    *(f"p{percentile:02d}_m" for percentile in PERCENTILES),
)


def _integer_column(arrays, column_name, path, expected_size=None, row_mask=None):
    if column_name not in arrays.files:
        raise ValueError(f"Missing {column_name!r} in reference shard: {path}")
    values = arrays[column_name]
    if values.ndim != 1 or values.dtype.kind not in "iu":
        raise ValueError(f"Expected a one-dimensional integer {column_name!r}: {path}")
    if expected_size is not None and values.size != expected_size:
        raise ValueError(f"Column length mismatch for {column_name!r}: {path}")
    validated_values = values if row_mask is None else values[row_mask]
    if validated_values.size and np.any(validated_values < 0):
        raise ValueError(f"Negative {column_name!r} value in reference shard: {path}")
    return values


def _frame_count(arrays, frame_indices, path):
    if "frame_count" in arrays.files:
        value = arrays["frame_count"]
        if value.size != 1 or value.dtype.kind not in "iu" or int(value.reshape(-1)[0]) < 0:
            raise ValueError(f"Expected a nonnegative integer frame_count: {path}")
        return int(value.reshape(-1)[0])
    if "joint_actions" in arrays.files:
        joint_actions = arrays["joint_actions"]
        if joint_actions.ndim < 1:
            raise ValueError(f"Expected joint_actions to have a frame axis: {path}")
        return int(joint_actions.shape[0])
    return int(np.unique(frame_indices).size)


def _read_json(path, *, label):
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise ValueError(f"Cannot read {label}: {path}") from error
    if not isinstance(value, dict):
        raise ValueError(f"Expected a JSON object for {label}: {path}")
    return value


def _sha256_file(path):
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _nonnegative_integer(value, field, path):
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise ValueError(f"Expected nonnegative integer {field!r}: {path}")
    return value


def _validate_geometry_collection(records_dir, references_dir, shard_paths):
    """Fail closed if a manifested eight-map collection is partial or uncommitted."""
    manifest_path = records_dir / "geometry_manifest.json"
    if not manifest_path.is_file():
        return
    manifest = _read_json(manifest_path, label="geometry manifest")
    signature = manifest.get("signature")
    settings = manifest.get("settings")
    if not isinstance(signature, str) or not isinstance(settings, dict):
        raise ValueError(f"Geometry manifest lacks a signature or settings object: {manifest_path}")
    try:
        canonical_settings = json.dumps(
            settings, sort_keys=True, separators=(",", ":"), allow_nan=False
        ).encode("utf-8")
    except (TypeError, ValueError) as error:
        raise ValueError(f"Geometry manifest settings are not canonical JSON: {manifest_path}") from error
    expected_signature = hashlib.sha256(canonical_settings).hexdigest()
    if signature != expected_signature:
        raise ValueError(f"Geometry manifest signature does not match its settings: {manifest_path}")

    map_records = settings.get("maps")
    if not isinstance(map_records, list) or len(map_records) != 8:
        raise ValueError(f"Geometry manifest must describe all eight map budgets: {manifest_path}")
    target_frames = settings.get("steps_per_map")
    if isinstance(target_frames, bool) or not isinstance(target_frames, int) or target_frames <= 0:
        raise ValueError(f"Geometry manifest has an invalid steps_per_map budget: {manifest_path}")
    map_names = []
    for map_index, map_record in enumerate(map_records):
        if not isinstance(map_record, dict) or map_record.get("map_index") != map_index:
            raise ValueError(f"Geometry manifest map ordering is invalid at index {map_index}: {manifest_path}")
        map_name = map_record.get("map_name")
        if not isinstance(map_name, str) or not map_name or map_name in map_names:
            raise ValueError(f"Geometry manifest map names must be unique: {manifest_path}")
        map_names.append(map_name)

    expected_map_names = set(map_names)
    shard_map_names = {path.parent.name for path in shard_paths}
    unexpected_shards = sorted(shard_map_names - expected_map_names)
    if unexpected_shards:
        raise ValueError(f"Geometry shards belong to maps absent from the manifest: {unexpected_shards}")

    for child in references_dir.iterdir() if references_dir.exists() else ():
        if not child.is_dir() or child.name in expected_map_names:
            continue
        if list(child.glob("*.npz")) or list(child.glob("episode_*.json")) or (child / "progress.json").exists():
            raise ValueError(f"Unmanifested geometry map data found under {child}")

    manifest_format_version = manifest.get("format_version")
    if isinstance(manifest_format_version, bool) or not isinstance(manifest_format_version, int):
        raise ValueError(f"Geometry manifest has an invalid format_version: {manifest_path}")
    for map_index, map_record in enumerate(map_records):
        map_name = map_record["map_name"]
        map_dir = references_dir / map_name
        progress_path = map_dir / "progress.json"
        if not progress_path.is_file():
            raise ValueError(f"Incomplete geometry map {map_name}: missing progress.json")
        progress = _read_json(progress_path, label="geometry map progress")
        expected_progress = {
            "run_signature": signature,
            "map_index": map_index,
            "map_name": map_name,
            "target_joint_frames": target_frames,
        }
        progress_mismatches = [
            field for field, expected in expected_progress.items()
            if progress.get(field) != expected
        ]
        if progress_mismatches:
            raise ValueError(
                f"Geometry progress for {map_name} does not match the manifest: {progress_mismatches}"
            )
        if progress.get("complete") is not True:
            raise ValueError(f"Incomplete geometry map budget for {map_name}")

        map_shards = sorted(map_dir.glob("*.npz"))
        map_sidecars = sorted(map_dir.glob("episode_*.json"))
        shard_by_stem = {path.stem: path for path in map_shards}
        sidecar_by_stem = {path.stem: path for path in map_sidecars}
        if len(shard_by_stem) != len(map_shards) or len(sidecar_by_stem) != len(map_sidecars):
            raise ValueError(f"Duplicate geometry episode names in {map_dir}")
        if set(shard_by_stem) != set(sidecar_by_stem):
            missing_sidecars = sorted(set(shard_by_stem) - set(sidecar_by_stem))
            missing_shards = sorted(set(sidecar_by_stem) - set(shard_by_stem))
            raise ValueError(
                f"Uncommitted/orphan geometry episodes in {map_dir}: "
                f"missing_sidecars={missing_sidecars}, missing_shards={missing_shards}"
            )
        indexed_stems = []
        for stem in shard_by_stem:
            match = re.fullmatch(r"episode_(\d{5})", stem)
            if match is None:
                raise ValueError(f"Unexpected geometry episode filename in {map_dir}: {stem}")
            indexed_stems.append((int(match.group(1)), stem))
        indexed_stems.sort()
        episode_indices = [index for index, _ in indexed_stems]
        if episode_indices != list(range(len(episode_indices))):
            raise ValueError(f"Geometry episodes are not contiguous from zero in {map_dir}")

        completed_frames = 0
        for episode_index, stem in indexed_stems:
            sidecar_path = sidecar_by_stem[stem]
            shard_path = shard_by_stem[stem]
            sidecar = _read_json(sidecar_path, label="geometry episode sidecar")
            expected_sidecar = {
                "format_version": manifest_format_version,
                "run_signature": signature,
                "map_index": map_index,
                "map_name": map_name,
                "episode_index": episode_index,
                "data_file": shard_path.name,
                "data_sha256": _sha256_file(shard_path),
            }
            for field in ("map_path", "map_sha256"):
                if field in map_record:
                    expected_sidecar[field] = map_record[field]
            sidecar_mismatches = [
                field for field, expected in expected_sidecar.items()
                if sidecar.get(field) != expected
            ]
            if sidecar_mismatches:
                raise ValueError(
                    f"Geometry episode sidecar does not commit {shard_path}: {sidecar_mismatches}"
                )

            frame_count = _nonnegative_integer(sidecar.get("frame_count"), "frame_count", sidecar_path)
            frame_start = _nonnegative_integer(
                sidecar.get("cumulative_frame_start"), "cumulative_frame_start", sidecar_path
            )
            frame_end = _nonnegative_integer(
                sidecar.get("cumulative_frame_end"), "cumulative_frame_end", sidecar_path
            )
            if frame_count <= 0 or frame_start != completed_frames or frame_end != frame_start + frame_count:
                raise ValueError(f"Geometry episode frame ranges are not contiguous in {map_dir}")
            completed_frames = frame_end

        progress_frames = _nonnegative_integer(
            progress.get("completed_joint_frames"), "completed_joint_frames", progress_path
        )
        progress_episodes = _nonnegative_integer(
            progress.get("completed_episodes"), "completed_episodes", progress_path
        )
        if (completed_frames != progress_frames or len(indexed_stems) != progress_episodes
                or completed_frames < target_frames):
            raise ValueError(
                f"Geometry map {map_name} does not meet its {target_frames}-frame budget: "
                f"sidecars={completed_frames} frames/{len(indexed_stems)} episodes, "
                f"progress={progress_frames} frames/{progress_episodes} episodes"
            )


def load_reference_shards(model_dir):
    """Read one flat, row-aligned geometry shard per episode, grouped by map."""
    records_dir = records_directory(model_dir)
    references_dir = records_dir / "references"
    shard_paths = sorted(references_dir.glob("*/*.npz"))
    _validate_geometry_collection(records_dir, references_dir, shard_paths)
    if not shard_paths:
        raise ValueError(f"No reference shards found under {references_dir}")

    episodes_by_map = {}
    for path in shard_paths:
        with np.load(path, allow_pickle=False) as arrays:
            if "eligible" in arrays.files:
                eligible_rows = arrays["eligible"]
                if (eligible_rows.ndim != 1 or eligible_rows.dtype.kind not in "bu"
                        or (eligible_rows.dtype.kind == "u" and np.any(eligible_rows > 1))):
                    raise ValueError(f"Expected a one-dimensional boolean eligible mask: {path}")
                eligible_rows = eligible_rows.astype(bool, copy=False)
            else:
                # Shards without a row mask follow the eligible-only reference contract.
                eligible_rows = None
            identifiers = {
                name: _integer_column(arrays, name, path,
                                      expected_size=None if eligible_rows is None else eligible_rows.size,
                                      row_mask=eligible_rows)
                for name in IDENTIFIER_COLUMNS
            }
            row_count = identifiers["frame_idx"].size
            for name in IDENTIFIER_COLUMNS[1:]:
                if identifiers[name].size != row_count:
                    raise ValueError(f"Column length mismatch for {name!r}: {path}")
            if eligible_rows is not None and eligible_rows.size != row_count:
                raise ValueError(f"Column length mismatch for 'eligible': {path}")

            distances = {}
            for name in DISTANCE_COLUMNS:
                if name not in arrays.files:
                    raise ValueError(f"Missing {name!r} in reference shard: {path}")
                values = arrays[name]
                if values.ndim != 1 or values.size != row_count or values.dtype.kind not in "fiu":
                    raise ValueError(f"Expected a row-aligned numeric {name!r}: {path}")
                values = values.astype(np.float64, copy=False)
                if eligible_rows is not None:
                    values = values[eligible_rows]
                if np.isinf(values).any() or np.any(values[np.isfinite(values)] < 0):
                    raise ValueError(f"Expected nonnegative finite distances or NaN in {name!r}: {path}")
                distances[name] = values

            episode_frame_count = _frame_count(arrays, identifiers["frame_idx"], path)
            if eligible_rows is not None:
                identifiers = {name: values[eligible_rows] for name, values in identifiers.items()}

            episode = {
                "path": path,
                "frame_idx": identifiers["frame_idx"],
                "agent_id": identifiers["agent_id"],
                "slot": identifiers["slot"],
                "frame_count": episode_frame_count,
                "distances": distances,
            }
        episodes_by_map.setdefault(path.parent.name, []).append(episode)

    return episodes_by_map


def _coverage_row(map_name, episodes, metric):
    values = np.concatenate([episode["distances"][metric] for episode in episodes])
    finite_values = values[np.isfinite(values)]
    row = {
        "map": map_name,
        "metric": metric,
        "episode_count": len(episodes),
        # Agent IDs can repeat between episodes, so count episode-agent pairs.
        "agent_count": sum(np.unique(episode["agent_id"]).size for episode in episodes),
        # Scene length may exceed the frames with eligible agents; use saved metadata when present.
        "frame_count": sum(episode["frame_count"] for episode in episodes),
        "total_count": values.size,
        "finite_count": finite_values.size,
        "missing_count": values.size - finite_values.size,
        "missing_percent": (100.0 * (values.size - finite_values.size) / values.size
                            if values.size else 0.0),
        "minimum_m": "",
        "maximum_m": "",
    }
    row.update({f"p{percentile:02d}_m": "" for percentile in PERCENTILES})
    if finite_values.size:
        row["minimum_m"] = float(np.min(finite_values))
        row["maximum_m"] = float(np.max(finite_values))
        for percentile, value in zip(PERCENTILES, np.percentile(finite_values, PERCENTILES)):
            row[f"p{percentile:02d}_m"] = float(value)
    return row


def build_coverage_rows(episodes_by_map):
    """Return map and pooled coverage rows for all four recorded distances."""
    rows = []
    pooled_episodes = []
    for map_name, episodes in sorted(episodes_by_map.items()):
        rows.extend(_coverage_row(map_name, episodes, metric) for metric in DISTANCE_COLUMNS)
        pooled_episodes.extend(episodes)
    rows.extend(_coverage_row("__pooled__", pooled_episodes, metric) for metric in DISTANCE_COLUMNS)
    return rows


def _plot_ecdf(axis, values, *, label, color, linewidth, linestyle="-"):
    values = np.sort(values[np.isfinite(values)])
    if values.size == 0:
        return
    unique_values, counts = np.unique(values, return_counts=True)
    cumulative_fraction = np.cumsum(counts, dtype=np.float64) / values.size
    axis.step(np.concatenate(([0.0], unique_values)),
              np.concatenate(([0.0], cumulative_fraction)),
              where="post", label=label, color=color, linewidth=linewidth,
              linestyle=linestyle)


def _save_ecdf(episodes_by_map, model_dir, output_dir):
    map_names = sorted(episodes_by_map)
    color_map = plt.get_cmap("tab20", max(1, len(map_names)))
    map_colors = {name: color_map(index) for index, name in enumerate(map_names)}
    figure, axes = plt.subplots(len(PLOTTED_COLUMNS), 1, figsize=(11, 12), squeeze=False)
    axes = axes[:, 0]
    legend_handles = []
    legend_labels = []

    for axis, metric in zip(axes, PLOTTED_COLUMNS):
        all_values = []
        total_count = 0
        finite_count = 0
        missing_count = 0
        for map_name in map_names:
            values = np.concatenate([episode["distances"][metric]
                                     for episode in episodes_by_map[map_name]])
            finite_values = values[np.isfinite(values)]
            all_values.append(finite_values)
            total_count += values.size
            finite_count += finite_values.size
            missing_count += values.size - finite_values.size
            _plot_ecdf(axis, finite_values, label=map_name, color=map_colors[map_name],
                       linewidth=1.5)

        pooled_values = np.concatenate(all_values) if all_values else np.empty(0, dtype=np.float64)
        _plot_ecdf(axis, pooled_values, label="Pooled", color="#111827", linewidth=2.3,
                   linestyle="--")
        axis.set_title(
            f"{PLOT_LABELS[metric]}  |  {finite_count:,} finite, "
            f"{missing_count:,}/{total_count:,} missing",
            loc="left", fontsize=12, fontweight="semibold")
        axis.set_xlabel("Distance (m)")
        axis.set_ylabel("Cumulative fraction")
        axis.set_ylim(0.0, 1.0)
        axis.yaxis.set_major_formatter(PercentFormatter(1.0))
        axis.yaxis.set_major_locator(MaxNLocator(nbins=5))
        axis.grid(True, which="major", color="#d1d5db", linewidth=0.7, alpha=0.7)
        axis.spines["top"].set_visible(False)
        axis.spines["right"].set_visible(False)
        finite_pooled = pooled_values[np.isfinite(pooled_values)]
        x_maximum = float(np.max(finite_pooled)) if finite_pooled.size else 0.0
        axis.set_xlim(0.0, x_maximum if x_maximum > 0 else 1.0)
        decimals = 1 if x_maximum <= 100 else 0
        axis.xaxis.set_major_formatter(
            FuncFormatter(lambda value, _position, digits=decimals: f"{value:,.{digits}f}"))
        axis.xaxis.set_major_locator(MaxNLocator(nbins=7))
        if finite_count == 0:
            axis.text(0.5, 0.5, "No finite measurements", transform=axis.transAxes,
                      ha="center", va="center", color="#6b7280", fontsize=11)

    for map_name in map_names:
        legend_handles.append(plt.Line2D([], [], color=map_colors[map_name], linewidth=1.8))
        legend_labels.append(map_name)
    legend_handles.append(plt.Line2D([], [], color="#111827", linewidth=2.3, linestyle="--"))
    legend_labels.append("Pooled")
    legend_columns = min(4, len(legend_labels))
    legend_rows = (len(legend_labels) + legend_columns - 1) // legend_columns
    bottom_margin = 0.09 + 0.025 * max(0, legend_rows - 1)
    figure.suptitle(f"{model_dir.parent.name} / {model_dir.name}: reference geometry ECDFs",
                    fontsize=16, fontweight="semibold")
    figure.legend(legend_handles, legend_labels, loc="lower center", ncol=legend_columns,
                  frameon=False, title="Map directory (dashed line: pooled)",
                  bbox_to_anchor=(0.5, 0.01), fontsize=9)
    figure.tight_layout(rect=(0, bottom_margin, 1, 0.95))
    figure.savefig(output_dir / "geometry_ecdf.png", dpi=180, facecolor="white")
    plt.close(figure)


def plot_geometry(model_dir):
    """Create the geometry ECDF image and complete coverage CSV for a model run."""
    model_dir = model_directory(model_dir)
    episodes_by_map = load_reference_shards(model_dir)
    plot_dir = plot_output_dir(model_dir, "geometry_plots")
    analysis_dir = plot_output_dir(model_dir, "geometry_plots", artifact_type="analysis")

    image_path = plot_dir / "geometry_ecdf.png"
    _save_ecdf(episodes_by_map, model_dir, plot_dir)
    rows = build_coverage_rows(episodes_by_map)
    with (analysis_dir / "coverage.csv").open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=COVERAGE_COLUMNS)
        writer.writeheader()
        writer.writerows(rows)

    (plot_dir / "README.md").write_text(
        "# Reference geometry ECDF\n\n"
        "`geometry_ecdf.png` shows empirical cumulative distributions for nearest observed "
        "partner center, nearest observed road boundary, and nearest visible goal distances. "
        "Each solid curve is a map directory; the dashed curve pools finite observations "
        "across maps. Axes show the full finite range from zero to the largest observed value. "
        "NaN values are excluded from ECDFs and reported as missing in the coverage table.\n\n"
        "Coverage for all four saved distance columns, including current goal distance, is in "
        "[coverage.csv](../../records/analysis/geometry_plots/coverage.csv). Percentiles are "
        "computed over finite values. `__pooled__` is the combined row across map directories.\n"
    )
    (analysis_dir / "README.md").write_text(
        "# Geometry coverage\n\n"
        "`coverage.csv` has one row per map and metric, plus `__pooled__` rows. Distances use "
        "finite values in meters; NaN values count as missing. Agent counts sum unique agent IDs "
        "within each episode, and frame counts use saved `frame_count`, then `joint_actions` "
        "length, then distinct `frame_idx` as a fallback.\n"
    )
    print(f"Wrote {image_path}")
    print(f"Wrote {analysis_dir / 'coverage.csv'}")


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-dir", type=Path, required=True,
                        help="Model run directory containing records/references/<map>/*.npz")
    args = parser.parse_args(argv)
    plot_geometry(args.model_dir)


if __name__ == "__main__":
    main()
