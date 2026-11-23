import hashlib
import json

import numpy as np

from project.soft_actor_critic_q_calibration.development.probe_returns import compute_probe_measurements
from project.soft_actor_critic_q_calibration.development.verify_binned_probes import (
    ACTION_IDS,
    AXIS_EDGES,
    DISTANCE_FIELDS,
    REQUIRED_EVAL_METRICS,
    _verify_probe_bundle,
    _verify_selection_states,
)


def _sha256(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _tag(axis, index):
    edges = AXIS_EDGES[axis]
    upper = edges[index + 1]
    return {
        "bin_index": index,
        "bin_id": f"{axis}:{index:02d}",
        "lower_m": edges[index],
        "upper_m": None if np.isinf(upper) else upper,
    }


def _reference_selection(tmp_path):
    records_dir = tmp_path / "source" / "records"
    references_dir = records_dir / "references"
    states = []
    summaries = {}
    for axis, edges in AXIS_EDGES.items():
        axis_summary = []
        for bin_index, (lower, upper) in enumerate(zip(edges[:-1], edges[1:])):
            episode_index = len(states)
            value = lower + 1.0 if np.isinf(upper) else (lower + upper) / 2.0
            distance_values = {
                field: np.nan for field in AXIS_EDGES
            }
            distance_values[axis] = value
            distance_values["current_goal_distance_m"] = 13.0
            map_dir = references_dir / "Town01"
            map_dir.mkdir(parents=True, exist_ok=True)
            shard_path = map_dir / f"episode_{episode_index:05d}.npz"
            state_id = f"Town01/episode_{episode_index:05d}/frame_000000/agent_{episode_index:04d}_slot_0000"
            arrays = {
                "frame_idx": np.array([0], dtype=np.int32),
                "agent_id": np.array([episode_index], dtype=np.int32),
                "slot": np.array([0], dtype=np.int32),
                "current_goal_idx": np.array([1], dtype=np.int32),
                "goal_count": np.array([3], dtype=np.int32),
                "eligible": np.array([True], dtype=np.bool_),
                "preaction_eligible": np.array([[True]], dtype=np.bool_),
                "fingerprints": np.array([f"scene-{episode_index}"], dtype="U64"),
                "observation_fingerprints": np.array([f"obs-{episode_index}"], dtype="U64"),
            }
            for field in DISTANCE_FIELDS:
                arrays[field] = np.array([distance_values[field]], dtype=np.float32)
            np.savez_compressed(shard_path, **arrays)
            sidecar_path = shard_path.with_suffix(".json")
            sidecar_path.write_text(json.dumps({
                "run_signature": "geometry-signature",
                "data_file": shard_path.name,
                "data_sha256": _sha256(shard_path),
                "map_index": 0,
                "map_name": "Town01",
                "map_path": "/maps/Town01.bin",
                "episode_index": episode_index,
            }))
            state = {
                "state_id": state_id,
                "scene_key": state_id,
                "assigned_bin": {"axis": axis, **_tag(axis, bin_index)},
                "bin_tags": {
                    tag_axis: (_tag(tag_axis, bin_index) if tag_axis == axis else {
                        "bin_index": None, "bin_id": None, "lower_m": None, "upper_m": None, "missing": True
                    })
                    for tag_axis in AXIS_EDGES
                },
                "map_index": 0,
                "map_name": "Town01",
                "map_path": "/maps/Town01.bin",
                "episode_index": episode_index,
                "frame_idx": 0,
                "agent_id": episode_index,
                "slot": 0,
                "current_goal_idx": 1,
                "goal_count": 3,
                "source_shard": f"references/Town01/{shard_path.name}",
                "source_metadata": f"references/Town01/{sidecar_path.name}",
                "fingerprint": f"scene-{episode_index}",
                "observation_fingerprint": f"obs-{episode_index}",
                "distances_m": {
                    name: (None if np.isnan(value) else float(value))
                    for name, value in distance_values.items()
                },
            }
            states.append(state)
            axis_summary.append({
                **_tag(axis, bin_index),
                "candidate_count": 2,
                "selected_count": 1,
                "shortage_count": 0,
            })
        summaries[axis] = axis_summary
    return records_dir, states, summaries


def test_selection_is_joined_to_unique_raw_rows_and_every_bin_is_counted(tmp_path):
    records_dir, states, summaries = _reference_selection(tmp_path)
    axes = {
        axis: {"edges_m": [None if np.isinf(edge) else edge for edge in edges]}
        for axis, edges in AXIS_EDGES.items()
    }
    selection = {
        "axes": axes,
        "requested_per_bin": 1,
        "requested_state_count": len(states),
        "selected_state_count": len(states),
        "states": states,
        "bin_summary": summaries,
        "shortages": [],
    }

    errors, state_by_id, counts, unique_count, _ = _verify_selection_states(
        selection,
        expected_per_bin=1,
        references_dir=records_dir / "references",
        geometry_signature="geometry-signature",
    )

    assert errors == []
    assert len(state_by_id) == 19
    assert unique_count == 19
    assert set(counts.values()) == {1}


def _complete_probe(tmp_path):
    state = {
        "state_id": "Town01/episode_00000/frame_000000/agent_0000_slot_0000",
        "source_shard": "references/Town01/episode_00000.npz",
        "fingerprint": "scene-0",
        "observation_fingerprint": "obs-0",
        "frame_idx": 0,
        "agent_id": 0,
        "slot": 0,
        "current_goal_idx": 0,
        "goal_count": 1,
        "distances_m": {
            next(iter(AXIS_EDGES)): 2.5,
            "nearest_observed_road_boundary_distance_m": None,
            "nearest_visible_goal_distance_m": None,
            "current_goal_distance_m": 10.0,
        },
        "assigned_bin": {
            "axis": next(iter(AXIS_EDGES)),
            "bin_index": 0,
            "bin_id": f"{next(iter(AXIS_EDGES))}:00",
            "lower_m": 0.0,
            "upper_m": 5.0,
        },
    }
    action_count = len(ACTION_IDS)
    shape = (2, action_count)
    arrays = {
        "rewards": np.ones(shape, dtype=np.float64),
        "log_probs": np.full(shape, -0.2, dtype=np.float64),
        "valid": np.ones(shape, dtype=np.bool_),
        "terminals": np.zeros(shape, dtype=np.bool_),
        "truncations": np.zeros(shape, dtype=np.bool_),
        "next_soft_values": np.full(shape, np.nan, dtype=np.float64),
        "goal_events": np.zeros(shape, dtype=np.bool_),
        "action_idx": np.asarray(ACTION_IDS, dtype=np.int32),
        "repeat_idx": np.zeros(action_count, dtype=np.int32),
        "continuation_seed": np.full(action_count, 917, dtype=np.int64),
        "task_id": np.asarray([f"task-{action}" for action in ACTION_IDS], dtype="U160"),
        "q1": np.arange(action_count, dtype=np.float64) + 0.1,
        "q2": np.arange(action_count, dtype=np.float64) + 0.2,
        "initial_ego_observation": np.array([0.0], dtype=np.float32),
    }
    arrays["terminals"][-1] = True
    arrays["event_collision"] = np.zeros(shape, dtype=np.bool_)
    arrays["event_at_fault_collision"] = np.zeros(shape, dtype=np.bool_)
    arrays["event_offroad"] = np.zeros(shape, dtype=np.bool_)
    arrays["event_red_light"] = np.zeros(shape, dtype=np.bool_)
    arrays["event_collision"][0, 2] = True
    arrays["goal_events"][0, 3] = True
    model = {"gamma": 0.99, "alpha": 0.2, "return_estimator": "pql", "pql_horizon": 2, "pql_lambda": 0.95}
    measurements = compute_probe_measurements(
        arrays["rewards"], arrays["log_probs"], arrays["valid"], arrays["terminals"],
        arrays["truncations"], arrays["next_soft_values"], **model,
    )
    metric_names = sorted(REQUIRED_EVAL_METRICS)
    metric_arrays = {name: f"eval_metrics__{name.replace('/', '__')}" for name in metric_names}
    metrics_mean = {name: 0.0 for name in metric_names}
    for array_name in metric_arrays.values():
        arrays[array_name] = np.zeros(action_count, dtype=np.float64)
    branches = []
    for action in ACTION_IDS:
        branch = {
            name: values[action].item()
            for name, values in measurements.items()
            if isinstance(values, np.ndarray) and values.ndim == 1
        }
        branch.update({
            "action_idx": action,
            "repeat_idx": 0,
            "continuation_seed": 917,
            "task_id": f"task-{action}",
            "q1": float(arrays["q1"][action]),
            "q2": float(arrays["q2"][action]),
            "q_min": float(min(arrays["q1"][action], arrays["q2"][action])),
            "terminal": True,
            "truncated": False,
            "validity_gap": False,
            "scene_boundary": False,
            "probe_horizon_reached": False,
            "ending_reason": "terminal",
            "partial_return": False,
            "goals_reached": int(np.count_nonzero(arrays["goal_events"][:, action])),
            "goal_reached": bool(np.any(arrays["goal_events"][:, action])),
            "collision": bool(np.any(arrays["event_collision"][:, action])),
            "at_fault_collision": False,
            "offroad": False,
            "red_light": False,
            "metrics_mean": dict(metrics_mean),
        })
        branches.append(branch)

    source_row = {
        "frame_idx": 0,
        "slot": 0,
        "agent_id": 0,
        "current_goal_idx": 0,
        "goal_count": 1,
        "fingerprint": state["fingerprint"],
        "observation_fingerprint": state["observation_fingerprint"],
        "distances_m": dict(state["distances_m"]),
    }

    bundle_dir = tmp_path / "probes"
    bundle_dir.mkdir()
    json_path = bundle_dir / "state.json"
    npz_path = bundle_dir / "state.npz"
    np.savez_compressed(npz_path, **arrays)
    metadata = {
        "status": "complete",
        "state_id": state["state_id"],
        "probe_id": state["state_id"],
        "data_file": npz_path.name,
        "data_sha256": _sha256(npz_path),
        "selection_state": state,
        "assigned_bin": state["assigned_bin"],
        "probe": {
            "frame_idx": 0,
            "agent_id": 0,
            "ego_idx": 0,
            "active_slot": 0,
            "reference_fingerprint": state["fingerprint"],
            "reference_observation_fingerprint": state["observation_fingerprint"],
        },
        "source": {
            "shard": state["source_shard"],
            "geometry_manifest_signature": "geometry-signature",
            "shard_sha256": "source-hash",
            "sidecar_sha256": "sidecar-hash",
            "sidecar": "/source/references/episode_00000.json",
            "source_geometry_row": source_row,
        },
        "source_row": source_row,
        "collection": {"continuation_horizon": 2560},
        "model": model,
        "metrics_schema": {"keys": metric_arrays},
        "branches": branches,
    }
    json_path.write_text(json.dumps(metadata))
    return json_path, state


def test_probe_bundle_reconstructs_targets_events_and_metric_arrays(tmp_path):
    json_path, state = _complete_probe(tmp_path)

    errors, counts = _verify_probe_bundle(
        json_path, state, state["state_id"],
        source_info={
            "shard_sha256": "source-hash",
            "sidecar_sha256": "sidecar-hash",
            "sidecar_path": "/source/references/episode_00000.json",
        },
        geometry_signature="geometry-signature",
    )

    assert errors == []
    assert counts["branch_count"] == 12
    assert counts["action_count"] == 12
    assert counts["repeat_count"] == 1
    assert counts["max_observed_horizon"] == 2

    metadata = json.loads(json_path.read_text())
    metadata["branches"][0]["discounted_reward_return"] += 1.0
    json_path.write_text(json.dumps(metadata))
    errors, _ = _verify_probe_bundle(
        json_path, state, state["state_id"],
        source_info={
            "shard_sha256": "source-hash",
            "sidecar_sha256": "sidecar-hash",
            "sidecar_path": "/source/references/episode_00000.json",
        },
        geometry_signature="geometry-signature",
    )
    assert any("discounted_reward_return" in error for error in errors)
