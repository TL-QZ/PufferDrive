import csv
import json
from types import SimpleNamespace

import numpy as np

from project.soft_actor_critic_q_calibration.development import summarize_probes
from project.soft_actor_critic_q_calibration.development.run_probes import (
    SAFETY_EVENTS,
    _assemble_probe_bundle,
    _run_branch_batch,
    observation_fingerprint,
    probe_paths,
    save_probe,
)


class _FakeRng:
    def random(self, size):
        return np.zeros(size, dtype=np.float64)


class _FakeModels:
    def __init__(self):
        self.action_table = np.zeros((12, 2), dtype=np.float32)
        self.gamma = 0.99
        self.alpha = 0.1

    def policy(self, observations):
        probabilities = np.zeros((len(observations), 12), dtype=np.float32)
        probabilities[:, 0] = 1.0
        return probabilities, np.zeros_like(probabilities)

    def q_values(self, observations):
        return np.zeros((len(observations), 12)), np.ones((len(observations), 12))

    def next_values(self, observations):
        return np.zeros(len(observations), dtype=np.float32)


class _FakeSceneBatch:
    instances = []

    @classmethod
    def from_template(cls, template, map_path, agent_count, copies, scene_seed):
        batch = cls(agent_count, copies)
        cls.instances.append(batch)
        return batch

    def __init__(self, agent_count, copies):
        self.agent_count = agent_count
        self.copies = copies
        self.action_table = np.zeros((12, 2), dtype=np.float32)
        self.observations = np.zeros((copies, agent_count, 1), dtype=np.float32)
        self.step_count = 0
        self.metric_calls = []

    def reset(self):
        self.step_count = 0
        return self.observations

    def replay_prefix(self, joint_actions, frame_idx, expected_fingerprint=None):
        assert joint_actions.shape == (frame_idx, self.agent_count)

    def fingerprint(self, copy_idx):
        return "same-reference"

    def set_continuation_seeds(self, seeds):
        assert len(seeds) == self.copies
        return [_FakeRng() for _ in seeds]

    def begin_metrics(self, ego_idx):
        assert ego_idx == 0
        return {"step_at_baseline": self.step_count}

    def step(self, actions):
        assert actions.shape == (self.copies, self.agent_count, 2)
        self.step_count += 1
        self.observations.fill(self.step_count)
        rewards = np.ones((self.copies, self.agent_count), dtype=np.float32)
        terminals = np.zeros((self.copies, self.agent_count), dtype=np.bool_)
        truncations = np.zeros_like(terminals)
        eligible = np.ones_like(terminals)
        events = {
            name: np.zeros_like(terminals)
            for name in (*SAFETY_EVENTS, "reached_goal")
        }
        if self.step_count == 1:
            terminals[0, 0] = True
            events["collision"][0, 0] = True
            events["offroad"][1, 0] = True
            eligible[0, 0] = False
        elif self.step_count == 2:
            truncations[1, 0] = True
            eligible[1, 0] = False
            # This occurs after copy 0 has ended and must not enter its interval flags.
            events["red_light"][0, 0] = True
        else:
            raise AssertionError("fake branch ran beyond its expected completion")
        metadata = {
            "eligible": eligible,
            "scene_done": np.zeros(self.copies, dtype=np.bool_),
        }
        return {
            "observations": self.observations.copy(),
            "rewards": rewards,
            "terminals": terminals,
            "truncations": truncations,
            "events": events,
            "metadata": metadata,
        }

    def metrics(self, baseline, event_flags):
        assert baseline == {"step_at_baseline": 0}
        flags = np.asarray(event_flags).copy()
        self.metric_calls.append((self.step_count, flags))
        return [
            {"episode_timestep": float(self.step_count), "score": float(copy_idx)}
            for copy_idx in range(self.copies)
        ]

    def close(self):
        pass


def _branch_reference(agent_count=2):
    initial_observations = np.zeros((agent_count, 1), dtype=np.float32)
    return {
        "joint_actions": np.zeros((0, agent_count), dtype=np.int32),
        "fingerprints": ["same-reference"],
        "observation_fingerprints": [observation_fingerprint(initial_observations)],
    }


def test_branch_metrics_are_snapshotted_at_each_branch_completion():
    _FakeSceneBatch.instances.clear()
    task_records = _run_branch_batch(
        _FakeSceneBatch,
        SimpleNamespace(probe_config={"scenario_length": 4}),
        "map.bin",
        2,
        1,
        _branch_reference(),
        {"frame_idx": 0, "ego_idx": 0},
        [
            {"action_idx": 0, "repeat_idx": 0, "continuation_seed": 1, "task_id": "early"},
            {"action_idx": 1, "repeat_idx": 0, "continuation_seed": 1, "task_id": "late"},
        ],
        _FakeModels(),
        pql_horizon=8,
        record_horizons=(1, 2),
    )

    batch = _FakeSceneBatch.instances[-1]
    np.testing.assert_array_equal(task_records[0]["initial_ego_observation"], [0.0])
    assert batch.observations[0, 0, 0] == 2
    assert [step for step, _ in batch.metric_calls] == [1, 2]
    np.testing.assert_array_equal(batch.metric_calls[0][1], [[True, False, False, False], [False, False, True, False]])
    np.testing.assert_array_equal(batch.metric_calls[1][1], [[True, False, False, False], [False, False, True, False]])
    assert task_records[0]["metrics_mean"]["episode_timestep"] == 1.0
    assert task_records[1]["metrics_mean"]["episode_timestep"] == 2.0
    assert task_records[0]["horizon_metrics"]["1"]["discounted_reward_return"] == 1.0
    assert "2" not in task_records[0]["horizon_metrics"]
    assert task_records[1]["horizon_metrics"]["1"]["discounted_reward_return"] == 1.0
    assert task_records[1]["horizon_metrics"]["2"]["discounted_reward_return"] == 1.99
    assert task_records[1]["horizon_metrics"]["2"]["standard_soft_return"] == 1.99
    assert all(record["metrics_agents_per_batch"] == 4 for record in task_records)


def test_probe_bundle_stores_scalar_metrics_in_json_and_npz(tmp_path):
    metric_values = {"driving/score": 0.75, "average_speed": 3.5}
    record = {
        "task": {
            "action_idx": 2,
            "repeat_idx": 0,
            "continuation_seed": 17,
            "task_id": "probe/repeat_000/action_02",
        },
        "valid": [True],
        "rewards": [1.0],
        "log_probs": [-0.25],
        "terminals": [True],
        "truncations": [False],
        "next_soft_values": [np.nan],
        "events": {name: [False] for name in SAFETY_EVENTS},
        "goals": [False],
        "validity_gap": False,
        "scene_boundary": False,
        "initial_ego_observation": np.array([2.5, 0.0], dtype=np.float32),
        "initial_q1": 1.0,
        "initial_q2": 2.0,
        "metrics_mean": metric_values,
        "metrics_agents_per_batch": 4,
    }
    models = SimpleNamespace(
        gamma=0.99,
        alpha=0.1,
        return_estimator="n_step",
        audit={"checkpoint": "checkpoint.pt", "checkpoint_sha256": "hash"},
    )
    reference = {
        "metadata_series": {},
        "fingerprints": ["reference"],
        "observation_fingerprints": ["observation"],
        "frame_count": 1,
        "boundary_captured": True,
        "initial_metadata": {},
    }
    probe = {"frame_idx": 0, "ego_idx": 0}

    metadata, arrays = _assemble_probe_bundle(
        "episode_000/frame_00000/ego_0000",
        0,
        "map.bin",
        "map-hash",
        3,
        reference,
        probe,
        2,
        [record],
        models,
        lambda *args, **kwargs: {"terminal": np.array([True])},
        pql_horizon=8,
        pql_lambda=0.95,
        num_goals=1,
    )
    save_probe(tmp_path, metadata["probe_id"], "signature", metadata, arrays)

    metadata_path, data_path = probe_paths(tmp_path, metadata["probe_id"])
    stored = json.loads(metadata_path.read_text())
    with np.load(data_path) as data:
        np.testing.assert_array_equal(data["initial_ego_observation"], record["initial_ego_observation"])
    stored_metrics = stored["branches"][0]["metrics_mean"]
    assert stored_metrics == metric_values
    assert all(not isinstance(value, list) for value in stored_metrics.values())
    assert stored["metrics_scope"]["n"] == 1
    assert stored["metrics_scope"]["active_agent_count"] == 2
    assert stored["metrics_scope"]["agents_per_batch"] == [4]
    assert stored["metrics_schema"]["keys"] == {
        "average_speed": "eval_metrics__average_speed",
        "driving/score": "eval_metrics__driving__score",
    }
    with np.load(data_path) as data:
        assert data["eval_metrics__driving__score"].shape == (1,)
        assert data["eval_metrics__driving__score"].dtype == np.float64
        assert data["eval_metrics__driving__score"].tolist() == [0.75]
        assert data["eval_metrics__average_speed"].tolist() == [3.5]


def test_round_four_bundle_keeps_empty_horizon_metrics():
    metric_values = {"episode_timestep": 1.0}
    record = {
        "task": {
            "action_idx": 2,
            "repeat_idx": 0,
            "continuation_seed": 17,
            "task_id": "probe/repeat_000/action_02",
        },
        "valid": [True],
        "rewards": [1.0],
        "log_probs": [-0.25],
        "terminals": [True],
        "truncations": [False],
        "next_soft_values": [np.nan],
        "events": {name: [False] for name in SAFETY_EVENTS},
        "goals": [False],
        "validity_gap": False,
        "scene_boundary": False,
        "initial_ego_observation": np.array([2.5, 0.0], dtype=np.float32),
        "initial_q1": 1.0,
        "initial_q2": 2.0,
        "metrics_mean": metric_values,
        "metrics_agents_per_batch": 4,
        "horizon_metrics": {},
    }
    models = SimpleNamespace(
        gamma=0.99,
        alpha=0.1,
        return_estimator="n_step",
        audit={"checkpoint": "checkpoint.pt", "checkpoint_sha256": "hash"},
    )
    reference = {
        "metadata_series": {},
        "fingerprints": ["reference"],
        "observation_fingerprints": ["observation"],
        "frame_count": 1,
        "boundary_captured": True,
        "initial_metadata": {},
    }
    metadata, _ = _assemble_probe_bundle(
        "episode_000/frame_00000/ego_0000", 0, "map.bin", "map-hash", 3,
        reference, {"frame_idx": 0, "ego_idx": 0}, 2, [record], models,
        lambda *args, **kwargs: {"terminal": np.array([True])},
        pql_horizon=8, pql_lambda=0.95, num_goals=1, record_horizons=(8, 32),
    )
    assert metadata["branches"][0]["horizon_metrics"] == {}


def test_probe_bundle_labels_explicit_horizon_truncation():
    record = {
        "task": {
            "action_idx": 0,
            "repeat_idx": 0,
            "continuation_seed": 19,
            "task_id": "probe/repeat_000/action_00",
        },
        "valid": [True],
        "rewards": [0.5],
        "log_probs": [-0.1],
        "terminals": [False],
        "truncations": [True],
        "next_soft_values": [0.0],
        "events": {name: [False] for name in SAFETY_EVENTS},
        "goals": [False],
        "validity_gap": False,
        "scene_boundary": False,
        "probe_horizon_reached": True,
        "initial_ego_observation": np.array([1.0], dtype=np.float32),
        "initial_q1": 1.0,
        "initial_q2": 2.0,
        "metrics_mean": {"average_speed": 3.0},
        "metrics_agents_per_batch": 12,
    }
    models = SimpleNamespace(
        gamma=0.99,
        alpha=0.1,
        return_estimator="n_step",
        audit={"checkpoint": "checkpoint.pt", "checkpoint_sha256": "hash"},
    )
    reference = {
        "metadata_series": {},
        "fingerprints": ["reference"],
        "observation_fingerprints": ["observation"],
        "frame_count": 1,
        "boundary_captured": False,
        "initial_metadata": {},
    }

    metadata, _ = _assemble_probe_bundle(
        "probe",
        0,
        "map.bin",
        "map-hash",
        3,
        reference,
        {"frame_idx": 0, "ego_idx": 0},
        12,
        [record],
        models,
        lambda *args, **kwargs: {
            "terminal": np.array([False]),
            "truncated": np.array([True]),
        },
        pql_horizon=8,
        pql_lambda=0.95,
        num_goals=1,
    )

    assert metadata["branches"][0]["ending_reason"] == "probe_horizon"
    assert metadata["branches"][0]["probe_horizon_reached"] is True


def test_summary_adds_dynamic_metrics_and_keeps_legacy_fields_blank(tmp_path, monkeypatch):
    records_dir = tmp_path / "records"
    probes_dir = records_dir / "probes"
    probes_dir.mkdir(parents=True)
    legacy_path = probes_dir / "a_legacy.json"
    legacy_probe = {
        "status": "complete",
        "branches": [{"action_idx": 0, "repeat_idx": 0, "q_min": 1.0}],
    }
    legacy_path.write_text(json.dumps(legacy_probe))
    (probes_dir / "z_new.json").write_text(json.dumps({
        "status": "complete",
        "branches": [
            {"action_idx": 0, "repeat_idx": 0, "q_min": 2.0,
             "metrics_mean": {"route/success": 0.25, "score": 1.0}},
            {"action_idx": 0, "repeat_idx": 1, "q_min": 3.0,
             "metrics_mean": {"route/success": 0.75, "score": 0.0}},
        ],
    }))
    monkeypatch.setattr(summarize_probes, "records_directory", lambda _: records_dir)

    summary = summarize_probes.summarize(tmp_path)

    assert summary["probe_count"] == 2
    with (records_dir / "action_summary.csv").open(newline="") as stream:
        rows = {row["probe_id"]: row for row in csv.DictReader(stream)}
    success_mean = "eval_metrics/route/success_mean"
    success_sem = "eval_metrics/route/success_sem"
    success_count = "eval_metrics/route/success_count"
    assert success_mean in rows["z_new"]
    assert rows["z_new"][success_mean] == "0.5"
    assert rows["z_new"][success_sem] == "0.25"
    assert rows["z_new"][success_count] == "2"
    assert rows["a_legacy"][success_mean] == ""
    assert rows["a_legacy"][success_count] == ""
    assert json.loads(legacy_path.read_text()) == legacy_probe
