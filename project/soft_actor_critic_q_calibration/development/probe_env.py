"""Deterministic same-scene self-play branching over a private Drive binding.

This module deliberately does not change Drive's production binding. Call
``create_template`` once from saved Drive kwargs, then create one native scene
copy per branch with :meth:`SceneBatch.from_template`.
"""

from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass
import hashlib
import inspect
from pathlib import Path
import sys
from types import ModuleType
from typing import Any

import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parents[3]
NATIVE_DIR = Path(__file__).resolve().parent / "native_probe"
METRIC_EVENT_NAMES = ("collision", "at_fault_collision", "offroad", "red_light")
_private_binding: ModuleType | None = None


def _as_drive_kwargs(saved_env_config: dict[str, Any]) -> dict[str, Any]:
    """Apply Drive defaults while rejecting non-Drive config keys."""
    from pufferlib.ocean.drive.drive import Drive

    signature = inspect.signature(Drive.__init__)
    names = {name for name in signature.parameters if name != "self"}
    unknown = sorted(set(saved_env_config) - names)
    if unknown:
        raise ValueError(f"Unknown Drive environment options: {unknown}")
    defaults = {
        name: parameter.default
        for name, parameter in signature.parameters.items()
        if name != "self" and parameter.default is not inspect.Parameter.empty
    }
    defaults.update(dict(saved_env_config))
    return defaults


def _map_files(map_dir: str | Path, num_maps: int) -> list[str]:
    path = Path(map_dir).expanduser().resolve()
    if path.is_file() and path.suffix == ".bin":
        files = [str(path)]
    elif path.is_dir():
        files = sorted(str(item.resolve()) for item in path.glob("*.bin"))
    else:
        raise FileNotFoundError(path)
    if not files:
        raise FileNotFoundError(f"No .bin maps found in {path}")
    if num_maps < 1 or num_maps > len(files):
        raise ValueError(f"num_maps={num_maps} outside [1, {len(files)}] for {path}")
    return files


def _enum_value(value: Any, choices: dict[str, int], name: str) -> int:
    if isinstance(value, str):
        if value not in choices:
            raise ValueError(f"Unsupported {name}: {value!r}")
        return choices[value]
    return int(value)


def _sample_scene_layout(
    config: dict[str, Any], seed: int
) -> tuple[list[str], str, int, int, dict[str, int]]:
    """Replay Drive's first training-layout draw without allocating every scene."""
    from pufferlib.ocean.drive import binding

    files = _map_files(config["map_dir"], int(config["num_maps"]))
    rng = np.random.default_rng(seed)
    shared_seed = int(rng.integers(0, 2**63, dtype=np.int64))
    mode = _enum_value(
        config["simulation_mode"],
        {"gigaflow": binding.SIMULATION_MODE_GIGAFLOW, "replay": binding.SIMULATION_MODE_REPLAY},
        "simulation_mode",
    )
    init_mode = _enum_value(
        config["init_mode"],
        {"create_all_valid": binding.INIT_MODE_CREATE_ALL_VALID,
         "create_only_controlled": binding.INIT_MODE_CREATE_ONLY_CONTROLLED},
        "init_mode",
    )
    control_mode = _enum_value(
        config["control_mode"],
        {"control_vehicles": binding.CONTROL_MODE_VEHICLES,
         "control_agents": binding.CONTROL_MODE_AGENTS,
         "control_wosac": binding.CONTROL_MODE_WOSAC,
         "control_sdc_only": binding.CONTROL_MODE_SDC_ONLY},
        "control_mode",
    )
    controller_options = {
        "static": binding.CONTROLLER_STATIC,
        "policy": binding.CONTROLLER_POLICY,
        "replay": binding.CONTROLLER_REPLAY,
        "idm": binding.CONTROLLER_IDM,
    }
    sdc_controller = _enum_value(config["sdc_controller"], controller_options, "sdc_controller")
    non_sdc_controller = _enum_value(
        config["non_sdc_controller"], controller_options, "non_sdc_controller"
    )
    non_vehicle_name = config["non_vehicle_controller"]
    if non_vehicle_name == "auto":
        non_vehicle_name = "replay" if config["non_sdc_controller"] == "idm" else config["non_sdc_controller"]
    non_vehicle_controller = _enum_value(non_vehicle_name, controller_options, "non_vehicle_controller")

    shared_kwargs = {
        "map_files": files,
        "num_agents": int(config["num_agents"]),
        "num_maps": int(config["num_maps"]),
        "starting_map_counter": int(config["starting_map"]),
        "eval_mode": 0,
        "init_mode": init_mode,
        "control_mode": control_mode,
        "sdc_controller": sdc_controller,
        "non_sdc_controller": non_sdc_controller,
        "non_vehicle_controller": non_vehicle_controller,
        "simulation_mode": mode,
        "init_step": int(config["init_step"]),
        "init_step_spread": bool(config["init_step_spread"]),
        "resample_replay_to_dt": bool(config["resample_replay_to_dt"]),
        "dt": float(config["dt"]),
        "scenario_length": int(config["scenario_length"] or 0),
        "seed": shared_seed,
        "min_agents_per_env": int(config["min_agents_per_env"]),
        "max_agents_per_env": int(config["max_agents_per_env"]),
        "num_eval_scenarios": int(config["num_eval_scenarios"]),
        "eval_map_indices": config["eval_map_indices"],
        "goal_radius": float(config["goal_radius"]),
    }
    offsets, map_ids, scene_count = binding.shared(**shared_kwargs)
    if scene_count < 1:
        raise RuntimeError("Drive training layout produced no scenes")
    agent_count = int(offsets[1]) - int(offsets[0])
    if agent_count < 1:
        raise RuntimeError("Drive training layout produced an empty first scene")
    map_agent_counts: dict[str, int] = {}
    for scene_idx in range(scene_count):
        scene_map = files[int(map_ids[scene_idx])]
        scene_agent_count = int(offsets[scene_idx + 1]) - int(offsets[scene_idx])
        if scene_agent_count > 0:
            map_agent_counts.setdefault(scene_map, scene_agent_count)

    # Drive consumes one Python RNG seed per native scene before sampling that
    # scene's optional replay start frame. Reproduce just the first-scene draws.
    first_native_seed = int(rng.integers(0, 2**63, dtype=np.int64))
    init_step = int(config["init_step"])
    if config["init_step_spread"]:
        upper = int(config["scenario_length"]) - int(config["init_step_min_horizon"])
        if upper <= 0:
            raise ValueError("init_step_spread leaves no valid initial frame")
        init_step = int(rng.integers(0, upper))
    return files, files[int(map_ids[0])], agent_count, init_step, map_agent_counts


def create_template(saved_env_config: dict[str, Any], seed: int):
    """Create a small Drive model template for a sampled training scene.

    The returned Drive owns one ordinary production scene and must be closed by
    the caller. The private branch copies are created separately by
    ``SceneBatch`` and preserve the saved ``eval_mode`` and termination rules.
    """
    from pufferlib.ocean.drive.drive import Drive

    config = _as_drive_kwargs(saved_env_config)
    config["seed"] = int(seed)
    config["eval_mode"] = 0
    files, map_path, agent_count, init_step, map_agent_counts = _sample_scene_layout(config, int(seed))

    template_config = dict(config)
    template_config.update(
        {
            "map_dir": map_path,
            "num_maps": 1,
            "num_agents": agent_count,
            "min_agents_per_env": agent_count,
            "max_agents_per_env": int(config["max_agents_per_env"]),
            "starting_map": 0,
            "init_step": init_step,
            # The first scene's start draw above is already fixed. Keep the
            # saved native flag below when building branch env kwargs.
            "init_step_spread": False,
        }
    )
    template = Drive(**template_config)
    template.model_env = template
    template.map_paths = tuple(
        scene_map for scene_map in files[: int(config["num_maps"])] if scene_map in map_agent_counts
    )
    template.template_map_path = map_path
    template.template_agent_count = agent_count
    template.map_agent_counts = map_agent_counts
    template.template_seed = int(seed)
    template.original_init_step_spread = bool(config["init_step_spread"])
    template.probe_env_kwargs = template._env_init_kwargs(map_path, agent_count)
    template.probe_env_kwargs["init_step_spread"] = int(bool(config["init_step_spread"]))
    template.probe_env_kwargs["init_step"] = init_step
    template.probe_env_kwargs["max_agents_per_env"] = int(config["max_agents_per_env"])
    template.probe_config = config
    template.initial_native_seed = int(seed)
    return template


def _load_private_binding(force_rebuild: bool = False) -> ModuleType:
    global _private_binding
    if force_rebuild and "probe_binding" in sys.modules:
        raise RuntimeError("Cannot force-rebuild probe_binding after it was loaded in this process")
    from .native_probe.build_private_binding import build_private_binding

    extension_path = build_private_binding(force_rebuild=force_rebuild)
    module = sys.modules.get("probe_binding")
    if module is None:
        import importlib.util

        spec = importlib.util.spec_from_file_location("probe_binding", extension_path)
        if spec is None or spec.loader is None:
            raise ImportError(f"Cannot import private Drive binding at {extension_path}")
        module = importlib.util.module_from_spec(spec)
        sys.modules["probe_binding"] = module
        try:
            spec.loader.exec_module(module)
        except Exception:
            sys.modules.pop("probe_binding", None)
            raise
    required = (
        "vec_set_probe_mode", "vec_get_probe_metadata", "vec_get_probe_fingerprints",
        "vec_start_probe_horizon",
        "vec_begin_probe_metrics", "vec_get_probe_metrics",
    )
    missing = [name for name in required if not hasattr(module, name)]
    if missing:
        raise RuntimeError(f"Private Drive binding is missing probe methods: {missing}")
    _private_binding = module
    return module


def ensure_private_binding(force_rebuild: bool = False) -> ModuleType:
    """Explicitly build/load the isolated native binding used by SceneBatch."""
    global _private_binding
    if _private_binding is not None and not force_rebuild:
        return _private_binding
    return _load_private_binding(force_rebuild=force_rebuild)


def native_build_fingerprint() -> str:
    """Return the source digest used to build the isolated native extension."""
    from .native_probe.build_private_binding import source_fingerprint

    return source_fingerprint()


def map_sha256(map_path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(map_path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _jerk_action_table() -> np.ndarray:
    from pufferlib.ocean.drive import binding

    long_values = np.asarray(binding.JERK_LONG, dtype=np.float32)
    lateral_values = np.asarray(binding.JERK_LAT, dtype=np.float32)
    long_normalized = np.where(long_values < 0.0, long_values / -long_values[0], long_values / long_values[-1])
    lateral_normalized = lateral_values / lateral_values[-1]
    return np.stack(
        [
            long_normalized[np.repeat(np.arange(long_values.size), lateral_values.size)],
            np.tile(lateral_normalized, long_values.size),
        ],
        axis=-1,
    ).astype(np.float32)


@dataclass
class SceneBatch:
    """A vector of identical native scene copies with a stable agent layout."""

    template: Any
    map_path: str
    agent_count: int
    copies: int
    scene_seed: int

    @classmethod
    def from_template(
        cls, template, map_path: str | Path, agent_count: int, copies: int, scene_seed: int
    ) -> "SceneBatch":
        if copies < 1 or agent_count < 1:
            raise ValueError("copies and agent_count must be positive")
        return cls(template, str(Path(map_path).resolve()), int(agent_count), int(copies), int(scene_seed))._initialize()

    def _initialize(self) -> "SceneBatch":
        from pufferlib.ocean.drive import drive as drive_module

        self.binding = ensure_private_binding()
        if self.template.probe_config["action_type"] != "continuous":
            raise ValueError("SceneBatch requires saved Drive action_type='continuous'")
        if self.template.probe_config["dynamics_model"] != "jerk":
            raise ValueError("SceneBatch currently supports the twelve jerk actions only")
        self.num_obs = int(self.template.num_obs)
        total_agents = self.copies * self.agent_count
        self.observations = np.zeros((self.copies, self.agent_count, self.num_obs), dtype=np.float32)
        self.actions = np.zeros((self.copies, self.agent_count, 2), dtype=np.float32)
        self.rewards = np.zeros((self.copies, self.agent_count), dtype=np.float32)
        self.terminals = np.zeros((self.copies, self.agent_count), dtype=np.uint8)
        self.truncations = np.zeros((self.copies, self.agent_count), dtype=np.uint8)
        self.masks = np.zeros((self.copies, self.agent_count), dtype=np.uint8)
        self.final_observations = np.zeros_like(self.observations)
        self.final_observation_valid = np.zeros((self.copies, self.agent_count), dtype=np.uint8)
        self.action_table = _jerk_action_table()
        self._handles = []
        base_kwargs = deepcopy(self.template.probe_env_kwargs)
        base_kwargs.update(
            {
                "map_file": self.map_path,
                "max_agents": self.agent_count,
                "max_agents_per_env": int(self.template.probe_config["max_agents_per_env"]),
                "capture_final_observations": True,
                "eval_mode": int(self.template.probe_config["eval_mode"]),
                "compute_eval_metrics": int(self.template.probe_config["compute_eval_metrics"]),
            }
        )
        try:
            for copy_idx in range(self.copies):
                env_kwargs = dict(base_kwargs)
                env_kwargs["final_observations"] = self.final_observations[copy_idx]
                env_kwargs["final_observation_valid"] = self.final_observation_valid[copy_idx]
                self._handles.append(
                    self.binding.env_init(
                        self.observations[copy_idx],
                        self.actions[copy_idx],
                        self.rewards[copy_idx],
                        self.terminals[copy_idx],
                        self.truncations[copy_idx],
                        self.masks[copy_idx],
                        self.scene_seed,
                        **env_kwargs,
                    )
                )
            self.c_envs = self.binding.vectorize(*self._handles)
            self.binding.vec_set_probe_mode(self.c_envs, 1, 1)
            self.reset()
            active_counts = self.metadata()["scene_active_agent_count"]
            if np.any(active_counts != self.agent_count):
                raise RuntimeError(
                    "Native scene active-agent count differs from the sampled Drive layout: "
                    f"expected {self.agent_count}, got {active_counts.tolist()}"
                )
        except Exception:
            if getattr(self, "c_envs", None) is not None:
                self.binding.vec_close(self.c_envs)
                self.c_envs = None
            else:
                for handle in self._handles:
                    self.binding.env_close(handle)
            self._handles.clear()
            raise
        return self

    def reset(self) -> np.ndarray:
        if getattr(self, "c_envs", None) is None:
            raise RuntimeError("SceneBatch is closed")
        self.binding.vec_reset(self.c_envs, [self.scene_seed] * self.copies)
        self._verified_prefix_fingerprint = None
        self._scene_done = np.zeros(self.copies, dtype=bool)
        self.rewards.fill(0)
        self.terminals.fill(0)
        self.truncations.fill(0)
        self.final_observation_valid.fill(0)
        return self.observations

    def metadata(self, *, include_traffic: bool = True) -> dict[str, np.ndarray]:
        if getattr(self, "c_envs", None) is None:
            raise RuntimeError("Cannot query a closed SceneBatch")
        flat = self.binding.vec_get_probe_metadata(self.c_envs, include_traffic)
        result: dict[str, np.ndarray] = {}
        for name, values in flat.items():
            array = np.asarray(values)
            if name in {"timestep", "episode_seed", "scene_done", "scene_active_agent_count"}:
                result[name] = array.reshape(self.copies)
            elif name in {"pose_xyz", "goal_xyz"}:
                result[name] = array.reshape(self.copies, self.agent_count, 3)
            elif name == "reward_coefficients":
                result[name] = array.reshape(self.copies, self.agent_count, -1)
            elif name in {
                "traffic_control_type",
                "traffic_control_state_size",
                "traffic_control_pose",
                "traffic_control_state",
            }:
                result[name] = array.reshape(self.copies, *array.shape[1:])
            elif name in {"traffic_control_count", "traffic_control_max_state_size"}:
                result[name] = array.reshape(self.copies)
            else:
                result[name] = array.reshape(self.copies, self.agent_count)
        result["eligible"] = (
            result["active"].astype(bool)
            & result["sim_valid"].astype(bool)
            & ~result["stopped"].astype(bool)
            & ~result["removed"].astype(bool)
        )
        return result

    def begin_metrics(self, ego_idx: int):
        """Snapshot cumulative counters before the forced action, without resetting them."""
        if getattr(self, "c_envs", None) is None:
            raise RuntimeError("Cannot query a closed SceneBatch")
        if not 0 <= ego_idx < self.agent_count:
            raise IndexError(f"ego_idx={ego_idx} outside [0, {self.agent_count})")
        return self.binding.vec_begin_probe_metrics(self.c_envs, ego_idx)

    def start_probe_horizon(self, steps: int) -> None:
        """Extend a fingerprint-verified gigaflow prefix before starting probe metrics."""
        if getattr(self, "c_envs", None) is None:
            raise RuntimeError("Cannot extend a closed SceneBatch")
        if isinstance(steps, (bool, np.bool_)) or not isinstance(steps, (int, np.integer)):
            raise TypeError("Probe horizon steps must be an integer")
        steps = int(steps)
        if steps <= 0:
            raise ValueError("Probe horizon steps must be positive")
        expected_fingerprint = getattr(self, "_verified_prefix_fingerprint", None)
        if not isinstance(expected_fingerprint, str) or len(expected_fingerprint) != 64:
            raise RuntimeError("Call replay_prefix with a verified fingerprint before starting the probe horizon")
        words = np.asarray(self.binding.vec_get_probe_fingerprints(self.c_envs), dtype=np.uint64)
        if words.shape != (self.copies, 4):
            raise RuntimeError(f"Unexpected native fingerprint shape: {words.shape}")
        fingerprints = ["".join(f"{int(word):016x}" for word in row) for row in words]
        if any(fingerprint != expected_fingerprint for fingerprint in fingerprints):
            raise AssertionError("Probe scene fingerprint changed after prefix verification")
        self.binding.vec_start_probe_horizon(self.c_envs, steps)

    def metrics(self, baseline, event_flags: np.ndarray) -> list[dict[str, float]]:
        """Read interval metrics; flags are collision, at-fault, offroad, red-light."""
        if getattr(self, "c_envs", None) is None:
            raise RuntimeError("Cannot query a closed SceneBatch")
        flags = np.asarray(event_flags)
        expected_shape = (self.copies, len(METRIC_EVENT_NAMES))
        if flags.dtype != np.bool_ or flags.shape != expected_shape:
            raise ValueError(f"Expected boolean event flags with shape {expected_shape}")
        return self.binding.vec_get_probe_metrics(
            self.c_envs, baseline, np.ascontiguousarray(flags)
        )

    def fingerprint(self, copy_idx: int) -> str:
        if not 0 <= copy_idx < self.copies:
            raise IndexError(f"copy_idx={copy_idx} outside [0, {self.copies})")
        words = np.asarray(self.binding.vec_get_probe_fingerprints(self.c_envs), dtype=np.uint64)
        if words.shape != (self.copies, 4):
            raise RuntimeError(f"Unexpected native fingerprint shape: {words.shape}")
        return "".join(f"{int(word):016x}" for word in words[copy_idx])

    def step(self, normalized_actions: np.ndarray, *, compact_metadata: bool = False) -> dict[str, Any]:
        actions = np.asarray(normalized_actions, dtype=np.float32)
        expected_shape = (self.copies, self.agent_count, 2)
        if actions.shape != expected_shape:
            raise ValueError(f"Expected normalized actions with shape {expected_shape}, got {actions.shape}")
        if not np.all(np.isfinite(actions)):
            raise ValueError("Actions contain NaN or infinity")
        if np.any(actions < -1.0) or np.any(actions > 1.0):
            raise ValueError("Normalized actions must be in [-1, 1]")
        done_before = self._scene_done.copy()
        self.actions[:] = actions
        self.final_observation_valid.fill(0)
        self.binding.vec_step(self.c_envs)
        metadata = self.metadata(include_traffic=not compact_metadata)
        self._scene_done = np.asarray(metadata["scene_done"], dtype=bool)
        output_observations = self.observations.copy()
        valid_final = self.final_observation_valid.astype(bool)
        output_observations[valid_final] = self.final_observations[valid_final]
        event_names = ("collision", "at_fault_collision", "offroad", "red_light", "reached_goal")
        events = {name: metadata[name].astype(bool).copy() for name in event_names}
        if np.any(done_before):
            for values in events.values():
                values[done_before] = False
        return {
            "observations": output_observations,
            "rewards": self.rewards.copy(),
            "terminals": self.terminals.astype(bool).copy(),
            "truncations": self.truncations.astype(bool).copy(),
            "events": events,
            "metadata": metadata,
        }

    def replay_prefix(
        self,
        joint_actions: np.ndarray,
        frame_idx: int,
        expected_fingerprint: str | None = None,
        *,
        compact: bool = False,
    ) -> list[str]:
        prefix = np.asarray(joint_actions, dtype=np.int32)
        if frame_idx < 0 or prefix.shape != (frame_idx, self.agent_count):
            raise ValueError(
                f"Expected discrete prefix shape ({frame_idx}, {self.agent_count}), got {prefix.shape}"
            )
        if np.any(prefix < 0) or np.any(prefix >= len(self.action_table)):
            raise ValueError("Prefix actions must be discrete jerk action classes in [0, 12)")
        if compact and expected_fingerprint is None:
            raise ValueError("Compact prefix replay requires the recorded pre-action fingerprint")
        self.reset()
        if compact:
            # Only the endpoint is consumed. The native binding holds completed scenes;
            # a premature boundary therefore fails the endpoint checks.
            for action_indices in prefix:
                self.actions[:] = self.action_table[action_indices]
                self.final_observation_valid.fill(0)
                self.binding.vec_step(self.c_envs)
            self._scene_done = self.metadata(include_traffic=False)["scene_done"].astype(bool)
            if np.any(self._scene_done):
                raise RuntimeError("Recorded pre-action prefix ended in a held scene")
        else:
            for prefix_step, action_indices in enumerate(prefix):
                result = self.step(self.action_table[action_indices][None, ...].repeat(self.copies, axis=0))
                if prefix_step + 1 < frame_idx and np.any(result["metadata"]["scene_done"]):
                    raise RuntimeError(f"Joint-action prefix crosses the original scene boundary at frame {prefix_step + 1}")
        fingerprints = [self.fingerprint(copy_idx) for copy_idx in range(self.copies)]
        if len(set(fingerprints)) != 1:
            raise AssertionError("Deterministic prefix replay diverged across native scene copies")
        if expected_fingerprint is not None and any(value != expected_fingerprint for value in fingerprints):
            raise AssertionError("Replayed scene fingerprint does not match the reference scene")
        self._verified_prefix_fingerprint = expected_fingerprint
        return fingerprints

    def set_continuation_seeds(self, seeds: list[int] | np.ndarray) -> list[np.random.Generator]:
        values = np.asarray(seeds, dtype=np.int64)
        if values.shape != (self.copies,) or np.any(values < 0):
            raise ValueError(f"Expected {self.copies} non-negative continuation seeds")
        self.continuation_seeds = values.copy()
        self.policy_rngs = [np.random.default_rng(int(seed)) for seed in values]
        return self.policy_rngs

    def close(self) -> None:
        if getattr(self, "c_envs", None) is not None:
            self.binding.vec_close(self.c_envs)
            self.c_envs = None

    def __enter__(self) -> "SceneBatch":
        return self

    def __exit__(self, exc_type, exc, traceback) -> None:
        self.close()

