"""Condition B teacher configuration and checkpoint loading.

The teacher is an existing PufferDrive ``Drive`` policy.  This module keeps
the configuration boundary separate from the simulator: resolving a saved
configuration does not construct an environment, and loading a checkpoint
can construct the policy from the small set of environment attributes the
network reads.
"""

import hashlib
from copy import deepcopy
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Mapping, Optional, Union

import torch
import torch.nn as nn


PathLike = Union[str, Path]
TeacherConfig = Mapping[str, Any]

_REPO_ROOT = Path(__file__).resolve().parents[2]


def _as_plain_mapping(value: Any, *, name: str) -> dict[str, Any]:
    if not isinstance(value, Mapping):
        raise ValueError(f"{name} must be a mapping, got {type(value).__name__}")
    return deepcopy(dict(value))


def _resolve_repo_path(path: PathLike) -> Path:
    candidate = Path(path).expanduser()
    if not candidate.is_absolute():
        candidate = _REPO_ROOT / candidate
    return candidate.resolve()


def _load_yaml(path: Path) -> dict[str, Any]:
    try:
        import yaml
    except ImportError as exc:  # pragma: no cover - project dependencies provide PyYAML
        raise RuntimeError("PyYAML is required to load the saved PufferDrive config") from exc

    if not path.is_file():
        raise FileNotFoundError(f"Saved teacher config does not exist: {path}")
    with path.open("r", encoding="utf-8") as handle:
        loaded = yaml.safe_load(handle)
    return _as_plain_mapping(loaded, name=f"saved teacher config {path}")


def _condition_config(config: TeacherConfig) -> tuple[dict[str, Any], Optional[Path]]:
    """Return a condition config copy and its explicit saved-config path."""

    condition = _as_plain_mapping(config, name="config")
    teacher_section = condition.get("teacher")
    if teacher_section is not None and not isinstance(teacher_section, Mapping):
        raise ValueError("config.teacher must be a mapping")

    config_path = None
    if isinstance(teacher_section, Mapping) and teacher_section.get("config") is not None:
        config_path = _resolve_repo_path(teacher_section["config"])
    return condition, config_path


def _apply_env_overrides(
    saved_config: dict[str, Any],
    condition_config: Mapping[str, Any],
) -> dict[str, Any]:
    """Merge condition overrides into the saved full PufferDrive mapping."""

    resolved = deepcopy(saved_config)
    env = resolved.get("env")
    if not isinstance(env, Mapping):
        raise ValueError("saved teacher config must contain an env mapping")
    env = deepcopy(dict(env))

    overrides = condition_config.get("env_overrides", {})
    if not isinstance(overrides, Mapping):
        raise ValueError("config.env_overrides must be a mapping")
    for raw_key, value in overrides.items():
        key = str(raw_key)
        if key == "env":
            if not isinstance(value, Mapping):
                raise ValueError("config.env_overrides.env must be a mapping")
            nested = value
        elif key.startswith("env."):
            nested = {key[4:]: value}
        else:
            nested = {key: value}
        for env_key, env_value in nested.items():
            if env_key not in env:
                raise ValueError(f"env_overrides references unknown saved env key '{env_key}'")
            env[env_key] = deepcopy(env_value)

    resolved["env"] = env
    if not isinstance(resolved.get("policy"), Mapping):
        raise ValueError("saved teacher config must contain a policy mapping")
    return resolved


def _validate_positive_finite(value: Any, *, name: str) -> float:
    try:
        numeric = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{name} must be a finite positive number") from exc
    if not torch.isfinite(torch.tensor(numeric)) or numeric <= 0:
        raise ValueError(f"{name} must be a finite positive number, got {value!r}")
    return numeric


def _validate_effective_env(resolved: Mapping[str, Any]) -> None:
    env = resolved["env"]
    action_type = env.get("action_type")
    if action_type not in ("continuous", "discrete"):
        raise ValueError(f"env.action_type must be 'continuous' or 'discrete', got {action_type!r}")
    dynamics_model = env.get("dynamics_model")
    if dynamics_model not in ("jerk", "classic"):
        raise ValueError(f"env.dynamics_model must be 'jerk' or 'classic', got {dynamics_model!r}")
    _validate_positive_finite(env.get("dt"), name="env.dt")

    for name in (
        "obs_slots_lane_n",
        "obs_slots_boundary_n",
        "obs_slots_partners_n",
        "obs_slots_traffic_controls_n",
    ):
        value = env.get(name)
        if not isinstance(value, int) or isinstance(value, bool) or value < 0:
            raise ValueError(f"env.{name} must be a non-negative integer, got {value!r}")
    for name in ("obs_dropout_lane", "obs_dropout_boundary"):
        value = env.get(name)
        try:
            numeric = float(value)
        except (TypeError, ValueError) as exc:
            raise ValueError(f"env.{name} must be finite in [0, 1]") from exc
        if not torch.isfinite(torch.tensor(numeric)) or not 0.0 <= numeric <= 1.0:
            raise ValueError(f"env.{name} must be finite in [0, 1], got {value!r}")

    policy = resolved["policy"]
    required_policy_keys = (
        "ego_input_size",
        "partner_input_size",
        "lane_input_size",
        "boundary_input_size",
        "traffic_control_input_size",
        "context_input_size",
        "backbone_hidden_size",
        "backbone_num_layers",
        "encoder_activation",
        "encoder_layer_norm",
        "backbone_activation",
        "backbone_layer_norm",
        "mask_padded_features",
    )
    missing = [key for key in required_policy_keys if key not in policy]
    if missing:
        raise ValueError(f"saved teacher policy is missing required keys: {', '.join(missing)}")


def _effective_road_count(max_count: int, dropout: float) -> int:
    return int(max_count * (1.0 - float(dropout)))


def observation_layout_from_config(resolved_config: TeacherConfig) -> dict[str, int]:
    """Derive the flat observation layout consumed by ``DriveBackbone``.

    The returned values are metadata only.  In particular, this helper does
    not create a ``Drive`` environment or inspect a map.
    """

    resolved = _as_plain_mapping(resolved_config, name="resolved_config")
    env = resolved.get("env")
    if not isinstance(env, Mapping):
        raise ValueError("resolved_config must contain an env mapping")

    # These feature counts are compile-time constants in the Drive binding.
    # Import lazily so reading a YAML config remains independent of simulator
    # construction and does not initialize a neural network.
    from pufferlib.ocean.drive import binding

    ego_features = int(env.get("ego_features", binding.EGO_FEATURES))
    partner_features = int(env.get("partner_features", binding.PARTNER_FEATURES))
    lane_features = int(env.get("lane_features", binding.LANE_FEATURES))
    boundary_features = int(env.get("boundary_features", binding.BOUNDARY_FEATURES))
    traffic_control_features = int(env.get("traffic_control_features", binding.TRAFFIC_CONTROL_FEATURES))
    obs_valid_count_features = int(env.get("obs_valid_count_features", binding.OBS_VALID_COUNT_FEATURES))
    goal_features = int(env.get("goal_features", binding.GOAL_FEATURES))

    num_goals = int(env.get("num_goals", 0))
    num_reward_coefs = int(
        env.get(
            "num_reward_coefs",
            binding.NUM_REWARD_COEFS if bool(env.get("reward_conditioning", False)) else 0,
        )
    )
    goal_dim = int(env.get("goal_dim", num_goals * goal_features))
    if "obs_slots_lane_n" in env and "obs_dropout_lane" in env:
        lane_slots = _effective_road_count(int(env["obs_slots_lane_n"]), float(env["obs_dropout_lane"]))
    elif "obs_slots_lane_kept" in env:
        lane_slots = int(env["obs_slots_lane_kept"])
    else:
        raise ValueError("resolved env must define lane slot count metadata")
    if "obs_slots_boundary_n" in env and "obs_dropout_boundary" in env:
        boundary_slots = _effective_road_count(
            int(env["obs_slots_boundary_n"]), float(env["obs_dropout_boundary"])
        )
    elif "obs_slots_boundary_kept" in env:
        boundary_slots = int(env["obs_slots_boundary_kept"])
    else:
        raise ValueError("resolved env must define boundary slot count metadata")
    partners = int(env["obs_slots_partners_n"])
    traffic_controls = int(env["obs_slots_traffic_controls_n"])
    context_dim = num_reward_coefs + goal_dim
    observation_dim = (
        ego_features
        + context_dim
        + partners * partner_features
        + lane_slots * lane_features
        + boundary_slots * boundary_features
        + traffic_controls * traffic_control_features
        + obs_valid_count_features
    )
    return {
        "ego_features": ego_features,
        "partner_features": partner_features,
        "lane_features": lane_features,
        "boundary_features": boundary_features,
        "traffic_control_features": traffic_control_features,
        "obs_valid_count_features": obs_valid_count_features,
        "goal_features": goal_features,
        "num_reward_coefs": num_reward_coefs,
        "goal_dim": goal_dim,
        "context_dim": context_dim,
        "obs_slots_partners_n": partners,
        "obs_slots_lane_kept": lane_slots,
        "obs_slots_boundary_kept": boundary_slots,
        "obs_slots_traffic_controls_n": traffic_controls,
        "observation_dim": observation_dim,
    }


def _resolve_from_existing_args(config: TeacherConfig) -> dict[str, Any]:
    """Accept a previously resolved full PufferDrive mapping unchanged."""

    resolved = _as_plain_mapping(config, name="config")
    if not isinstance(resolved.get("env"), Mapping) or not isinstance(resolved.get("policy"), Mapping):
        raise ValueError("resolved teacher config must contain env and policy mappings")
    _validate_effective_env(resolved)
    return resolved


def resolve_teacher_config(
    config: TeacherConfig,
    *,
    config_path: Optional[PathLike] = None,
) -> TeacherConfig:
    """Return the full saved PufferDrive args with clean env overrides applied.

    ``config`` may be the Condition B mapping from ``condition_b.yaml`` or a
    full already-resolved PufferDrive mapping containing ``env`` and
    ``policy``.  Relative saved-config paths are repository-relative.  The
    input mapping is never mutated, and derived observation metadata is
    available through :func:`observation_layout_from_config`.
    """

    condition, configured_path = _condition_config(config)
    selected_path = _resolve_repo_path(config_path) if config_path is not None else configured_path

    if selected_path is None and isinstance(condition.get("env"), Mapping):
        if "env_overrides" in condition:
            resolved = _apply_env_overrides(condition, condition)
            _validate_effective_env(resolved)
        else:
            resolved = _resolve_from_existing_args(condition)
    elif selected_path is None:
        raise ValueError(
            "teacher config path is required unless config already contains full env and policy mappings"
        )
    else:
        saved = _load_yaml(selected_path)
        resolved = _apply_env_overrides(saved, condition)
        _validate_effective_env(resolved)

    # Verify the derived layout now, while malformed external metadata still
    # belongs to the configuration boundary.
    observation_layout_from_config(resolved)
    return resolved


def _metadata_environment(resolved: TeacherConfig) -> SimpleNamespace:
    """Build the attributes read by ``Drive`` and ``DriveBackbone``."""

    from pufferlib.ocean.drive import binding

    layout = observation_layout_from_config(resolved)
    env_config = resolved["env"]
    dynamics_model = env_config["dynamics_model"]
    dynamics_flag = (
        binding.DYNAMICS_MODEL_JERK
        if dynamics_model == "jerk"
        else binding.DYNAMICS_MODEL_CLASSIC
    )
    metadata = dict(layout)
    metadata.update(
        {
            "dynamics_model": dynamics_model,
            "dynamics_model_flag": dynamics_flag,
            "single_action_space": SimpleNamespace(shape=(2,)),
        }
    )
    return SimpleNamespace(**metadata)


def _policy_constructor_args(resolved: TeacherConfig) -> dict[str, Any]:
    policy = resolved["policy"]
    names = (
        "ego_input_size",
        "partner_input_size",
        "lane_input_size",
        "boundary_input_size",
        "traffic_control_input_size",
        "context_input_size",
        "backbone_hidden_size",
        "backbone_num_layers",
        "actor_hidden_size",
        "actor_num_layers",
        "critic_hidden_size",
        "critic_num_layers",
        "encoder_activation",
        "encoder_layer_norm",
        "backbone_activation",
        "backbone_layer_norm",
        "shared_network",
        "mask_padded_features",
        "action_type",
    )
    missing = [name for name in names if name not in policy]
    if missing:
        raise ValueError(f"saved teacher policy is missing constructor keys: {', '.join(missing)}")
    args = {name: deepcopy(policy[name]) for name in dict.fromkeys(names)}
    for optional_name in ("actor_head_layer_norm", "critic_head_layer_norm"):
        if optional_name in policy:
            args[optional_name] = deepcopy(policy[optional_name])
    return args


def _checkpoint_state_dict(checkpoint: Any) -> Mapping[str, torch.Tensor]:
    if isinstance(checkpoint, Mapping):
        for wrapper_key in ("state_dict", "model_state_dict", "model"):
            nested = checkpoint.get(wrapper_key)
            if isinstance(nested, Mapping):
                checkpoint = nested
                break
    if not isinstance(checkpoint, Mapping) or not checkpoint:
        raise ValueError("teacher checkpoint must contain a non-empty state dict")
    if not all(isinstance(key, str) and isinstance(value, torch.Tensor) for key, value in checkpoint.items()):
        raise ValueError("teacher checkpoint state dict must map string keys to tensors")
    return checkpoint


def load_teacher(
    config: TeacherConfig,
    *,
    resolved_config: Optional[TeacherConfig] = None,
    resolved: Optional[TeacherConfig] = None,
    checkpoint_path: Optional[PathLike] = None,
    device: Union[str, torch.device] = "cpu",
    env: Optional[Any] = None,
) -> nn.Module:
    """Load and freeze the existing Drive teacher from weights only.

    The default path constructs the policy from metadata and never creates a
    live simulator.  ``env`` is accepted for callers that already own a Drive
    environment; it is not required for offline loading.  The complete Drive
    state dict is loaded strictly so actor and critic checkpoint compatibility
    is checked, while Condition B later copies only ``actor_backbone``.
    """

    if resolved_config is not None and resolved is not None:
        raise ValueError("pass only one of resolved_config or resolved")
    resolved_mapping = resolved_config if resolved_config is not None else resolved
    resolved_args = (
        _resolve_from_existing_args(resolved_mapping)
        if resolved_mapping is not None
        else resolve_teacher_config(config)
    )
    condition, _ = _condition_config(config)
    teacher_section = condition.get("teacher", {})
    if teacher_section is not None and not isinstance(teacher_section, Mapping):
        raise ValueError("config.teacher must be a mapping")

    selected_checkpoint = checkpoint_path
    if selected_checkpoint is None and isinstance(teacher_section, Mapping):
        selected_checkpoint = teacher_section.get("checkpoint")
    if selected_checkpoint is None:
        raise ValueError("teacher checkpoint path is required")
    checkpoint_file = _resolve_repo_path(selected_checkpoint)
    if not checkpoint_file.is_file():
        raise FileNotFoundError(f"Teacher checkpoint does not exist: {checkpoint_file}")

    from pufferlib.ocean.torch import Drive

    model_env = env
    if model_env is not None and hasattr(model_env, "driver_env"):
        model_env = model_env.driver_env
    if model_env is None:
        model_env = _metadata_environment(resolved_args)

    policy_args = _policy_constructor_args(resolved_args)
    teacher = Drive(model_env, **policy_args)
    checkpoint = torch.load(checkpoint_file, map_location=device)
    state_dict = _checkpoint_state_dict(checkpoint)
    try:
        teacher.load_state_dict(state_dict, strict=True)
    except RuntimeError as exc:
        raise RuntimeError(f"Teacher checkpoint is incompatible with the resolved Drive policy: {exc}") from exc

    teacher.to(device)
    teacher.eval()
    for parameter in teacher.parameters():
        parameter.requires_grad_(False)

    # Keep reconstruction metadata outside the model state dict.  These
    # attributes are intentionally plain values so a student checkpoint can
    # serialize them without serializing the teacher or an environment.
    with checkpoint_file.open('rb') as checkpoint_stream:
        teacher.condition_b_checkpoint_sha256 = hashlib.file_digest(checkpoint_stream, 'sha256').hexdigest()
    teacher.condition_b_checkpoint_path = str(checkpoint_file)
    teacher.condition_b_resolved_config = deepcopy(dict(resolved_args))
    teacher.condition_b_observation_layout = observation_layout_from_config(resolved_args)
    # Short aliases make the metadata contract convenient for callers while
    # retaining the explicit Condition B names above.
    teacher.resolved_config = deepcopy(teacher.condition_b_resolved_config)
    teacher.observation_layout = deepcopy(teacher.condition_b_observation_layout)
    return teacher
