"""Student policy adapter and bounded evaluation entrypoint.

The adapter is intentionally small: Condition B predicts a four-slot chunk,
while PufferLib's evaluator asks for one categorical head and an unused value
head. ``evaluate_student`` accepts a caller-built student and delegates CARLA
evaluation to :func:`pufferlib.pufferl.eval`; a tiny injected environment path
is provided for deterministic smoke tests and local toy runs.
"""

from __future__ import annotations

import argparse
import copy
import json
import math
from pathlib import Path
from typing import Any, Mapping, Optional, Sequence, Tuple, TYPE_CHECKING, Union

import torch
import torch.nn as nn

if TYPE_CHECKING:
    from pufferlib.ocean.drive.drive import Drive
    from .monitoring import WandbMonitor


PathLike = Union[str, Path]


class StudentPolicyAdapter(nn.Module):
    """Expose student slot-zero logits through PufferLib's policy API.

    The wrapped model is registered as ``student`` so its state and device
    movement follow normal ``nn.Module`` semantics. The policy remains
    categorical even when the environment accepts continuous controls. Mean
    controls use the physical action table first, matching Drive's piecewise
    longitudinal normalization exactly.
    """

    student: nn.Module
    is_continuous: bool = False

    def __init__(
        self,
        student: nn.Module,
        *,
        action_table: Optional[torch.Tensor] = None,
        action_table_physical: Optional[torch.Tensor] = None,
    ) -> None:
        super().__init__()
        if not isinstance(student, nn.Module):
            raise TypeError("student must be a torch.nn.Module")
        self.student = student

        normalized_source = action_table
        physical_source = action_table_physical
        if normalized_source is None:
            normalized_source = getattr(student, "action_table", None)
        if physical_source is None:
            physical_source = getattr(student, "action_table_physical", None)

        if normalized_source is None and physical_source is None:
            # Condition B's default is Drive's 4x3 jerk table. Other dynamics
            # models must provide their exact tables at construction time.
            physical_source = _default_jerk_physical_table()
        physical = _as_action_table(physical_source, "action_table_physical") if physical_source is not None else None
        normalized = _as_action_table(normalized_source, "action_table") if normalized_source is not None else None
        if normalized is None:
            assert physical is not None
            scales = _resolve_scales(student, physical)
            normalized = _normalize_physical_table(physical, scales)
        if physical is None:
            scales = _resolve_scales(student, normalized)
            if normalized.shape[0] == 12 and not _has_explicit_scales(student):
                # A bare 12-class table is the normalized jerk table from
                # Drive; retain its asymmetric physical longitudinal scales.
                scales = (15.0, 4.0, 4.0)
            physical = _physical_from_normalized_table(normalized, scales)
        if normalized.shape != physical.shape:
            raise ValueError(
                "action_table and action_table_physical must have the same shape, "
                f"got {tuple(normalized.shape)} and {tuple(physical.shape)}"
            )
        if normalized.shape[1] != 2 or normalized.shape[0] <= 0:
            raise ValueError("action tables must have shape [num_classes, 2]")
        if not torch.isfinite(normalized).all() or not torch.isfinite(physical).all():
            raise ValueError("action tables must contain only finite values")

        self.register_buffer("action_table", normalized, persistent=False)
        self.register_buffer("action_table_physical", physical, persistent=False)
        self._action_long_neg_scale, self._action_long_pos_scale, self._action_lat_scale = _resolve_scales(
            student, physical
        )
        self.num_action_classes = int(normalized.shape[0])

    def forward_eval(
        self,
        observations: torch.Tensor,
        state: Optional[Mapping[str, torch.Tensor]] = None,
    ) -> Tuple[Tuple[torch.Tensor], torch.Tensor]:
        """Return ``((slot_zero_logits,), zeros)`` for PufferLib evaluation."""

        if state is not None:
            raise ValueError("StudentPolicyAdapter is feed-forward and does not accept recurrent state")
        if not isinstance(observations, torch.Tensor) or observations.ndim < 2:
            raise ValueError("observations must be a batched torch tensor")
        logits = self._student_chunk_logits(observations)
        if logits.ndim == 2:
            slot_zero_logits = logits
        elif logits.ndim == 3:
            if logits.shape[1] <= 0:
                raise ValueError("student chunk logits have no action slots")
            slot_zero_logits = logits[:, 0, :]
        else:
            raise ValueError(
                "student logits must have shape [B, classes] or [B, slots, classes], "
                f"got {tuple(logits.shape)}"
            )
        if slot_zero_logits.shape[0] != observations.shape[0]:
            raise ValueError("student logits batch dimension does not match observations")
        if slot_zero_logits.shape[-1] != self.num_action_classes:
            raise ValueError(
                "student logits class count does not match the action table: "
                f"{slot_zero_logits.shape[-1]} != {self.num_action_classes}"
            )
        unused_value = torch.zeros(
            (observations.shape[0], 1), dtype=slot_zero_logits.dtype, device=slot_zero_logits.device
        )
        return (slot_zero_logits,), unused_value

    def forward(
        self,
        observations: torch.Tensor,
        state: Optional[Mapping[str, torch.Tensor]] = None,
    ) -> Tuple[Tuple[torch.Tensor], torch.Tensor]:
        """Mirror ``forward_eval`` for ordinary ``nn.Module`` callers."""

        return self.forward_eval(observations, state)

    def _student_chunk_logits(self, observations: torch.Tensor) -> torch.Tensor:
        # ConditionBModel exposes the two deployment stages explicitly. The
        # fallback keeps the adapter useful for small test/exported modules.
        encode_context = getattr(self.student, "encode_context", None)
        decode_chunk = getattr(self.student, "decode_chunk", None)
        if callable(encode_context) and callable(decode_chunk):
            raw_logits = decode_chunk(encode_context(observations))
        else:
            raw_logits = self.student(observations)
        return _extract_chunk_logits(raw_logits)

    def discrete_actions_to_continuous(self, actions: torch.Tensor) -> torch.Tensor:
        """Map arbitrary-shaped class IDs through Drive's normalized table."""

        if not isinstance(actions, torch.Tensor):
            actions = torch.as_tensor(actions)
        index = actions.to(device=self.action_table.device).long()
        if index.numel() and (index.min() < 0 or index.max() >= self.num_action_classes):
            raise ValueError("discrete action index is outside the action table")
        return self.action_table[index]

    def discrete_probs_to_continuous_mean(self, probabilities: torch.Tensor) -> torch.Tensor:
        """Average physical controls, then apply Drive's piecewise scaling."""

        if not isinstance(probabilities, torch.Tensor):
            probabilities = torch.as_tensor(probabilities)
        if not (probabilities.is_floating_point() or probabilities.is_complex()):
            probabilities = probabilities.float()
        if probabilities.ndim == 0 or probabilities.shape[-1] != self.num_action_classes:
            raise ValueError(
                "probabilities must have final dimension equal to the action table: "
                f"{self.num_action_classes}"
            )
        physical_table = self.action_table_physical.to(
            device=probabilities.device, dtype=probabilities.dtype
        )
        mean_physical = probabilities @ physical_table
        neg_scale = torch.as_tensor(
            self._action_long_neg_scale, dtype=mean_physical.dtype, device=mean_physical.device
        )
        pos_scale = torch.as_tensor(
            self._action_long_pos_scale, dtype=mean_physical.dtype, device=mean_physical.device
        )
        lat_scale = torch.as_tensor(
            self._action_lat_scale, dtype=mean_physical.dtype, device=mean_physical.device
        )
        mean_long = mean_physical[..., 0]
        mean_long_norm = torch.where(
            mean_long < 0.0,
            mean_long / neg_scale,
            mean_long / pos_scale,
        )
        return torch.stack([mean_long_norm, mean_physical[..., 1] / lat_scale], dim=-1).clamp(-1.0, 1.0)


def evaluate_student(
    config: Mapping[str, Any],
    *,
    student: Optional[nn.Module] = None,
    env: Optional["Drive"] = None,
    monitor: Optional["WandbMonitor"] = None,
) -> Mapping[str, Any]:
    """Evaluate an injected student adapter with matched scenarios.

    For CARLA, ``config`` must provide a caller-built PufferLib argument
    mapping under ``puffer_args`` (or contain its full ``env/policy/eval/...``
    mapping). This keeps Condition B checkpoint reconstruction in the model
    worker/root, where its exported model metadata is available, while this
    function only injects the adapter into the existing ``pufferl.eval`` path.
    An injected ``env`` uses the bounded toy rollout path and requires an
    explicit positive ``episode_timesteps`` (or ``max_episode_steps``).
    """

    if not isinstance(config, Mapping):
        raise TypeError("evaluation config must be a mapping")
    evaluation_config = dict(_evaluation_section(config))
    _validate_evaluation_protocol(evaluation_config)
    checkpoint = None
    checkpoint_path = None
    if student is None:
        checkpoint_path = _checkpoint_path(config, evaluation_config)
        if checkpoint_path is None:
            raise ValueError(
                "evaluate_student requires an injected student or student_checkpoint/checkpoint path"
            )
        student, checkpoint = _load_student_checkpoint(
            checkpoint_path,
            model_config=_model_config(config, evaluation_config),
            device=evaluation_config.get("device", "cpu"),
        )
    evaluation_config = _checkpoint_output_defaults(
        config, evaluation_config, checkpoint, checkpoint_path
    )
    if not isinstance(student, nn.Module):
        raise TypeError("student must be a torch.nn.Module")

    adapter = student if isinstance(student, StudentPolicyAdapter) else StudentPolicyAdapter(
        student,
        action_table=_config_action_table(config, evaluation_config, checkpoint, "action_table"),
        action_table_physical=_config_action_table(
            config, evaluation_config, checkpoint, "action_table_physical"
        ),
    )
    adapter.eval()

    num_scenarios = _positive_int(
        evaluation_config.get("num_scenarios"), "evaluation.num_scenarios"
    )
    benchmark_names = _benchmark_names(evaluation_config)
    progress = _evaluation_progress(config, evaluation_config, checkpoint)
    owned_monitor = False
    active_monitor = monitor
    if active_monitor is None:
        from .monitoring import WandbMonitor

        run_dir = _evaluation_run_dir(config, evaluation_config)
        wandb_config = evaluation_config.get("wandb", {"enabled": False, "mode": "disabled"})
        checkpoint_state = checkpoint.get("monitoring_state") if isinstance(checkpoint, Mapping) else None
        active_monitor = WandbMonitor(
            wandb_config,
            config,
            run_dir,
            checkpoint_state=checkpoint_state,
        )
        owned_monitor = True

    try:
        runtime_args = _runtime_args(config, evaluation_config, checkpoint)
        _apply_evaluation_runtime_overrides(adapter, runtime_args, evaluation_config)
        if env is not None and runtime_args is None:
            results = _evaluate_injected_env(
                adapter,
                env,
                evaluation_config,
                num_scenarios=num_scenarios,
            )
        else:
            if runtime_args is None:
                raise ValueError(
                    "CARLA evaluation requires config['puffer_args'] (the root/model worker "
                    "must reconstruct and export the resolved PufferLib args)"
                )
            results = _evaluate_with_pufferl(
                adapter,
                runtime_args,
                evaluation_config,
                benchmark_names=benchmark_names,
            )
        _log_benchmark_results(active_monitor, results, benchmark_names, progress, evaluation_config)
        return results
    finally:
        if owned_monitor:
            active_monitor.finish()


def load_student_checkpoint(
    checkpoint_path: PathLike,
    *,
    model_config: Optional[Mapping[str, Any]] = None,
    device: Union[str, torch.device] = "cpu",
) -> nn.Module:
    """Load a Condition B checkpoint for callers that already own metadata.

    The checkpoint may contain a serialized module (toy runs), ``model_state``
    plus ``model_config`` (normal Condition B runs), or ``student_state_dict``.
    PPO trainer state is never consumed.
    """

    student, _ = _load_student_checkpoint(checkpoint_path, model_config=model_config, device=device)
    return student


def _load_student_checkpoint(
    checkpoint_path: PathLike,
    *,
    model_config: Optional[Mapping[str, Any]],
    device: Union[str, torch.device],
) -> tuple[nn.Module, Mapping[str, Any]]:
    path = Path(checkpoint_path)
    if not path.is_file() and not path.is_absolute():
        try:
            from .runtime import resolve_path

            path = resolve_path(path)
        except (ImportError, OSError):
            pass
    if not path.is_file():
        raise FileNotFoundError(f"student checkpoint not found: {path}")
    try:
        checkpoint = torch.load(path, map_location=device, weights_only=False)
    except TypeError:  # PyTorch versions before ``weights_only``
        checkpoint = torch.load(path, map_location=device)
    if isinstance(checkpoint, nn.Module):
        return checkpoint.to(device), {"model": checkpoint}
    if not isinstance(checkpoint, Mapping):
        raise ValueError("student checkpoint must contain a module or mapping")

    for key in ("student", "model"):
        candidate = checkpoint.get(key)
        if isinstance(candidate, nn.Module):
            return candidate.to(device), checkpoint

    state = None
    for key in ("model_state", "student_state_dict", "student_state", "state_dict"):
        candidate = checkpoint.get(key)
        if isinstance(candidate, Mapping):
            state = candidate
            break
    if state is None:
        raise ValueError(
            "student checkpoint has no serialized module or model_state/student_state_dict mapping"
        )
    resolved_model_config = checkpoint.get("model_config") or model_config
    if not isinstance(resolved_model_config, Mapping):
        raise ValueError(
            "checkpoint reconstruction requires model_config exported with the student checkpoint"
        )
    from .model import ConditionBModel

    student = ConditionBModel(resolved_model_config).to(device)
    try:
        student.load_state_dict(state, strict=True)
    except RuntimeError:
        # A wrapper may carry one leading module or student prefix. Strip
        # exactly one known prefix, then keep strict loading semantics.
        cleaned = {}
        for key, value in state.items():
            for prefix in ("module.", "student.", "model."):
                if key.startswith(prefix):
                    key = key[len(prefix) :]
                    break
            cleaned[key] = value
        student.load_state_dict(cleaned, strict=True)
    return student, checkpoint


def _evaluation_section(config: Mapping[str, Any]) -> Mapping[str, Any]:
    section = config.get("evaluation")
    if section is None:
        section = config
    if not isinstance(section, Mapping):
        raise TypeError("evaluation section must be a mapping")
    return section


def _validate_evaluation_protocol(evaluation_config: Mapping[str, Any]) -> None:
    population = evaluation_config.get("population", "student_self_play")
    if population != "student_self_play":
        raise ValueError(
            "evaluation.population must be 'student_self_play'; "
            f"got {population!r}"
        )
    execution_horizon = evaluation_config.get("execution_horizon", 1)
    if isinstance(execution_horizon, bool) or not isinstance(execution_horizon, int):
        raise ValueError("evaluation.execution_horizon must be the integer 1")
    if execution_horizon != 1:
        raise ValueError(
            "evaluation.execution_horizon must be 1 for the slot-zero policy adapter; "
            "use a separate evaluator for another horizon"
        )
    transfer = evaluation_config.get("transfer")
    if transfer is None:
        return
    if not isinstance(transfer, Mapping):
        raise ValueError("evaluation.transfer must be a mapping")
    enabled = transfer.get("enabled", False)
    if not isinstance(enabled, bool):
        raise ValueError("evaluation.transfer.enabled must be a boolean")
    if enabled:
        raise ValueError(
            "evaluation.transfer.enabled=true is unsupported by evaluate_student; "
            "run a separate transfer evaluation config with transfer disabled here"
        )


def _checkpoint_output_defaults(
    config: Mapping[str, Any],
    evaluation_config: Mapping[str, Any],
    checkpoint: Optional[Mapping[str, Any]],
    checkpoint_path: Optional[PathLike],
) -> dict[str, Any]:
    """Fill nullable output fields from the saved student run recipe."""

    resolved = dict(evaluation_config)
    checkpoint_config = checkpoint.get("config") if isinstance(checkpoint, Mapping) else None
    training = checkpoint_config.get("training") if isinstance(checkpoint_config, Mapping) else None
    if not isinstance(training, Mapping):
        training = {}

    if not isinstance(resolved.get("run_id"), str) or not resolved["run_id"].strip():
        candidates = (
            training.get("run_id"),
            config.get("run_id"),
            Path(checkpoint_path).resolve().parent.name if checkpoint_path is not None else None,
        )
        for candidate in candidates:
            if isinstance(candidate, str) and candidate.strip() and Path(candidate).name == candidate:
                resolved["run_id"] = candidate
                break
        else:
            resolved["run_id"] = "student_eval"

    if not isinstance(resolved.get("output_root"), (str, Path)) or not str(
        resolved["output_root"]
    ).strip():
        output_root = training.get("output_root")
        if isinstance(output_root, (str, Path)) and str(output_root).strip():
            resolved["output_root"] = output_root
        elif checkpoint_path is not None:
            resolved["output_root"] = Path(checkpoint_path).resolve().parent.parent
        else:
            resolved["output_root"] = "experiments/jepa_distill/runs"
    return resolved


def _checkpoint_path(config: Mapping[str, Any], evaluation_config: Mapping[str, Any]) -> Optional[PathLike]:
    for key in ("checkpoint", "student_checkpoint", "checkpoint_path"):
        if key in evaluation_config and evaluation_config[key] is not None:
            return evaluation_config[key]
        if key in config and config[key] is not None:
            return config[key]
    return None


def _model_config(config: Mapping[str, Any], evaluation_config: Mapping[str, Any]) -> Optional[Mapping[str, Any]]:
    for key in ("model_config", "model"):
        value = evaluation_config.get(key, config.get(key))
        if isinstance(value, Mapping):
            return value
    return None


def _config_action_table(
    config: Mapping[str, Any],
    evaluation_config: Mapping[str, Any],
    checkpoint: Optional[Mapping[str, Any]],
    name: str,
) -> Optional[torch.Tensor]:
    sources = [evaluation_config.get(name), config.get(name)]
    if isinstance(checkpoint, Mapping):
        sources.extend(
            [
                checkpoint.get(name),
                checkpoint.get("action_metadata", {}).get(name)
                if isinstance(checkpoint.get("action_metadata"), Mapping)
                else None,
            ]
        )
    for value in sources:
        if value is not None:
            return torch.as_tensor(value, dtype=torch.float32)
    return None


def _evaluation_progress(
    config: Mapping[str, Any],
    evaluation_config: Mapping[str, Any],
    checkpoint: Optional[Mapping[str, Any]],
) -> "MetricProgress":
    from .monitoring import MetricProgress

    source = evaluation_config.get("progress") or config.get("progress")
    source = source if isinstance(source, Mapping) else {}
    checkpoint_counters = checkpoint.get("counters", {}) if isinstance(checkpoint, Mapping) else {}
    collection_state = checkpoint.get("collection_state", {}) if isinstance(checkpoint, Mapping) else {}
    sampler_state = checkpoint.get("sampler_state", {}) if isinstance(checkpoint, Mapping) else {}
    if not isinstance(checkpoint_counters, Mapping):
        checkpoint_counters = {}
    if not isinstance(collection_state, Mapping):
        collection_state = {}
    if not isinstance(sampler_state, Mapping):
        sampler_state = {}
    checkpoint_step = checkpoint.get("step", 0) if isinstance(checkpoint, Mapping) else 0
    return MetricProgress(
        optimizer_step=int(
            source.get(
                "optimizer_step",
                checkpoint_counters.get("optimizer_step", checkpoint_step),
            )
        ),
        simulator_transitions=int(
            source.get(
                "simulator_transitions",
                checkpoint_counters.get(
                    "simulator_transitions", collection_state.get("simulator_transitions", 0)
                ),
            )
        ),
        collection_round_idx=int(
            source.get(
                "collection_round_idx",
                checkpoint_counters.get("collection_round_idx", sampler_state.get("collection_round_idx", 0)),
            )
        ),
        update_epoch_idx=int(
            source.get(
                "update_epoch_idx",
                checkpoint_counters.get("update_epoch_idx", sampler_state.get("update_epoch_idx", 0)),
            )
        ),
    )


def _positive_int(value: Any, label: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise ValueError(f"{label} must be a positive integer")
    return value


def _benchmark_names(evaluation_config: Mapping[str, Any]) -> list[str]:
    value = evaluation_config.get("benchmarks", ["carla"])
    if isinstance(value, str):
        value = [item.strip() for item in value.split(",") if item.strip()]
    if not isinstance(value, Sequence) or isinstance(value, (bytes, str)):
        raise ValueError("evaluation.benchmarks must be a non-empty sequence or comma-separated string")
    names = [name for name in value if isinstance(name, str) and name.strip()]
    if not names:
        raise ValueError("evaluation.benchmarks must select at least one benchmark")
    return names


def _runtime_args(
    config: Mapping[str, Any],
    evaluation_config: Mapping[str, Any],
    checkpoint: Optional[Mapping[str, Any]] = None,
) -> Optional[dict[str, Any]]:
    for key in ("puffer_args", "runtime_args", "args"):
        value = evaluation_config.get(key, config.get(key))
        if isinstance(value, Mapping):
            return _disable_external_loggers(copy.deepcopy(dict(value)))
    required = {"env", "policy", "eval", "train", "vec", "package", "policy_name"}
    if required.issubset(config):
        return _disable_external_loggers(copy.deepcopy(dict(config)))
    checkpoint_sources = []
    if isinstance(checkpoint, Mapping):
        checkpoint_config = checkpoint.get("config")
        if isinstance(checkpoint_config, Mapping):
            checkpoint_sources.append(checkpoint_config.get("teacher_config"))
            if required.issubset(checkpoint_config):
                checkpoint_sources.append(checkpoint_config)
        model_config = checkpoint.get("model_config")
        if isinstance(model_config, Mapping):
            checkpoint_sources.append(model_config.get("teacher_config"))
        checkpoint_sources.append(checkpoint.get("teacher_config"))
    for source in checkpoint_sources:
        if isinstance(source, Mapping) and required.issubset(source):
            runtime_args = copy.deepcopy(dict(source))
            # The saved args describe the PPO teacher. Reuse its architecture
            # and environment settings, while preventing the evaluator from
            # attaching to or creating any teacher tracker session.
            return _disable_external_loggers(runtime_args)
    return None


def _apply_evaluation_runtime_overrides(
    adapter: StudentPolicyAdapter,
    runtime_args: Optional[Mapping[str, Any]],
    evaluation_config: Mapping[str, Any],
) -> None:
    """Make the standalone evaluation settings authoritative at runtime.

    The checkpoint carries the teacher's resolved PPO arguments so the native
    evaluator can reconstruct the environment. Device, precision, compiler,
    and seed settings belong to the evaluation invocation, however; inheriting
    those fields silently can move observations away from the student or
    enable teacher-only runtime features.
    """

    target_device = evaluation_config.get("device")
    if target_device is None and isinstance(runtime_args, Mapping):
        train_args = runtime_args.get("train")
        if isinstance(train_args, Mapping):
            target_device = train_args.get("device")
    if target_device is not None:
        adapter.to(_as_torch_device(target_device))

    if runtime_args is None:
        return
    train_args = runtime_args.get("train")
    if not isinstance(train_args, Mapping):
        raise ValueError("puffer_args.train must be a mapping")
    train_args = dict(train_args)
    runtime_args["train"] = train_args  # type: ignore[index]
    if evaluation_config.get("device") is not None:
        train_args["device"] = evaluation_config["device"]
    # A standalone evaluation must not inherit AMP or torch.compile from the
    # teacher. They are opt-in in the evaluation config and default off.
    train_args["amp"] = _evaluation_bool(evaluation_config.get("amp", False), "evaluation.amp")
    train_args["compile"] = _evaluation_bool(
        evaluation_config.get("compile", False), "evaluation.compile"
    )
    for key in ("precision", "compile_mode", "compile_fullgraph"):
        if key in evaluation_config:
            train_args[key] = evaluation_config[key]

    seed = _optional_seed(evaluation_config.get("seed"), "evaluation.seed")
    if seed is not None:
        train_args["seed"] = seed
        vec_args = runtime_args.get("vec")
        if not isinstance(vec_args, Mapping):
            vec_args = {}
        vec_args = dict(vec_args)
        vec_args["seed"] = seed
        runtime_args["vec"] = vec_args  # type: ignore[index]


def _disable_external_loggers(runtime_args: dict[str, Any]) -> dict[str, Any]:
    """Keep standalone Condition B evaluation out of the PPO teacher run."""

    runtime_args["wandb"] = False
    runtime_args["neptune"] = False
    runtime_args["tb"] = False
    runtime_args["load_id"] = None
    runtime_args["load_model_path"] = None
    return runtime_args


def _as_torch_device(value: Any) -> Union[str, torch.device]:
    if isinstance(value, bool):
        raise ValueError("evaluation.device must be a torch device or device index")
    if isinstance(value, int):
        return torch.device("cuda", value)
    if isinstance(value, (str, torch.device)):
        return value
    raise ValueError("evaluation.device must be a torch device or device index")


def _evaluation_bool(value: Any, label: str) -> bool:
    if not isinstance(value, bool):
        raise ValueError(f"{label} must be a boolean")
    return value


def _optional_seed(value: Any, label: str) -> Optional[int]:
    if value is None:
        return None
    if isinstance(value, bool) or not isinstance(value, int) or not 0 <= value <= 2**31 - 1:
        raise ValueError(f"{label} must be an integer in [0, {2**31 - 1}] or null")
    return value


def _evaluate_with_pufferl(
    adapter: StudentPolicyAdapter,
    runtime_args: Mapping[str, Any],
    evaluation_config: Mapping[str, Any],
    *,
    benchmark_names: Sequence[str],
) -> Mapping[str, Any]:
    from pufferlib import pufferl

    args = copy.deepcopy(dict(runtime_args))
    if not isinstance(args.get("eval"), Mapping):
        raise ValueError("puffer_args.eval must be a mapping")
    _apply_evaluation_runtime_overrides(adapter, args, evaluation_config)
    args["eval"] = dict(args["eval"])
    args["eval"]["benchmarks"] = list(benchmark_names)
    if evaluation_config.get("num_agents") is not None:
        args["eval"]["num_agents"] = _positive_int(
            evaluation_config["num_agents"], "evaluation.num_agents"
        )
    args["eval"]["action_selection"] = evaluation_config.get(
        "action_selection", args["eval"].get("action_selection", "mean")
    )
    if evaluation_config.get("benchmark_config") is not None:
        args["eval"]["benchmark_config"] = evaluation_config["benchmark_config"]
    if evaluation_config.get("output_name") is not None:
        args["eval"]["output_name"] = evaluation_config["output_name"]
    env_overrides = evaluation_config.get("env_overrides")
    if env_overrides is not None:
        if not isinstance(env_overrides, Mapping):
            raise ValueError("evaluation.env_overrides must be a mapping")
        args["env"] = dict(args.get("env", {}))
        args["env"].update(copy.deepcopy(dict(env_overrides)))
    vec_overrides = evaluation_config.get("vec_overrides", evaluation_config.get("vec"))
    if vec_overrides is not None:
        if not isinstance(vec_overrides, Mapping):
            raise ValueError("evaluation.vec_overrides/evaluation.vec must be a mapping")
        args["vec"] = dict(args.get("vec", {}))
        args["vec"].update(copy.deepcopy(dict(vec_overrides)))
    env_name = (
        evaluation_config.get("env_name")
        or runtime_args.get("env_name")
        or args.get("env_name", "drive")
    )
    output_dir = _evaluation_run_dir({}, evaluation_config)
    output_dir.mkdir(parents=True, exist_ok=True)
    benchmark_config_path = _materialize_benchmark_config(
        args["eval"].get("benchmark_config"),
        output_dir=output_dir,
        benchmark_names=benchmark_names,
        env_overrides=env_overrides if isinstance(env_overrides, Mapping) else {},
        episode_timesteps=evaluation_config.get("episode_timesteps"),
        num_scenarios=_positive_int(evaluation_config.get("num_scenarios"), "evaluation.num_scenarios"),
        seed=_optional_seed(evaluation_config.get("seed"), "evaluation.seed"),
    )
    if benchmark_config_path is not None:
        args["eval"]["benchmark_config"] = str(benchmark_config_path)
    output_subdir = evaluation_config.get("output_subdir", "standalone")
    return pufferl.eval(
        env_name=env_name,
        args=args,
        policy=adapter,
        eval_output_dir=str(output_dir / "eval"),
        eval_output_subdir=str(output_subdir),
        use_training_config=True,
        benchmark_names=list(benchmark_names),
    )


def _materialize_benchmark_config(
    benchmark_config_path: Any,
    *,
    output_dir: Path,
    benchmark_names: Sequence[str],
    env_overrides: Mapping[str, Any],
    episode_timesteps: Any,
    num_scenarios: int,
    seed: Optional[int] = None,
) -> Optional[Path]:
    """Apply native timing/step overrides without mutating the shared catalog."""

    if not isinstance(benchmark_config_path, (str, Path)):
        return None
    source_path = Path(benchmark_config_path)
    if not source_path.is_file() and not source_path.is_absolute():
        try:
            from .runtime import resolve_path

            source_path = resolve_path(source_path)
        except (ImportError, OSError):
            pass
    if not source_path.is_file() or (not env_overrides and episode_timesteps is None and num_scenarios is None):
        return None
    try:
        import yaml
    except ImportError as exc:  # pragma: no cover - project dependency
        raise RuntimeError("PyYAML is required to resolve benchmark overrides") from exc
    with source_path.open("r", encoding="utf-8") as config_file:
        benchmark_config = yaml.safe_load(config_file)
    if not isinstance(benchmark_config, Mapping):
        raise ValueError("benchmark config must contain a mapping")
    benchmark_config = copy.deepcopy(dict(benchmark_config))
    global_env = benchmark_config.get("env")
    if not isinstance(global_env, Mapping):
        raise ValueError("benchmark config must contain an env mapping")
    global_env = dict(global_env)
    original_global_env = dict(global_env)
    global_env.update(copy.deepcopy(dict(env_overrides)))
    benchmark_config["env"] = global_env
    benchmarks = benchmark_config.get("benchmarks")
    if not isinstance(benchmarks, list):
        raise ValueError("benchmark config must contain a benchmarks list")
    selected = set(benchmark_names)
    for benchmark in benchmarks:
        if not isinstance(benchmark, Mapping) or benchmark.get("name") not in selected:
            continue
        benchmark_env = benchmark.get("env")
        if not isinstance(benchmark_env, Mapping):
            raise ValueError(f"benchmark {benchmark.get('name')!r} must contain an env mapping")
        benchmark_env = dict(benchmark_env)
        benchmark["num_scenarios"] = _positive_int(num_scenarios, "evaluation.num_scenarios")
        if seed is not None:
            benchmark["seed"] = seed
        requested_dt = env_overrides.get("dt")
        if episode_timesteps is None and requested_dt is not None and benchmark_env.get("simulation_mode") == "gigaflow":
            old_dt = float(benchmark_env.get("dt", original_global_env.get("dt", requested_dt)))
            new_dt = float(requested_dt)
            if old_dt <= 0.0 or new_dt <= 0.0:
                raise ValueError("benchmark dt values must be positive")
            old_steps = benchmark_env.get("scenario_length")
            if isinstance(old_steps, int) and old_steps > 0 and old_dt != new_dt:
                converted_steps = old_steps * old_dt / new_dt
                if not math.isclose(converted_steps, round(converted_steps), rel_tol=0.0, abs_tol=1e-9):
                    raise ValueError(
                        f"benchmark {benchmark.get('name')!r} scenario_length does not convert "
                        f"integrally from dt={old_dt} to dt={new_dt}"
                    )
                benchmark_env["scenario_length"] = int(round(converted_steps))
        if episode_timesteps is not None:
            benchmark_env["scenario_length"] = _positive_int(
                episode_timesteps, "evaluation.episode_timesteps"
            )
        for key, value in env_overrides.items():
            if key == "dt" and benchmark_env.get("simulation_mode") != "gigaflow":
                continue
            benchmark_env[key] = copy.deepcopy(value)
        benchmark["env"] = benchmark_env
    resolved_path = output_dir / "benchmark_config_resolved.yaml"
    with resolved_path.open("w", encoding="utf-8") as config_file:
        yaml.safe_dump(benchmark_config, config_file, sort_keys=False)
    return resolved_path


def _evaluation_run_dir(config: Mapping[str, Any], evaluation_config: Mapping[str, Any]) -> Path:
    output_root = evaluation_config.get(
        "output_root", config.get("output_root", "experiments/jepa_distill/runs")
    )
    run_id = evaluation_config.get("run_id") or config.get("run_id") or "student_eval"
    if not isinstance(output_root, (str, Path)) or not isinstance(run_id, str) or not run_id.strip():
        raise ValueError("evaluation output_root/run_id must identify a directory")
    return Path(output_root) / run_id


def _evaluate_injected_env(
    adapter: StudentPolicyAdapter,
    env: Any,
    evaluation_config: Mapping[str, Any],
    *,
    num_scenarios: int,
) -> Mapping[str, Any]:
    max_steps = evaluation_config.get(
        "episode_timesteps", evaluation_config.get("max_episode_steps")
    )
    max_steps = _positive_int(max_steps, "evaluation.episode_timesteps")
    action_selection = evaluation_config.get("action_selection", "mean")
    if action_selection not in ("mean", "mode", "sample"):
        raise ValueError("evaluation.action_selection must be mean, mode, or sample")
    infos: list[Mapping[str, Any]] = []
    total_steps = 0
    completed = 0
    while completed < num_scenarios:
        reset_result = env.reset()
        observations = reset_result[0] if isinstance(reset_result, tuple) else reset_result
        for _ in range(max_steps):
            observation_tensor = torch.as_tensor(observations)
            if observation_tensor.ndim == 1:
                observation_tensor = observation_tensor.unsqueeze(0)
            with torch.no_grad():
                (logits,), _ = adapter.forward_eval(observation_tensor)
                probabilities = torch.softmax(logits, dim=-1)
                if action_selection == "mean":
                    action = adapter.discrete_probs_to_continuous_mean(probabilities)
                elif action_selection == "mode":
                    action = adapter.discrete_actions_to_continuous(torch.argmax(probabilities, dim=-1))
                else:
                    action = adapter.discrete_actions_to_continuous(
                        torch.multinomial(probabilities, num_samples=1).squeeze(-1)
                    )
            environment_action = action.detach().cpu().numpy()
            step_result = env.step(environment_action)
            if not isinstance(step_result, tuple) or len(step_result) not in (4, 5):
                raise ValueError("injected evaluation env.step must return a 4- or 5-tuple")
            observations = step_result[0]
            if len(step_result) == 5:
                _, _, terminated, truncated, info = step_result
                done = _done_any(terminated) or _done_any(truncated)
            else:
                _, _, done, info = step_result
                done = _done_any(done)
            if isinstance(info, Mapping):
                infos.append(info)
            total_steps += 1
            if done:
                completed += 1
                break
        else:
            completed += 1

    benchmark_name = _benchmark_names(evaluation_config)[0]
    metrics: dict[str, Any] = {
        "num_scenarios": completed,
        "num_timesteps": total_steps,
    }
    numeric_values: dict[str, list[float]] = {}
    for info in infos:
        for name, value in info.items():
            if isinstance(value, bool) or not isinstance(value, (int, float)):
                continue
            if math.isfinite(float(value)):
                numeric_values.setdefault(name, []).append(float(value))
    metrics.update({name: sum(values) / len(values) for name, values in numeric_values.items() if values})
    return {
        benchmark_name: {
            "episodes": infos,
            "summary": {"num_scenarios": completed, "num_episodes": completed, "metrics_mean": metrics},
        }
    }


def _done_any(value: Any) -> bool:
    if isinstance(value, torch.Tensor):
        return bool(value.any().item())
    try:
        return bool(value.any())
    except AttributeError:
        return bool(value)


def _log_benchmark_results(
    monitor: Any,
    results: Mapping[str, Any],
    benchmark_names: Sequence[str],
    progress: Any,
    evaluation_config: Mapping[str, Any],
) -> None:
    output_name = str(evaluation_config.get("output_name", "student_eval"))
    for benchmark_name in benchmark_names:
        result = results.get(benchmark_name)
        if not isinstance(result, Mapping):
            continue
        summary = result.get("summary")
        if not isinstance(summary, Mapping):
            continue
        metrics = summary.get("metrics_mean")
        if not isinstance(metrics, Mapping):
            continue
        monitor.log_evaluation(
            metrics,
            benchmark_name=benchmark_name,
            output_name=output_name,
            policy_role="student",
            progress=progress,
        )


def _extract_chunk_logits(raw_logits: Any) -> torch.Tensor:
    if hasattr(raw_logits, "chunk_logits"):
        raw_logits = raw_logits.chunk_logits
    elif isinstance(raw_logits, Mapping):
        for key in ("chunk_logits", "logits"):
            if key in raw_logits:
                raw_logits = raw_logits[key]
                break
    elif isinstance(raw_logits, (tuple, list)):
        if not raw_logits:
            raise ValueError("student returned an empty output tuple")
        raw_logits = raw_logits[0]
        if isinstance(raw_logits, (tuple, list)):
            if not raw_logits:
                raise ValueError("student returned an empty logits tuple")
            raw_logits = raw_logits[0]
    if not isinstance(raw_logits, torch.Tensor):
        raise TypeError("student output must expose a tensor of chunk logits")
    return raw_logits


def _as_action_table(value: Any, name: str) -> torch.Tensor:
    table = torch.as_tensor(value, dtype=torch.float32).detach().clone()
    if table.ndim != 2 or table.shape[1] != 2 or table.shape[0] <= 0:
        raise ValueError(f"{name} must have shape [num_classes, 2]")
    return table


def _default_jerk_physical_table() -> torch.Tensor:
    longitudinal = torch.tensor([-15.0, -4.0, 0.0, 4.0], dtype=torch.float32)
    lateral = torch.tensor([-4.0, 0.0, 4.0], dtype=torch.float32)
    indices = torch.arange(12)
    return torch.stack([longitudinal[indices // 3], lateral[indices % 3]], dim=-1)


def _resolve_scales(student: nn.Module, table: torch.Tensor) -> tuple[float, float, float]:
    values = []
    for name in ("action_long_neg_scale", "action_long_pos_scale", "action_lat_scale"):
        value = getattr(student, name, None)
        try:
            values.append(float(value))
        except (TypeError, ValueError):
            values.append(float("nan"))
    long_values = table[:, 0]
    lat_values = table[:, 1]
    inferred = (
        float(max(abs(float(long_values.min())), 1.0)),
        float(max(abs(float(long_values.max())), 1.0)),
        float(max(abs(float(lat_values.min())), abs(float(lat_values.max())), 1.0)),
    )
    return tuple(
        value if math.isfinite(value) and value > 0 else inferred[index]
        for index, value in enumerate(values)
    )


def _has_explicit_scales(student: nn.Module) -> bool:
    return all(
        getattr(student, name, None) is not None
        for name in ("action_long_neg_scale", "action_long_pos_scale", "action_lat_scale")
    )


def _normalize_physical_table(
    physical: torch.Tensor, scales: tuple[float, float, float]
) -> torch.Tensor:
    neg, pos, lat = scales
    long_values = physical[:, 0]
    normalized_long = torch.where(long_values < 0, long_values / neg, long_values / pos)
    return torch.stack([normalized_long, physical[:, 1] / lat], dim=-1).clamp(-1.0, 1.0)


def _physical_from_normalized_table(
    normalized: torch.Tensor, scales: tuple[float, float, float]
) -> torch.Tensor:
    neg, pos, lat = scales
    normalized_long = normalized[:, 0]
    physical_long = torch.where(normalized_long < 0, normalized_long * neg, normalized_long * pos)
    return torch.stack([physical_long, normalized[:, 1] * lat], dim=-1)


def _json_default(value: Any) -> Any:
    if isinstance(value, torch.Tensor):
        return value.detach().cpu().tolist()
    if isinstance(value, Mapping):
        return {str(key): _json_default(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_default(item) for item in value]
    return value


def _load_yaml(path: Path) -> Mapping[str, Any]:
    try:
        import yaml
    except ImportError as exc:  # pragma: no cover - project dependency
        raise RuntimeError("CLI config loading requires PyYAML") from exc
    with path.open("r", encoding="utf-8") as config_file:
        config = yaml.safe_load(config_file)
    if not isinstance(config, Mapping):
        raise ValueError(f"evaluation config must contain a mapping: {path}")
    return config


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = argparse.ArgumentParser(description="Evaluate a Condition B student checkpoint")
    parser.add_argument("--config", required=True, help="YAML evaluation config")
    parser.add_argument("--checkpoint", help="Condition B student checkpoint")
    parser.add_argument("--num-scenarios", type=int, help="Positive scenario limit")
    parser.add_argument("--episode-timesteps", type=int, help="Bounded toy/native episode steps")
    parser.add_argument("--device", help="Checkpoint/device override")
    parser.add_argument("--wandb-disabled", action="store_true", help="Use local metrics only")
    args = parser.parse_args(argv)
    config = copy.deepcopy(dict(_load_yaml(Path(args.config))))
    evaluation_config = dict(config.get("evaluation", config))
    config["evaluation"] = evaluation_config
    if args.checkpoint:
        evaluation_config["student_checkpoint"] = args.checkpoint
    if args.num_scenarios is not None:
        evaluation_config["num_scenarios"] = args.num_scenarios
    if args.episode_timesteps is not None:
        evaluation_config["episode_timesteps"] = args.episode_timesteps
    if args.device:
        evaluation_config["device"] = args.device
    if args.wandb_disabled:
        evaluation_config["wandb"] = {"enabled": False, "mode": "disabled"}
    results = evaluate_student(config)
    print(json.dumps(results, indent=2, default=_json_default, sort_keys=True))
    return 0


if __name__ == "__main__":  # pragma: no cover - exercised by the launcher
    raise SystemExit(main())
