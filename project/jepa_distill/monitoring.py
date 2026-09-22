"""Small, checkpointable monitoring adapter for Condition B.

The training code owns one :class:`WandbMonitor` for the whole student run.
The W&B import is deliberately kept inside ``__init__`` so local and disabled
training do not require a working W&B installation.  Every event is also
written to ``metrics.jsonl`` below the student run directory; this makes
disabled/offline runs inspectable and gives checkpoint recovery a local cursor
to reconcile.
"""

from __future__ import annotations

import json
import math
import numbers
import uuid
from pathlib import Path
from typing import Any, Literal, Mapping, NamedTuple, Optional, Union


class MetricProgress(NamedTuple):
    """Scientific counters, separate from the monotonically increasing log event.

    ``optimizer_step`` counts completed updates; ``simulator_transitions`` counts
    training agent-transitions, not reused windows or evaluation interactions.
    Round and reuse-epoch indices are zero-based. See implementation_plan.md §8.
    """

    optimizer_step: int
    simulator_transitions: int
    collection_round_idx: int
    update_epoch_idx: int


class WandbMonitor:
    """Own a dedicated student run and its metric history.

    Fresh training creates a new student run.  When a checkpoint contains
    monitoring state, its project/entity/run ID are authoritative and online
    resume uses W&B's strict ``resume="must"`` mode.  Teacher run metadata is
    never consulted.
    """

    def __init__(
        self,
        config: Mapping[str, Any],
        resolved_config: Mapping[str, Any],
        run_dir: Union[str, Path],
        *,
        checkpoint_state: Optional[Mapping[str, Any]] = None,
    ) -> None:
        """Create a local logger and, when enabled, one dedicated W&B run.

        ``checkpoint_state`` is the value previously returned by
        :meth:`state_dict` (or a checkpoint mapping containing that value).
        Passing a state for an enabled monitor therefore means *resume*; an
        absent run ID is an error instead of silently creating another run.
        """

        if not isinstance(config, Mapping):
            raise TypeError("WandbMonitor config must be a mapping")
        if not isinstance(resolved_config, Mapping):
            raise TypeError("WandbMonitor resolved_config must be a mapping")

        self.run_dir = Path(run_dir)
        self.run_dir.mkdir(parents=True, exist_ok=True)
        self.metrics_path = self.run_dir / "metrics.jsonl"
        self._metrics_file = self.metrics_path.open("a", encoding="utf-8")
        self._finished = False
        self._wandb = None
        self.wandb = None  # Publicly useful for diagnostics; absent when disabled.
        self.run = None
        self._owns_run = False
        self._last_progress: Optional[dict[str, int]] = None
        self._reconciled_ahead = False
        self._remote_last_event_index: Optional[int] = None

        state = _unwrap_checkpoint_state(checkpoint_state)
        requested_mode = _monitor_mode(config)
        saved_mode = _optional_string(state.get("mode")) if state is not None else None
        if requested_mode != "disabled" and saved_mode is not None:
            if saved_mode not in _VALID_MODES:
                raise ValueError(f"checkpoint monitoring mode is invalid: {saved_mode!r}")
            if saved_mode == "disabled":
                raise ValueError("cannot resume an enabled W&B monitor from disabled checkpoint state")
            if requested_mode != saved_mode:
                raise ValueError(
                    "monitoring mode mismatch on resume: "
                    f"checkpoint={saved_mode!r}, requested={requested_mode!r}"
                )
        self.mode = requested_mode
        self.enabled = self.mode != "disabled"

        saved_run_id = _checkpoint_run_id(state) if state is not None else None
        self._offline_parent_run_id = saved_run_id if self.mode == "offline" else None
        if self.enabled and state is not None and not saved_run_id:
            raise ValueError("enabled monitoring resume requires the saved student W&B run ID")

        saved_project = (
            _optional_string(state.get("project") or state.get("wandb_project"))
            if state is not None
            else None
        )
        saved_entity = (
            _optional_string(state.get("entity") or state.get("wandb_entity"))
            if state is not None
            else None
        )
        requested_project = _optional_string(config.get("project"))
        requested_entity = _optional_string(config.get("entity"))
        if saved_project and requested_project and saved_project != requested_project:
            raise ValueError(
                "student W&B project mismatch on resume: "
                f"checkpoint={saved_project!r}, requested={requested_project!r}"
            )
        if saved_entity and requested_entity and saved_entity != requested_entity:
            raise ValueError(
                "student W&B entity mismatch on resume: "
                f"checkpoint={saved_entity!r}, requested={requested_entity!r}"
            )
        self.project = saved_project or requested_project or "pufferdrive"
        self.entity = saved_entity or requested_entity
        self.group = (
            _optional_string(state.get("group")) if state is not None else None
        ) or _optional_string(config.get("group"))
        self.name = (
            _optional_string(state.get("name")) if state is not None else None
        ) or _optional_string(config.get("name")) or self.run_dir.name

        checkpoint_last_event = _checkpoint_last_event(state)
        local_last_event = _read_last_event_index(self.metrics_path)
        checkpoint_metrics_path = _checkpoint_metrics_path(state)
        checkpoint_local_last_event = (
            _read_last_event_index(checkpoint_metrics_path)
            if checkpoint_metrics_path is not None and checkpoint_metrics_path != self.metrics_path
            else -1
        )
        self._last_event_index = max(checkpoint_last_event, local_last_event, checkpoint_local_last_event)
        self._reconciled_ahead = max(local_last_event, checkpoint_local_last_event) > checkpoint_last_event
        if state is not None and isinstance(state.get("last_progress"), Mapping):
            self._last_progress = _coerce_progress_mapping(state["last_progress"])

        if not self.enabled:
            self.run_id = saved_run_id if saved_run_id and saved_mode == "disabled" else None
            self.run_url = _optional_string(state.get("run_url")) if state is not None else None
            return

        # This is the first and only SDK import in this module.
        try:
            import wandb as wandb_sdk
        except ImportError as exc:  # pragma: no cover - exercised by environment setup
            self._metrics_file.close()
            raise RuntimeError(
                "W&B monitoring is enabled but the 'wandb' package is unavailable; "
                "set wandb.enabled=false or wandb.mode=disabled for local logging"
            ) from exc

        self._wandb = wandb_sdk
        self.wandb = wandb_sdk
        is_resume = state is not None
        self.run_id = saved_run_id or _generate_run_id(wandb_sdk)
        init_kwargs = {
            "project": self.project,
            "name": self.name,
            "id": self.run_id,
            # W&B intentionally creates a new offline segment. Its SDK may
            # replace the requested ID, so strict identity applies to online
            # resume only.
            "resume": "must" if is_resume and self.mode == "online" else "allow",
            "config": _sanitize_config(resolved_config),
        }
        if self.entity is not None:
            init_kwargs["entity"] = self.entity
        if self.group is not None:
            init_kwargs["group"] = self.group
        tags = config.get("tags")
        if tags is not None:
            if isinstance(tags, str):
                tags = [tags]
            if not isinstance(tags, (list, tuple)) or any(not isinstance(tag, str) for tag in tags):
                raise ValueError("wandb.tags must be a string or a sequence of strings")
            init_kwargs["tags"] = list(tags)
        if self.mode == "offline":
            init_kwargs["mode"] = "offline"
        try:
            self.run = wandb_sdk.init(**init_kwargs)
        except Exception:
            self._metrics_file.close()
            raise
        if self.run is None:
            self.run = getattr(wandb_sdk, "run", None)
        self._owns_run = self.run is not None or hasattr(wandb_sdk, "finish")
        actual_run_id = _optional_string(getattr(self.run, "id", None))
        if actual_run_id is not None:
            if is_resume and self.mode == "online" and actual_run_id != self.run_id:
                self._close_remote_after_init_failure()
                raise RuntimeError(
                    "W&B returned a different run ID while resuming the student run: "
                    f"expected={self.run_id!r}, actual={actual_run_id!r}"
                )
            self.run_id = actual_run_id
        self.run_url = _optional_string(getattr(self.run, "url", None))
        remote_last_event = _remote_last_event_index(self.run)
        self._remote_last_event_index = remote_last_event
        if remote_last_event is not None and remote_last_event > self._last_event_index:
            self._last_event_index = remote_last_event
            self._reconciled_ahead = True

    def log_metrics(
        self,
        metrics: Mapping[str, Union[float, int]],
        *,
        progress: MetricProgress,
    ) -> None:
        """Log one finite scalar event and advance only the event cursor.

        The caller supplies already aggregated values.  This method intentionally
        does not average minibatches: unequal-sample aggregation belongs to the
        training loop where sample counts are available.
        """

        self._emit(metrics, progress=progress, event_type="metrics")

    def log_evaluation(
        self,
        metrics: Mapping[str, Union[float, int]],
        *,
        benchmark_name: str,
        output_name: str,
        policy_role: Literal["student", "teacher"],
        progress: MetricProgress,
    ) -> None:
        """Record namespaced driving metrics in the same student run."""

        if policy_role not in ("student", "teacher"):
            raise ValueError("policy_role must be 'student' or 'teacher'")
        if not isinstance(benchmark_name, str) or not benchmark_name.strip():
            raise ValueError("benchmark_name must be a non-empty string")
        if not isinstance(output_name, str) or not output_name.strip():
            raise ValueError("output_name must be a non-empty string")
        if not isinstance(metrics, Mapping):
            raise TypeError("metrics must be a mapping")
        namespaced = {
            f"eval/{output_name}/{benchmark_name}/{policy_role}/{name}": value
            for name, value in metrics.items()
        }
        self._emit(namespaced, progress=progress, event_type="evaluation")

    def state_dict(self) -> Mapping[str, Any]:
        """Return checkpoint-safe identity and cursor metadata.

        ``event_index`` is the last emitted event; ``event_cursor`` is the next
        value.  They are deliberately separate from
        ``last_progress.optimizer_step`` and the other scientific counters.
        """

        return {
            "version": 1,
            "mode": self.mode,
            "enabled": self.enabled,
            "project": self.project,
            "entity": self.entity,
            "group": self.group,
            "name": self.name,
            "run_id": self.run_id,
            "run_name": self.name,
            "wandb_project": self.project,
            "wandb_entity": self.entity,
            "wandb_run_id": self.run_id,
            "run_url": self.run_url,
            "offline_parent_run_id": self._offline_parent_run_id,
            "event_index": self._last_event_index,
            "event_cursor": self._last_event_index + 1,
            "last_event_index": self._last_event_index,
            "last_progress": dict(self._last_progress) if self._last_progress is not None else None,
            "reconciled_local_history_ahead": self._reconciled_ahead,
            "remote_last_event_index": self._remote_last_event_index,
            "metrics_path": str(self.metrics_path),
        }

    def finish(self, *, exit_code: int = 0) -> None:
        """Flush local events and close this monitor once."""

        if self._finished:
            return
        self._finished = True
        try:
            self._metrics_file.flush()
        finally:
            self._metrics_file.close()
        if not self.enabled or not self._owns_run:
            return
        finish = getattr(self.run, "finish", None)
        if callable(finish):
            try:
                finish(exit_code=exit_code)
            except TypeError:
                finish()
            return
        sdk_finish = getattr(self._wandb, "finish", None)
        if callable(sdk_finish):
            try:
                sdk_finish(exit_code=exit_code)
            except TypeError:
                sdk_finish()

    def _emit(
        self,
        metrics: Mapping[str, Union[float, int]],
        *,
        progress: MetricProgress,
        event_type: str,
    ) -> None:
        if self._finished:
            raise RuntimeError("cannot log after WandbMonitor.finish()")
        normalized_metrics = _normalize_metrics(metrics)
        progress_dict = _progress_dict(progress)
        event_index = self._last_event_index + 1
        payload = dict(normalized_metrics)
        payload.update({f"progress/{name}": value for name, value in progress_dict.items()})
        payload["event_index"] = event_index

        local_event = dict(payload)
        local_event["event_type"] = event_type
        local_event["metrics"] = dict(normalized_metrics)
        local_event["progress"] = progress_dict
        self._metrics_file.write(json.dumps(local_event, sort_keys=True) + "\n")
        self._metrics_file.flush()

        # Reserve the cursor before the remote call. A transient upload error
        # must not cause the next local event to reuse this history index.
        self._last_event_index = event_index
        self._last_progress = progress_dict
        if self.enabled:
            self._remote_log(payload, event_index)

    def _remote_log(self, payload: Mapping[str, Any], event_index: int) -> None:
        logger = self.run if self.run is not None else self._wandb
        log = getattr(logger, "log", None)
        if not callable(log) and self._wandb is not None:
            logger = self._wandb
            log = getattr(logger, "log", None)
        if not callable(log):
            raise RuntimeError("enabled W&B monitor has no log method")
        try:
            log(dict(payload), step=event_index)
        except TypeError:
            # Tiny SDK fakes and older wrappers may not accept W&B's step kwarg.
            log(dict(payload))

    def _close_remote_after_init_failure(self) -> None:
        finish = getattr(self.run, "finish", None)
        if callable(finish):
            finish()


_VALID_MODES = {"online", "offline", "disabled"}


def _monitor_mode(config: Mapping[str, Any]) -> str:
    enabled = config.get("enabled", True)
    if not isinstance(enabled, bool):
        raise ValueError("wandb.enabled must be a boolean")
    mode = config.get("mode", "online")
    if not isinstance(mode, str) or mode.lower() not in _VALID_MODES:
        raise ValueError("wandb.mode must be one of online, offline, or disabled")
    mode = mode.lower()
    return "disabled" if not enabled or mode == "disabled" else mode


def _optional_string(value: Any) -> Optional[str]:
    if value is None:
        return None
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"expected a non-empty string, got {value!r}")
    return value.strip()


def _unwrap_checkpoint_state(state: Optional[Mapping[str, Any]]) -> Optional[Mapping[str, Any]]:
    if state is None:
        return None
    if not isinstance(state, Mapping):
        raise TypeError("checkpoint_state must be a mapping")
    nested = state.get("monitoring_state")
    if nested is not None:
        if not isinstance(nested, Mapping):
            raise TypeError("checkpoint_state.monitoring_state must be a mapping")
        return nested
    return state


def _checkpoint_run_id(state: Optional[Mapping[str, Any]]) -> Optional[str]:
    if state is None:
        return None
    for key in ("run_id", "wandb_run_id", "id"):
        value = state.get(key)
        if value is not None:
            return _optional_string(value)
    return None


def _checkpoint_metrics_path(state: Optional[Mapping[str, Any]]) -> Optional[Path]:
    if state is None:
        return None
    value = state.get("metrics_path")
    if value is None:
        return None
    if not isinstance(value, (str, Path)) or not str(value).strip():
        raise ValueError("checkpoint metrics_path must be a non-empty path")
    return Path(value)


def _checkpoint_last_event(state: Optional[Mapping[str, Any]]) -> int:
    if state is None:
        return -1
    candidates = []
    for key in ("last_event_index", "event_index"):
        value = state.get(key)
        if value is not None:
            if value == -1:
                candidates.append(-1)
            else:
                candidates.append(_nonnegative_int(value, f"checkpoint {key}", allow_zero=True))
    cursor = state.get("event_cursor")
    if cursor is not None:
        cursor_value = _nonnegative_int(cursor, "checkpoint event_cursor", allow_zero=True)
        candidates.append(cursor_value - 1)
    return max(candidates, default=-1)


def _nonnegative_int(value: Any, label: str, *, allow_zero: bool = True) -> int:
    if isinstance(value, bool) or not isinstance(value, numbers.Integral):
        raise ValueError(f"{label} must be an integer")
    value = int(value)
    if (value < 0 and allow_zero) or (value <= 0 and not allow_zero):
        raise ValueError(f"{label} must be {'non-negative' if allow_zero else 'positive'}")
    return value


def _progress_dict(progress: MetricProgress) -> dict[str, int]:
    if isinstance(progress, MetricProgress):
        values = progress._asdict()
    elif isinstance(progress, Mapping):
        values = dict(progress)
    else:
        raise TypeError("progress must be MetricProgress or a mapping")
    expected = set(MetricProgress._fields)
    if set(values) != expected:
        raise ValueError(f"progress must contain exactly {sorted(expected)}")
    return {
        name: _nonnegative_int(value, f"progress {name}")
        for name, value in values.items()
    }


def _coerce_progress_mapping(progress: Mapping[str, Any]) -> dict[str, int]:
    return _progress_dict(progress)


def _normalize_metrics(metrics: Mapping[str, Union[float, int]]) -> dict[str, Union[float, int]]:
    if not isinstance(metrics, Mapping):
        raise TypeError("metrics must be a mapping")
    normalized: dict[str, Union[float, int]] = {}
    for name, value in metrics.items():
        if not isinstance(name, str) or not name.strip():
            raise ValueError("metric names must be non-empty strings")
        scalar = _scalar_value(value)
        if not math.isfinite(float(scalar)):
            raise ValueError(f"non-finite metric {name!r}: {scalar!r}")
        normalized[name] = scalar
    return normalized


def _scalar_value(value: Any) -> Union[float, int]:
    if isinstance(value, bool):
        raise TypeError("boolean values are not valid scalar metrics")
    # Accept detached torch/numpy scalar values without importing either package.
    detach = getattr(value, "detach", None)
    if callable(detach):
        value = detach()
    numel = getattr(value, "numel", None)
    if callable(numel) and int(numel()) != 1:
        raise TypeError("metrics must contain scalar values")
    item = getattr(value, "item", None)
    if callable(item):
        value = item()
    if isinstance(value, bool) or not isinstance(value, numbers.Real):
        raise TypeError(f"metric values must be real scalars, got {type(value).__name__}")
    if isinstance(value, numbers.Integral):
        return int(value)
    return float(value)


def _generate_run_id(wandb_sdk: Any) -> str:
    util = getattr(wandb_sdk, "util", None)
    generator = getattr(util, "generate_id", None)
    if callable(generator):
        value = generator()
        if isinstance(value, str) and value:
            return value
    return uuid.uuid4().hex[:8]


def _read_last_event_index(path: Path) -> int:
    if not path.exists():
        return -1
    last = -1
    try:
        with path.open("r", encoding="utf-8") as metrics_file:
            for line in metrics_file:
                if not line.strip():
                    continue
                try:
                    event = json.loads(line)
                except json.JSONDecodeError:
                    continue
                value = event.get("event_index")
                if isinstance(value, int) and not isinstance(value, bool) and value >= 0:
                    last = max(last, value)
    except OSError:
        return -1
    return last


def _remote_last_event_index(run: Any) -> Optional[int]:
    """Read a cursor exposed by SDK fakes or W&B's internal run step.

    W&B history is remote state and is intentionally optional here. If an SDK
    exposes no cursor, checkpoint/local reconciliation remains conservative.
    """

    if run is None:
        return None
    # ``event_cursor`` names the next event, while the other fields mirror the
    # last history row. Check the explicit cursor first, then SDK/fake fields.
    for source in (run, getattr(run, "summary", None)):
        if source is None:
            continue
        for name in ("event_cursor", "history_event_cursor"):
            value = source.get(name) if isinstance(source, Mapping) else getattr(source, name, None)
            if isinstance(value, numbers.Integral) and not isinstance(value, bool) and value >= 0:
                return int(value) - 1
        for name in ("last_event_index", "event_index", "history_event_index", "_step", "step"):
            value = source.get(name) if isinstance(source, Mapping) else getattr(source, name, None)
            if isinstance(value, numbers.Integral) and not isinstance(value, bool) and value >= 0:
                return int(value)
    return None


def _sanitize_config(value: Any, *, key: Optional[str] = None) -> Any:
    """Copy JSON-like config while dropping credentials and SDK identities."""

    lowered = key.lower() if isinstance(key, str) else ""
    if any(secret in lowered for secret in ("api_key", "apikey", "token", "password", "secret")):
        return "<redacted>"
    if isinstance(value, Mapping):
        return {str(name): _sanitize_config(item, key=str(name)) for name, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_sanitize_config(item, key=key) for item in value]
    if isinstance(value, (str, int, float, bool)) or value is None:
        if isinstance(value, float) and not math.isfinite(value):
            raise ValueError(f"resolved config contains non-finite value under {key!r}")
        return value
    return str(value)
