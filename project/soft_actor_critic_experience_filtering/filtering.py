"""Distributed TD-error selection for SAC learner updates.

The learner owns target construction and replay access. This module owns the
filter's configuration, global threshold state, selection counters, and resume
state so the scoring contract can be tested independently of training.
"""

from collections.abc import Mapping
import math
import numbers

import torch
from torch import nn


FILTER_STATE_VERSION = 2
FILTER_SCORE = "twin_max_abs_td_error"
FILTER_TARGET = "actor_critic"
LEGACY_FILTER_TARGET = "critic"
FILTER_THRESHOLD_MODE = "ewma_max"
DEFAULT_EWMA_BETA = 0.25
DEFAULT_THRESHOLD_SCALE = 0.01

_CONFIG_KEYS = frozenset({
    "enabled", "score", "apply_to", "threshold_mode", "ewma_beta", "threshold_scale",
})
_COUNT_NAMES = (
    "candidate_count", "retained_count", "filtered_count", "empty_batches",
    "local_empty_ranks", "critic_steps", "actor_rows", "actor_steps",
    "alpha_steps", "target_steps",
)
_LEGACY_COUNT_NAMES = _COUNT_NAMES[:6]


def _finite_real(value):
    if not isinstance(value, numbers.Real) or isinstance(value, bool):
        return False
    try:
        return math.isfinite(float(value))
    except OverflowError:
        return False


def normalize_experience_filter_config(value):
    """Validate and normalize ``train.experience_filter``.

    ``None`` and an explicitly disabled mapping both normalize to ``None``.
    A present mapping opts in unless it explicitly sets ``enabled: false``.
    """
    if value is None:
        return None
    if not isinstance(value, Mapping):
        raise ValueError("experience_filter must be a mapping or None")
    unknown_keys = set(value) - _CONFIG_KEYS
    if unknown_keys:
        raise ValueError(
            "Unsupported experience_filter fields: " + ", ".join(sorted(unknown_keys))
        )

    enabled = value.get("enabled", True)
    if not isinstance(enabled, bool):
        raise ValueError("experience_filter.enabled must be a boolean")
    if value.get("score", FILTER_SCORE) != FILTER_SCORE:
        raise ValueError(f"experience_filter.score must be {FILTER_SCORE!r}")
    apply_to = value.get("apply_to", FILTER_TARGET)
    if apply_to not in (FILTER_TARGET, LEGACY_FILTER_TARGET):
        raise ValueError(
            "experience_filter.apply_to must be "
            f"{FILTER_TARGET!r} or {LEGACY_FILTER_TARGET!r}"
        )
    if value.get("threshold_mode", FILTER_THRESHOLD_MODE) != FILTER_THRESHOLD_MODE:
        raise ValueError(
            f"experience_filter.threshold_mode must be {FILTER_THRESHOLD_MODE!r}"
        )

    ewma_beta = value.get("ewma_beta", DEFAULT_EWMA_BETA)
    threshold_scale = value.get("threshold_scale", DEFAULT_THRESHOLD_SCALE)
    if not _finite_real(ewma_beta) or not 0 < float(ewma_beta) <= 1:
        raise ValueError("experience_filter.ewma_beta must be finite and in (0, 1]")
    if not _finite_real(threshold_scale) or float(threshold_scale) < 0:
        raise ValueError("experience_filter.threshold_scale must be finite and nonnegative")
    if not enabled:
        return None

    return {
        "enabled": True,
        "score": FILTER_SCORE,
        "apply_to": apply_to,
        "threshold_mode": FILTER_THRESHOLD_MODE,
        "ewma_beta": float(ewma_beta),
        "threshold_scale": float(threshold_scale),
    }


def _validate_state_free_filter_forward(module, role):
    """Reject modules whose filter-only participation forward changes state."""
    batch_norm_type = nn.modules.batchnorm._BatchNorm
    dropout_type = nn.modules.dropout._DropoutNd
    unsupported = []
    for module_name, module in module.named_modules():
        if not module_name:
            module_name = type(module).__name__
        if isinstance(module, batch_norm_type):
            unsupported.append(f"{module_name} ({type(module).__name__})")
        elif isinstance(module, dropout_type):
            unsupported.append(f"{module_name} ({type(module).__name__})")
    if unsupported:
        names = ", ".join(unsupported)
        raise ValueError(
            "experience_filter requires a state-free, deterministic "
            f"{role} forward; "
            f"found batch-normalization or dropout modules: {names}"
        )


def validate_critic_filter_forward(critic):
    """Reject built-in critic modules whose extra score forward changes state."""
    _validate_state_free_filter_forward(critic, "critic")


def validate_actor_filter_forward(actor):
    """Reject stateful actors used for empty-rank DDP participation."""
    _validate_state_free_filter_forward(actor, "actor")


class ExperienceFilter:
    """Maintain the shared EMA threshold and select retained learner rows."""

    def __init__(self, config, device):
        normalized = normalize_experience_filter_config(config)
        if normalized is None:
            raise ValueError("ExperienceFilter requires an enabled configuration")
        self.config = normalized
        self.device = torch.device(device)
        self.distributed = torch.distributed.is_available() and torch.distributed.is_initialized()
        self.world_size = torch.distributed.get_world_size() if self.distributed else 1
        self.ema_initialized = False
        self.ema_max = None
        self.scoring_batches = 0
        self.last_current_max = None
        self.last_threshold = None
        self._cumulative = self._empty_counters()
        self._cycle = self._empty_counters()
        self._actor_metric_history_complete = True

    @staticmethod
    def _empty_counters():
        return {name: 0 for name in _COUNT_NAMES}

    def begin_cycle(self):
        """Start counters for one learner train call without resetting history."""
        self._cycle = self._empty_counters()

    def validate_candidate_count(self, local_count):
        """Check the learner's synchronized nonempty logical-batch contract."""
        if not isinstance(local_count, int) or isinstance(local_count, bool) or local_count < 0:
            raise ValueError("SAC experience-filter candidate count must be a nonnegative integer")
        count_tensor = torch.tensor(local_count, device=self.device, dtype=torch.int64)
        if self.distributed:
            minimum_count = count_tensor.clone()
            maximum_count = count_tensor.clone()
            torch.distributed.all_reduce(minimum_count, op=torch.distributed.ReduceOp.MIN)
            torch.distributed.all_reduce(maximum_count, op=torch.distributed.ReduceOp.MAX)
            if minimum_count.item() == 0 or minimum_count.item() != maximum_count.item():
                raise RuntimeError(
                    "SAC experience filtering requires synchronized nonempty logical candidate "
                    "counts across ranks"
                )
        elif local_count == 0:
            raise ValueError("experience filtering requires a nonempty candidate batch")

    def select(self, scores, candidate_count_validated=False):
        """Return the local keep mask and shared metrics for one logical batch.

        ``scores`` contains one detached float32 twin-max absolute TD residual
        per original candidate row. Ranks must enter this method together and
        have the same nonempty candidate count, as required by the SAC update
        schedule.
        """
        if not isinstance(scores, torch.Tensor):
            raise TypeError("experience filter scores must be a torch.Tensor")
        scores = scores.detach().to(device=self.device, dtype=torch.float32).reshape(-1)
        if candidate_count_validated:
            if scores.numel() == 0:
                raise ValueError("validated experience-filter candidates must be nonempty")
        else:
            self.validate_candidate_count(scores.numel())

        finite_flag = (torch.isfinite(scores) & (scores >= 0)).all().to(dtype=torch.int32)
        if self.distributed:
            torch.distributed.all_reduce(finite_flag, op=torch.distributed.ReduceOp.MIN)
        if not finite_flag.item():
            raise FloatingPointError(
                "Nonfinite or negative SAC critic TD-error score detected on at least one rank; "
                "scores must be finite and nonnegative"
            )

        current_max_tensor = scores.max()
        if self.distributed:
            torch.distributed.all_reduce(current_max_tensor, op=torch.distributed.ReduceOp.MAX)
        if self.ema_initialized:
            beta = self.config["ewma_beta"]
            previous_ema = torch.tensor(self.ema_max, device=self.device, dtype=torch.float32)
            ema_tensor = beta * current_max_tensor + (1.0 - beta) * previous_ema
        else:
            ema_tensor = current_max_tensor
        threshold_tensor = self.config["threshold_scale"] * ema_tensor
        ema_max = float(ema_tensor.item())
        threshold = float(threshold_tensor.item())
        current_max = float(current_max_tensor.item())
        if not math.isfinite(ema_max) or not math.isfinite(threshold):
            raise FloatingPointError("Nonfinite SAC experience-filter EMA or threshold")
        keep_mask = scores >= threshold_tensor

        local_counts = torch.tensor(
            [scores.numel(), int(keep_mask.sum().item()), int((~keep_mask).sum().item()),
             int(keep_mask.sum().item() == 0)],
            device=self.device,
            dtype=torch.int64,
        )
        if self.distributed:
            torch.distributed.all_reduce(local_counts, op=torch.distributed.ReduceOp.SUM)
        candidate_count, retained_count, filtered_count, local_empty_ranks = (
            int(value) for value in local_counts.tolist()
        )
        empty_batch = int(retained_count == 0)
        row_counts = {
            "candidate_count": candidate_count,
            "retained_count": retained_count,
            "filtered_count": filtered_count,
            "empty_batches": empty_batch,
            "local_empty_ranks": local_empty_ranks,
        }
        for name, value in row_counts.items():
            self._cumulative[name] += value
            self._cycle[name] += value
        self.scoring_batches += 1
        self.ema_initialized = True
        self.ema_max = ema_max
        self.last_current_max = current_max
        self.last_threshold = threshold
        return keep_mask, {
            **row_counts,
            "current_max": current_max,
            "ema_max": ema_max,
            "threshold": threshold,
        }

    def mark_critic_step(self):
        """Record one globally shared optimizer step after a nonempty selection."""
        self._cumulative["critic_steps"] += 1
        self._cycle["critic_steps"] += 1

    def mark_actor_step(self, actor_rows):
        """Record a successful actor step and its global row exposure."""
        if not isinstance(actor_rows, int) or isinstance(actor_rows, bool) or actor_rows <= 0:
            raise ValueError("SAC experience-filter actor rows must be a positive integer")
        for counters in (self._cumulative, self._cycle):
            counters["actor_rows"] += actor_rows
            counters["actor_steps"] += 1

    def mark_alpha_step(self):
        """Record one successful temperature optimizer step."""
        self._cumulative["alpha_steps"] += 1
        self._cycle["alpha_steps"] += 1

    def mark_target_step(self):
        """Record one completed Polyak target update."""
        self._cumulative["target_steps"] += 1
        self._cycle["target_steps"] += 1

    def metrics(self):
        """Return row-weighted cycle and lifetime counters for production logs."""
        cycle_candidates = self._cycle["candidate_count"]
        cumulative_candidates = self._cumulative["candidate_count"]
        return {
            "filter/ewma_beta": self.config["ewma_beta"],
            "filter/threshold_scale": self.config["threshold_scale"],
            "filter/candidate_count": self._cycle["candidate_count"],
            "filter/retained_count": self._cycle["retained_count"],
            "filter/filtered_count": self._cycle["filtered_count"],
            "filter/retained_fraction": (
                self._cycle["retained_count"] / cycle_candidates if cycle_candidates else 0.0
            ),
            "filter/filtered_fraction": (
                self._cycle["filtered_count"] / cycle_candidates if cycle_candidates else 0.0
            ),
            "filter/empty_batches": self._cycle["empty_batches"],
            "filter/local_empty_ranks": self._cycle["local_empty_ranks"],
            "filter/update_steps": self._cycle["critic_steps"],
            "filter/actor_rows": self._cycle["actor_rows"],
            "filter/cumulative_candidate_count": self._cumulative["candidate_count"],
            "filter/cumulative_retained_count": self._cumulative["retained_count"],
            "filter/cumulative_filtered_count": self._cumulative["filtered_count"],
            "filter/cumulative_retained_fraction": (
                self._cumulative["retained_count"] / cumulative_candidates
                if cumulative_candidates else 0.0
            ),
            "filter/cumulative_filtered_fraction": (
                self._cumulative["filtered_count"] / cumulative_candidates
                if cumulative_candidates else 0.0
            ),
            "filter/cumulative_empty_batches": self._cumulative["empty_batches"],
            "filter/cumulative_local_empty_ranks": self._cumulative["local_empty_ranks"],
            "filter/cumulative_update_steps": self._cumulative["critic_steps"],
            "filter/cumulative_actor_rows": self._cumulative["actor_rows"],
            "filter/actor_metric_history_complete": self._actor_metric_history_complete,
            "filter/scoring_batches": self.scoring_batches,
            "filter/current_max": self.last_current_max,
            "filter/ema_max": self.ema_max,
            "filter/threshold": self.last_threshold,
        }

    def state_dict(self):
        """Return the complete, configuration-bound resume state."""
        return {
            "version": FILTER_STATE_VERSION,
            "config": dict(self.config),
            "ema_initialized": self.ema_initialized,
            "ema_max": self.ema_max,
            "scoring_batches": self.scoring_batches,
            "last_current_max": self.last_current_max,
            "last_threshold": self.last_threshold,
            "cumulative": dict(self._cumulative),
            "cycle": dict(self._cycle),
            "actor_metric_history_complete": self._actor_metric_history_complete,
        }

    def validate_state_dict(self, state):
        """Validate and normalize resume state without changing this object."""
        if not isinstance(state, Mapping):
            raise ValueError("Enabled SAC experience filter checkpoint state is missing")
        state_version = state.get("version")
        if (not isinstance(state_version, int) or isinstance(state_version, bool)
                or state_version not in (1, FILTER_STATE_VERSION)):
            raise ValueError("Unsupported SAC experience-filter checkpoint state version")
        if state.get("config") != self.config:
            raise ValueError("SAC experience-filter checkpoint settings do not match configuration")
        if state_version == 1 and self.config["apply_to"] != LEGACY_FILTER_TARGET:
            raise ValueError(
                "Legacy SAC experience-filter checkpoints require apply_to='critic'"
            )

        ema_initialized = state.get("ema_initialized")
        if not isinstance(ema_initialized, bool):
            raise ValueError("Invalid SAC experience-filter EMA initialization flag")
        ema_max = state.get("ema_max")
        if ema_initialized:
            if not _finite_real(ema_max) or float(ema_max) < 0:
                raise ValueError("Invalid SAC experience-filter EMA value")
            ema_max = float(ema_max)
        elif ema_max is not None:
            raise ValueError("Uninitialized SAC experience-filter EMA must not have a value")

        scoring_batches = state.get("scoring_batches")
        if not isinstance(scoring_batches, int) or isinstance(scoring_batches, bool) or scoring_batches < 0:
            raise ValueError("Invalid SAC experience-filter scoring-batch count")
        if ema_initialized != (scoring_batches > 0):
            raise ValueError("Inconsistent SAC experience-filter EMA and scoring-batch state")

        last_current_max = state.get("last_current_max")
        last_threshold = state.get("last_threshold")
        if scoring_batches == 0:
            if last_current_max is not None or last_threshold is not None:
                raise ValueError("Uninitialized SAC experience-filter metrics must be empty")
        else:
            if not _finite_real(last_current_max) or float(last_current_max) < 0:
                raise ValueError("Invalid SAC experience-filter current maximum")
            if not _finite_real(last_threshold) or float(last_threshold) < 0:
                raise ValueError("Invalid SAC experience-filter threshold")
            last_current_max = float(last_current_max)
            last_threshold = float(last_threshold)

        validated_counts = {}
        for counter_name in ("cumulative", "cycle"):
            counters = state.get(counter_name)
            expected_names = _LEGACY_COUNT_NAMES if state_version == 1 else _COUNT_NAMES
            if not isinstance(counters, Mapping) or set(counters) != set(expected_names):
                raise ValueError(f"Invalid SAC experience-filter {counter_name} counters")
            normalized_counts = {name: 0 for name in _COUNT_NAMES}
            for name in expected_names:
                value = counters[name]
                if not isinstance(value, int) or isinstance(value, bool) or value < 0:
                    raise ValueError(f"Invalid SAC experience-filter {counter_name} counter {name}")
                normalized_counts[name] = value
            if (normalized_counts["retained_count"] + normalized_counts["filtered_count"]
                    != normalized_counts["candidate_count"]):
                raise ValueError(f"Inconsistent SAC experience-filter {counter_name} row counts")
            if normalized_counts["critic_steps"] > normalized_counts["candidate_count"]:
                raise ValueError(f"Invalid SAC experience-filter {counter_name} critic-step count")
            if (normalized_counts["actor_rows"] < normalized_counts["actor_steps"]
                    or normalized_counts["actor_steps"] > normalized_counts["candidate_count"]
                    or normalized_counts["alpha_steps"] > normalized_counts["actor_steps"]
                    or normalized_counts["target_steps"] > normalized_counts["candidate_count"]):
                raise ValueError(f"Invalid SAC experience-filter {counter_name} update counters")
            actor_row_limit = (
                normalized_counts["retained_count"]
                if self.config["apply_to"] == FILTER_TARGET
                else normalized_counts["candidate_count"]
            )
            if normalized_counts["actor_rows"] > actor_row_limit:
                raise ValueError(f"Invalid SAC experience-filter {counter_name} actor-row count")
            validated_counts[counter_name] = normalized_counts

        if validated_counts["cumulative"]["candidate_count"] < validated_counts["cycle"]["candidate_count"]:
            raise ValueError("SAC experience-filter cycle counts exceed cumulative counts")
        for name in _COUNT_NAMES:
            if validated_counts["cycle"][name] > validated_counts["cumulative"][name]:
                raise ValueError(f"SAC experience-filter cycle counter exceeds cumulative counter: {name}")
        if validated_counts["cumulative"]["candidate_count"] < scoring_batches:
            raise ValueError("SAC experience-filter score batches exceed cumulative candidates")
        if validated_counts["cumulative"]["critic_steps"] > scoring_batches:
            raise ValueError("SAC experience-filter critic steps exceed score batches")
        if scoring_batches < validated_counts["cumulative"]["empty_batches"]:
            raise ValueError("SAC experience-filter empty-batch count exceeds score batches")
        actor_metric_history_complete = state.get("actor_metric_history_complete", False)
        if state_version == FILTER_STATE_VERSION:
            if not isinstance(actor_metric_history_complete, bool):
                raise ValueError("Invalid SAC experience-filter actor metric history flag")
        else:
            actor_metric_history_complete = False
        return {
            "version": FILTER_STATE_VERSION,
            "config": dict(self.config),
            "ema_initialized": ema_initialized,
            "ema_max": ema_max,
            "scoring_batches": scoring_batches,
            "last_current_max": last_current_max,
            "last_threshold": last_threshold,
            **validated_counts,
            "actor_metric_history_complete": actor_metric_history_complete,
        }

    def load_state_dict(self, state):
        """Restore a previously validated shared state."""
        validated = self.validate_state_dict(state)
        self.ema_initialized = validated["ema_initialized"]
        self.ema_max = validated["ema_max"]
        self.scoring_batches = validated["scoring_batches"]
        self.last_current_max = validated["last_current_max"]
        self.last_threshold = validated["last_threshold"]
        self._cumulative = dict(validated["cumulative"])
        self._cycle = dict(validated["cycle"])
        self._actor_metric_history_complete = validated["actor_metric_history_complete"]
