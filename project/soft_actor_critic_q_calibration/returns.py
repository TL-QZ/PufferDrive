"""Reference return calculations for Q-value calibration rollouts."""

from typing import NamedTuple

import numpy as np


class ReturnEstimates(NamedTuple):
    """Per-transition return estimates, with the same ``[time, agent]`` shape.

    Floating-point fields are NaN on invalid transitions. ``censored`` marks
    returns whose continuation was not observed; ``truncation_endpoint``
    distinguishes explicit scenario timeouts from collection or validity cuts.
    ``bootstrap_value`` is the endpoint V_soft before discounting, while
    ``bootstrap_tail`` is its discounted contribution. Missing endpoint values
    remain NaN when their nonzero tail discount makes the estimate unavailable.
    """

    reward_partial: np.ndarray
    soft_partial: np.ndarray
    entropy_contribution: np.ndarray
    censored: np.ndarray
    truncation_endpoint: np.ndarray
    observed_horizon: np.ndarray
    tail_discount: np.ndarray
    bootstrap_value: np.ndarray
    bootstrap_tail: np.ndarray
    soft_bootstrapped: np.ndarray


def _as_float_array(name: str, values: np.ndarray) -> np.ndarray:
    try:
        result = np.asarray(values, dtype=np.float64)
    except (TypeError, ValueError) as error:
        raise ValueError(f"{name} must be a numeric array") from error
    if result.ndim != 2:
        raise ValueError(f"{name} must have shape [time, agent]")
    return result


def _as_bool_array(name: str, values: np.ndarray) -> np.ndarray:
    result = np.asarray(values)
    if result.dtype != np.bool_:
        raise ValueError(f"{name} must have boolean dtype")
    if result.ndim != 2:
        raise ValueError(f"{name} must have shape [time, agent]")
    return result


def compute_returns(
    rewards: np.ndarray,
    log_probs: np.ndarray,
    valid: np.ndarray,
    terminals: np.ndarray,
    truncations: np.ndarray,
    next_soft_values: np.ndarray,
    gamma: float,
    alpha: float,
) -> ReturnEstimates:
    """Compute backward reward and sampled-entropy returns for each agent.

    A transition at ``t`` includes its reward and excludes entropy for its own
    action. A following sampled action contributes ``-alpha * log_probs[t+1]``.
    True terminals have no tail. A truncation or a missing contiguous valid row
    ends the observed trajectory and is marked censored; its endpoint value is
    taken from ``next_soft_values`` on the final included transition. Invalid
    gaps never connect to a later valid row.

    ``next_soft_values`` may be NaN where it is unused or unavailable. Infinite
    endpoint values are rejected. A missing endpoint value leaves the
    bootstrapped estimate NaN when the retained discount is nonzero.
    """

    reward_values = _as_float_array("rewards", rewards)
    log_prob_values = _as_float_array("log_probs", log_probs)
    next_value_array = _as_float_array("next_soft_values", next_soft_values)
    valid_mask = _as_bool_array("valid", valid)
    terminal_mask = _as_bool_array("terminals", terminals)
    truncation_mask = _as_bool_array("truncations", truncations)

    shape = reward_values.shape
    for name, values in (
        ("log_probs", log_prob_values),
        ("valid", valid_mask),
        ("terminals", terminal_mask),
        ("truncations", truncation_mask),
        ("next_soft_values", next_value_array),
    ):
        if values.shape != shape:
            raise ValueError(f"{name} must have the same shape as rewards")

    try:
        discount_factor = float(gamma)
        temperature = float(alpha)
    except (TypeError, ValueError) as error:
        raise ValueError("gamma and alpha must be finite scalars") from error
    if not np.isfinite(discount_factor) or not 0.0 <= discount_factor <= 1.0:
        raise ValueError("gamma must be finite and in [0, 1]")
    if not np.isfinite(temperature) or temperature < 0.0:
        raise ValueError("alpha must be finite and nonnegative")

    if np.any((terminal_mask | truncation_mask) & ~valid_mask):
        raise ValueError("terminal and truncation flags require a valid transition")
    if not np.all(np.isfinite(reward_values[valid_mask])):
        raise ValueError("rewards must be finite on valid transitions")
    if not np.all(np.isfinite(log_prob_values[valid_mask])):
        raise ValueError("log_probs must be finite on valid transitions")

    # An endpoint is censored when this transition does not terminate and the
    # next time row is either absent, invalid, or separated by a truncation.
    has_contiguous_successor = np.zeros(shape, dtype=np.bool_)
    if shape[0] > 1:
        has_contiguous_successor[:-1] = (
            valid_mask[:-1]
            & valid_mask[1:]
            & ~terminal_mask[:-1]
            & ~truncation_mask[:-1]
        )
    censored_endpoints = valid_mask & ~terminal_mask & ~has_contiguous_successor
    if np.any(np.isinf(next_value_array[censored_endpoints])):
        raise ValueError("next_soft_values must not be infinite at censored endpoints")

    reward_partial = np.full(shape, np.nan, dtype=np.float64)
    soft_partial = np.full(shape, np.nan, dtype=np.float64)
    entropy_contribution = np.full(shape, np.nan, dtype=np.float64)
    censored = np.zeros(shape, dtype=np.bool_)
    truncation_endpoint = np.zeros(shape, dtype=np.bool_)
    observed_horizon = np.zeros(shape, dtype=np.int64)
    tail_discount = np.full(shape, np.nan, dtype=np.float64)
    bootstrap_value = np.full(shape, np.nan, dtype=np.float64)
    bootstrap_tail = np.full(shape, np.nan, dtype=np.float64)
    soft_bootstrapped = np.full(shape, np.nan, dtype=np.float64)

    for time_idx in range(shape[0] - 1, -1, -1):
        row_valid = valid_mask[time_idx]
        if not np.any(row_valid):
            continue

        row_terminal = row_valid & terminal_mask[time_idx]
        row_continues = row_valid & ~terminal_mask[time_idx] & ~truncation_mask[time_idx]
        if time_idx + 1 < shape[0]:
            row_continues &= valid_mask[time_idx + 1]
        else:
            row_continues[:] = False
        row_censored = row_valid & ~row_terminal & ~row_continues
        row_truncated = row_censored & truncation_mask[time_idx]

        if np.any(row_terminal):
            reward_partial[time_idx, row_terminal] = reward_values[time_idx, row_terminal]
            entropy_contribution[time_idx, row_terminal] = 0.0
            soft_partial[time_idx, row_terminal] = reward_values[time_idx, row_terminal]
            observed_horizon[time_idx, row_terminal] = 1
            tail_discount[time_idx, row_terminal] = 0.0
            bootstrap_tail[time_idx, row_terminal] = 0.0
            soft_bootstrapped[time_idx, row_terminal] = reward_values[time_idx, row_terminal]

        if np.any(row_censored):
            reward_partial[time_idx, row_censored] = reward_values[time_idx, row_censored]
            entropy_contribution[time_idx, row_censored] = 0.0
            soft_partial[time_idx, row_censored] = reward_values[time_idx, row_censored]
            censored[time_idx, row_censored] = True
            truncation_endpoint[time_idx, row_truncated] = True
            observed_horizon[time_idx, row_censored] = 1
            tail_discount[time_idx, row_censored] = discount_factor
            bootstrap_value[time_idx, row_censored] = next_value_array[
                time_idx, row_censored
            ]
            row_tail = np.zeros(np.count_nonzero(row_censored), dtype=np.float64)
            np.multiply(
                tail_discount[time_idx, row_censored],
                bootstrap_value[time_idx, row_censored],
                out=row_tail,
                where=tail_discount[time_idx, row_censored] != 0.0,
            )
            bootstrap_tail[time_idx, row_censored] = row_tail
            soft_bootstrapped[time_idx, row_censored] = (
                soft_partial[time_idx, row_censored] + row_tail
            )

        if np.any(row_continues):
            next_idx = time_idx + 1
            reward_partial[time_idx, row_continues] = (
                reward_values[time_idx, row_continues]
                + discount_factor * reward_partial[next_idx, row_continues]
            )
            entropy_contribution[time_idx, row_continues] = discount_factor * (
                -temperature * log_prob_values[next_idx, row_continues]
                + entropy_contribution[next_idx, row_continues]
            )
            soft_partial[time_idx, row_continues] = (
                reward_partial[time_idx, row_continues]
                + entropy_contribution[time_idx, row_continues]
            )
            censored[time_idx, row_continues] = censored[next_idx, row_continues]
            truncation_endpoint[time_idx, row_continues] = truncation_endpoint[
                next_idx, row_continues
            ]
            observed_horizon[time_idx, row_continues] = (
                observed_horizon[next_idx, row_continues] + 1
            )
            tail_discount[time_idx, row_continues] = (
                discount_factor * tail_discount[next_idx, row_continues]
            )
            bootstrap_value[time_idx, row_continues] = bootstrap_value[
                next_idx, row_continues
            ]
            row_tail = np.zeros(np.count_nonzero(row_continues), dtype=np.float64)
            np.multiply(
                tail_discount[time_idx, row_continues],
                bootstrap_value[time_idx, row_continues],
                out=row_tail,
                where=tail_discount[time_idx, row_continues] != 0.0,
            )
            bootstrap_tail[time_idx, row_continues] = row_tail
            soft_bootstrapped[time_idx, row_continues] = (
                soft_partial[time_idx, row_continues] + row_tail
            )

    if not np.all(np.isfinite(reward_partial[valid_mask])):
        raise FloatingPointError("reward return overflowed on valid transitions")
    if not np.all(np.isfinite(soft_partial[valid_mask])):
        raise FloatingPointError("soft return overflowed on valid transitions")
    if not np.all(np.isfinite(entropy_contribution[valid_mask])):
        raise FloatingPointError("entropy contribution overflowed on valid transitions")
    if not np.all(np.isfinite(bootstrap_tail[valid_mask & (tail_discount == 0.0)])):
        raise FloatingPointError("zero-discount bootstrap tail must be finite")
    finite_bootstrap_estimates = valid_mask & np.isfinite(bootstrap_value)
    if not np.all(np.isfinite(bootstrap_tail[finite_bootstrap_estimates])):
        raise FloatingPointError("bootstrapped tail overflowed on valid transitions")

    return ReturnEstimates(
        reward_partial=reward_partial,
        soft_partial=soft_partial,
        entropy_contribution=entropy_contribution,
        censored=censored,
        truncation_endpoint=truncation_endpoint,
        observed_horizon=observed_horizon,
        tail_discount=tail_discount,
        bootstrap_value=bootstrap_value,
        bootstrap_tail=bootstrap_tail,
        soft_bootstrapped=soft_bootstrapped,
    )
