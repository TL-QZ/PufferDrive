"""Numeric return measurements for frozen-policy scene probes.

Arrays use ``[time, branch]`` layout.  Each branch must contain one non-empty
valid prefix; a terminal or truncation flag, when present, belongs to its final
valid transition.  NaN next-state values are retained as missing bootstraps and
propagate into any target that needs them.
"""

import math
import numbers

import numpy as np


def _numeric_array(name, values):
    array = np.asarray(values)
    if array.ndim != 2 or array.shape[0] < 1 or array.shape[1] < 1:
        raise ValueError(f"{name} must have shape [time, branch] with non-empty axes")
    if array.dtype.kind not in "fiu":
        raise ValueError(f"{name} must contain numeric values")
    return array.astype(np.float64, copy=False)


def _flag_array(name, values, shape):
    array = np.asarray(values)
    if array.shape != shape or array.dtype.kind != "b":
        raise ValueError(f"{name} must be a boolean array with shape {shape}")
    return array


def _finite_real(name, value, minimum=None, maximum=None):
    if not isinstance(value, numbers.Real) or isinstance(value, (bool, np.bool_)):
        raise ValueError(f"{name} must be a finite real number")
    value = float(value)
    if not math.isfinite(value):
        raise ValueError(f"{name} must be a finite real number")
    if minimum is not None and value < minimum:
        raise ValueError(f"{name} must be at least {minimum}")
    if maximum is not None and value > maximum:
        raise ValueError(f"{name} must be at most {maximum}")
    return value


def compute_probe_measurements(
    rewards,
    log_probs,
    valid,
    terminals,
    truncations,
    next_soft_values,
    gamma,
    alpha,
    return_estimator,
    pql_horizon=128,
    pql_lambda=0.95,
):
    """Measure sampled returns and the configured first-state SAC target.

    ``next_soft_values[t, b]`` is the target-policy soft value of the state
    following transition ``t``.  It may be NaN where no bootstrap was captured;
    that missing value stays visible in the target and bootstrapped-tail outputs.

    The standard sampled soft return includes ``-alpha * log_prob`` only for
    sampled continuation actions at times ``t >= 1``.  This omits entropy from
    the forced action at time zero.  ``critic_bootstrapped_soft_return`` adds a
    separately reported soft critic tail after the observed prefix, unless a
    terminal transition takes precedence.

    Both actual target forms are returned for clarity.  ``actual_critic_target``
    selects the one named by ``return_estimator``.  The PQL target delegates to
    the repository implementation, which uses its float32 target contract.
    Sampled return arithmetic is performed in CPU float64.
    """
    reward_array = _numeric_array("rewards", rewards)
    shape = reward_array.shape
    log_prob_array = _numeric_array("log_probs", log_probs)
    next_value_array = _numeric_array("next_soft_values", next_soft_values)
    valid_array = _flag_array("valid", valid, shape)
    terminal_array = _flag_array("terminals", terminals, shape)
    truncation_array = _flag_array("truncations", truncations, shape)
    for name, array in (
        ("log_probs", log_prob_array),
        ("next_soft_values", next_value_array),
    ):
        if array.shape != shape:
            raise ValueError(f"{name} must have shape {shape}")
        if np.any(np.isinf(array[valid_array])):
            raise ValueError(f"{name} cannot contain infinity on valid transitions")

    gamma = _finite_real("gamma", gamma, minimum=0.0, maximum=1.0)
    alpha = _finite_real("alpha", alpha, minimum=0.0)
    if return_estimator not in ("n_step", "pql"):
        raise ValueError("return_estimator must be 'n_step' or 'pql'")
    if not isinstance(pql_horizon, int) or isinstance(pql_horizon, bool) or pql_horizon <= 0:
        raise ValueError("pql_horizon must be a positive integer")
    pql_lambda = _finite_real("pql_lambda", pql_lambda, minimum=0.0, maximum=1.0)

    if np.any(terminal_array & ~valid_array) or np.any(truncation_array & ~valid_array):
        raise ValueError("terminal and truncation flags must be false on invalid padding")
    if np.any(~np.isfinite(reward_array[valid_array])):
        raise ValueError("rewards must be finite on valid transitions")

    time_count, branch_count = shape
    observed_horizon = np.empty(branch_count, dtype=np.int64)
    terminal = np.zeros(branch_count, dtype=np.bool_)
    truncated = np.zeros(branch_count, dtype=np.bool_)
    discounted_reward_return = np.zeros(branch_count, dtype=np.float64)
    standard_soft_return = np.zeros(branch_count, dtype=np.float64)
    return_tail_discount = np.zeros(branch_count, dtype=np.float64)
    critic_bootstrapped_soft_tail = np.zeros(branch_count, dtype=np.float64)
    critic_bootstrapped_soft_return = np.zeros(branch_count, dtype=np.float64)
    n_step_1_target = np.zeros(branch_count, dtype=np.float64)
    n_step_1_discount = np.zeros(branch_count, dtype=np.float64)
    pql_horizon_used = np.empty(branch_count, dtype=np.int64)
    pql_horizon_capped = np.zeros(branch_count, dtype=np.bool_)
    pql_terminal = np.zeros(branch_count, dtype=np.bool_)
    pql_truncated = np.zeros(branch_count, dtype=np.bool_)
    pql_tail_discount = np.zeros(branch_count, dtype=np.float64)

    pql_rewards = []
    pql_discounts = []
    pql_next_values = []
    pql_segment_offsets = [0]

    for branch_idx in range(branch_count):
        valid_column = valid_array[:, branch_idx]
        invalid_indices = np.flatnonzero(~valid_column)
        if invalid_indices.size and np.any(valid_column[invalid_indices[0] + 1:]):
            raise ValueError(f"valid must be a prefix for branch {branch_idx}")
        horizon = int(np.count_nonzero(valid_column))
        if horizon == 0:
            raise ValueError(f"branch {branch_idx} must have at least one valid transition")
        observed_horizon[branch_idx] = horizon

        terminal_positions = np.flatnonzero(terminal_array[:horizon, branch_idx])
        truncation_positions = np.flatnonzero(truncation_array[:horizon, branch_idx])
        boundary_positions = np.flatnonzero(
            terminal_array[:horizon, branch_idx] | truncation_array[:horizon, branch_idx]
        )
        if boundary_positions.size and (
            boundary_positions.size != 1 or boundary_positions[0] != horizon - 1
        ):
            raise ValueError(
                f"terminal or truncation must occur only on the final valid transition "
                f"for branch {branch_idx}"
            )

        has_terminal = terminal_positions.size == 1
        has_truncation = truncation_positions.size == 1 and not has_terminal
        terminal[branch_idx] = has_terminal
        truncated[branch_idx] = has_truncation

        if alpha > 0.0 and horizon > 1:
            continuation_log_probs = log_prob_array[1:horizon, branch_idx]
            if np.any(~np.isfinite(continuation_log_probs)):
                raise ValueError(
                    f"log_probs must be finite for sampled continuation actions "
                    f"in branch {branch_idx}"
                )

        discount = 1.0
        for time_idx in range(horizon):
            reward = reward_array[time_idx, branch_idx]
            discounted_reward_return[branch_idx] += discount * reward
            standard_soft_return[branch_idx] += discount * reward
            if time_idx >= 1 and alpha > 0.0:
                standard_soft_return[branch_idx] -= (
                    discount * alpha * log_prob_array[time_idx, branch_idx]
                )
            discount *= gamma

        return_tail_discount[branch_idx] = 0.0 if has_terminal else discount
        if has_terminal or discount == 0.0:
            critic_bootstrapped_soft_tail[branch_idx] = 0.0
        else:
            critic_bootstrapped_soft_tail[branch_idx] = (
                discount * next_value_array[horizon - 1, branch_idx]
            )
        critic_bootstrapped_soft_return[branch_idx] = (
            standard_soft_return[branch_idx]
            + critic_bootstrapped_soft_tail[branch_idx]
        )

        n_step_1_discount[branch_idx] = 0.0 if terminal_array[0, branch_idx] else gamma
        if terminal_array[0, branch_idx] or gamma == 0.0:
            n_step_1_target[branch_idx] = reward_array[0, branch_idx]
        else:
            n_step_1_target[branch_idx] = (
                reward_array[0, branch_idx]
                + gamma * next_value_array[0, branch_idx]
            )

        pql_length = min(horizon, pql_horizon)
        pql_horizon_used[branch_idx] = pql_length
        pql_horizon_capped[branch_idx] = horizon > pql_horizon
        pql_ends_at_terminal = has_terminal and pql_length == horizon
        pql_ends_at_truncation = has_truncation and pql_length == horizon
        pql_terminal[branch_idx] = pql_ends_at_terminal
        pql_truncated[branch_idx] = pql_ends_at_truncation
        pql_tail_discount[branch_idx] = (
            0.0 if pql_ends_at_terminal else gamma ** pql_length
        )

        for time_idx in range(pql_length):
            is_terminal = terminal_array[time_idx, branch_idx]
            pql_rewards.append(reward_array[time_idx, branch_idx])
            pql_discounts.append(0.0 if is_terminal else gamma)
            pql_next_values.append(
                0.0 if is_terminal or gamma == 0.0 else next_value_array[time_idx, branch_idx]
            )
        pql_segment_offsets.append(len(pql_rewards))

    # PQL matches the learner's real raw-reward recurrence, including its
    # float32 target arithmetic and segment-end bootstrap convention.
    import torch

    from pufferlib.sac_returns import compute_pql_targets

    pql_targets = compute_pql_targets(
        torch.as_tensor(np.asarray(pql_rewards, dtype=np.float32), dtype=torch.float32),
        torch.as_tensor(np.asarray(pql_discounts, dtype=np.float32), dtype=torch.float32),
        torch.as_tensor(np.asarray(pql_next_values, dtype=np.float32), dtype=torch.float32),
        torch.as_tensor(pql_segment_offsets, dtype=torch.int64),
        pql_lambda,
    ).cpu().numpy().astype(np.float64, copy=False)
    pql_target = np.empty(branch_count, dtype=np.float64)
    target_offset = 0
    for branch_idx, pql_length in enumerate(pql_horizon_used):
        pql_target[branch_idx] = pql_targets[target_offset]
        target_offset += int(pql_length)

    actual_critic_target = n_step_1_target if return_estimator == "n_step" else pql_target
    return {
        "return_estimator": return_estimator,
        "observed_horizon": observed_horizon,
        "terminal": terminal,
        "truncated": truncated,
        "censored": ~terminal,
        "discounted_reward_return": discounted_reward_return,
        "standard_soft_return": standard_soft_return,
        "return_tail_discount": return_tail_discount,
        "critic_bootstrapped_soft_tail": critic_bootstrapped_soft_tail,
        "critic_bootstrapped_soft_return": critic_bootstrapped_soft_return,
        "n_step_1_target": n_step_1_target,
        "n_step_1_discount": n_step_1_discount,
        "pql_target": pql_target,
        "pql_horizon_used": pql_horizon_used,
        "pql_horizon_capped": pql_horizon_capped,
        "pql_terminal": pql_terminal,
        "pql_truncated": pql_truncated,
        "pql_censored": ~pql_terminal,
        "pql_tail_discount": pql_tail_discount,
        "actual_critic_target": actual_critic_target,
    }
