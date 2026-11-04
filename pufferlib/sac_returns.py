"""Return estimators used by discrete SAC."""

import math

import torch


def compute_pql_targets(rewards, discounts, next_values, segment_offsets, trace_lambda):
    """Build detached PQL targets for packed, contiguous replay segments.

    ``rewards``, ``discounts``, and ``next_values`` are packed one-dimensional
    tensors. ``segment_offsets`` contains the first row and one-past-last row
    for each segment. The scan runs once per suffix depth, with each depth
    updating every segment that is long enough; it never loops over rows.
    """
    if not isinstance(trace_lambda, (int, float)) or isinstance(trace_lambda, bool):
        raise ValueError("trace_lambda must be a finite number between zero and one")
    if not math.isfinite(trace_lambda) or trace_lambda < 0 or trace_lambda > 1:
        raise ValueError("trace_lambda must be a finite number between zero and one")

    rewards = torch.as_tensor(rewards)
    if rewards.ndim != 1:
        raise ValueError("rewards must be a one-dimensional tensor")
    device = rewards.device
    rewards = rewards.detach().to(dtype=torch.float32)
    discounts = torch.as_tensor(discounts, device=device).detach().to(dtype=torch.float32)
    next_values = torch.as_tensor(next_values, device=device).detach().to(dtype=torch.float32)
    if discounts.ndim != 1 or tuple(discounts.shape) != tuple(rewards.shape):
        raise ValueError("discounts must have the same one-dimensional shape as rewards")
    if next_values.ndim != 1 or tuple(next_values.shape) != tuple(rewards.shape):
        raise ValueError("next_values must have the same one-dimensional shape as rewards")

    offsets = torch.as_tensor(segment_offsets)
    if offsets.ndim != 1 or offsets.dtype not in (
        torch.int8,
        torch.int16,
        torch.int32,
        torch.int64,
        torch.uint8,
    ):
        raise ValueError("segment_offsets must be a one-dimensional integer tensor")
    offsets_cpu = offsets.detach().to(device="cpu", dtype=torch.int64)
    if offsets_cpu.numel() == 0:
        raise ValueError("segment_offsets must include the initial offset")
    if int(offsets_cpu[0]) != 0 or int(offsets_cpu[-1]) != rewards.numel():
        raise ValueError("segment_offsets must start at zero and end at the packed row count")
    if offsets_cpu.numel() > 1 and bool(torch.any(offsets_cpu[1:] <= offsets_cpu[:-1])):
        raise ValueError("segment_offsets must be strictly increasing")

    targets = torch.empty_like(rewards)
    if rewards.numel() == 0:
        return targets.detach()

    starts_cpu = offsets_cpu[:-1]
    lengths_cpu = offsets_cpu[1:] - starts_cpu
    maximum_segment_length = int(lengths_cpu.max())
    row_groups = []
    group_offsets = [0]
    for suffix_depth in range(1, maximum_segment_length + 1):
        active_segments = lengths_cpu >= suffix_depth
        row_groups.append(
            starts_cpu[active_segments] + lengths_cpu[active_segments] - suffix_depth
        )
        group_offsets.append(group_offsets[-1] + int(active_segments.sum()))
    grouped_row_indices = torch.cat(row_groups).to(device=device)

    for suffix_depth in range(1, maximum_segment_length + 1):
        row_indices = grouped_row_indices[
            group_offsets[suffix_depth - 1]:group_offsets[suffix_depth]
        ]
        if suffix_depth == 1:
            targets[row_indices] = (
                rewards[row_indices] + discounts[row_indices] * next_values[row_indices]
            )
            continue

        successor_targets = targets[row_indices + 1]
        targets[row_indices] = rewards[row_indices] + discounts[row_indices] * (
            (1.0 - trace_lambda) * next_values[row_indices]
            + trace_lambda * successor_targets
        )

    return targets.detach()
