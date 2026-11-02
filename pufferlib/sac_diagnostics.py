"""Low-overhead, opt-in SAC collapse diagnostics.

The tracker only aggregates detached tensors already produced by the SAC
losses. It performs no model calls and transfers data to the host only when a
training log is emitted.
"""

import torch
import torch.distributed as dist


NEAR_DETERMINISTIC_PROBABILITY = 0.99


class SACCollapseDiagnostics:
    """Accumulate device-side statistics for one SAC logging interval."""

    _ACTOR_GRAD = 0
    _ACTOR_CLIP = 1
    _CRITIC_GRAD = 2
    _CRITIC_CLIP = 3
    _MAX_ACTION_PROBABILITY = 4
    _NEAR_DETERMINISTIC = 5
    _LOGIT_RANGE = 6
    _Q_ACTION_GAP_OVER_ALPHA = 7
    _ABS_TD_ERROR = 8
    _ENTROPY_UPDATE_MEAN = 9
    _METRIC_COUNT = 10

    _METRICS = (
        (_ACTOR_GRAD, "actor/grad_norm_mean", "actor/grad_norm_max", None),
        (_ACTOR_CLIP, "actor/grad_clip_fraction", None, None),
        (_CRITIC_GRAD, "critic/grad_norm_mean", "critic/grad_norm_max", None),
        (_CRITIC_CLIP, "critic/grad_clip_fraction", None, None),
        (_MAX_ACTION_PROBABILITY, "policy/max_action_probability_mean", None, None),
        (_NEAR_DETERMINISTIC, "policy/near_deterministic_fraction", None, None),
        (_LOGIT_RANGE, "policy/logit_range_mean", "policy/logit_range_max", None),
        (_Q_ACTION_GAP_OVER_ALPHA, "policy/q_action_gap_over_alpha_mean",
         "policy/q_action_gap_over_alpha_max", None),
        (_ABS_TD_ERROR, "critic/abs_td_error_mean", "critic/abs_td_error_max", None),
        (_ENTROPY_UPDATE_MEAN, None, None, "entropy_update_mean_min"),
    )

    def __init__(self, device):
        self.device = torch.device(device)
        self.sums = torch.zeros(self._METRIC_COUNT, device=self.device, dtype=torch.float64)
        self.counts = torch.zeros_like(self.sums)
        self.maxima = torch.full_like(self.sums, -torch.inf)
        self.minima = torch.full_like(self.sums, torch.inf)

    @torch.no_grad()
    def add_policy(self, logits, probabilities, minimum_q, alpha):
        """Add row-weighted policy statistics from an existing forward pass."""
        logits = logits.detach().float()
        probabilities = probabilities.detach().float()
        minimum_q = minimum_q.detach().float()
        maximum_probability = probabilities.amax(dim=-1)
        near_deterministic = (maximum_probability >= NEAR_DETERMINISTIC_PROBABILITY).to(torch.float32)
        logit_range = logits.amax(dim=-1) - logits.amin(dim=-1)
        if minimum_q.shape[-1] > 1:
            top_q = torch.topk(minimum_q, k=2, dim=-1).values
            q_gap_over_alpha = (top_q[:, 0] - top_q[:, 1]) / alpha.detach()
        else:
            # There is no second action in a one-action policy, so define its gap as zero.
            q_gap_over_alpha = torch.zeros_like(maximum_probability)

        values = (
            maximum_probability,
            near_deterministic,
            logit_range,
            q_gap_over_alpha,
        )
        policy_slice = slice(self._MAX_ACTION_PROBABILITY, self._Q_ACTION_GAP_OVER_ALPHA + 1)
        batch_sums = torch.stack(tuple(value.sum(dtype=torch.float64) for value in values))
        row_count = maximum_probability.numel()
        self.sums[policy_slice].add_(batch_sums)
        self.counts[policy_slice].add_(row_count)
        self.maxima[self._LOGIT_RANGE].copy_(torch.maximum(
            self.maxima[self._LOGIT_RANGE], logit_range.max().to(torch.float64),
        ))
        self.maxima[self._Q_ACTION_GAP_OVER_ALPHA].copy_(torch.maximum(
            self.maxima[self._Q_ACTION_GAP_OVER_ALPHA], q_gap_over_alpha.max().to(torch.float64),
        ))

    @torch.no_grad()
    def add_td_errors(self, q1, q2, targets):
        """Aggregate absolute TD errors from both critics on replay actions."""
        q1_error = (q1.detach().float() - targets.detach().float()).abs()
        q2_error = (q2.detach().float() - targets.detach().float()).abs()
        error_sum = q1_error.sum(dtype=torch.float64) + q2_error.sum(dtype=torch.float64)
        error_max = torch.maximum(q1_error.max(), q2_error.max()).to(torch.float64)
        self.sums[self._ABS_TD_ERROR].add_(error_sum)
        self.counts[self._ABS_TD_ERROR].add_(2 * q1_error.numel())
        self.maxima[self._ABS_TD_ERROR].copy_(torch.maximum(
            self.maxima[self._ABS_TD_ERROR], error_max,
        ))

    @torch.no_grad()
    def add_grad_norm(self, network, preclip_norm, max_grad_norm):
        """Record the norm returned by clip_grad_norm_ before clipping."""
        if network == "actor":
            norm_index, clip_index = self._ACTOR_GRAD, self._ACTOR_CLIP
        elif network == "critic":
            norm_index, clip_index = self._CRITIC_GRAD, self._CRITIC_CLIP
        else:
            raise ValueError(f"Unknown SAC diagnostic network: {network}")
        norm = preclip_norm.detach().to(dtype=torch.float64)
        clipped = (preclip_norm.detach() > max_grad_norm).to(dtype=torch.float64)
        self.sums[norm_index].add_(norm)
        self.counts[norm_index].add_(1)
        self.maxima[norm_index].copy_(torch.maximum(self.maxima[norm_index], norm))
        self.sums[clip_index].add_(clipped)
        self.counts[clip_index].add_(1)

    @torch.no_grad()
    def add_entropy_update_mean(self, entropy_update_mean):
        """Track the minimum rank-local mean entropy across logical updates."""
        value = entropy_update_mean.detach().to(dtype=torch.float64)
        index = self._ENTROPY_UPDATE_MEAN
        self.minima[index].copy_(torch.minimum(self.minima[index], value))
        self.counts[index].add_(1)

    @torch.no_grad()
    def reset(self):
        self.sums.zero_()
        self.counts.zero_()
        self.maxima.fill_(-torch.inf)
        self.minima.fill_(torch.inf)

    @torch.no_grad()
    def reduce_and_reset(self, *, distributed=False):
        """Reduce a packed interval and transfer its result to host once."""
        packed = torch.stack((self.sums, self.counts, self.maxima, self.minima))
        if distributed:
            if not dist.is_available() or not dist.is_initialized():
                raise RuntimeError("SAC collapse diagnostics require an initialized process group")
            # Sum/count, max, and min use distinct reduction operators. Each call
            # carries all metrics in one small fixed-size device tensor.
            dist.all_reduce(packed[:2], op=dist.ReduceOp.SUM)
            dist.all_reduce(packed[2], op=dist.ReduceOp.MAX)
            dist.all_reduce(packed[3], op=dist.ReduceOp.MIN)

        host_values = packed.cpu().tolist()
        sums, counts, maxima, minima = host_values
        logs = {}
        for index, mean_name, max_name, min_name in self._METRICS:
            count = counts[index]
            if count == 0:
                continue
            if mean_name is not None:
                logs[f"diagnostics/{mean_name}"] = sums[index] / count
            if max_name is not None:
                logs[f"diagnostics/{max_name}"] = maxima[index]
            if min_name is not None:
                logs[f"diagnostics/{min_name}"] = minima[index]

        self.reset()
        return logs
