"""Learner-local SAC replay gather candidates for the profiling harness.

The candidate update loop is copied from
``pufferlib.pufferl_sac.PuffeSAC._train_sac_transition`` at repository revision
``cd4a75432882d64bee8977eba473b35bd6c9e7e9`` (source lines 1818-1980). All
variants preserve the stock minibatch sampling, backward, optimizer, clipping,
target-update, DDP, and accounting order. ``obs_only`` and ``cached`` change
replay gathering only; ``fixed_targets`` and ``fixed_targets_cached`` also
reuse cycle-start bootstrap targets. Rebase this copy when the source method
changes.

These candidates intentionally support the baseline ``n_step`` estimator only.
PQL has a different sequence-sampling and target-preparation path and remains on
the stock learner method. The ``fixed_targets`` variants are algorithmic
experiments: they compute targets with the current actor, target critic, and
entropy coefficient once at the start of a training cycle, then keep those
scalar targets fixed while the actor, critic, entropy coefficient, and target
critic continue to update during the cycle. This matches the PPO-like target
reuse idea, but makes bootstrap targets stale relative to those updates.
``fixed_targets_actor_q`` also caches the cycle-start online twin-critic minimum
for every action and replay observation. Its actor optimizes current policy
probabilities against those Q scores for the whole cycle, so the scores become
stale as the online critic changes. Both caches are rebuilt from the full replay
at the start of each cycle, including rows retained since the previous collection.
"""

import math
from contextlib import nullcontext
from types import MethodType

import torch
import torch.nn.functional as F


_VARIANTS = {
    "stock",
    "obs_only",
    "cached",
    "fixed_targets",
    "fixed_targets_cached",
    "fixed_targets_actor_q",
}


def _gather_replay_field(learner, field_name, indices):
    """Gather one replay field while preserving TensorReplay pinned copies."""
    replay = learner.replay
    storage = getattr(replay, field_name)
    storage_indices = indices.to(device=replay.device)
    if replay.pin_memory:
        gathered = torch.empty(
            (storage_indices.numel(), *storage.shape[1:]),
            dtype=storage.dtype,
            device=replay.device,
            pin_memory=True,
        )
        torch.index_select(storage, 0, storage_indices, out=gathered)
        return gathered
    return storage[storage_indices]


def _gather_replay_fields(learner, field_names, indices):
    return tuple(_gather_replay_field(learner, name, indices) for name in field_names)


def _policy_observations(learner, indices):
    """Gather only observations, matching TensorReplay's pinned-memory path."""
    replay = learner.replay
    storage = replay.observations
    storage_indices = indices.to(device=replay.device)
    if replay.pin_memory:
        observations = torch.empty(
            (storage_indices.numel(), *storage.shape[1:]),
            dtype=storage.dtype,
            device=replay.device,
            pin_memory=True,
        )
        torch.index_select(storage, 0, storage_indices, out=observations)
        return observations
    return storage[storage_indices]


def _check_candidate_support(learner):
    if learner.config.get("return_estimator", "n_step") != "n_step":
        raise ValueError("SAC replay gather candidates support return_estimator='n_step' only")
    replay = learner.replay
    for attribute in ("observations", "device", "pin_memory"):
        if not hasattr(replay, attribute):
            raise TypeError(
                "SAC replay gather candidates require TensorReplay-style observation storage"
            )


def _check_fixed_target_support(learner):
    _check_candidate_support(learner)
    replay = learner.replay
    for attribute in ("actions", "rewards", "next_observations", "discounts"):
        if not hasattr(replay, attribute):
            raise TypeError(
                "SAC fixed-target candidates require TensorReplay-style transition storage"
            )


@torch.no_grad()
def _precompute_fixed_cycle_targets(learner, replay_size, profile, epoch, target_policy):
    """Compute targets from contiguous replay views while storage is immutable."""
    targets = torch.empty(replay_size, device=learner.device, dtype=torch.float32)
    learner.fixed_target_precompute_rows = 0
    learner.fixed_target_precompute_calls = 0
    for start_idx in range(0, replay_size, learner.max_minibatch_size):
        end_idx = min(start_idx + learner.max_minibatch_size, replay_size)
        profile("train_copy", epoch)
        # The learner owns the full train call, so replay cannot be appended or
        # replaced here. Contiguous views avoid allocating gather buffers; pinned
        # CPU views remain alive and unchanged until their asynchronous copies
        # finish, and GPU replay slices need no copy at all.
        rewards = learner.replay.rewards[start_idx:end_idx]
        next_observations = learner.replay.next_observations[start_idx:end_idx]
        discounts = learner.replay.discounts[start_idx:end_idx]
        rewards = rewards.to(learner.device, non_blocking=True)
        next_observations = next_observations.to(learner.device, non_blocking=True)
        discounts = discounts.to(learner.device, non_blocking=True)
        profile("train_forward", epoch)
        if target_policy is None:
            chunk_targets = learner._compute_targets(rewards, next_observations, discounts)
        else:
            chunk_targets = learner._compute_targets(
                rewards,
                next_observations,
                discounts,
                target_policy=target_policy,
            )
        targets[start_idx:end_idx].copy_(chunk_targets.detach().float())
        learner.fixed_target_precompute_rows += end_idx - start_idx
        learner.fixed_target_precompute_calls += 1
    return targets


@torch.no_grad()
def _precompute_fixed_cycle_actor_q(learner, replay_size, profile, epoch):
    """Cache cycle-start online min-Q scores for all actions and replay rows."""
    minimum_q_cache = None
    learner.fixed_actor_q_precompute_rows = 0
    learner.fixed_actor_q_precompute_calls = 0
    learner.actor_q_forward_rows = 0
    learner.actor_q_forward_calls = 0
    for start_idx in range(0, replay_size, learner.max_minibatch_size):
        end_idx = min(start_idx + learner.max_minibatch_size, replay_size)
        profile("train_copy", epoch)
        observations = learner.replay.observations[start_idx:end_idx]
        observations = observations.to(learner.device, non_blocking=True)
        profile("train_forward", epoch)
        # Use the same online critic path and AMP policy as the stock actor loss.
        # The cache is detached float32 even when inference ran under BF16 AMP.
        with learner.amp_context:
            q1, q2 = learner.critic(observations)
        minimum_q = torch.minimum(q1.float(), q2.float())
        if minimum_q_cache is None:
            minimum_q_cache = torch.empty(
                (replay_size, *minimum_q.shape[1:]),
                device=learner.device,
                dtype=torch.float32,
            )
        minimum_q_cache[start_idx:end_idx].copy_(minimum_q)
        row_count = end_idx - start_idx
        learner.fixed_actor_q_precompute_rows += row_count
        learner.fixed_actor_q_precompute_calls += 1
        learner.actor_q_forward_rows += row_count
        learner.actor_q_forward_calls += 1
    return minimum_q_cache


def _fixed_actor_q_policy_loss(learner, mb_obs, cached_minimum_q):
    """Train current policy probabilities against detached cycle-start Q scores."""
    with learner.amp_context:
        logits = learner.policy_train(mb_obs)
    logprobs = F.log_softmax(logits.float(), dim=-1)
    probs = logprobs.exp()
    alpha = learner.alpha
    loss = (probs * (alpha * logprobs - cached_minimum_q)).sum(-1).mean()
    entropy = -(probs * logprobs).sum(-1).mean()
    diagnostics = getattr(learner, "_collapse_diagnostics_tracker", None)
    if diagnostics is not None:
        diagnostics.add_policy(logits, probs, cached_minimum_q, alpha)
    return loss, entropy.detach()


def _train_sac_transition_obs_only(self, losses, profile, epoch, update_policy=True, target_policy=None):
    return _train_sac_transition_candidate(
        self, losses, profile, epoch, update_policy, target_policy,
        cache_actor_observations=False,
    )


def _train_sac_transition_cached(self, losses, profile, epoch, update_policy=True, target_policy=None):
    return _train_sac_transition_candidate(
        self, losses, profile, epoch, update_policy, target_policy,
        cache_actor_observations=True,
    )


def _train_sac_transition_fixed_targets(
        self, losses, profile, epoch, update_policy=True, target_policy=None):
    return _train_sac_transition_candidate(
        self, losses, profile, epoch, update_policy, target_policy,
        cache_actor_observations=False,
        fixed_cycle_targets=True,
    )


def _train_sac_transition_fixed_targets_cached(
        self, losses, profile, epoch, update_policy=True, target_policy=None):
    return _train_sac_transition_candidate(
        self, losses, profile, epoch, update_policy, target_policy,
        cache_actor_observations=True,
        fixed_cycle_targets=True,
    )


def _train_sac_transition_fixed_targets_actor_q(
        self, losses, profile, epoch, update_policy=True, target_policy=None):
    return _train_sac_transition_candidate(
        self, losses, profile, epoch, update_policy, target_policy,
        cache_actor_observations=False,
        fixed_cycle_targets=True,
        fixed_cycle_actor_q=True,
    )


def _train_sac_transition_candidate(
        self, losses, profile, epoch, update_policy, target_policy, *,
        cache_actor_observations, fixed_cycle_targets=False, fixed_cycle_actor_q=False):
    """SAC update loop with replay-gather and optional fixed-target candidates."""
    _check_candidate_support(self)
    self.fixed_cycle_targets = None
    self.fixed_target_precompute_rows = 0
    self.fixed_target_precompute_calls = 0
    self.fixed_cycle_actor_q = None
    self.fixed_actor_q_precompute_rows = 0
    self.fixed_actor_q_precompute_calls = 0
    self.actor_q_forward_rows = 0
    self.actor_q_forward_calls = 0
    self.fixed_actor_loss_rows = 0
    if fixed_cycle_targets:
        _check_fixed_target_support(self)
    if getattr(self, "config", {}).get("log_collapse_diagnostics", False):
        diagnostics = getattr(self, "_collapse_diagnostics_tracker", None)
        if diagnostics is None:
            from pufferlib.sac_diagnostics import SACCollapseDiagnostics

            diagnostics = SACCollapseDiagnostics(self.device)
            self._collapse_diagnostics_tracker = diagnostics
    else:
        diagnostics = None
        self._collapse_diagnostics_tracker = None
    replay_size = len(self.replay)
    if replay_size < self.batch_size:
        return 0
    fixed_targets = None
    if fixed_cycle_targets:
        fixed_targets = _precompute_fixed_cycle_targets(
            self, replay_size, profile, epoch, target_policy,
        )
        # Keep the detached cache available for harness audits until the next
        # training cycle replaces it.
        self.fixed_cycle_targets = fixed_targets
    fixed_actor_q = None
    if fixed_cycle_actor_q and update_policy:
        fixed_actor_q = _precompute_fixed_cycle_actor_q(self, replay_size, profile, epoch)
        # Keep the detached cycle cache available for row-alignment audits.
        self.fixed_cycle_actor_q = fixed_actor_q
    minibatches_per_epoch = math.ceil(replay_size / self.minibatch_size)
    total_minibatches = self.config["update_epochs"] * minibatches_per_epoch
    total_sampled_transitions = replay_size * self.config["update_epochs"]
    replay_indices = None
    for logical_update_idx in range(total_minibatches):
        minibatch_idx = logical_update_idx % minibatches_per_epoch
        if minibatch_idx == 0:
            replay_indices = torch.randperm(replay_size, device="cpu")
        start_idx = minibatch_idx * self.minibatch_size
        mb_indices = replay_indices[start_idx:start_idx + self.minibatch_size]
        logical_batch_count = mb_indices.numel()

        self.critic_optimizer.zero_grad(set_to_none=True)
        critic_loss_value = 0.0
        q1_value = 0.0
        q2_value = 0.0
        target_value = 0.0
        cached_observations = [] if cache_actor_observations and update_policy else None
        for micro_start in range(0, logical_batch_count, self.max_minibatch_size):
            micro_indices = mb_indices[micro_start:micro_start + self.max_minibatch_size]
            micro_count = micro_indices.numel()
            profile("train_copy", epoch)
            if fixed_targets is None:
                mb_obs, mb_actions, mb_rewards, mb_next_obs, mb_discounts = (
                    tensor.to(self.device, non_blocking=True)
                    for tensor in self.replay.get_batch(micro_indices)
                )
            else:
                mb_obs, mb_actions = _gather_replay_fields(
                    self, ("observations", "actions"), micro_indices,
                )
                mb_obs = mb_obs.to(self.device, non_blocking=True)
                mb_actions = mb_actions.to(self.device, non_blocking=True)
                target_indices = micro_indices.to(self.device, non_blocking=True)
                mb_targets = fixed_targets.index_select(0, target_indices)
            if cached_observations is not None:
                cached_observations.append(mb_obs)
            profile("train_forward", epoch)
            last_microbatch = micro_start + micro_count == logical_batch_count
            sync_context = (
                self.ddp_critic.no_sync()
                if self.distributed and not last_microbatch
                else nullcontext()
            )
            with sync_context:
                if fixed_targets is None:
                    if target_policy is None:
                        mb_targets = self._compute_targets(mb_rewards, mb_next_obs, mb_discounts)
                    else:
                        mb_targets = self._compute_targets(
                            mb_rewards, mb_next_obs, mb_discounts, target_policy=target_policy,
                        )
                critic_loss, q1, q2 = self._critic_loss(mb_obs, mb_actions, mb_targets)
                profile("learn", epoch)
                (critic_loss * (micro_count / logical_batch_count)).backward()
            critic_loss_value += critic_loss.detach().item() * micro_count / logical_batch_count
            q1_value += q1.item() * micro_count / logical_batch_count
            q2_value += q2.item() * micro_count / logical_batch_count
            target_value += mb_targets.mean().item() * micro_count / logical_batch_count

        if self.config["max_grad_norm"] is not None:
            critic_grad_norm = torch.nn.utils.clip_grad_norm_(
                self.critic.parameters(), self.config["max_grad_norm"],
            )
            if diagnostics is not None:
                diagnostics.add_grad_norm("critic", critic_grad_norm, self.config["max_grad_norm"])
        self.critic_optimizer.step()

        update_losses = dict(
            critic_loss=critic_loss_value,
            q1=q1_value,
            q2=q2_value,
            target=target_value,
        )
        if update_policy:
            self.optimizer.zero_grad(set_to_none=True)
            policy_loss_value = 0.0
            entropy_sum = torch.zeros((), device=self.device, dtype=torch.float32)
            for microbatch_idx, micro_start in enumerate(
                    range(0, logical_batch_count, self.max_minibatch_size)):
                micro_indices = mb_indices[micro_start:micro_start + self.max_minibatch_size]
                micro_count = micro_indices.numel()
                profile("train_copy", epoch)
                if cached_observations is None:
                    mb_obs = _policy_observations(self, micro_indices).to(
                        self.device, non_blocking=True,
                    )
                else:
                    mb_obs = cached_observations[microbatch_idx]
                profile("train_forward", epoch)
                last_microbatch = micro_start + micro_count == logical_batch_count
                sync_context = (
                    self.ddp_policy.no_sync()
                    if self.distributed and not last_microbatch
                    else nullcontext()
                )
                with sync_context:
                    if fixed_actor_q is None:
                        policy_loss, entropy = self._policy_loss(mb_obs)
                    else:
                        cached_minimum_q = fixed_actor_q.index_select(
                            0, micro_indices.to(self.device, non_blocking=True),
                        )
                        policy_loss, entropy = _fixed_actor_q_policy_loss(
                            self, mb_obs, cached_minimum_q,
                        )
                        self.fixed_actor_loss_rows += int(mb_obs.shape[0])
                    profile("learn", epoch)
                    (policy_loss * (micro_count / logical_batch_count)).backward()
                policy_loss_value += policy_loss.detach().item() * micro_count / logical_batch_count
                entropy_sum += entropy * micro_count

            if self.config["max_grad_norm"] is not None:
                actor_grad_norm = torch.nn.utils.clip_grad_norm_(
                    self.policy.parameters(), self.config["max_grad_norm"],
                )
                if diagnostics is not None:
                    diagnostics.add_grad_norm("actor", actor_grad_norm, self.config["max_grad_norm"])
            self.optimizer.step()

            mean_entropy = entropy_sum / logical_batch_count
            if diagnostics is not None:
                diagnostics.add_entropy_update_mean(mean_entropy)
            update_losses["policy_loss"] = policy_loss_value
            update_losses["entropy"] = mean_entropy.item()
            if self.config.get("learn_alpha", True):
                target_entropy_controller = getattr(self, "target_entropy_controller", None)
                target_entropy = (
                    target_entropy_controller.target_entropy
                    if target_entropy_controller is not None
                    else self.config["target_entropy"]
                )
                alpha_loss = self.log_alpha * (mean_entropy - target_entropy)
                self.alpha_optimizer.zero_grad(set_to_none=True)
                alpha_loss.backward()
                if self.distributed:
                    torch.distributed.all_reduce(self.log_alpha.grad, op=torch.distributed.ReduceOp.SUM)
                    self.log_alpha.grad.div_(self.world_size)
                self.alpha_optimizer.step()
                update_losses["alpha_loss"] = alpha_loss.detach().item()
                self._update_target_entropy(entropy_sum, logical_batch_count)
        update_losses["alpha"] = self.alpha.item()

        profile("train_misc", epoch)
        # Match stock Polyak updates for parameters and buffers.
        with torch.no_grad():
            online = dict(self.uncompiled_critic.named_parameters()) | dict(
                self.uncompiled_critic.named_buffers()
            )
            targets = dict(self.uncompiled_target_critic.named_parameters()) | dict(
                self.uncompiled_target_critic.named_buffers()
            )
            for key, target in targets.items():
                if target.is_floating_point():
                    target.lerp_(online[key], self.config["tau"])
                else:
                    target.copy_(online[key])

        self.learner_updates += 1
        self.sampled_transitions += logical_batch_count * self.world_size
        profile("train_misc", epoch)
        for key, value in update_losses.items():
            losses[key] += value * logical_batch_count / total_sampled_transitions
    return total_minibatches


def install_variant(learner, variant):
    """Install one benchmark candidate on a single SAC learner instance.

    ``stock`` restores the learner's original bound method. Candidate installation
    replaces only that instance's ``_train_sac_transition`` method and keeps loss
    wrappers and counters attached to the learner intact.
    """
    if variant not in _VARIANTS:
        raise ValueError(f"Unknown SAC replay gather variant {variant!r}; expected one of {sorted(_VARIANTS)}")

    stock_method = getattr(learner, "_sac_profile_stock_train_transition", None)
    if stock_method is None:
        stock_method = learner._train_sac_transition
        learner._sac_profile_stock_train_transition = stock_method
    if variant == "stock":
        learner._train_sac_transition = stock_method
        return learner

    _check_candidate_support(learner)
    candidate = (
        _train_sac_transition_obs_only if variant == "obs_only"
        else _train_sac_transition_cached
    )
    if variant in {"fixed_targets", "fixed_targets_cached"}:
        candidate = (
            _train_sac_transition_fixed_targets
            if variant == "fixed_targets"
            else _train_sac_transition_fixed_targets_cached
        )
    elif variant == "fixed_targets_actor_q":
        candidate = _train_sac_transition_fixed_targets_actor_q
    learner._train_sac_transition = MethodType(candidate, learner)
    return learner
