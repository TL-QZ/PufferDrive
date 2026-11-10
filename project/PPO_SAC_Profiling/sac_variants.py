"""Learner-local SAC replay gather candidates for the profiling harness.

The candidate update loop is copied from
``pufferlib.pufferl_sac.PuffeSAC._train_sac_transition`` at repository revision
``cd4a75432882d64bee8977eba473b35bd6c9e7e9`` (source lines 1818-1980). The
variants retain its backward, optimizer, clipping, target-update, DDP, and
accounting order. ``obs_only`` and ``cached`` change replay gathering only;
``fixed_targets`` and ``fixed_targets_cached`` also reuse cycle-start targets.
Rebase this copy when the source method changes.

The observation-gather candidates intentionally support the baseline
``n_step`` estimator only. The fixed-target candidates also support PQL by
retaining one cycle-start SequenceReplay plan and preparing its ordered targets
before any gradient updates. The ``fixed_targets`` variants are algorithmic
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
    storage = _replay_storage(replay, field_name)
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
    storage = _replay_storage(replay, "observations")
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


def _replay_storage(replay, field_name):
    """Resolve transition fields on TensorReplay or SequenceReplay storage."""
    if hasattr(replay, field_name):
        return getattr(replay, field_name)
    wrapped_replay = getattr(replay, "_replay", None)
    if wrapped_replay is not None and hasattr(wrapped_replay, field_name):
        return getattr(wrapped_replay, field_name)
    raise TypeError(f"SAC replay storage does not expose {field_name!r}")


def _resident_replay_indices(replay, replay_size):
    """Return physical indices for all currently resident rows."""
    valid_rows = getattr(replay, "_valid", None)
    if valid_rows is not None:
        return torch.nonzero(valid_rows, as_tuple=False).flatten().to(device="cpu")
    return torch.arange(replay_size, dtype=torch.int64, device="cpu")


def _check_candidate_support(learner, *, allow_pql=False):
    estimator = learner.config.get("return_estimator", "n_step")
    if estimator == "pql" and not allow_pql:
        raise ValueError("SAC replay gather candidates support return_estimator='n_step' only")
    if estimator not in {"n_step", "pql"}:
        raise ValueError(f"Unsupported SAC return_estimator {estimator!r}")
    replay = learner.replay
    for attribute in ("device", "pin_memory"):
        if not hasattr(replay, attribute):
            raise TypeError(
                "SAC replay gather candidates require TensorReplay-style observation storage"
            )
    _replay_storage(replay, "observations")
    if estimator == "pql":
        if not callable(getattr(replay, "build_epoch_batches", None)):
            raise TypeError("PQL fixed-target candidates require SequenceReplay planning")
        if not callable(getattr(learner, "_prepare_pql_targets", None)):
            raise TypeError("PQL fixed-target candidates require the stock PQL target preparer")


def _check_fixed_target_support(learner):
    _check_candidate_support(learner, allow_pql=True)
    replay = learner.replay
    for attribute in ("actions", "rewards", "next_observations", "discounts"):
        _replay_storage(replay, attribute)


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
        rewards = _replay_storage(learner.replay, "rewards")[start_idx:end_idx]
        next_observations = _replay_storage(learner.replay, "next_observations")[start_idx:end_idx]
        discounts = _replay_storage(learner.replay, "discounts")[start_idx:end_idx]
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
def _precompute_fixed_cycle_pql_targets(learner, sequence_batches, replay_size, profile, epoch):
    """Prepare every PQL segment batch before updates and scatter by replay row."""
    replay = learner.replay
    capacity = replay.capacity
    resident_indices = _resident_replay_indices(replay, replay_size)
    cache_size = int(resident_indices.max()) + 1 if resident_indices.numel() else 0
    targets = torch.empty(cache_size, device=learner.device, dtype=torch.float32)
    coverage = torch.zeros(capacity, dtype=torch.bool, device="cpu")
    learner.fixed_target_precompute_rows = 0
    learner.fixed_target_precompute_calls = 0
    for replay_indices, segment_offsets in sequence_batches:
        if replay_indices.numel() == 0:
            continue
        if bool(coverage[replay_indices].any()):
            raise RuntimeError("PQL epoch plan contains duplicate replay rows")
        profile("train_forward", epoch)
        batch_targets = learner._prepare_pql_targets(replay_indices, segment_offsets)
        if batch_targets.shape != replay_indices.shape:
            raise RuntimeError("PQL target preparer returned a misaligned batch")
        target_indices = replay_indices.to(learner.device, non_blocking=True)
        targets.index_copy_(0, target_indices, batch_targets.detach().float())
        coverage[replay_indices] = True
        learner.fixed_target_precompute_rows += replay_indices.numel()
        learner.fixed_target_precompute_calls += 1

    resident_mask = getattr(replay, "_valid", None)
    if resident_mask is None:
        resident_mask = torch.zeros(cache_size, dtype=torch.bool, device="cpu")
        resident_mask[resident_indices] = True
    if int(coverage.sum()) != replay_size or not torch.equal(coverage, resident_mask):
        raise RuntimeError("PQL epoch plan must cover every resident replay row exactly once")
    return targets


@torch.no_grad()
def _precompute_fixed_cycle_actor_q(learner, replay_size, profile, epoch, replay_indices=None):
    """Cache cycle-start online min-Q scores for all actions and replay rows."""
    minimum_q_cache = None
    learner.fixed_actor_q_precompute_rows = 0
    learner.fixed_actor_q_precompute_calls = 0
    learner.actor_q_forward_rows = 0
    learner.actor_q_forward_calls = 0
    if replay_indices is None:
        replay_indices = torch.arange(replay_size, dtype=torch.int64, device="cpu")
    cache_size = int(replay_indices.max()) + 1 if replay_indices.numel() else 0
    replay_storage = _replay_storage(learner.replay, "observations")
    for start_idx in range(0, replay_indices.numel(), learner.max_minibatch_size):
        end_idx = min(start_idx + learner.max_minibatch_size, replay_indices.numel())
        profile("train_copy", epoch)
        chunk_indices = replay_indices[start_idx:end_idx]
        if torch.equal(
                chunk_indices,
                torch.arange(start_idx, end_idx, dtype=torch.int64, device="cpu")):
            observations = replay_storage[start_idx:end_idx]
        else:
            observations = replay_storage[chunk_indices.to(replay_storage.device)]
        observations = observations.to(learner.device, non_blocking=True)
        profile("train_forward", epoch)
        # Use the same online critic path and AMP policy as the stock actor loss.
        # The cache is detached float32 even when inference ran under BF16 AMP.
        with learner.amp_context:
            q1, q2 = learner.critic(observations)
        minimum_q = torch.minimum(q1.float(), q2.float())
        if minimum_q_cache is None:
            minimum_q_cache = torch.empty(
                (cache_size, *minimum_q.shape[1:]),
                device=learner.device,
                dtype=torch.float32,
            )
        cache_indices = chunk_indices.to(learner.device, non_blocking=True)
        minimum_q_cache.index_copy_(0, cache_indices, minimum_q)
        row_count = end_idx - start_idx
        learner.fixed_actor_q_precompute_rows += row_count
        learner.fixed_actor_q_precompute_calls += 1
        learner.actor_q_forward_rows += row_count
        learner.actor_q_forward_calls += 1
    return minimum_q_cache


@torch.no_grad()
def _precompute_fixed_filter_scores(
        learner, replay_indices, fixed_targets, fixed_actor_q, score_name, profile, epoch):
    """Score every resident row once using cycle-start actor and critic values."""
    replay = learner.replay
    score_chunks = []
    for start_idx in range(0, replay_indices.numel(), learner.max_minibatch_size):
        end_idx = min(start_idx + learner.max_minibatch_size, replay_indices.numel())
        chunk_indices = replay_indices[start_idx:end_idx]
        profile("train_copy", epoch)
        observations, actions = _gather_replay_fields(
            learner, ("observations", "actions"), chunk_indices,
        )
        observations = observations.to(learner.device, non_blocking=True)
        actions = actions.to(learner.device, non_blocking=True)
        cache_indices = chunk_indices.to(learner.device, non_blocking=True)
        minimum_q = fixed_actor_q.index_select(0, cache_indices)
        profile("train_forward", epoch)
        with learner.amp_context:
            logits = learner.policy(observations)
        probabilities = F.softmax(logits.float(), dim=-1)
        baseline = (probabilities * minimum_q).sum(-1)
        if score_name == "q_advantage":
            recorded_q = minimum_q.gather(1, actions.unsqueeze(-1)).squeeze(-1)
            scores = (recorded_q - baseline).abs()
        elif score_name == "target_advantage":
            targets = fixed_targets.index_select(0, cache_indices)
            scores = (targets - baseline).abs()
        else:
            scores = (probabilities * (minimum_q - baseline.unsqueeze(-1)).abs()).sum(-1)
        score_chunks.append(scores.detach().float())
    scores = torch.cat(score_chunks) if score_chunks else torch.empty(
        0, device=learner.device, dtype=torch.float32,
    )
    if not bool(torch.isfinite(scores).all()):
        raise ValueError("SAC profile filter produced nonfinite scores")
    return scores


def _resolve_profile_filter(learner, fixed_cycle_actor_q, update_policy):
    """Validate opt-in cycle-local filtering configuration."""
    config = getattr(learner, "config", {})
    score_name = config.get("profile_filter_score", "none")
    keep_fraction = config.get("profile_keep_fraction", 1.0)
    filter_mode = config.get("profile_filter_mode", "fraction")
    ewma_beta = config.get("profile_filter_ewma_beta", 0.25)
    threshold_scale = config.get("profile_filter_threshold_scale", 0.01)
    valid_scores = {"none", "q_advantage", "target_advantage", "expected_abs_advantage"}
    if not isinstance(score_name, str) or score_name not in valid_scores:
        raise ValueError(
            "profile_filter_score must be one of "
            "'none', 'q_advantage', 'target_advantage', or 'expected_abs_advantage'"
        )
    if isinstance(keep_fraction, bool) or not isinstance(keep_fraction, (int, float)):
        raise ValueError("profile_keep_fraction must be a finite number in (0, 1]")
    keep_fraction = float(keep_fraction)
    if not math.isfinite(keep_fraction) or not 0 < keep_fraction <= 1:
        raise ValueError("profile_keep_fraction must be a finite number in (0, 1]")
    if not isinstance(filter_mode, str) or filter_mode not in {"fraction", "ewma_threshold"}:
        raise ValueError("profile_filter_mode must be 'fraction' or 'ewma_threshold'")
    if isinstance(ewma_beta, bool) or not isinstance(ewma_beta, (int, float)):
        raise ValueError("profile_filter_ewma_beta must be finite and in [0, 1]")
    ewma_beta = float(ewma_beta)
    if not math.isfinite(ewma_beta) or not 0 <= ewma_beta <= 1:
        raise ValueError("profile_filter_ewma_beta must be finite and in [0, 1]")
    if isinstance(threshold_scale, bool) or not isinstance(threshold_scale, (int, float)):
        raise ValueError("profile_filter_threshold_scale must be finite and nonnegative")
    threshold_scale = float(threshold_scale)
    if not math.isfinite(threshold_scale) or threshold_scale < 0:
        raise ValueError("profile_filter_threshold_scale must be finite and nonnegative")
    if score_name == "none" and keep_fraction != 1.0:
        raise ValueError("profile_keep_fraction requires a non-'none' profile_filter_score")
    if filter_mode == "ewma_threshold":
        if score_name == "none":
            raise ValueError("ewma_threshold filtering requires a non-'none' profile_filter_score")
        if keep_fraction != 1.0:
            raise ValueError("ewma_threshold filtering requires profile_keep_fraction=1.0")
    filter_requested = score_name != "none"
    if filter_requested and not fixed_cycle_actor_q:
        raise ValueError("profile filtering is supported only by fixed_targets_actor_q")
    if filter_requested and not update_policy:
        raise ValueError("profile filtering requires update_policy=True")
    return score_name, keep_fraction, filter_mode, ewma_beta, threshold_scale


def _select_profile_filter_rows(learner, replay_indices, scores, keep_fraction, threshold=None):
    """Select eligible rows by score and synchronize retained counts for DDP."""
    if threshold is None:
        eligible_positions = torch.arange(scores.numel(), device=scores.device)
        requested_count = math.ceil(replay_indices.numel() * keep_fraction)
    else:
        eligible_positions = torch.nonzero(scores >= threshold, as_tuple=False).flatten()
        requested_count = int(eligible_positions.numel())
    eligible_scores = scores.index_select(0, eligible_positions)
    ranking = eligible_positions.index_select(
        0, torch.argsort(eligible_scores, descending=True, stable=True),
    )
    retained_before_sync = requested_count
    retained_count = retained_before_sync
    if learner.distributed:
        retained_count_tensor = torch.tensor(
            [retained_count], dtype=torch.int64, device=learner.device,
        )
        torch.distributed.all_reduce(
            retained_count_tensor, op=torch.distributed.ReduceOp.MIN,
        )
        retained_count = int(retained_count_tensor.item())
    retained_positions = ranking[:retained_count].to(device="cpu")
    retained_indices = replay_indices[retained_positions]
    return retained_indices, retained_count, retained_before_sync


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
    self.fixed_cycle_targets = None
    self.fixed_target_precompute_rows = 0
    self.fixed_target_precompute_calls = 0
    self.fixed_pql_sequence_batches = None
    self.fixed_cycle_actor_q = None
    self.fixed_actor_q_precompute_rows = 0
    self.fixed_actor_q_precompute_calls = 0
    self.actor_q_forward_rows = 0
    self.actor_q_forward_calls = 0
    self.fixed_actor_loss_rows = 0
    self.filter_candidate_rows = 0
    self.filter_retained_rows = 0
    self.filter_score_min = 0.0
    self.filter_score_max = 0.0
    self.filter_score_mean = 0.0
    self.filter_current_max = None
    self.filter_threshold = None
    self.filter_retained_before_sync = None
    self.fixed_filter_scores = None
    self.fixed_filter_indices = None
    self.fixed_filter_candidate_indices = None

    if fixed_cycle_targets:
        _check_fixed_target_support(self)
    else:
        _check_candidate_support(self)
    score_name, keep_fraction, filter_mode, ewma_beta, threshold_scale = _resolve_profile_filter(
        self, fixed_cycle_actor_q, update_policy,
    )
    pql_enabled = self.config.get("return_estimator", "n_step") == "pql"
    if pql_enabled and target_policy is not None:
        raise ValueError("PQL requires the current SAC actor for bootstrap values")

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

    sequence_batches = None
    if pql_enabled:
        sequence_batches = self.replay.build_epoch_batches(
            self.config["pql_horizon"], self.minibatch_size,
        )
        self.fixed_pql_sequence_batches = sequence_batches

    fixed_targets = None
    if fixed_cycle_targets:
        if pql_enabled:
            fixed_targets = _precompute_fixed_cycle_pql_targets(
                self, sequence_batches, replay_size, profile, epoch,
            )
        else:
            fixed_targets = _precompute_fixed_cycle_targets(
                self, replay_size, profile, epoch, target_policy,
            )
        # Keep the detached cache available for harness audits until the next
        # training cycle replaces it.
        self.fixed_cycle_targets = fixed_targets

    resident_indices = _resident_replay_indices(self.replay, replay_size)
    fixed_actor_q = None
    if fixed_cycle_actor_q and update_policy:
        fixed_actor_q = _precompute_fixed_cycle_actor_q(
            self, replay_size, profile, epoch, replay_indices=resident_indices,
        )
        # Keep the detached cycle cache available for row-alignment audits.
        self.fixed_cycle_actor_q = fixed_actor_q

    retained_indices = None
    retained_count = replay_size
    if score_name != "none":
        scores = _precompute_fixed_filter_scores(
            self, resident_indices, fixed_targets, fixed_actor_q,
            score_name, profile, epoch,
        )
        self.filter_current_max = float(scores.max().item())
        if filter_mode == "ewma_threshold":
            previous_ema_max = getattr(self, "filter_ema_max", None)
            self.filter_ema_max = (
                self.filter_current_max
                if previous_ema_max is None
                else ewma_beta * self.filter_current_max + (1.0 - ewma_beta) * previous_ema_max
            )
            self.filter_threshold = threshold_scale * self.filter_ema_max
            if not math.isfinite(self.filter_threshold):
                raise ValueError("profile_filter_threshold_scale produced a nonfinite threshold")
            retained_indices, retained_count, retained_before_sync = _select_profile_filter_rows(
                self, resident_indices, scores, keep_fraction,
                threshold=self.filter_threshold,
            )
        else:
            retained_indices, retained_count, retained_before_sync = _select_profile_filter_rows(
                self, resident_indices, scores, keep_fraction,
            )
        self.filter_retained_before_sync = retained_before_sync
        self.filter_candidate_rows = int(resident_indices.numel())
        self.filter_retained_rows = retained_count
        self.filter_score_min = float(scores.min().item())
        self.filter_score_max = float(scores.max().item())
        self.filter_score_mean = float(scores.mean().item())
        self.fixed_filter_scores = scores.detach()
        self.fixed_filter_indices = retained_indices.detach().clone()
        self.fixed_filter_candidate_indices = resident_indices.clone()
    else:
        self.filter_candidate_rows = int(resident_indices.numel())
        self.filter_retained_rows = int(resident_indices.numel())
        self.fixed_filter_indices = resident_indices.clone()
        self.fixed_filter_candidate_indices = resident_indices.clone()

    filtering_rows = retained_count < replay_size
    training_rows = (
        torch.sort(retained_indices).values
        if filtering_rows
        else resident_indices
    )
    training_row_count = training_rows.numel()
    if training_row_count == 0:
        return 0
    minibatches_per_epoch = math.ceil(training_row_count / self.minibatch_size)

    total_minibatches = self.config["update_epochs"] * minibatches_per_epoch
    total_sampled_transitions = retained_count * self.config["update_epochs"]
    replay_indices = None
    for logical_update_idx in range(total_minibatches):
        minibatch_idx = logical_update_idx % minibatches_per_epoch
        if minibatch_idx == 0:
            shuffled_positions = torch.randperm(training_row_count, device="cpu")
            replay_indices = training_rows[shuffled_positions]
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

    if variant in {"fixed_targets", "fixed_targets_cached", "fixed_targets_actor_q"}:
        _check_fixed_target_support(learner)
    else:
        _check_candidate_support(learner)
    _resolve_profile_filter(
        learner, variant == "fixed_targets_actor_q", update_policy=True,
    )
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
