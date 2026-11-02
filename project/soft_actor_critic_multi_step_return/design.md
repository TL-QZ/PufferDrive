# SAC n-step return design

## Target and scope

For a valid start `(s_t, a_t)` and an uninterrupted sequence of `k <= n_steps` outcomes:

```text
R = r[t+1] + gamma*r[t+2] + ... + gamma**(k-1)*r[t+k]
D = 0 if a true terminal was reached, otherwise gamma**k
target = R + D * V_soft(s[t+k])

V_soft(s) = sum_a policy(a|s) * (
    min(target_Q1(s,a), target_Q2(s,a)) - alpha*log(policy(a|s))
)
```

With `critic_entropy_in_target=false`, omit the endpoint entropy term. Actor optimization, temperature optimization, target-network updates, and replay sampling remain unchanged. No entropy rewards are inserted between the start and endpoint. There is no importance correction for intervening actions from older policies. This deliberately matches the practical SB3-style target, rather than claiming an unbiased multi-step soft-policy evaluation operator.

`train.n_steps` must be a positive Python integer; booleans, floating-point values, zero, and negative values are rejected. `1` bypasses accumulation, preserving the one-step collection path and its RNG consumption. Higher values require standard SAC and empty initial replay. PPO behavior factories, directly attached behavior providers, and PPO warm-start workflows are rejected for this mode.

## Boundary contract

Every emitted entry represents one uninterrupted sequence for one stable agent-slot ID. Received rows are associated with the compact one-step batch through exactly `pending[env_ids] & masks`, not through row positions from a previous receive. Explicit terminal flags determine terminality: discount zero is insufficient because `gamma=0` is valid.

| Event | Accumulator action | Endpoint and discount |
|---|---|---|
| Valid nonfinal row | Append; at `n_steps`, emit oldest start and keep overlapping newer starts | Latest valid successor; `gamma**n_steps` |
| True terminal | Include current reward; emit every remaining start; clear | Finite placeholder permitted; all discounts zero |
| Timeout or map resampling | Include valid final reward; emit every remaining start; clear | Captured pre-reset final observation; each prefix's `gamma**k` |
| Both terminal and truncation | Handle as terminal | Zero discount; terminal wins |
| Invalid/masked row | Exclude this row; flush earlier valid starts; clear | Cached last valid successor; prior reward sums and `gamma**k` |
| Ordinary collection/training boundary | Keep pending starts | No forced shortening |
| Global received-step budget exhausted | Flush already-received prefixes on every rank; apply remaining local quota | Cached valid successor; `gamma**k` |
| Cancellation/error shutdown | Discard staging | No additional receive, emission, or training |

Missing required final observations for valid nonterminal truncations remain an error. A masked reset row does not make its reward or reset observation eligible: earlier tails bootstrap from their last valid successor. Subsequent valid data begin a fresh sequence. No episode may bridge a masked gap or reset. This relies on the existing Drive transition contract; reset-only rows from a different vector backend must not be presented as valid completed transitions.

## Overlapping windows

For one agent, `n_steps=3`, `gamma=0.9`, and rewards `1, 2, 3, 4`:

| Outcome received | Emitted start | Stored reward | Stored discount | Starts retained |
|---|---|---|---|---|
| `r1=1`, successor `s1` | None | — | — | `s0` |
| `r2=2`, successor `s2` | None | — | — | `s0, s1` |
| `r3=3`, successor `s3` | `s0` | `1+.9*2+.81*3 = 5.23` | `.729` | `s1, s2` |
| `r4=4`, successor `s4` | `s1` | `2+.9*3+.81*4 = 7.94` | `.729` | `s2, s3` |

If the fourth outcome is terminal, it instead emits `s1`, `s2`, and `s3`, with rewards `7.94`, `6.6`, and `4`, all discounts zero. If it is a timeout, the same rewards use the true final `s4` with discounts `.729`, `.81`, and `.9`.

If the third row is invalid, no reward `3` is consumed. Flush `s0` and `s1` with rewards `2.8` and `2`, endpoint `s2`, and discounts `.81` and `.9`. The invalid row's observation is never the endpoint.

A collection stop after the second valid outcome does not flush: the sequence may mature after training. Policy changes between collection calls are allowed; the stored observed rewards still came from the behavior policies that generated those actions.

## Storage and collection

`NStepAccumulator` uses rank-local, per-agent fixed-size rings. The observation ring has shape `[agent_slots, n_steps, *observation_shape]`; actions and reward data use `[agent_slots, n_steps]`. Per-agent metadata tracks ring positions and occupancy; one `[agent_slots, *observation_shape]` tensor owns the latest valid successor. Tensor operations process received batches; bounded horizon loops handle prefix emission. There are no Python per-agent queues.

The staging tensors live on `replay.device` and follow replay's pinned-memory setting. Shared simulator observations and caller tensors are copied into owned storage before a send can overwrite them. Observation storage alone costs:

```text
agent_slots * (n_steps + 1) * observation_features * 4 bytes
64,000 * (3 + 1) * 923 * 4 / 1024**3 = 0.880 GiB per rank
```

This excludes scalar metadata, emitted candidate tensors, transfer temporaries, the actor's pending observations, and the existing replay allocation. CPU offload moves this additional storage to host memory; it does not remove it.

Replay continues to store five fields: `(observation, action, reward, next_observation, discount)`. Reward and discount now encode the aggregate. Effective horizons are carried only with emitted candidates for telemetry; they are not added to replay samples. `_compute_targets()` therefore needs no new target formula or sampling interface.

All received rows advance or clear their sequences before quota subsampling. A boundary can emit multiple starts per agent. Randomly excluded candidates are permanently discarded; their starts have already left staging and cannot reappear. Remaining newer starts are preserved. Replay capacity and refresh quotas count stored entries, not raw transitions or episodes. Initial filling can require more simulator outcomes while windows mature.

At collection end, received-row counters are synchronized across ranks. When the resulting global counter reaches the training budget, every rank flushes its pending prefixes using only outcomes it has already received, then selects within its remaining local insertion quota. Additional insertion counts are synchronized. This does not call `recv()` again or invent the outcome of the last action sent. Existing receive-batch granularity can overshoot the requested budget. No collective occurs inside per-agent boundary processing.

## Metrics

`agent_steps` retains its global count of all received agent rows, including initial observations and masked rows. `replay/valid_inserts` counts globally inserted replay entries, now aggregated for multi-step mode. Existing environment metrics keep their episode-based meaning.

| Metric | Units and aggregation |
|---|---|
| `sac/n_steps` | Configured maximum horizon; identical on all ranks |
| `replay/mean_effective_horizon` | Sum of inserted horizons divided by inserted entry count across all ranks since the preceding log; zero when no entries were inserted |
| `replay/pending_starts` | Latest number of unfinished starts after collection, summed across ranks |
| `replay/quota_discarded` | Emitted candidates permanently dropped by insertion quotas across all ranks since the preceding log |

Horizon units are simulator transitions for one agent; counts are entries, not episodes. Logging can cover multiple collection calls, so the mean is weighted by actual insertions rather than averaging per-collection means. The pending-start count is a snapshot, not a cumulative count. Horizon and discard counters restart after logging and after learner-only resume.

## Resume and compatibility

Checkpoints remain learner-only in the current format. Configuration records `n_steps`; older checkpoints without the field resolve to `1`. Resuming with a different horizon is rejected. Replay, simulator state, pending actions, and accumulator staging are not checkpointed; resumed collection begins with fresh versions of all four. Consequently resume is not exact trajectory continuation. Existing one-step PPO integration remains supported, but multi-step PPO integrations are intentionally rejected before their collection hooks run.

## Validation

Activate the repository environment before running tests:

```bash
source .venv/bin/activate
python -m pytest -q tests/unit_tests/test_sac_nstep.py tests/unit_tests/test_sac_nstep_collection.py
python -m pytest -q tests/unit_tests/test_sac_distributed.py
python -m pytest -q tests/unit_tests/test_sac_*.py tests/unit_tests/test_pufferl_sac_startup.py
```

Required evidence includes an independent numerical oracle, terminal/timeout/mask boundaries, overlapping starts, copied ownership, interleaved stable IDs, quota dropping, persistence across collection calls, final-budget flushes without extra receives, one-step parity, checkpoints, and unequal-rank Gloo collection. CPU tests are mandatory. CUDA/pinned-offload parity is conditional on a usable CUDA device. Synthetic correctness tests do not establish simulator throughput or policy improvement.

Validated on 2026-10-30, branch `tony/sac_multi_step_return`:

| Check | Actual result |
|---|---|
| Full SAC regression command above, including startup | **200 passed**, no skips, in **67.32 seconds** |
| Focused accumulator/replay CPU, CUDA, and pinned-offload tests | **31 passed** in **3.21 seconds**, also included in the full run |
| Collector/target/compatibility/checkpoint tests | **24 passed**, also included in the full run |
| Two-rank CPU/Gloo suite | All four tests passed in the full run, including the new n-step final-budget test |
| Final diff and independent review | `git diff --check` passed; no unresolved blocking findings |

The full command ran with host permissions because the restricted sandbox blocks Gloo loopback sockets and CUDA access. The host exposes NVIDIA RTX A5000 GPUs. CUDA parity used small synthetic sequences to compare GPU staging, pinned CPU staging receiving CUDA inputs, and a CPU reference through overlapping windows, terminal/timeout boundaries, selective mask-gap flushes, and final flushes. Zero-length tensors have no required pinned allocation; nonempty offloaded outputs were checked for pinning.

The full run reported one existing PyTorch/Heavyball deprecation warning from the Muon optimizer test (`torch._prims_common.check`). No C source changed, so no extension rebuild was needed. These tests did not launch live Drive training, a throughput measurement, or a driving benchmark. The subsequently prepared [32-step trial configurations](configs/round1_nstep32/README.md) have separate dry-run validation. These results establish the tested data and learner contracts; they do not establish policy improvement or full-scale memory/throughput behavior.

## References

- [SB3 SAC target implementation](https://stable-baselines3.readthedocs.io/en/master/_modules/stable_baselines3/sac/sac.html): endpoint entropy and replay-provided bootstrap discount.
- [SB3 NStepReplayBuffer source](https://github.com/DLR-RM/stable-baselines3/blob/master/stable_baselines3/common/buffers.py): discounted observed rewards and shorter boundary windows, computed at sample time.
- [Raffin, Getting SAC to Work on a Massive Parallel Simulator, Part II](https://araffin.github.io/post/tune-sac-isaac-sim/): motivating continuous-control experiments, not proof of driving performance.
