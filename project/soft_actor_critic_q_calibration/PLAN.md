# Q-value calibration by distance to the next waypoint

Original status: analysis plan only, prepared 2026-11-02. Checkpoint metadata and source
were inspected read-only. No rollout, training, analysis implementation, or
simulator modification has been performed. All future analysis artifacts
belong in this folder; source experiments remain read-only.

Execution update: the user subsequently authorized this analysis, limited to
this folder and physical GPUs 6 and 7. The text below preserves the original
plan; [README.md](README.md) describes the implemented workflow, and
[REPORT.md](REPORT.md) records the final evidence and deviations from the plan.

Question: does the completed TES-SAC policy have well-calibrated Q estimates
near a waypoint but inaccurate or uninformative estimates farther away?
This is a testable hypothesis, not a conclusion from flat Q curves alone.

## 1. Verified checkpoint and comparison contract

Source run:

`../soft_actor_critic_target_entropy_scheduling/runs/round2_tes_sac_20261029T044847.014369240_539543/trial_98/`

| Item | Verified value |
|---|---|
| Completion | summary.json: completed, exit code 0, no missing artifacts |
| Final progress | Epoch 1615; 4,000,032,000 global agent rows; 77,520 learner updates |
| Primary checkpoint | learner/trainer_state.pt, approximately 161 MiB |
| Checkpoint contents | policy_state_dict, critic_state_dict, target_critic_state_dict, log_alpha, learner config, counters, RNG/controller/optimizer states |
| Discount | gamma=0.999 |
| Temperature | exp(log_alpha)=0.0005304708029143512 |
| Critic convention | critic_entropy_in_target=true; n_steps absent in this historical state, corresponding to legacy one-step behavior |
| Environment | dt=0.3 s; scenario_length=2560; finite goals; 3 goal observations; spacing 20–200 m; randomized reward conditioning |
| Existing evaluation | eval.action_selection=mean; final carla_fast puffer_score about 0.4627 |

Use the actor, online twin critics, target twin critics, and alpha from the
SAME final learner checkpoint. Freeze all of them: no optimizer, Polyak,
alpha, TES, or normalization-statistic updates during the diagnostic.

The best evaluation actor is epoch 1506, score about 0.5375. The separately
named `best_trainer_state_001574.pt` was inspected and is also an actor-only
state dict despite its name. Neither is a substitute for a matching full
critic checkpoint. Do not combine either actor with the final critics and
call the resulting error a same-policy calibration measurement.

The trusted locally generated full state contains NumPy RNG metadata, so a
weights-only torch.load rejected it. CPU inspection succeeded using the
repository-style full-state loader with mmap; no GPU allocation was needed.
Future loading should reconstruct networks directly without allocating
training replay or restoring a running optimizer/training loop.

### Which Q and which action?

Record Q1(s_t,a_t), Q2(s_t,a_t), their minimum and mean for the discrete action
actually selected at s_t. Report each critic separately; twin-min clipping is
an intentional conservative estimator. Target-critic estimates are a secondary
diagnostic for lag, not the sole definition of learned Q.

Sample categorical actions from the frozen actor for the main experiment,
including subsequent actions. Use the native discrete-to-continuous decoder.
Mean physical jerk can lie between action classes: the Q of an arbitrary
class is not its value. Therefore existing mean-action benchmark episodes
cannot directly calibrate this discrete Q(s,a).

## 2. Reference returns, boundaries, and experimental conditions

Let r_t be the raw environment reward resulting from action a_t. For a
trajectory ending at a TRUE terminal T, compute backward:

```text
G_reward_(T-1) = r_(T-1)
G_soft_(T-1)   = r_(T-1)

G_reward_t = r_t + gamma * G_reward_(t+1)
G_soft_t   = r_t + gamma * [G_soft_(t+1) - alpha*log pi(a_(t+1)|s_(t+1))]
```

Thus G_soft excludes the entropy of the initial action a_t, matching SAC's
Q convention, and includes entropy of later sampled policy actions. Freeze
alpha at the saved value. Record G_reward and the discounted entropy
contribution separately. Exact categorical entropy at future states can be
an additional estimator/check, but must be clearly labeled rather than mixed
with sampled log probabilities silently.

Primary residuals are `Q1-G_soft`, `Q2-G_soft`, and `min(Q1,Q2)-G_soft`.
Reward-only returns answer a different question and must not be used alone
to claim this soft critic is overestimating. A single realized return is a
noisy sample of Q's conditional expectation, not an exact state-action value.

### Three endpoint categories

| Endpoint | Treatment |
|---|---|
| True agent terminal | Zero continuation; complete sampled return for that trajectory |
| Scenario timeout/map reset | Use the true pre-reset successor; mark incomplete continuation; never bridge into the reset episode |
| Analysis collection cap / missing valid continuation | Mark right-censored; do not pretend unseen future reward is zero |

At gamma=.999, gamma^2560 is about .077, so even a full 2560-step timeout can
leave material continuation value. A 128-step window is emphatically not a
complete-return reference for this checkpoint.

Report terminal-complete trajectories separately and disclose their selection
bias: they may overrepresent collisions or other early endings. For censored
trajectories, show observed discounted reward/entropy sums, retained discount,
and censorship rate by distance. These partial sums are neither exact returns
nor general lower bounds when rewards can be negative.

A secondary estimate may append the frozen target critic's soft value at a
nonterminal cut. Label it `bootstrapped_tail_return`, never Monte Carlo ground
truth: it partly tests the critic against itself. Compare tail estimates
across available rollout lengths and report the fraction attributable to
bootstrap. Do not change scenario_length or reinterpret a timeout as terminal
to manufacture a complete target. If censored tails dominate, the report must
say the full-return calibration question remains unresolved there.

### Match the training task first

Primary rollouts use the saved training environment/reward/observation config,
including dt=.3, original spacing, finite-goal semantics, reward conditioning
and randomization, traffic composition, and termination/reset rules. Preserve
within-scene density; reducing worker count is acceptable only if it does not
change those scene semantics. Freeze the shared policy for all policy-driven
agents so the data describe that checkpoint's self-play continuation.

Log actual reward coefficients for each trajectory. The critic is conditioned
on them; distance alone cannot explain values when goal and penalty weights
vary. Use new rollout RNG streams, saved map/scenario identifiers, and recorded
initialization seeds. This measures calibration on the selected distribution,
not automatically held-out map generalization.

Optional carla_fast rollouts are a SEPARATE distribution-shift analysis using
the run's saved resolved_benchmark.yaml. Differences in timestep, rewards,
traffic, goal placement, and action mode must be tabulated. Do not pool these
rows with training-matched calibration or quietly reuse current default
benchmark catalogs.

## 3. Distance measurement and data collection

Bin using distance at the same pre-action state where Q(s_t,a_t) is evaluated.
Identify the current waypoint before advancing goals. Goal switches do not
end the Q return unless the environment itself terminates the agent. Q values
all subsequent rewards, not merely reward until the next waypoint. Record
return-to-next-waypoint, waypoint success, and time-to-waypoint separately as
behavior diagnostics rather than treating them as total-Q ground truth.

Source inspection establishes that this run uses map-generated goals from a
random lane walk with NO retained route (`drive.h:1132`). The exact length of
that generated walk cannot be recovered from a route object after collection.
Use the raw precomputed lane-graph distance from current lane to active goal
lane as the proposed primary road-network PROXY, explicitly named as such.
It is not exact remaining distance along the discarded generated walk and may
omit within-lane progress. Plot Euclidean distance alongside it.

The native graph matrix is available internally (`drive.h:3892`), but Python
does not expose a direct current-goal distance scalar. Observation lane-distance
columns are normalized/clipped and map invalid/unreachable distances to a
maximum; do not invert those into purported exact distances. Goal observation
offsets (`drive.h:3751`) are ego-frame coordinates normalized by 200 m in this
run, with consumed slots zeroed. Deriving Euclidean distance requires the
correct active goal index rather than assuming the first slot is current.

`Drive.get_state()` (`drive.py:1008`) exports geometry and current goal position
but omits active goal index/count, goal-lane IDs, and the graph matrix. Minimal
future diagnostic export should supply pre-action agent/episode identity,
current lane, current goal index/count/lane, raw graph distance, reachability,
and goal coordinates. Missing/unreachable distance remains a separate category.
If shared native instrumentation is not approved, a Python-only Euclidean
diagnostic can be a preliminary result, clearly labeled with its limitations.

There is also an alignment issue to validate before collecting results:
current `c_step` computes observations before updating/regenerating goals
(`drive.h:4638`). At finite goal-set exhaustion the policy's returned goal
observation may describe the old exhausted set while native state already has
new goals. Capture/flag goal-switch rows and distinguish policy-observed from
native active-goal metadata; report a sensitivity analysis excluding affected
rows. Do not repair simulation behavior as part of a calibration analysis.
Check whether the checkpoint-era source had the same ordering before treating
this as a property of its training data.

Any future metadata instrumentation must be read-only with respect to dynamics,
RNG, observations, and reward, and export stable agent/episode IDs, waypoint
index, and distances. Prefer an analysis-side/native diagnostic query over
altering the observation vector or policy architecture. This is a planning
dependency, not authorization to modify C now; any eventual C/H change requires
the repository's mandatory rebuild and transition-parity checks.

Collect transition-level scalar records, not every full observation:

| Group | Required fields |
|---|---|
| Identity | checkpoint hash, config hash, map/scenario, initialization seed, action RNG stream, stable agent ID, episode generation, timestep |
| Policy/critic | action class, sampled log probability, exact entropy, Q1/Q2, target Q1/Q2, alpha |
| Task state | pre-action waypoint ID, route-distance value and definition, Euclidean distance, speed, goal count/progress, reward coefficients |
| Outcome | raw reward and components, terminal/truncation/validity flags, endpoint reason, final-observation availability |
| Derived | observed return length, tail discount, G_reward, G_soft, entropy contribution, censoring and residuals |

Keep bounded observation snapshots only if needed to reproduce selected
examples. Capture any potential fork state only through a verified simulator
snapshot mechanism; do not imply that cloning an observation restores a state.
No complete state-restore/clone API was found: get_state is a serializer, and
capture_replay records rendering data, not a restorable native/RNG snapshot.
Replaying an initialization plus a recorded joint-action prefix is a possible
future alternative, but requires an explicit deterministic replay check.
If exact same-state/action branching is unavailable, use ordinary on-policy
rollouts and describe the result as binned calibration, not per-state Monte
Carlo value estimation from repeated futures.

Proposed collection stages after implementation is authorized:

1. Metadata/load audit and synthetic return-indexing checks, without simulation.
2. A bounded pilot of 8 training-matched scene initializations and 2 independent
   action seeds per initialization, up to their configured 2560-step boundary.
   Confirm runtime, storage, distance availability, terminal alignment, and
   censoring before selecting a larger workload.
3. A main sample across the saved map set, targeting at least 30 independent
   scene initializations in each interpretable distance bin. This is a coverage
   goal, not a promise of statistical precision; sparse bins remain labeled.
4. Optional repeated-future studies at identical state/action pairs, only if
   simulator/environment RNG restoration and multi-agent state cloning are
   proven correct. Freeze the first action and resample subsequent policy RNG.
5. Optional separate benchmark-distribution analysis after the primary result.

No GPU allocation, pilot, or main rollout is requested by this plan alone.

## 4. Analysis, plots, and interpretation rules

Predeclare meter bins: [0,5), [5,10), [10,20), [20,40), [40,80), [80,160),
and [160,infinity). Keep invalid/unreachable/no-active-goal cases separate;
never turn an infinity/sentinel into a valid far-distance observation.
Use the same bins for every compared estimator. Also report sampled distance
coverage, scene counts, trajectory counts, and timestep counts.

Main output is one concise report with five figure groups:

1. **Distance calibration:** mean Q1, Q2, min-Q, and mean G_soft by distance;
   reward-only return shown separately. Include scenario-clustered confidence
   intervals, sample counts, and censoring fractions.
2. **Residuals by distance:** mean signed error, MAE/RMSE to realized returns,
   and twin-critic disagreement. Label RMSE as containing return stochasticity;
   it is not pure critic approximation error without repeated futures.
3. **Value reliability:** bin predictions themselves and compare predicted
   versus empirical soft return, with identity line and distance strata.
   Distance-only averaging can hide action-specific errors.
4. **Behavior and confounders:** progress/waypoint success, speed, goals, safety,
   and reward-coefficient strata. Compare matched strata where coverage permits;
   do not infer causal distance effects from unmatched mixtures.
5. **Tail sensitivity:** complete versus censored samples, observed horizons,
   bootstrap fraction, and return estimates under available horizon choices.

Use cluster bootstrap over independent scene initializations, grouping all
agents, timesteps, and repeated action seeds from a shared initialization.
Agents in a scene interact and neighboring returns overlap. Do not treat
millions of transition rows as millions of independent experiments. Report
both on-policy occupancy-weighted summaries and a scene-balanced sensitivity
summary so unusually long surviving trajectories do not silently dominate.

Decision table for the advisor's hypothesis:

| Observation | Supported interpretation |
|---|---|
| Accurate near goal; systematic residuals far away with adequate uncensored support | Consistent with a distance-related value-learning weakness; motivates testing n-step/PQL/shaping |
| Q and empirical returns both flat far away | May accurately reflect this policy's weak progress; flat Q alone does not prove critic failure |
| Q flat but empirical returns differ substantially within matched strata | Evidence of poor value discrimination/calibration |
| Large realized-return variance but small mean residual | May be environmental/policy uncertainty rather than systematic bias |
| Errors mainly on censored trajectories or changed benchmark settings | Cannot isolate critic accuracy from missing tails or distribution mismatch |

Even a positive diagnostic does not prove which intervention will help.
Keep reward shaping, n-step changes, and PQL as separate experiment factors.

## 5. Implementation boundary and deliverables

Future scripts/configs/data/reports belong here. Reuse existing model creation,
action decoding, transition-mask, and final-observation conventions; do not
modify trained checkpoint files, training config folders, or existing results.
Any required shared-source instrumentation must be described explicitly before
implementation. Do not launch training, fine-tune critics, or resume TES.

Planned artifacts:

| Path | Purpose |
|---|---|
| audit.json | Checkpoint/config hashes, source revision, dimensions, alpha, gamma, environment semantics, and distance definition |
| configs/ | Frozen diagnostic settings and recorded seeds; training-matched and benchmark conditions separated |
| scripts/ | Future checkpoint reader, rollout recorder, return calculator, and plotting entrypoint |
| data/ | Transition records and trajectory/scene summaries; no bulk observations by default |
| REPORT.md and figures/ | Findings, five plot groups, coverage/uncertainty, and explicit unresolved comparisons |

Acceptance checks before collecting the main sample: strict network-state
loading; same saved actor/critics/alpha; valid action indexing; independently
checked reward/entropy-return arithmetic; no initial-action entropy in Q
reference; no cross-reset returns; complete and censored rows separated;
distance attached to the correct waypoint and timestep; no train-mode updates;
cluster-aware uncertainty; original versus soft returns named distinctly.

The current deliverable is PLAN.md only. No calibration conclusion is yet
available from this run.
