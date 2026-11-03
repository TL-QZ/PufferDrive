# Round2 TES-SAC trial 98: Q calibration by waypoint distance

**The “accurate near the waypoint, inaccurate far away” hypothesis remains
unresolved.** Q exceeds realized soft return on the terminal-selected subset,
but **99.31% of sampled returns are censored**, and complete returns near
waypoints have sparse independent support. Sampled movement is slow. These
observations justify further controlled investigation; they do not establish
population-wide Q overestimation or prove that n-step, PQL, or shaping will help.

Status: completed frozen-checkpoint diagnostic, 2026-11-02 local time. No
training, checkpoint changes, or shared simulator-source edits were made.

## 1. What was measured

| Item | Actual experiment |
|---|---|
| Checkpoint | Final `round2_tes_sac_20261029T044847.014369240_539543/trial_98/learner/trainer_state.pt` |
| Training progress | Epoch 1615; 4,000,032,000 agent rows |
| Matching state | Actor, online twin critics, target twin critics, and alpha from that same full checkpoint |
| Policy | Frozen categorical sampling, including subsequent actions; all policy-controlled agents use it |
| Temperature / discount | alpha = 0.0005304708029143512; gamma = 0.999 |
| Environment | Saved training configuration: Carla maps, dt = 0.3 s, 3 finite goals, randomized reward conditioning, 3200 slots per process |
| Workload | Two independent seeds, physical GPUs 6 and 7, 2560 ticks each; 104 native scene initializations |
| Data | 15,192,262 valid transitions; 1,899,135 scalar records, sampled every 8 ticks **after full-resolution return calculation** |
| Runtime | 368.6 and 370.9 seconds including return calculation and output |
| Endpoint counts | 233 valid terminal transitions and 5894 valid nonterminal truncation transitions across both full-resolution rollouts |

The two runs use environment/action seeds 61011/71011 and 61012/71012.
The 64-tick pilot is excluded. This is a training-matched diagnostic, not a
rerun of the mean-action CARLA benchmark; the original final benchmark score
about 0.4627 is not directly comparable to these sampled-action trajectories.

The primary target includes future entropy, matching the saved soft critic:

\[
G_t^{\mathrm{soft}}=r_t+\gamma\left(G_{t+1}^{\mathrm{soft}}
-\alpha\log\pi(a_{t+1}\mid s_{t+1})\right).
\]

At a true terminal, the return is the final reward. The starting action's
entropy is excluded. Timeouts use the true pre-reset final observation for a
separately labeled target-critic bootstrap; they are never treated as zero-tail
Monte Carlo truth. A complete row means its observed suffix reaches a terminal,
not that the row itself must be terminal. No return bridges a reset or invalid
transition gap.

## 2. Q versus complete sampled returns

There are **13,055 complete rows from 79 scene/episode clusters**. On this
subset, mean min-Q is **1.087**, mean soft return is **−0.877**, and their
mean residual is **+1.964**. Reward-only return averages −1.021; future entropy
contributes +0.143. Omitting entropy would exaggerate the discrepancy.

The following Euclidean-distance table uses identical complete rows for Q and
return. Intervals are 95% percentile intervals from 250 scene-cluster bootstrap
replicates, not independent-row intervals.

| Distance (m) | Complete rows / clusters | Mean min-Q | Mean soft return | Mean residual [95% CI] |
|---|---:|---:|---:|---:|
| [0,5) | 1 / 1 | 1.336 | −1.286 | +2.622 [not estimable] |
| [5,10) | 9 / 3 | 1.007 | −1.662 | +2.669 [2.539, 2.751] |
| [10,20) | 67 / 10 | 0.180 | −2.455 | +2.636 [2.226, 2.726] |
| [20,40) | 464 / 18 | 1.132 | −0.766 | +1.898 [1.520, 2.682] |
| [40,80) | 920 / 36 | 0.644 | −0.963 | +1.606 [1.111, 2.193] |
| [80,160) | 3548 / 51 | 0.969 | −0.813 | +1.782 [1.567, 2.045] |
| [160,infinity) | 8046 / 68 | 1.195 | −0.888 | +2.083 [1.963, 2.208] |

**Bins below 40 m fail the predeclared 30-cluster coverage target.** A narrow
interval from only three clusters does not establish reliable near-goal
calibration. Nor is the residual a clean monotonic distance effect.

For the road-network proxy, only the ≥160 m complete bin reaches 30 clusters:
8907 rows / 73 clusters; min-Q 0.961 versus soft return −0.947; residual
+1.909 [1.761, 2.071]. Giving each scene/episode equal weight yields +1.875,
so occupancy weighting does not explain that subset's positive residual.

These are **terminal-selected** outcomes. A perfectly calibrated expectation
can exceed returns conditional on eventual failure. Consequently these numbers
must not be described as an unbiased estimate of population critic bias.
MAE/RMSE also contain future policy/environment randomness; we did not replay
multiple futures from the exact same state/action pair.

![Distance calibration](analysis/main/distance_calibration.png)

## 3. What the censored majority shows

All **1,886,080 censored sampled suffixes end at scenario truncation** in these
two runs. No missing-validity endpoint contributed to this main result.
The observed horizon varies with the sampled starting tick: median 1280 ticks.
At gamma = .999, the retained discount after 128, 1280, and 2560 steps is
approximately .880, .278, and .077 respectively. The mean retained discount
across censored rows is .359, so unseen continuation matters.

| Quantity, censored rows only | Mean |
|---|---:|
| min-Q | 1.461 |
| Observed partial soft return | −0.200 |
| Discounted target-critic tail | +0.508 |
| Partial return plus that tail | 0.308 |
| min-Q minus bootstrapped return | +1.153 |

This is a substantial **multi-step consistency discrepancy using a
critic-dependent tail**, not independent evidence of true-return bias. Initial
tick rows, which have up to all 2560 ticks available, average min-Q 0.997 and
bootstrapped soft return −0.244. They are a different state distribution from
late-rollout rows; the horizon plot compares suffix groups, not paired
re-truncations of identical starting states.

![Tail sensitivity](analysis/main/tail_sensitivity.png)

## 4. Behavior, distance, and confounders

Across all sampled valid rows, median speed is **0.174 m/s** and **88.29%**
have speed below 1 m/s. This establishes slow movement in these rollouts.
There are 67 next-transition waypoint-hit events among the every-eight-tick
records; this is an incomplete event sample, **not** a total-goals count or
episode success rate. We did not export collision/offroad event causes or
time-to-next-waypoint, so those outcomes cannot be inferred from this count.

The raw lane-graph distance is a proxy: the simulator does not retain the
random lane-walk route that generated the goals. It omits within-lane progress;
zero graph distance can mean the same lane, not being at the waypoint.
1,575,353 rows have reachable graph distances. Another 319,914 have
unreachable/invalid graph entries, 3840 have no active goal, and 28 lack a
current/goal lane. Those are separated from valid far-distance bins.

Reward conditioning matters: collision/offroad coefficients span roughly
0–3, goal radius 2–12 m, and goal speed roughly 0–20 m/s. Goal reward itself is
fixed at 1. The [confounder table](analysis/main/confounder_summary.csv)
stratifies speed and available reward coefficients, but is not a jointly
matched causal analysis. Many terminal-selected strata are sparse.

Native waypoint state disagrees with the policy-observed goal distance on
3860 rows (0.203%). This existing goal-regeneration ordering is also present
in the checkpoint-era source. Only one complete row is flagged; excluding it
changes the complete-subset residual from +1.964302 to +1.964315. It does not
explain the observed complete-subset discrepancy.

Twin critics agree closely: their mean absolute difference over all sampled
rows is 0.0104. Agreement between critics is not evidence that either is
calibrated.

## 5. Interpretation and reproducibility

The advisor's proposed diagnostic is useful, but this first execution **does
not confirm the near-accurate/far-inaccurate pattern**. It identifies slow
sampled movement, a positive residual on terminal-selected trajectories, and
a large discrepancy from a long multi-step target with a frozen critic tail.
The near-goal ground-truth comparison remains under-supported, and most
continuations remain censored under the training task's timeout semantics.

For the next diagnostic, prioritize matched near/far state coverage and
repeated continuations from verified reproducible simulator states. The native
API currently lacks full state/RNG restore; replaying initializations and joint
action prefixes would need its own determinism check. Merely collecting more
uniform ticks will mostly add slow, censored rows. Extending scenario length
would change the saved task and must be identified as a separate experiment.
PQL, fixed n-step, and shaping should remain separate controlled interventions;
this analysis alone cannot select the remedy.

All changes and artifacts are inside this analysis folder. The private
diagnostic extension passed bitwise production/private CPU parity, padded-row
ordering, and timeout/final-observation checks; see [native verification](native/README.md).
All **13 local tests pass**, both GPU processes exited successfully, and
before/after parameter hashes prove the actor and critics remained frozen.
Shared simulator and SAC model sources have no Git differences from the
checkpoint revision for the relevant files.

The original pilot plan was adjusted to a 64-tick instrumentation check,
followed by two full-size independent runs. No repeated-state branching,
separate benchmark evaluation, complete safety-event analysis, or paired
horizon-ablation experiment is claimed.

Use [README.md](README.md) for commands. Audits are in
[GPU 6 run.json](runs/main_gpu6/run.json) and [GPU 7 run.json](runs/main_gpu7/run.json);
[map_inventory.json](map_inventory.json) maps recorded map IDs to filenames and
content hashes. Main results are in [analysis.json](analysis/main/analysis.json),
[descriptive_summary.json](analysis/main/descriptive_summary.json), and the
linked figures/tables. Additional figure groups are
[residuals](analysis/main/residuals_by_distance.png),
[prediction reliability](analysis/main/value_reliability.png), and
[behavior/confounders](analysis/main/behavior_confounders.png).

Next action: open the distance-calibration figure and compare its complete
cluster counts before interpreting the near-waypoint curve.
