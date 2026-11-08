# GPU replay with fixed critic targets and actor Q scores

Status: complete. With GPU replay, fixing critic targets and actor Q scores cuts
SAC learner time by about **20%**. It remains **1.70x PPO's learner cost per
trained sample** in the matched comparison. All 40 tests and all timed cases
passed. Production learners are unchanged.

## Exact experiment

At each train call, after replay replacement, compute two caches over **every**
replay row: the usual discrete SAC bootstrap target at the cycle-start actor,
target critics and alpha; and the action-score vector
`min(Q1_online(observation), Q2_online(observation))` from the cycle-start online
critics. Reuse both detached float32 caches through three shuffled epochs.
Refresh both caches on the next train call, including retained replay rows.

Actor probabilities, entropy and gradients remain current. Its objective is
`sum_a pi_current(a|s) * (alpha_current * log(pi_current(a|s)) - cached_Q(s,a))`.
Alpha/TES, online critics, actor and Polyak target updates keep their existing
order and cadence. Q supervision is fixed; the entropy coefficient is not.
This is a changed SAC learning procedure, not PPO or an exact stock-SAC optimization.

## Validation and timing protocol

Independent tests must verify actor loss/gradients against an oracle, detached
caches, shuffled row alignment and tails, cache refresh after partial replay
replacement, unchanged optimizer/Polyak activity, and no online critic forwards
inside the actor loop. Run CUDA tests before timing, not concurrently.

Main comparison: trained model weights, one A6000, 4 environments x 1,000 agent
slots, 512k-row rollout/replay, GPU storage for every case, 128k logical batches,
32k microbatches, three passes, compiled networks. Run PPO with its normal filter,
fresh-target SAC, critic-target-only caching, and both caches sequentially.
Exclude two warmup cycles and measure three cycles. Include both precompute
passes inside synchronized learner timing. Alpha/optimizer/replay state is fresh.

Report learner time per sampled transition as well as received throughput:
PPO filters samples, so raw cycle times alone do not isolate learner efficiency.
Any full-replay confirmation is labeled separately and uses the smaller simulator
geometry because earlier full-environment startup attempts did not complete.
Short probes do not establish convergence, driving quality or 4B wall time.

## Correctness evidence

All **40 tests passed** with `SAC_PROFILE_CUDA_TEST=1 CUDA_VISIBLE_DEVICES=4`,
including the new GPU-resident replay test. The actor loss and gradients match
the current-policy/current-alpha cached-Q oracle. Actual critic-forward hooks
in tests verify one replay-wide actor-Q precompute plus ordinary critic training,
with no critic evaluation inside the actor loop. Partial replay replacement
tests check refreshed targets and Q scores for both replaced and retained rows.

The initial CUDA test exposed a bitwise-comparison issue between different
batch shapes (one float32 difference of 5.96e-8). Matching the oracle's contiguous
microbatch/tail shapes resolved it without changing implementation or loosening
the exact cache comparison. Independent review found no material code issue.

The benchmark's actor-row counter explicitly includes the new cached-Q loss
path. Neither cached supervision nor avoiding critic inference removes actor
gradient updates from the workload accounting.

## Matched GPU-storage results

All four cases use one A6000, GPU data storage and the main protocol above.
The fresh-target SAC control gathers only observations for its actor phase,
matching the candidate's input path; it avoids charging the control for the
known redundant all-field actor gather. Each SAC case trains 1,536,000 sampled
transitions per measured cycle. PPO's ordinary filtering retains fewer rows.

| Case | Learn s/cycle | Learn s per 1M sampled transitions | Received rows/s |
|---|---:|---:|---:|
| PPO, normal advantage filter | 1.287 | 5.079 | 118,900 |
| SAC, fresh targets and actor Q | 16.664 | 10.849 | 17,265 |
| SAC, fixed critic targets only | 14.610 | 9.511 | 19,514 |
| SAC, fixed critic targets and actor Q | 13.290 | 8.652 | 20,519 |

The proposed variant reduces learner time **20.2%** versus the matched fresh
control, and **9.0%** beyond critic-target-only caching. Per sampled transition,
SAC falls from **2.14x** PPO's learner cost to **1.70x**. Received-row throughput
remains **5.79x below PPO** in this probe because the algorithms train on different
numbers of samples per received row. These short-run results are not 4B-runtime
or learning-quality estimates.

All final parameters are finite. Runtime counters show 512,000 critic-target
precompute rows and 512,000 actor-Q precompute rows per measured candidate cycle;
both actor and critic still process 1,536,000 training rows and make 12 optimizer
steps each, with 12 alpha steps. The actor-Q cache is 24,576,000 bytes (23.44 MiB).
The fresh and critic-only controls reproduced their earlier final model hashes.

Raw measurements, ranges, memory peaks and cycle paths are in
[fixed_both_gpu_summary.json](results/fixed_both_gpu_summary.json) and
[fixed_both_gpu_summary.md](results/fixed_both_gpu_summary.md). Exact commands and
source hashes are preserved in each controls/candidate matrix directory.

## Remaining work per sample

Across three replay passes, fresh SAC evaluates eight backbones per exposure:
24 forwards and nine network backward paths per replay row. Fixing critic targets
reduces forwards to 18 per row across the cycle. Fixing actor Q as well reduces
them to 14: five cycle-start precompute forwards (next actor, two target critics,
two online critics) plus three training forwards per pass (actor and two critics).
All nine network backward paths remain. PPO still trains only its actor and
single value network, and reuses its rollout supervision. These counts explain
remaining work but are not a FLOP or timing decomposition.

## Full-size GPU replay confirmation

One A6000, **2,048,000 replay rows**, GPU replay, 128k logical / 32k microbatches,
three passes and trained model weights. Both cases retain the smaller
4-environment x 1,000-agent simulator geometry; this is full SAC replay size,
not full production simulator geometry. One warmup cycle is excluded and two
warmed cycles are measured. Both cases use the same observation-only actor
input gathering. Both commands exited 0.

| Full replay case | Learn s/cycle | Collect s/cycle | Learn s per 1M samples | Peak allocated GiB |
|---|---:|---:|---:|---:|
| Fresh critic targets and actor Q | 66.623 | 9.056 | 10.844 | 22.27 |
| Fixed critic targets and actor Q | 53.199 | 8.400 | 8.659 | 22.15 |

Learner time drops **20.1%**, consistent with the smaller matched experiment.
The candidate's actor-Q cache is **93.75 MiB**. Both actor and critic train on
6,144,000 sampled rows and make 48 optimizer steps per cycle, with 48 alpha
steps. Both caches are rebuilt over all 2,048,000 rows (64 microbatches each).
All final parameters are finite. This establishes fit on the tested A6000,
not the historical A5000.

Full evidence: [fixed_both_full_gpu_summary.md](results/fixed_both_full_gpu_summary.md),
its JSON, and [exact commands](results/fixed_both_full_gpu/matrix.json).
Raw cycles preserve precompute counts, train counts, optimizer counts, memory,
configs and source provenance. The post-timing critic-target staleness audit
remains diagnostic; it does not establish stock-SAC equivalence or policy quality.

## Completion checks

All six timed cases exited successfully, all expected caches and update counts
were verified, all parameters were finite, and no matrix remained active.
No production training, evaluation sweep, remote logging or model checkpoint
write was performed. The experiment lives only in this profiling folder.
