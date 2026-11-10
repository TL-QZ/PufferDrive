# PQL cache and filtering throughput experiment

Goal: make bounded SAC throughput experiments support PQL, fixed critic targets,
fixed actor Q scores, GPU replay, and optional sample filtering. Optimize the
development/sweep cycle first; long-run policy-quality evaluation is a later task.
All implementation stays in this profiling project; production trainers stay unchanged.

## Implementation contract

1. Build PQL sequence plans over the full resident replay at cycle start. Compute
   targets using the existing ordered-segment PQL return path, scatter them back
   to physical replay indices, and keep them fixed for all training epochs.
   Preserve terminal, truncation, overwrite, gap, horizon, and logical-batch cuts.
2. Cache cycle-start online twin-min Q scores for all actions and replay rows.
   Continue updating actor probabilities, critics, alpha/TES, and target networks.
   Refresh caches for replaced and retained rows every cycle.
3. Optionally rank replay rows by one of three detached cycle-start scores:
   recorded-action absolute Q advantage; absolute PQL-target minus expected Q;
   or policy-weighted absolute action advantage. The baseline is expected min-Q
   under the cycle-start actor, excluding the current-state entropy bonus.
4. Select rows with either a configured top fraction or a PPO-style EWMA maximum
   threshold (beta 0.25, scale 0.01). Train both actor and
   critic on them. Alpha/TES sees the retained states too. No importance correction
   is applied. Filtering happens AFTER full sequence targets are computed, never
   by dropping transitions from PQL recursion. This is an algorithmic experiment.
   EWMA state persists across cycles, updates once per cycle, and remains local
   to each rank. Synchronize survivor counts with MIN and trim by score. No padding
   or retention floor: zero survivors skip all optimizer and target-network updates.
5. Measure synchronized collection and full learner wall time, including cache
   construction and selection, plus candidate/retained rows, actual training
   exposures, optimizer steps, target forwards, and GPU memory. Keep source config,
   estimator, lambda, horizon, score, mode, fraction, beta, and scale in provenance.

## Verification gates

- Independent PQL target oracle and production target parity on identical segment
  plans; terminal/truncation/gap/overwrite and microbatch-tail coverage.
- Physical row alignment through shuffled training, full cache refresh across
  cycles, no actor critic forwards beyond precompute, and live alpha updates.
- Filter score arithmetic, deterministic ties, fractional retained counts, 100%
  equivalence, nonfinite/invalid rejection, real reduction in optimizer work,
  and equal distributed backward counts. EWMA tests cover first-cycle initialization,
  cross-cycle decay, inclusive threshold comparison, zero-survivor recovery,
  unequal rank survivor counts, and zero-exposure analysis.
- Existing n-step profiling regressions, CPU/CUDA tests, bounded real-Drive GPU
  stock/optimized/filtered smoke runs, and independent code review.
- Ready-to-run matched matrix and analysis command; raw outputs remain ignored.

## Comparison interpretation

Stock PQL recomputes targets and sequence plans while training. Fixed PQL freezes
both target values and the target construction plan for a cycle. These are not
mathematically identical update sequences. Filtering further changes training
data and optimizer/target/temperature update counts. Smoke timings demonstrate
that the machinery runs; they are not production-scale speed estimates.
