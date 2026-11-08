# Fixed-cycle SAC targets experiment

## Question

Does computing SAC Bellman targets once per training cycle and reusing them
across three replay passes approach PPO throughput while preserving SAC computation?

Correctness has two separate meanings here: the discrete SAC target formula
evaluated at cycle-start parameters, and equality to the existing learner's
fresh targets after actor/alpha/target-network updates. The first can be tested
and preserved; the second is not guaranteed and must be tested explicitly.

## Acceptance evidence

1. Independent oracle tests of soft action expectation, twin-Q minimum, entropy,
   discount/terminal handling, detached targets, shuffled indexing, tails and
   rebuilding the cache each cycle. Original candidates retain their parity tests.
2. Demonstrate and quantify deviation from fresh targets after updates. Preserve
   actor, critic, entropy and Polyak update counts/order; do not silently freeze
   them to obtain artificial parity. Compare cached/uncached fixed-target variants.
3. Run bounded real Drive measurements against fresh-target SAC and PPO on the
   same hardware. Count target precomputation inside total learner time, exclude
   warmup, record actual exposures, and audit final target drift outside timing.
4. Confirm useful variants with distributed/production-size probes, review the
   implementation and report measured benefits and correctness limits. No claim
   of learning-quality preservation follows from short profiling runs.

Production learner files remain unchanged. These are opt-in profiling variants,
not a new default. No long production job is part of this experiment.

## Completion evidence

1. All 37 profiling tests passed, including opted-in CUDA checks and independent
   Bellman, reuse, update-order/parity and audit batch-context tests.
2. Explicit divergence tests and a full-replay post-timing audit establish
   cycle-start versus cycle-end staleness; fresh-target stock equivalence is not
   claimed. Full-replay per-rank optimizer counts remain 48/48/48.
3. Six one-GPU cases completed with trained weights and three measured cycles.
   See `results/fixed_matched_summary.md` for timing and bootstrap-row counts.
4. Two-rank smoke and full-replay control/candidate runs passed with finite,
   rank-matching final states. Full replay uses smaller simulator geometry;
   full-environment startup attempts were interrupted without measurements.
   Independent review found no material code issue. See `FIXED_TARGET_REPORT.md`.
