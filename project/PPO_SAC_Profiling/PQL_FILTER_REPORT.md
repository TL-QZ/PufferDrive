# PQL caching and advantage filtering

Status: ready for bounded throughput experiments. PQL caching, GPU replay, and
opt-in filtering are implemented in the profiling project. Production learner
files are unchanged. The complete suite, including EWMA coverage, passes
(112 tests in 15.24 seconds), including CUDA and two-process Gloo. Real-Drive
EWMA GPU smoke runs passed; earlier PQL/fraction single-GPU, two-GPU, compiled,
and CPU-preset smokes also completed. Larger-scale speed and policy quality
remain unmeasured.

## What is being compared

The baseline is the existing discrete SAC learner configured with PQL raw
one-step insertion, horizon 128, and lambda 0.95. Other model/environment/reward
settings come from the profiling harness's saved trial-98 source config. This is
a controlled estimator override, not a claim to reproduce every official PQL
experiment setting. Each bounded run records its resolved config and provenance.

The optimized candidate uses GPU replay, fixed cycle-start PQL critic targets,
and fixed cycle-start online twin-min Q vectors for actor training. The PQL
sequence plan is frozen when constructing targets; subsequent optimizer passes
shuffle cached row targets. The actor probabilities, alpha/TES, critic parameters,
and Polyak targets continue to update. These are algorithmic changes, not exact
equivalence to stock PQL's update sequence.

## Filtering choices

Let `q = min(Q1, Q2)` for all actions, `p` be the cycle-start actor probabilities,
and `b = sum(p * q)` be expected Q without the current-state entropy bonus.
All scores are detached and computed once per cycle, after full target caching.

| CLI score | Per-row score |
|---|---|
| `none` | No filtering or scoring pass |
| `q_advantage` | `abs(q[recorded_action] - b)` |
| `target_advantage` | `abs(cached_return_target - b)` |
| `expected_abs_advantage` | `sum(p * abs(q - b))` |

`--sac-filter-mode fraction` keeps the highest-scoring
`ceil(replay_rows * keep_fraction)` rows. Ties are resolved stably. This remains
the backward-compatible default and provides a controlled workload axis for
throughput sweeps. A 100% retention setting measures scoring overhead without
changing the training rows.

`--sac-filter-mode ewma_threshold` mirrors PPO's adaptive selection rule:

```text
current_max = max(raw_scores)
ema_max = current_max                         # first scored cycle
ema_max = beta * current_max + (1-beta)*ema_max # later cycles
threshold = threshold_scale * ema_max
keep = raw_scores >= threshold
```

Defaults match the inspected PPO run: beta **0.25**, threshold scale **0.01**,
no minimum-batch padding and no retention floor. The moving average is over the
cycle maximum, not the mean score or a quantile. It updates once per complete
collection/training cycle, not once per optimization epoch or microbatch. Each
rank keeps its own EMA; ranks synchronize retained counts with MIN and trim
larger survivor sets to their highest-scoring rows. A zero maximum with zero EMA
keeps zero-score rows because the comparison is inclusive. A historical EMA
above the current scores can instead produce zero survivors.

The same retained rows train actor and critic, including entropy and alpha/TES
updates. Zero-survivor cycles perform no optimizer, Polyak, or alpha/TES updates;
collection and future score/EMA refresh continue. There is no importance weighting
or actor-advantage normalization. Both modes retain partial final minibatches.

EWMA state persists on the learner between cycles and is recorded in result
metrics/final state. The profiling harness intentionally does not resume runs or
write training checkpoints; checkpoint persistence is not added to production.

Full replay targets are computed before filtering so a removed training row
still contributes its observed reward to preceding PQL returns. Caches refresh
each collection/training cycle, including replay rows not replaced by collection.

## Timing interpretation

Compare synchronized learner seconds per cycle and total collection+learner
seconds per cycle. Learner timing includes PQL planning, all target/Q precompute,
filter scoring/selection, and optimization. Sample exposures and optimizer steps
show the actual amount of skipped work. Filtering can reduce alpha/TES and target
network update counts because there are fewer logical minibatches.

Raw outputs live under ignored `results/`. Short smoke measurements verify the
pipeline; they do not establish production-scale speedups or model quality.

## Executed GPU smoke measurements (2026-11-09)

### EWMA threshold validation

The default EWMA mode (beta 0.25, scale 0.01) completed a three-cycle real-Drive
GPU smoke with the same small geometry below. The EMA maxima were 2.536565,
2.592140, and 2.810951, following the specified recurrence across warmup and
measured cycles. The two measured cycles retained 2,040 and 1,856 of 2,048 rows
(95.1% combined); learner time averaged 0.612 seconds per cycle. This fresh-policy
smoke does not reproduce PPO's historical retention fraction or establish a
large-scale speedup.

A separate two-cycle run with threshold scale 2 deliberately retained zero rows.
Each cycle still cached all 2,048 target/Q rows, performed zero actor/critic/alpha
optimizer steps, and finished with finite parameters. Both analyzers produced
valid JSON and Markdown, reporting undefined per-sample costs as null / dash.
Recovery after an empty cycle and unequal local thresholds are covered by the
controlled two-rank Gloo test.

Ignored outputs: `results/pql_ewma_gpu_smoke_20261109/` and
`results/pql_ewma_empty_gpu_smoke_20261109/`.

### Earlier fixed-fraction comparison

One RTX A6000, GPU replay, 2 environments x 32 agent slots, 2,048 replay rows,
1,024 logical batch, 256 microbatch, three epochs, PQL horizon 128/lambda 0.95,
uncompiled networks, one warmup and two measured cycles. All five cases completed
with finite model parameters. These deliberately tiny runs validate measurement
and workload accounting; use the larger command below for performance experiments.

| Case | Retained rows/cycle | Critic exposures/cycle | Actor/critic/alpha steps each | Learner seconds/cycle | Collection seconds/cycle |
|---|---:|---:|---:|---:|---:|
| Stock PQL, GPU replay | 2,048 | 6,144 | 6 | 0.782 | 0.196 |
| Fixed PQL targets + fixed actor Q | 2,048 | 6,144 | 6 | 0.598 | 0.187 |
| Same + target-advantage scoring, keep 100% | 2,048 | 6,144 | 6 | 0.611 | 0.189 |
| Same + target-advantage filtering, keep 50% | 1,024 | 3,072 | 3 | 0.365 | 0.193 |
| Same + target-advantage filtering, keep 25% | 512 | 1,536 | 3 | 0.251 | 0.189 |

Each cached case still precomputes all 2,048 critic targets and all 2,048 actor Q
vectors per cycle. Actor exposures equal the critic exposures shown. At 25%, the
logical batch is smaller rather than removing another optimizer step; that is why
the 50% and 25% rows both have three steps per optimizer.

The 100%-retention scoring control and unfiltered cached PQL produced identical
final model hashes. A separate two-GPU smoke (GPUs 4 and 5, two cycles) retained
1,024 rows per rank and produced identical final hashes across ranks; global
critic exposures were 6,144 per cycle. A separate compiled single-GPU smoke also
completed two cycles with finite parameters. Compilation startup is excluded
from warmed timings and is not represented by the uncompiled table above.

Raw artifacts: `results/pql_filter_matrix_smoke_20261109/`,
`results/pql_filter_ddp_smoke_20261109/`, and
`results/pql_filter_compiled_smoke_20261109/`. The first single-run smoke at
`results/pql_filter_gpu_smoke_20261109/` also exercised the same-segment PQL
target-staleness audit after training. That audit is not a claim that cycle-start
and end-of-cycle targets should be equal.

## Correctness and readiness checks

- Independent scalar PQL recursion and production-preparer parity on the exact
  saved segments, including lambda endpoints, terminal/truncation/gap cuts,
  FIFO overwrite, shuffled physical indices, and logical/microbatch tails.
- Full target/Q cache refresh for old and replaced replay rows; detached cached
  values, current actor gradients and alpha, score arithmetic, stable ties,
  invalid/nonfinite rejection, and actual retained-row/update accounting.
- 100%-retention parity in semantic tests and matching final hashes in real
  Drive; Gloo tests and real two-GPU checks verify equal update counts and synced
  models for the equal-capacity replay used by the harness.
- Pinned CPU/GPU replay oracle tests; CLI/preset validation; same-segment PQL
  staleness audit; old result-schema and PPO analysis compatibility; independent
  review with no remaining material findings.
- Complete profiling suite: **81 passed in 10.50 seconds**. Results remain ignored
  and GPUs were released after validation. No official training run was launched.

```bash
source .venv/bin/activate
SAC_PROFILE_CUDA_TEST=1 CUDA_VISIBLE_DEVICES=4 GLOO_SOCKET_IFNAME=lo \
  python -m pytest project/PPO_SAC_Profiling -q
```

The Gloo test requires permission to open local sockets. Under the managed
sandbox it failed at socket initialization; the complete suite passed outside
that restriction. CUDA and Gloo were actually executed, not skipped, in the
81-test validation. The CPU preset ran one complete real-Drive cycle separately.

## Run a matched GPU comparison

Activate the environment, select an idle GPU, and use a new output directory.
This matrix compares stock PQL, caching alone, scoring overhead at 100% retention,
and combined caching/filtering at 50% and 25% retention. It uses the same source
config, replay placement, simulator geometry, batch sizes, and epochs throughout.

```bash
source .venv/bin/activate
CUDA_VISIBLE_DEVICES=4 python project/PPO_SAC_Profiling/run_matrix.py \
  --output-dir project/PPO_SAC_Profiling/results/my_pql_comparison \
  --cases pql_stock_gpu pql_optimized_gpu pql_optimized_scored_all_gpu \
          pql_target_advantage_50_gpu pql_target_advantage_25_gpu
python project/PPO_SAC_Profiling/analyze_runs.py \
  project/PPO_SAC_Profiling/results/my_pql_comparison \
  --output project/PPO_SAC_Profiling/results/my_pql_comparison/comparison
```

Default comparison size: 4 environments x 1,000 agent slots, 512,000 replay rows,
128,000 logical batch, 32,000 microbatch, three epochs, two warmup cycles and three
measured cycles, compiled networks. Use `--sac-weights /path/to/trainer_state.pt`
to initialize every case from the same trained model weights; replay, optimizers,
alpha and training counters remain fresh. Short fresh-policy runs can have a
different score distribution than late-training policies.

For a tiny pipeline smoke, add these matrix options (not a performance estimate):

```bash
--num-envs 2 --agents 32 --replay-capacity 2048 \
--logical-batch 1024 --microbatch 256 --compile off --warmup 1 --cycles 2
```

For other scores or retention fractions, invoke `profile_training.py` directly
with `--return-estimator pql --variant fixed_targets_actor_q --replay-location gpu`
and `--sac-filter-score SCORE --sac-keep-fraction FRACTION`. The PQL GPU and CPU
smoke presets are in `configs/pql_gpu.yaml` and `configs/pql_cpu_smoke.yaml`.

For PPO-style moving-threshold filtering, use the dedicated preset:

```bash
source .venv/bin/activate
CUDA_VISIBLE_DEVICES=4 python project/PPO_SAC_Profiling/profile_training.py \
  --profile-config project/PPO_SAC_Profiling/configs/pql_ewma_gpu.yaml \
  --output-dir project/PPO_SAC_Profiling/results/my_pql_ewma
```

The equivalent selection flags are `--sac-filter-score target_advantage
--sac-filter-mode ewma_threshold --sac-filter-ewma-beta 0.25
--sac-filter-threshold-scale 0.01`. Do not combine EWMA with a nondefault
`--sac-keep-fraction`: its retained count is determined by the threshold.
Add `pql_target_advantage_ewma_gpu` to the matrix cases above to compare it with
stock PQL, caching alone, and fixed-fraction filtering under the same settings.
Per-rank cycle output records the current maximum, EWMA maximum, threshold, and
survivor counts. Zero-exposure runs report undefined learner time per sample as
null in JSON and a dash in the summary table; their measured cycle times remain
available.
