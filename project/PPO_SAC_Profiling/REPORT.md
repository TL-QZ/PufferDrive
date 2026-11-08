# Why PPO and SAC have different training times

SAC is slower mainly because it trains on many more replay samples and does more
network work per sample. Replay implementation overhead is real, but removing
much of it did not bring SAC close to PPO: full-size GPU replay improved SAC
throughput by 33%, while it remained about 10 times slower per received step in
the bounded two-GPU comparison. No production learner code has been changed.

## Batch sizes in the supplied runs

All sizes below are **per GPU/rank**, with two ranks in the historical runs.

| Quantity | PPO | SAC |
|---|---:|---:|
| Collected rollout / retained replay capacity | 8,192,000 | 2,048,000 |
| Fresh valid inserts before the next update | Full new rollout, then masks/filter | 1,024,000 after initial fill |
| Logical optimizer minibatch | 128,000 | 128,000 |
| Microbatch per forward/backward | 32,000 | 32,000 |
| Passes through retained training data | 3 | 3 |

PPO's rollout is `20 environments × 3,200 agent slots × 128 steps`. Its
8.192M rows are not all trained: inactive rows are masked, and the advantage
filter removes most remaining rows in these trained policies. Both supplied PPO
experiments use this geometry. SAC refreshes half its replay, but then trains
over **the entire replay three times**, not just the newly inserted half. That
is approximately six sampled exposures per newly inserted valid transition.

Changing replay capacity alone does not change that steady-state six-to-one
ratio while three passes and half-refresh remain fixed. A smaller buffer makes
each cycle shorter but causes more cycles per received-step budget. Increasing
the logical minibatch changes optimizer-step overhead, not the total number of
forward/backward rows in those three passes.

These are agent-transition counters, not counts of whole simulator worlds.
DDP combines the gradients across both ranks; a full logical update therefore
uses 256,000 sampled rows globally, except partial tail batches.

## What the historical records establish

| Quantity | November PPO | October SAC trial 98 |
|---|---:|---:|
| Received transitions, global | 4.014B | 4.000B |
| Logged elapsed time | 4h 15m | 57h 38m |
| Learner sampled exposures, global | 2.333B | 19.845B |
| Exposures / received transition | 0.581 | 4.961 |
| Estimated collection time | 1h 03m | 2h 46m |
| Estimated complete learner time | 1h 40m | 45h 03m |
| Replay/train copy time, nested inside learner | 21 seconds | 10h 39m |
| GPU model | RTX A6000 | RTX A5000 |

The largest difference is **learner workload**, not experience collection.
SAC sampled **8.50×** as many training transitions. Its historical learner time
was **27.08×** larger, equivalent to **3.18×** time per exposure. Hardware differs,
so the per-exposure ratio needs the same-hardware checks below.

The PPO retained fraction is reconstructed from **all 245 cycles**, not just the
final summary. Its cycle mean is 25.89% of valid rows, and its valid-row-weighted
mean is 26.23%. The August directory contains a **10.011B** run, not a 4B run; it
finished in 6h 39m and trained on 5.531B exposures. Its weighted retained fraction
was 24.34%. The complete curves, rounded sample reconstruction, source hashes,
and counter definitions are in [historical_summary.md](historical_summary.md).

PPO's total retained count is common across ranks because the implementation
trims to the distributed minimum. The global valid denominator is unavailable
from the rank-0-only history; the weighted fraction above is rank 0's value.

Historical phase times are the stock profiler's estimates: it samples every
fifth cycle and scales durations by five. Nested copy/forward/backward times
must not be added to the outer learner time. The broader elapsed time also
contains evaluation, startup, logging and unaccounted overhead. SAC ran 808
evaluations versus November PPO's 123 (August PPO: 7); evaluation durations were
not recorded reliably enough to assign the remaining time to evaluation.

## Why SAC costs more per sampled row

The saved policy dictionaries match. PPO trains an actor backbone and a value
backbone: two forwards and two backward paths. This discrete SAC implementation
does next-state actor plus two target-Q forwards, two online-Q forwards and
backwards, then an actor forward/backward plus two newly updated online-Q forwards.
That is eight backbone forwards and three backward paths per sampled row.

Counts of forwards are not elapsed-time multipliers, but they explain why equal
microbatch sizes do not imply equal network cost. The actor's Q evaluations
occur **after** the critic optimizer step. Reusing the pre-update Q output would
change the algorithm. Reusing the input observations is safe.

## Confirmed avoidable work

In [pufferl_sac.py](../../pufferlib/pufferl_sac.py), the actor path calls
`self.replay.get_batch(micro_indices)[0]`. The replay gathers and pins observations,
actions, rewards, next observations and discounts, then four fields are discarded.
It also gathers/transfers current observations again after the critic pass.

The isolated candidates in [sac_variants.py](sac_variants.py) either gather only
actor observations or retain the critic pass's current-observation microbatches
until the actor pass. Caching is bounded to one logical minibatch, approximately
451 MiB for 128,000 × 923 float32 observations. Sampling, target construction,
critic-before-actor optimizer order, gradient accumulation, clipping, entropy
scheduling and target updates are preserved. Production source stays unchanged.

Synthetic replay-only probes found actor gather/H2D time of 56.99 ms for stock
versus 29.03 ms for observation-only at one Torch CPU thread, and 20.41 versus
12.11 ms at four threads. These synthetic component timings are **not** end-to-end
speedups; the real training comparisons determine their practical value.

## Measurement protocol

Real Drive simulation and the full saved network run sequentially on the same
RTX A6000. The controlled matrix retains the production 128k logical/32k microbatch,
three passes, compiled networks and observation width 923. It uses 512k rollout
and replay capacities to keep probes bounded. Two warmup cycles are excluded;
three measured cycles synchronize CUDA at collection/train boundaries. Internal
stock profiling, evaluation, checkpoint writing and remote logging are disabled,
except the explicitly labeled stock-profiler ablation.

The host is shared; CPU/memory contention and the short measurement window limit
precision. Each case retains resolved config, code provenance, raw rank/cycle
counts, memory peaks and timing ranges. Samples count each SAC replay row **once**,
although it contributes to both actor and critic losses. Compilation/startup
are excluded from warmed throughput. Fresh policies retain more PPO advantages
than the historical trained policies, so raw received-rows/s in the small matrix
cannot be extrapolated to a complete 4B run or to policy quality.

## Controlled one-GPU results

Each row below has three warmed measured cycles. All SAC rows sample 1.536M
transitions per cycle. The two independent stock runs agree in their means
(22.63 and 22.68 seconds), while individual cycles span 21.04–23.65 seconds.

| Case | Mean learner seconds | Learner µs / sample | Peak allocated GiB |
|---|---:|---:|---:|
| PPO, advantage filter | 5.52 | 5.04 | 9.62 |
| PPO, filter disabled | 6.47 | 5.05 | 9.65 |
| SAC stock CPU replay, first / repeat | 22.63 / 22.68 | 14.73 / 14.77 | 8.04 / 8.15 |
| SAC, actor observations only | 21.63 | 14.08 | 8.15 |
| SAC, cache current observations | 18.88 | 12.29 | 8.48 |
| SAC, four Torch threads | 19.33 | 12.59 | 8.15 |
| SAC, cache plus four threads | 18.27 | 11.89 | 8.48 |
| SAC, GPU replay | 16.78 | 10.93 | 11.68 |
| SAC, stock internal profiler enabled | 23.82 | 15.51 | 8.15 |

Relative to the repeated stock baseline, caching cuts learner time by about 17%,
four threads by 15%, their combination by 19%, and GPU replay by 26%. GPU replay
raises end-to-end received throughput from 12.1k to 16.2k rows/second, about 34%.
These are short-run measurements with timing variation, not precise promises.
The profiler-on mean is about 5% slower. Its sampled cycle falls once in the
three measured cycles here; historical profiling samples once in five, so 5%
must not be extrapolated directly to historical profiler overhead.

The network cost remains substantial even with GPU replay: SAC takes about
2.17× PPO's learner time per sampled row (10.93 versus 5.04 µs). Stock CPU-replay
SAC takes about 2.93×. This same-hardware result supports the historical finding
that the difference is not primarily simulator collection size.

See [matched_summary.md](results/matched_summary.md) for collection times,
timing ranges, actual exposures per received row and source run paths.

## Production-size, two-GPU confirmation

These runs use 20 environments × 3,200 agents per rank, the original PPO rollout
and SAC replay capacities, and the saved trained model weights. One cold cycle
is excluded and two warmed cycles are measured. Optimizers, alpha, replay,
counters and schedules remain fresh. All three cases have finite parameters
and identical final hashes across ranks. CPU- and GPU-replay SAC also have
identical final hashes **across the two experiments**.

| Case, two A6000 GPUs | Collect s/cycle | Learn s/cycle | Samples / received | Global received rows/s | Peak allocated GiB/GPU |
|---|---:|---:|---:|---:|---:|
| PPO, trained weights | 15.54 | 30.48 | 0.733 | 356,042 | 37.00 |
| SAC, CPU replay | 7.15 | 85.39 | 4.923 | 26,970 | 8.81 |
| SAC, GPU replay | 2.64 | 67.00 | 4.923 | 35,842 | 23.04 |

SAC's stock learner consumes **92%** of collection-plus-learning time. The GPU
replay change cuts learner time **21.5%** and cycle time **24.8%**, increasing
received-step throughput **32.9%**. It also reduces collection/insertion time.
Nevertheless, received-step throughput remains **9.93× below PPO** in this
bounded comparison (stock SAC: 13.20× below PPO).

Normalized by actual global sampled rows, learner cost is 2.54 µs for PPO,
6.95 µs for CPU-replay SAC and 5.45 µs for GPU-replay SAC. All three numbers
describe two GPUs jointly; do not compare them directly to the one-GPU table
without accounting for GPU count. The optimized SAC cost per sample is still
2.15× PPO's, and its exposures per received row are 6.71× higher in these probes.

The GPU-replay memory peak is about **23.04 GiB allocated**, with approximately
26.54 GiB reserved during the run. This is not a memory-fit validation for the
historical A5000 hardware. CPU replay with the bounded observation cache is the
more conservative candidate when GPU capacity is limited. The large-buffer
GPU result validates speed on A6000s, not portability to every 24-GB GPU.

Two measured cycles are a short confirmation, not a long-run speed estimate.
PPO filtering varies with the trained policy and fresh state: this probe's
0.733 exposures/received differs from the full history's 0.581. Raw phase data
and exact commands are in [production_summary.md](results/production_summary.md)
and its linked run directories.

## Operator traces: compute versus replay work

Separate intrusive traces cover one warmed cycle each and are excluded from
throughput. The raw CUDA activity analyzer excludes annotation ranges and
host/device duplicate entries. Activity durations are sums of kernel/copy
durations, not wall-clock intervals; streams may overlap.

| Learner trace evidence, one GPU | PPO | SAC CPU replay |
|---|---:|---:|
| Actual sampled transitions | 1,124,721 | 1,536,000 |
| CUDA kernel activity | 5.610 s | 16.450 s |
| Kernel activity per sampled row | 4.99 µs | 10.71 µs |
| Pinned host-to-GPU transfer activity | none | 0.691 s |
| Pinned host-to-GPU bytes | none | 15.867 GiB |

SAC's trace also records **480 `aten::index_select` calls taking 3.581 seconds
of self CPU time** across the cycle. This matches 48 microbatches across the
three passes × two replay fetches (critic and actor) × five fields. Thus the
replay-copy region includes expensive CPU gathering/allocation, not just PCIe
transfer time. GPU replay eliminates that CPU work as well as the transfers.

The dominant kernels include BF16 matrix multiplication, fused normalization
and pooling work. SAC's kernel activity per sampled row is about **2.15× PPO's**,
consistent with the GPU-replay end-to-end learner comparison. Large
`cudaStreamSynchronize` CPU times mostly represent waiting for GPU work; adding
them to GPU activity would double-count elapsed time. Likewise, the profiler's
GPU annotation ranges must not be added to their contained kernels.

See [trace_summary.md](results/trace_summary.md) for activity counts/bytes and
the per-run `*_operators.json` files for operator tables. Chrome traces remain
local and are ignored by Git; the analyzer records their SHA-256 values.

## Can implementation changes bring SAC close to PPO?

The measured changes improve SAC materially, but do not remove the main
workload difference. Even GPU replay leaves roughly twice PPO's compute cost
per sampled transition, while the historical SAC workload samples 8.5 times
as many transitions. A 17–26% learner-time reduction cannot erase that gap.
This is evidence about the tested implementations, not a proof that every
possible SAC implementation has reached its speed limit.

The practical sequence is:

1. Use GPU replay (`train.cpu_offload=false`) where measured memory headroom
   allows it. This uses the existing production path and avoids CPU replay
   gathers/transfers; check capacity on the actual training GPU model.
2. If CPU replay is required, consider promoting the bounded observation cache
   after review. The isolated candidate has parity evidence and a measured gain;
   it is not yet installed in the production learner.
3. Tune Torch CPU threads against the real shared host. Four helped this probe,
   but gains overlap with caching and extra threads can contend with simulators.
4. If the target is close to PPO's wall time, investigate **less training work**:
   one pass instead of three reduces replay exposures threefold; a full refresh
   instead of half reduces steady-state exposures per new insert twofold.
   These change update/data reuse and potentially learning quality. They need
   learning-curve and evaluation comparisons; the profiling runs do not validate
   them. Changing critic architecture or update frequency has the same caveat.

The comparison at equal received steps answers the runtime question. It does
not establish which algorithm needs fewer environment samples to reach a given
driving score. No short probe here establishes a 4B completion time or long-run
quality for an altered update schedule.

## Reproduction and verification

See [README.md](README.md) for bounded commands, trained-weight initialization,
trace options and analysis. Model-only initialization keeps optimizer, replay,
alpha, schedules and run counters fresh; it is not a resumed training experiment.
The candidate update loops are experimental copies and must be kept synchronized
with production before adoption.

Final validation: **23 tests passed**, with the opt-in CUDA tests enabled,
in 4.68 seconds. This includes pinned CPU replay on CUDA and raw-trace accounting.
The candidate parity checks compare losses, model/target parameters,
optimizer and entropy state, multiple cycles, partial batches and frozen updates.
Independent review found no remaining material blocker in the bounded harness.

The real five-cycle matched-B runs also end with identical combined
policy/critic/target/log-alpha SHA-256 values for stock CPU replay, GPU replay,
and observation-only gathering. The four-thread stock and cached runs match
each other exactly; their hash differs from the one-thread group. Tiny two-GPU
PPO and cached-SAC runs both completed with finite parameters and identical
final hashes across ranks. These checks support update parity for these probes,
not long-run convergence equivalence after changing training hyperparameters.
