# Fixed-cycle SAC target experiment

Status: complete. Fixed-cycle targets reduced learner time by 12–21% relative
to matching fresh-target controls, but did not reach PPO throughput or preserve
the stock fresh-target update sequence. All 37 tests passed, and the full-replay
two-GPU control/candidate runs completed. Production SAC files remain unchanged.

## What was changed

The opt-in `fixed_targets` and `fixed_targets_cached` variants compute the
discrete SAC Bellman target once per replay storage row at the beginning of
each train call:

`y = reward + discount * sum_a pi(a|next_obs) * (min(Q1_target, Q2_target) - alpha * log pi(a|next_obs))`

Actor probabilities, target-Q outputs and alpha all use their cycle-start
values. The entropy term is omitted when `critic_entropy_in_target` is false.
The stored discount already includes terminal/n-step handling. Targets are
detached float32 scalars on the learner GPU, indexed by the original replay
storage row through all three shuffled training passes. Every new train call
rebuilds them, including rows whose old replay entries were replaced.

Precomputation uses bounded contiguous replay slices; replay cannot change
during a train call. Random minibatches gather current observations/actions
only, avoiding repeatedly fetching next observations. The cached variant also
retains each logical minibatch's current observations through its actor update.
Actor, twin critics, alpha/TES, clipping and Polyak updates continue in their
original order. Sampling and the number of gradient updates are unchanged.

Source: [sac_variants.py](sac_variants.py). These candidates support the existing
`n_step` replay path, not PQL sequence targets.

## What correctness means here

The cache preserves the **SAC target equation evaluated at cycle-start
parameters**. It does **not** preserve the existing learner's fresh-target
update sequence: later minibatches would otherwise see updated actor, alpha
and target-critic parameters. Targets can already differ within the first pass,
not only in the second and third passes.

Independent tests verify an explicit scalar softmax/min-Q/entropy Bellman
oracle, terminal rows, both entropy modes, an external target policy, detached
targets, shuffled row indexing, partial batches, cache rebuilding, continued
optimizer/Polyak updates and cached-versus-uncached fixed-variant parity. They
also demonstrate divergence from freshly recomputed targets after updates and
equivalence in the special case of one logical update with frozen bootstrap
inputs. Passing the first set does not establish general stock-SAC equivalence.

Validation: **37 tests passed** across the profiling folder: 34 with CUDA checks
enabled on GPU 4, followed by the three CPU analysis tests. A batch-context
regression covers contiguous audit chunks including a partial tail. Independent
review found no material implementation issue. These tests include deliberate
non-equivalence after parameter updates; they do not assert stock parity for
fixed targets.

A two-rank CUDA smoke run also completed: both ranks had finite parameters and
identical final actor/critic/target/alpha hashes. Its post-timing audit matched
the original contiguous batch contexts. Evidence is in
[`fixed_ddp_smoke/summary.json`](results/fixed_ddp_smoke/summary.json).

## Measurement protocol

Real Drive simulation and the saved trained model weights are used for every
algorithm. Optimizers, replay, alpha and schedules remain fresh. All cases use
128,000-row logical batches, 32,000-row microbatches, three passes, compiled
networks and sequential execution on the same A6000 GPU(s). Warmup is excluded.

The entire target-precomputation pass is inside synchronized learner timing.
Counters record bootstrap rows and learner sampled rows separately. Fresh-target
SAC should compute one bootstrap per sampled row; fixed-target SAC should
compute one per replay row, giving a bootstrap/sample ratio of 1/3 at three
passes. PPO's zero in this counter means it does not invoke SAC's bootstrap
networks; PPO's advantage/return computation is still timed.

After all timed cycles, a deterministic evenly spaced selection of 512 replay
rows is audited against targets recomputed with the final actor, target critics
and alpha. That audit quantifies staleness and is excluded from performance.
It is a sample, not a bound on maximum drift over the entire replay.
The full-size confirmation recomputes each audited row in its original
contiguous 32k-row (or tail) batch to preserve BF16 batch context. The earlier
one-GPU matrix used a 512-row audit batch, so its drift values may additionally
include batch-shape numerical differences; use the full-size audit for staleness.

Controls compare cached fresh-target SAC against cached fixed-target SAC on CPU
replay, and observation-only fresh-target SAC against fixed-target SAC on GPU
replay. This separates the earlier actor-gather improvement from target reuse.
Fixed-target cases also avoid repeated next-observation gathering; that is part
of the measured optimization, so speed gains cannot be assigned solely to fewer
network forwards. Stock SAC and PPO are included for end-to-end context.

## One-GPU measurements

Trained weights, 512,000-row replay/rollout, two warmup cycles and three measured
cycles on the same A6000. All SAC cases train 1,536,000 samples per measured
cycle; PPO's mask/filter leaves a smaller subset. Precomputation is included.

| Case | Learner seconds/cycle | Learner microseconds/sample | Received rows/second |
|---|---:|---:|---:|
| PPO with filtering | 1.274 | 5.031 | 127,849 |
| Stock SAC, CPU replay | 22.843 | 14.872 | 12,529 |
| SAC, cached observations | 19.004 | 12.373 | 15,119 |
| SAC, cached observations + fixed targets | 15.729 | 10.240 | 17,945 |
| SAC, GPU replay + observation-only actor gather | 16.670 | 10.853 | 17,390 |
| SAC, GPU replay + fixed targets | 14.676 | 9.555 | 19,257 |

Fixed targets reduce learner time **17.2%** beyond observation caching on CPU
replay and **12.0%** beyond observation-only gathering on GPU replay. These are
short sequential probes on a shared host, not precision estimates. The best
case still costs **1.90x PPO per trained sample**, and received-row throughput
is **6.64x below PPO** in this matrix because workload per received row differs.

Counters verify 48 bootstrap calls / 1,536,000 rows per fresh-target cycle
versus 16 calls / 512,000 rows for fixed targets. Every SAC case still performs
12 actor, 12 critic and 12 alpha optimizer steps per cycle. Final parameters
are finite. Stock, cached and GPU observation-only controls have identical
final model/alpha hashes; the two fixed-target cases match each other and
differ from the fresh-target controls, as expected.

Raw evidence: [fixed_matched_summary.md](results/fixed_matched_summary.md),
the corresponding JSON and per-case cycle files. Exact commands and source
hashes are in [fixed_matched/matrix.json](results/fixed_matched/matrix.json).

## Why PPO's CUDA advantage kernel is not the main opportunity

PPO already evaluates its value network during rollout and performs actor/value
forward/backward work during learning. Its custom
[`puff_advantage_kernel`](../../pufferlib/extensions/cuda/pufferlib.cu)
then applies the backward temporal recurrence to stored values and rewards.
In the earlier warmed operator trace, that kernel consumed **0.144 ms**, versus
**5.610 seconds** of total PPO learner CUDA kernel activity. This is not a
measurement of speedup over a Python implementation, but it shows that the
remaining recurrence cost is tiny in that trace.

Specifically, the supplied PPO transition learner calls `_compute_advantages`
once before its minibatch loop (`pufferlib/pufferl.py:731`). It then reuses those
advantages and returns; the current value estimate is recomputed for critic loss,
but does not refresh the actor advantage each minibatch. The separate trajectory
learner has different recomputation behavior and is not this comparison's path.

SAC's [`_compute_next_values`](../../pufferlib/pufferl_sac.py) additionally runs
the actor and both target-Q backbones before the soft action expectation.
Its PyTorch tensors and networks already execute on CUDA. A fused target
arithmetic kernel would not remove those network forwards. Reusing cycle-start
targets reduces three passes of bootstrap network work to one; actor/critic
gradient work and post-critic actor Q evaluations remain. This experiment also
removes repeated next-observation gathering, so it is broader than CUDA fusion.

## Full-replay confirmation protocol

The confirmation uses two A6000 GPUs, 2,048,000 replay rows per rank,
128k logical / 32k microbatches and three passes. One cold cycle is excluded
and two warmed cycles are measured. Simulator geometry is **4 environments x
1,000 agents per rank**, as in the small matrix. This confirms full learner
buffer size and DDP, not production simulator throughput.

Two attempts using 20 environments x 3,200 agents per rank were stopped before
their first cycle after long startup delays. They produced no measurements;
the precise startup cause was not established. Their logs and interruption
records remain in `results/fixed_production` and `results/fixed_production_sac`.
The successful small DDP smoke established that distributed target updates can
run; reducing simulator geometry then allowed the full-replay control to reach
compiled training. The earlier completed production PPO benchmark is historical
context, not a contemporaneous control for these follow-up measurements.

| Two GPUs, full replay | Learner seconds/cycle | Collect seconds/cycle | Received rows/second |
|---|---:|---:|---:|
| Cached observations, fresh targets | 80.078 | 11.131 | 27,782 |
| Cached observations, fixed targets | 63.612 | 10.936 | 34,219 |

Fixed targets cut learner time **20.6%** and collection-plus-learning cycle time
**18.3%**. Received-row throughput rose **23.2%** in this short probe; received
counts differ slightly as the learned policies diverge. Both cases trained
6,144,000 sampled transitions per rank per cycle and performed 48 actor,
48 critic and 48 alpha steps per rank. Fixed targets reduced bootstrap work
from 6,144,000 rows / 192 microbatch calls to 2,048,000 rows / 64 calls per rank.
All final parameters are finite, and final model/alpha hashes match across
ranks within each experiment. The cache itself is 7.81 MiB per rank.

The post-timing audit recomputed both full replay buffers in their original
32k-row batch contexts, then compared 512 evenly spaced rows per rank. Mean
absolute cycle-start versus cycle-end target differences were **0.04980** and
**0.04680**; RMS differences were **0.07191** and **0.06089**, and sampled maximum
absolute differences were **0.63607** and **0.32790**. Mean absolute current
targets were 1.18168 and 1.19309. These are observed target staleness, not an error
bound or a comparison against a separately evolved stock-SAC trajectory.

Raw evidence and exact commands/configs:
[fixed_full_replay_summary.md](results/fixed_full_replay_summary.md), its JSON,
and each linked case's `summary.json` and cycle files. Both commands exited 0.

## Limits

These bounded probes can establish implementation correctness, measured work,
timing and target drift. They cannot establish preserved driving score,
convergence, sample efficiency or a verified 4B runtime. If “preserve correct
SAC computing” requires unchanged per-update targets, this idea fails that
requirement by construction and by the divergence tests; it must remain an
explicit algorithmic experiment rather than be promoted as a transparent fix.
