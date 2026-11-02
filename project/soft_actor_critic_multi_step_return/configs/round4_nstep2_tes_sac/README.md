# Round 4: two-step SAC with adaptive target entropy and smaller minibatches

Round 4 derives from the isolated Round 3 configuration. It retains two-step
returns, adaptive target entropy, the trial 97/98 grid settings, and private
evaluation catalog. The only learner-size change is `train.minibatch_size`,
reduced from 128,000 to 64,000; `train.max_minibatch_size` remains 32,000.

At full replay, each training cycle samples the 2,048,000 rows per rank for
three update epochs. Round 3 therefore made
`3 * (2,048,000 / 128,000) = 48` logical learner updates per cycle. Round 4
makes `3 * (2,048,000 / 64,000) = 96`. Each Round 3 update accumulated four
32,000-row microbatches; Round 4 accumulates two. These are logical optimizer
updates, and each still uses the same total replay rows per epoch.

Each logical update steps learned alpha and observes entropy for adaptive
target-entropy scheduling once, then applies one target-critic Polyak update
with the unchanged `tau=0.005`. Round 4 therefore applies these per-update
operations twice as often per full-replay training cycle. Actor, critic, and
alpha learning rates remain unchanged, as does cosine annealing once per
training cycle. These cadence changes do not guarantee better policy or
evaluation performance.

Round 4 enables `train.log_collapse_diagnostics: true`. The diagnostics reuse
existing forward passes and record gradient norms before clipping. They add
reductions, and their GPU overhead has not been measured.

## Reading the collapse diagnostics

These metrics appear under `diagnostics/` in W&B and the trial's local
`metrics.jsonl`. They cover the interval since the previous learner log.

| Metric suffix after `diagnostics/` | Meaning and diagnostic use |
|---|---|
| `actor/grad_norm_mean`, `actor/grad_norm_max`, `actor/grad_clip_fraction` (also `critic/…`) | Total L2 gradient norm before clipping, and fraction of logical updates exceeding the configured 0.5 threshold. A critic spike preceding action concentration suggests a different failure sequence from a vanishing actor gradient after saturation. |
| `policy/max_action_probability_mean`, `policy/near_deterministic_fraction` | Mean largest action probability, and fraction of sampled states with largest probability at least 0.99. Detects concentration that an average entropy can hide. |
| `policy/logit_range_mean`, `policy/logit_range_max` | Largest minus smallest actor logit per state. A growing spread together with near-deterministic probabilities supports saturation. |
| `policy/q_action_gap_over_alpha_mean`, `policy/q_action_gap_over_alpha_max` | Gap between the highest and second-highest `min(Q1,Q2)` action values, divided by current alpha. Measures how decisive the critic is relative to entropy regularization; it is not a direct policy KL measurement. |
| `critic/abs_td_error_mean`, `critic/abs_td_error_max` | Absolute target error over both critics for replay actions. Maxima expose outliers that average critic loss can hide. |

`diagnostics/entropy_update_mean_min` is the minimum **rank-local logical
minibatch mean entropy** over the logging interval, not minimum per-state entropy
or a globally averaged update. Compare it with `losses/entropy`, alpha, target
entropy, and TES gate statistics already logged.

State-based means weight sampled rows, including tail microbatches; gradient
statistics weight logical updates. Distributed means reduce sums/counts, and
maxima/minima retain the worst value across ranks. The tracker uses detached
tensors from existing forwards, has no additional model forwards/backwards, and
copies one packed result to the CPU at log time. Three small distributed
reductions occur at logging time; there are no added per-microbatch scalar reads
from the GPU. Gradient metrics reuse clipping's returned norm; when clipping is
disabled they are omitted, rather than introducing a separate gradient scan.

These are replay-distribution measurements, not fixed-probe observations.
Interval maxima reveal spikes but do not identify the exact update sequence or
prove causality. No additional checkpoint/replay capture is enabled by this flag.

Two consecutive valid rewards contribute to each replay target, with a maximum
horizon of 0.6 simulation seconds at `env.dt=0.3`. Trajectory boundaries can
shorten the return. Use fresh replay with the standard SAC collector.

Training goal spacing remains 6.666666666666667–66.66666666666667 meters. The
private evaluation catalog retains 10-meter spacing for both bounds, including
scheduled `carla_fast` evaluation.

Adaptive target entropy starts at 1.863680 nats, with EMA decay 0.999,
mean/std thresholds 0.01/0.05, decay factor 0.9, and 200 qualifying updates.
Forced decay occurs after 500 million global received agent rows without a
reduction. Alpha remains learned.

| Trial | Replay refresh | Global received-row budget | Evaluation interval |
|---|---|---|---|
| 97 | 0.1 | 2 billion | 10 collection cycles |
| 98 | 0.5 | 4 billion | 2 collection cycles |

Both trials retain gamma 0.999, three update epochs, 2,048,000 replay rows per
rank, and two GPUs. W&B group and run folder are `round4_nstep2_tes_sac`.

Preview either trial (replace 97 with 98 for the second candidate):

```bash
source .venv/bin/activate
bash project/soft_actor_critic_multi_step_return/scripts/launch_round4.sh --trial 97 --dry-run --no-wandb
```

To train, remove `--dry-run --no-wandb`; use `--gpus ID,ID` to change GPU IDs.
No training was launched during preparation. Dry runs validate configuration
and command construction, not live GPU training or policy quality.

Validation on 2026-11-02: both trial dry runs, merged-configuration parity,
benchmark-catalog parity, and shell syntax passed. Eight diagnostics tests and
65 SAC learner/TES regression tests passed. Tests cover unchanged optimizer and
parameter updates, RNG state, forward-call counts, unequal microbatch weighting,
logging reset, checkpoint compatibility, and CPU compiled forward wrappers.
A separate real two-rank CPU/Gloo check passed weighted means, extrema, clipping
fractions, and reset with host socket access. CUDA throughput and diagnostic
overhead remain unmeasured; no claim of improved collapse resistance is made.

Standalone evaluation must explicitly select the private catalog:

```text
eval.benchmark_config=project/soft_actor_critic_multi_step_return/configs/round4_nstep2_tes_sac/evaluation_benchmarks.yaml
```

Checkpoint loading does not restore this catalog path. For `nuplan_multi`, also
set `eval.num_agents=300` or greater (the inherited default remains 128).
