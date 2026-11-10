# One-epoch, one-third-refresh PQL experiment

Derivative of `../large_run_1b/optimized.yaml`, with exactly these training
changes: disable filtering, use one epoch per cycle, and replace one third of
the replay each cycle. The copied source config differs only in
`train.replay_update_ratio` (0.5 to 0.3333333333333333). The preset overrides
the source's epoch count to one.

Preserved: 1B global received-agent-step budget, two GPUs, seed 0, 20 environments
and 3,200 agent slots per environment per rank, 2,048,000 replay rows per rank,
128,000 logical / 32,000 microbatch, compiled bfloat16 networks, PQL horizon 128
and lambda .95, fixed cycle-start critic targets and actor-Q values, GPU replay,
learning rates, adaptive entropy settings, environment/reward configuration.

Per rank, each full cycle trains all 2,048,000 resident rows once (16 logical
updates) after the initial fill. Subsequent cycles insert
`ceil(2048000 / 3) = 682667` new rows. Global exposures are 4,096,000 per cycle.
Targets and actor-Q caches refresh every cycle, including retained old rows.
The approximate steady-state reuse is three exposures per inserted row, subject
to FIFO boundaries and the finite run. This is not equivalent to three epochs
against one fixed set of targets: the new targets and policy change between cycles.

The existing cosine schedule derives its cycle count from the 1B budget and
refresh count, so its cycle length changes automatically. Adaptive entropy
statistics still update per actor optimizer step. No additional scheduler tuning
is introduced. This experiment changes filtering and update cadence together;
it does not isolate filtering's causal effect.

Outputs, training metrics, periodic learner snapshots and final actor artifacts
follow the prior run. No exact replay/environment resume is implemented.
Warmup counts toward the budget but is excluded from phase timing summaries.

```bash
CUDA_VISIBLE_DEVICES=4,5 bash project/PPO_SAC_Profiling/one_epoch_1b/launch_2gpu.sh \
  project/PPO_SAC_Profiling/results/pql_one_epoch_1b_UNIQUE_NAME
```

This launcher runs only the new candidate; the previous stock and filtered runs
remain intact. Inspect `optimized/progress.json` and `optimized.log` for progress.
