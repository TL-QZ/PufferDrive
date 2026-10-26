# Round 6: cosine learning-rate schedules for SAC candidates 97 and 98

Round 6 starts fresh runs from the round 4_1 configuration and selects the same
108-grid candidates. It changes the training budgets and enables cosine
learning-rate schedules for the actor, both critics, and learned temperature.
The base `train.total_timesteps` remains 2B; the launcher applies the 4B
override to trial 98. No checkpoint is resumed: `load_model_path` and
`resume_state_path` are null, and each launch writes into a new trial directory.

## Launch

Activate the repository environment, then launch one candidate at a time on two
GPUs (the default GPU pair is `0,1`):

```bash
source .venv/bin/activate
bash project/soft_actor_critic_tuning/scripts/launch_round6.sh --trial 97
bash project/soft_actor_critic_tuning/scripts/launch_round6.sh --trial 98
```

Preview either resolved config and command without starting training:

```bash
bash project/soft_actor_critic_tuning/scripts/launch_round6.sh --trial 97 --dry-run --no-wandb
bash project/soft_actor_critic_tuning/scripts/launch_round6.sh --trial 98 --dry-run --no-wandb
```

`--gpus ID,ID` selects two distinct nonnegative GPU IDs. Each invocation
launches one two-rank trial. `launcher.trial_hours` is null, so there is no
wall-time deadline or early stopping rule. The training budget counts global
received agent rows and can overshoot by a collection batch.

## Candidate comparison

| Candidate | Round 4_1 replay refresh | Evaluation interval | Round 4_1 budget | Round 6 budget | Nominal cosine `T_max` |
|---|---:|---:|---:|---:|---:|
| 97 | 0.1 | 10 cycles | 2B rows | 2B rows | 1,221 cycles |
| 98 | 0.5 | 2 cycles | 2B rows | 4B rows | 489 cycles |

The IDs preserve the round-three Cartesian grid mapping. Both candidates use
`gamma=0.999`, target entropy `1.863680`, and three replay passes per training
cycle. Trial 97 uses the 10-cycle evaluation interval, and trial 98 uses 2
cycles, matching one nominal replay turnover for each refresh ratio.

The copied round 4_1 resource settings remain in place: 8,192,000 replay rows
per rank on two GPUs, 128,000-row logical minibatches, 32,000-row microbatches,
CPU replay, and `carla_fast` evaluation. Initial actor and critic learning rates
are `5e-4`; initial log-alpha learning rate is `3e-4`. The new W&B group is
`vanilla_sac_round6`.

Default artifacts are isolated under
`project/soft_actor_critic_tuning/runs/round6/sac108_grid_<timestamp>/`.

## Cosine schedule estimate

`anneal_lr: true` enables the actor, critic, and alpha cosine schedulers. The
trainer defaults `learn_alpha` to true, so the alpha learning rate also follows
the schedule. Scheduler `T_max` is estimated at initialization as
`ceil(total_timesteps / (world_size * refresh_count))`, where
`refresh_count = ceil(replay_capacity_per_rank * replay_update_ratio)`. With
two ranks, that gives:

- Trial 97: `ceil(2,000,000,000 / (2 * 819,200)) = 1,221` cycles.
- Trial 98: `ceil(4,000,000,000 / (2 * 4,096,000)) = 489` cycles.

These are nominal scheduler steps. Each scheduler advances once after a
training cycle with learner updates. The initial cycle fills the entire replay
buffer before refreshing it, so it consumes more received rows than a
steady-state refresh cycle. The received-row budget includes reset, masked, and
otherwise invalid rows, while replay refresh counts only valid insertions. If
masking is high, a cycle can use more received rows to reach its refresh quota.
The initial fill and masked rows can leave fewer scheduler steps before the
budget ends than the nominal estimates above. Trial 98 also has a longer budget,
so its cosine schedule spans roughly twice the 245-cycle estimate it would
have with the same refresh ratio and a 2B budget.

Compare candidates across the full trajectory using return, progress, goals,
speed, infractions, and late-run stability. A dry run checks configuration and
scheduling; it does not establish GPU memory fit, throughput, or policy quality.
