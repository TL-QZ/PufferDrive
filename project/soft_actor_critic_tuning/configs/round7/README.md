# Round 7: smaller per-rank replay for SAC candidates 97 and 98

Round 7 starts fresh runs from the Round 6 candidates and keeps the two-GPU
setup, trial mapping, budgets, learning rates, and cosine schedules. The
experiment change is replay capacity: `train.batch_size` is 2,048,000
transitions **per rank**, down from 8,192,000 per rank in Round 6. With two
ranks, total replay capacity is 4,096,000 transitions across both ranks. Each
rank owns its replay independently; this is not a shared 4,096,000-row buffer.
The 128,000-row logical minibatch and 32,000-row microbatch remain unchanged
per learner rank.

The smaller capacity shortens the training rows between scheduled evaluations.
Both candidates still evaluate after one nominal replay turnover: candidate 97
refreshes 10% per cycle and evaluates every 10 cycles; candidate 98 refreshes
50% and evaluates every 2 cycles. In either case, a benchmark now follows
2,048,000 refreshed rows per rank (4,096,000 across both ranks), one quarter of
Round 6's 8,192,000 rows per rank (16,384,000 across both). The cycle-based
evaluation settings are unchanged, so each candidate gets about four times as
many scheduled evaluations over the same global row budget.

## Launch

Activate the repository environment and launch one candidate at a time on two
GPUs (the default pair is `0,1`):

```bash
source .venv/bin/activate
bash project/soft_actor_critic_tuning/scripts/launch_round7.sh --trial 97
bash project/soft_actor_critic_tuning/scripts/launch_round7.sh --trial 98
```

Preview either resolved config and command without starting training:

```bash
bash project/soft_actor_critic_tuning/scripts/launch_round7.sh --trial 97 --dry-run --no-wandb
bash project/soft_actor_critic_tuning/scripts/launch_round7.sh --trial 98 --dry-run --no-wandb
```

`--gpus ID,ID` selects two distinct nonnegative GPU IDs. Each invocation
launches one two-rank trial. There is no wall-time deadline. Budgets count
global received agent rows and can overshoot by a collection batch; replay
refresh counts valid inserted transitions.

## Candidates and scheduler horizons

| Candidate | Replay refresh | Global row budget | Evaluation interval | Nominal cosine `T_max` |
|---|---:|---:|---:|---:|
| 97 | 0.1 | 2B | 10 cycles | 4,883 cycles |
| 98 | 0.5 | 4B | 2 cycles | 1,954 cycles |

The trial IDs retain their Round 3 grid mappings: both use `gamma=0.999`,
target entropy `1.863680`, and three replay passes per training cycle. Candidate
97 uses a 2B global received-row budget; candidate 98 uses 4B. Both keep
`anneal_lr: true`, with cosine schedules for actor, critics, and learned alpha.
Other inherited settings include two GPUs, CPU replay, 20 environments per
rank with 3,200 agent slots each, `carla_fast` evaluation, actor and critic
learning rates of `5e-4`, alpha learning rate `3e-4`, and `tau=0.005`.
The W&B group is `vanilla_sac_round7`.

The nominal horizons use
`ceil(global_row_budget / (world_size * refresh_count_per_rank))`, where
`refresh_count_per_rank = ceil(replay_capacity_per_rank * replay_update_ratio)`:

- Candidate 97: `ceil(2,000,000,000 / (2 * 204,800)) = 4,883` cycles.
- Candidate 98: `ceil(4,000,000,000 / (2 * 1,024,000)) = 1,954` cycles.

The initial cycle fills the replay buffer before steady-state refreshes, and
masked or otherwise invalid rows do not count toward the refresh quota. The
received-row budget can therefore end collection before the schedulers reach
their nominal horizon. A dry run validates configuration and scheduling; it
does not establish GPU memory fit, throughput, or policy quality. No training
was launched while preparing this configuration.

Default outputs are isolated under
`project/soft_actor_critic_tuning/runs/round7/`; each invocation creates a new
sweep directory and per-trial artifacts.

## Checkpoints after evaluation

After each successful scheduled or final evaluation, SAC saves the latest
learner state even though `train.checkpoint_interval` remains null. Within each
trial's `learner/` directory:

- `best_evaluation.json` records the best mean `puffer_score` for each benchmark,
  its epoch, received agent steps, summary metrics, and relative actor path.
- `best_models/best_eval_*.pt` contains the selected actor's raw state dict,
  compatible with `load_model_path`. For this round, selection uses `carla_fast`;
  higher scores win, and ties retain the previous actor.
- `trainer_state.pt` contains the latest successfully saved learner state,
  including actor, critics, optimizers, temperature, and schedulers. It is not
  necessarily the best-evaluation state. Replay and simulator state are not
  saved; resuming starts with empty replay and a fresh environment.

The actor is written before its manifest is atomically published. The previous
winner is removed only after publication, and later evaluations read the saved
manifest before comparing scores. This preserves the recorded best actor
across an interrupted save or process restart. Failed evaluations, empty
reports, and nonfinite scores do not replace it. No evaluation checkpoint
exists until the first successful evaluation; the existing training-score
best-model mechanism remains separate.

Round 7 launches fresh runs by default. To resume an interrupted run, use the
native SAC resume interface with its `trainer_state.pt`; rerunning this launcher
creates a new sweep rather than resuming the old one.
