# Round 5: learning rates over long SAC training

Launch all six trials, one GPU each, from the repository root:

```bash
bash project/soft_actor_critic_tuning/scripts/launch_round5.sh
```

Preview the resolved configurations and commands without starting training:

```bash
bash project/soft_actor_critic_tuning/scripts/launch_round5.sh --dry-run --no-wandb
```

## Trial map

Round-three/round-four candidate IDs 97 and 98 identify the inherited non-LR
settings, not the new launcher's trial IDs. Both use gamma 0.999, target entropy
1.863680 nats, and three replay passes per training cycle.

| Round-5 trial ID | Source candidate | Replay refresh | Actor LR | Both critics LR | Log-alpha LR | Default GPU | Evaluation interval (cycles) |
|---|---|---|---|---|---|---|---|
| 1 | 97 | 0.1 | 5e-4 | 5e-4 | 3e-4 | 0 | 10 |
| 14 | 97 | 0.1 | 1e-4 | 1e-4 | 1e-4 | 1 | 10 |
| 27 | 97 | 0.1 | 5e-5 | 5e-5 | 3e-5 | 2 | 10 |
| 28 | 98 | 0.5 | 5e-4 | 5e-4 | 3e-4 | 3 | 2 |
| 41 | 98 | 0.5 | 1e-4 | 1e-4 | 1e-4 | 4 | 2 |
| 54 | 98 | 0.5 | 5e-5 | 5e-5 | 3e-5 | 5 | 2 |

The existing launcher enumerates 54 Cartesian combinations in axis order
replay refresh, actor LR, critic LR, alpha LR. `launcher.trial_ids` selects only
the six rows above; the remaining combinations are marked skipped. Actor and
critic rates are paired, and alpha changes alongside them. This experiment
compares joint rate settings; it cannot isolate alpha's contribution.

## Resources and training duration

Replay capacity is **512,000 transitions per GPU**, matching round 2 exactly.
Logical minibatches remain 128,000 transitions and microbatches remain 32,000.
The checked-in round 1 and round 4_1 capacities are 8,192,000 per rank, so this
is 1/16 of those capacities, not 1/8. Three full passes produce 12 logical
updates per cycle with a full replay buffer. Replay remains on CPU; each trial
uses 20 environments with 3,200 agent slots each, as in round 2. Six concurrent
trials therefore use 120 environment workers in addition to learner processes.

Every trial has `train.total_timesteps: 2000000000` and
`launcher.trial_hours: null`. There is no soft deadline, parent hard timeout,
or performance-based early stopping. Nonfinite losses and runtime errors still
fail a trial. The budget counts received agent rows, not optimizer updates or
accepted replay transitions, and collection batches can overshoot it.

Learning rates stay fixed (`anneal_lr: false`). Actor and critics use AdamW;
log-alpha uses Adam. Target critics retain Polyak coefficient `tau: 0.005`.
Initial alpha remains 0.1. Evaluation uses `carla_fast` after each nominal
replay turnover (10 or 2 cycles), with final evaluation handled by the learner.
Smaller replay and one GPU change training dynamics relative to round 4_1;
the controls provide a comparison at the same round-5 resource settings.

## Outputs and comparison

Outputs are isolated under `project/soft_actor_critic_tuning/runs/round5/`.
Each invocation creates a new sweep directory, unique run names, per-trial
resolved YAML, metrics, evaluation artifacts, and final checkpoints. Fresh runs
disable automatic checkpoint resume. W&B uses project
`pufferdrive_vanilla_sac_tuning`, group `vanilla_sac_round5`.

Compare each lower-rate trial with its same-refresh control across the full
2B-step trajectory: return, progress, goals, speed, infractions, and late-run
stability. Keep peak performance and late performance separate when assessing
collapse. Lower rates reduce optimizer step sizes; this is a hypothesis to
test, not a guarantee of stability.

Use native options for alternative GPU assignments or a selected subset:

```bash
bash project/soft_actor_critic_tuning/scripts/launch_round5.sh --gpu-groups '0;1;2;3;4;5'
bash project/soft_actor_critic_tuning/scripts/launch_round5.sh --trial-ids 14,41 --max-concurrent 2
```

Do not pass `--smoke` or `--trial-hours` for the long-training experiment: those
options intentionally introduce a time limit. A dry run verifies configuration
and scheduling only; it does not establish GPU memory fit or training quality.

Validation: the six resolved trial configs and GPU assignments passed a dry
run. The tuning test suite passed 34 tests and 75 subtests, including timeout
coverage and the round-5 launcher regression. No training was launched.
