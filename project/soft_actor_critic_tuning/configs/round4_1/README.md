# SAC round 4_1: longer FIFO runs for trials 97 and 98

Both candidates start from scratch with a **2,000,000,000 global
received-agent-row budget**, two GPUs, and no wall-time cutoff. Candidate IDs
reuse round-three hyperparameters, not checkpoints. Earlier round configs and
the round-four launcher remain unchanged.

## Candidate settings

| Source trial | Replay refresh per outer epoch | `carla_fast` interval | Global received-row budget |
| ---: | ---: | ---: | ---: |
| 97 | 0.1 | 10 outer epochs | 2,000,000,000 |
| 98 | 0.5 | 2 outer epochs | 2,000,000,000 |

Both use gamma 0.999, target entropy 1.863680 nats, tau 0.005, initial alpha
0.1, and 3 replay passes per outer epoch. Trial 98 replaces round four's
trial 99 (refresh ratio 0.75). The copied 108-combination grid preserves the
original IDs; evaluation cadence is a launcher override, not a new grid axis.

`SAC_run_config.yaml` defines `launcher.trial_evaluation_interval_epochs`:
97 maps to 10 and 98 maps to 2. The launcher applies these after grid parameters
and records the interval in each planned trial's manifest entry and resolved
YAML. `SAC_tunable_config.yaml` uses 10 as its base interval.

## Replay and evaluation cadence

The current SAC collector evicts the initial replay contents in a once-shuffled
slot order, then cycles through that order to evict subsequent transitions FIFO.
This is transition-level replay; episodes need not remain together. FIFO is
learner behavior, not a round-specific configuration flag.

Replay holds **8,192,000 transitions per GPU rank**, or **16,384,000 globally**.
One outer epoch collects transitions, then performs three shuffled passes over
the entire replay. A logical minibatch is 128,000 per rank (microbatch 32,000),
so each outer epoch has `8,192,000 / 128,000 * 3 = 192` synchronized updates.

In steady state, both evaluation schedules span one complete buffer turnover:

- Trial 97: `10 * 819,200 * 2 = 16,384,000` new transitions.
- Trial 98: `2 * 4,096,000 * 2 = 16,384,000` new transitions.

The first epoch fills replay rather than refreshing a fraction. The first
scheduled evaluation therefore follows 31,129,600 inserted transitions for
trial 97 (fill plus nine refreshes), or 24,576,000 for trial 98 (fill plus one
refresh). Later scheduled intervals match the full-buffer turnover above.
Evaluation runs on rank 0. Normal completion performs a final evaluation if
the last epoch was not already evaluated.

The stopping budget counts **received rows**, including masked/reset and
quota-discarded rows, rather than valid replay insertions or optimizer work.
With negligible discarded rows, the 2B budget gives approximately 1,212 outer
epochs for trial 97 and 244 for trial 98; both have about 122 evaluations
including the final evaluation. Actual counts depend on valid-row yield and
vector-batch rounding. Wall-clock evaluation frequency is not fixed.

## Launch one candidate

From the repository root, preview trial 97:

```bash
bash project/soft_actor_critic_tuning/scripts/launch_round4_1.sh \
  --trial 97 --gpus 0,1 --dry-run
```

Remove `--dry-run` to train. To select trial 98:

```bash
bash project/soft_actor_critic_tuning/scripts/launch_round4_1.sh \
  --trial 98 --gpus 0,1
```

`--trial` is required and accepts only 97 or 98. Each invocation runs one
two-GPU process group and waits for completion. GPUs default to `0,1`.
Optional `--output-dir PATH` selects a new artifact directory; `--no-wandb`
disables experiment loggers and remote model uploads. There is no duration or
concurrency override in the dedicated wrapper.

Default artifacts go under
`project/soft_actor_critic_tuning/runs/round4_1/sac108_grid_<timestamp>/`,
with `manifest.json`, one resolved trial config, and `trial_97/` or `trial_98/`.
The `sac108` prefix denotes retained grid indexing, not 108 launched jobs.
W&B group: `vanilla_sac_round4_1`.

Environment, model, optimizer, seed, BF16, compilation, CPU replay and resource
settings carry forward from round four. Replay alone uses about 112.9 GiB host
RAM across two ranks at 923 observation features, plus simulator/process memory.
Normal completion saves `final_model.pt` and `trainer_state.pt`;
`checkpoint_interval: null` keeps periodic checkpoints disabled. Nonfinite
losses, errors, or cancellation can stop a run before its budget.

Validation uses configuration checks and launcher dry runs; no training or
runtime throughput measurement is implied. Compare candidates at matched
received-agent-row counts using driving metrics as well as wall time.
