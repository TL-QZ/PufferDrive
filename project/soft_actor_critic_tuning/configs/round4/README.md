# SAC round four: full-scale trials 97 and 99

Prepared configuration only; training has not been launched by this setup.
Choose one round-three candidate per invocation. Each run starts from scratch
on two GPUs, with no wall-time cutoff and a 500-million received-agent-step
budget across both GPUs. Candidate selection reuses hyperparameters, not a
trained checkpoint.

## 1. How training ends

Normal termination occurs when `agent_steps >= train.total_timesteps`.
`agent_steps` counts received agent rows summed across GPU ranks, including
initial collection; it is not an optimizer-update count, epoch count, or
count of replay samples. The loop checks the budget between cycles, so vector
collection can slightly overshoot it.

The retained budget is **500,000,000 total received rows per run**, not per
GPU. `launcher.trial_hours: null` disables both the child deadline and the
parent hard timeout. No score threshold, convergence test, or plateau detector
stops these runs. A nonfinite learner loss still triggers early stopping;
errors, process termination, and manual cancellation can also end a run.

On normal completion, the learner runs final benchmark evaluation if needed,
then saves `final_model.pt` and `trainer_state.pt`. Abrupt cancellation or an
error does not guarantee these final artifacts. `checkpoint_interval: null`
retains the existing policy of no periodic checkpoints.

Source: [`pufferl_sac.py`](../../../../pufferlib/pufferl_sac.py), training loop
and `close()`; [`run_trial.py`](../../scripts/run_trial.py), `TrialMonitor`.

## 2. Candidate settings

| Retained source / trial ID | Tau | Gamma | Target entropy (nats) | Update epochs | Replay update ratio |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 97 | 0.005 | 0.999 | 1.863680 (H75) | 3 | 0.1 |
| 99 | 0.005 | 0.999 | 1.863680 (H75) | 3 | 0.75 |

Both descend from round-two candidate 31. They differ only in refresh ratio.
Initial alpha remains 0.1. Environment, rewards, model, optimizer settings,
seed 0, BF16, compilation, CPU replay, and per-cycle evaluation carry forward
from round three. W&B group is `vanilla_sac_round4`.

The copied 108-combination grid preserves source IDs. The dedicated launcher
selects exactly one ID; its manifest contains one planned trial and 107 skipped
entries. Only the selected trial gets a resolved config and training process.
The underlying round-four launcher profile lists both candidates with
`max_concurrent: 1`; the dedicated script requires an explicit choice and never
automatically queues the other candidate.

## 3. Full-scale resources

| Setting | Round three | Round four |
| --- | ---: | ---: |
| GPUs per run | 1 | 2 |
| Replay transitions per GPU rank | 512,000 | 8,192,000 |
| Aggregate replay transitions | 512,000 | 16,384,000 |
| Logical minibatch per rank | 128,000 | 128,000 |
| Maximum microbatch per rank | 32,000 | 32,000 |
| Aggregate logical minibatch | 128,000 | 256,000 |
| Concurrent training jobs per launcher invocation | 8 | 1 |
| Global received-agent-row budget | 500,000,000 | 500,000,000 |
| Wall-time cutoff | 3 hours | None |

The replay capacity matches the numeric rollout size of the two-GPU PPO
baseline used in the [parity audit](../../reports/ppo_parity_audit_2026-10-11.md):
20 environments × 3,200 agent slots × 128 rollout steps = **8,192,000 per
rank**. PPO derives a rollout size from its horizon; SAC uses this number as
replay capacity. Their collection and reuse semantics remain different.

Three passes over the larger replay now mean **192 optimizer updates per
cycle**, compared with 12 in round three. Tau stays 0.005 per optimizer update.
After the initial fill, trial 97 replaces **819,200 transitions per rank** per
refresh; trial 99 replaces **6,144,000**. The scaling therefore changes the
number of updates and collected transitions between evaluation cycles.

At 923 FP32 observation features, the two-rank replay alone needs about
**112.9 GiB host RAM**. The existing preflight includes 10% headroom (about
124.2 GiB); simulator/process memory is additional. Runtime memory and
throughput have not been measured for this round.

## 4. Select, preview, and launch

Run from the repository root. Preview trial 97 on GPUs 0 and 1:

```bash
bash project/soft_actor_critic_tuning/scripts/launch_round4.sh \
  --trial 97 --gpus 0,1 --dry-run
```

Launch that one candidate:

```bash
bash project/soft_actor_critic_tuning/scripts/launch_round4.sh \
  --trial 97 --gpus 0,1
```

To choose trial 99 instead:

```bash
bash project/soft_actor_critic_tuning/scripts/launch_round4.sh \
  --trial 99 --gpus 0,1
```

`--trial` is required and accepts only 97 or 99. `--gpus` defaults to `0,1`
and requires two distinct GPU indices. Optional `--output-dir` chooses a new
artifact directory; `--no-wandb` disables remote logging and model uploads.
There is no duration or concurrency override in this script. One invocation
launches one job and waits for it; starting another invocation yourself is a
separate job, so wait for completion before launching the other candidate.

Default output: `runs/round4/sac108_grid_<timestamp>/trial_97/` or `trial_99/`.
The `sac108` prefix reflects retained grid indexing, not the number of jobs.
Resolved configs and `manifest.json` are saved in the enclosing run directory.

To change the step budget before starting a new run, edit
`train.total_timesteps` in `SAC_tunable_config.yaml`. A running process uses its
saved resolved config. Compare the two candidates at matched received-agent
steps and inspect sustained return, progress, scores, goals, and infractions,
with wall time and Q/entropy diagnostics alongside them.
