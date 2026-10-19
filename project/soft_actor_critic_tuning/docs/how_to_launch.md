# Launch the 16-trial SAC experiment

For completed-checkpoint evaluation and comparison plots, follow
[the evaluation launch instructions](how_to_evaluate.md).

Run commands from the PufferDrive repository root. The shell wrapper activates
`.venv` and limits each process's numerical-library threads to one.

The default command below uses the **first-round experiment** in
[`configs/round1/`](../configs/round1/). The new
single-GPU resource profile lives separately in
[`configs/round2/`](../configs/round2/); see its
[configuration notes](../configs/round2/README.md). Select it explicitly:

```bash
bash project/soft_actor_critic_tuning/scripts/launch_sweep.sh \
  --config-dir project/soft_actor_critic_tuning/configs/round2 --dry-run
```

Round two uses one GPU per trial, eight concurrent trials, 512,000 replay
transitions, and a 1-billion received-agent-row budget. The logical minibatch
remains 128,000 and the microbatch remains 32,000. Its 64-trial grid sweeps
tau, gamma, and target entropy; its README contains the experiment plan and
launch commands.
Default artifacts go under `runs/round2/`, and its W&B group is
`vanilla_sac_round2`. The rest of this guide describes the original sweep.

## Round four: one full-scale candidate at a time

Round four restores two GPUs and 8,192,000 replay transitions per rank for
round-three candidates **97 or 99**. Choose one explicitly; the run has no
wall-time cutoff and retains the 500-million global received-agent-row budget.
See the [round-four README](../configs/round4/README.md) for stopping behavior,
resource requirements, and the candidate comparison.

```bash
bash project/soft_actor_critic_tuning/scripts/launch_round4.sh \
  --trial 97 --gpus 0,1 --dry-run
```

Remove `--dry-run` to train that one candidate; use `--trial 99` to choose the
other. The script waits for the selected run and does not queue another trial.

## 1. Inspect the grid without starting training

For the update-frequency sweep based on round-two candidates **17, 26 and 31**,
use [`configs/round3/`](../configs/round3/README.md). It uses round-two resources
(one GPU and 512,000 replay transitions per trial), a 500-million-row ceiling,
and a three-hour wall-time target. Each candidate sweeps epochs `[3, 15, 50, 100]`
and replay refresh ratios `[0.1, 0.5, 0.75]`: **36 selected trials**. The README
maps the new trial IDs; the manifest includes 72 additional skipped combinations.

```bash
bash project/soft_actor_critic_tuning/scripts/launch_sweep.sh \
  --config-dir project/soft_actor_critic_tuning/configs/round3 --dry-run
```

```bash
bash project/soft_actor_critic_tuning/scripts/launch_sweep.sh --dry-run
```

This writes 16 resolved YAML files, a `manifest.json`, and command logs under a
new `project/soft_actor_critic_tuning/runs/sac16_grid_<UTC timestamp>/` directory.
It creates no environments, CUDA contexts, or external logging sessions.
The dry run does **not** measure available GPUs or run the RAM preflight.

| Grid parameter | Values |
| --- | --- |
| `train.update_epochs` | `3`, `10` |
| `train.replay_update_ratio` | `0.05`, `0.1` |
| `train.initial_alpha` | `0.1`, `0.2` |
| `train.tau` | `0.005`, `0.01` |

There are **2 × 2 × 2 × 2 = 16 trials**, with one seed per combination.
Trial IDs are deterministic: epochs vary slowest, then replacement ratio, then
initial alpha, and tau varies fastest. Trials 1–8 use three epochs; 9–16 use ten.
The manifest records the exact parameters for each ID.

The launcher uses `controlled_exp` value lists, rather than the repository's
adaptive sweep searcher. The separate `sweep` search spaces and `max_runs` do not
affect this grid. The actor and critic learning rates are fixed at the baseline
`0.0005`; temperature LR is `0.0003`. Target entropy remains automatic
(`0.98 * log(K)`). Training seed `0` is fixed across all combinations.

## 2. Check the real two-GPU pipeline without W&B

```bash
bash project/soft_actor_critic_tuning/scripts/launch_sweep.sh \
  --smoke --gpu-pairs '0,1' --max-concurrent 1
```

Smoke mode runs trial 1 with the real Drive environment, actor, twin critics,
DDP, and optimizers. It uses a 60-second target plus a 600-second shutdown grace. The first compiled
cycle and its full benchmark can exceed the short target.
It disables W&B, Neptune, TensorBoard, and model uploads, and sets
`WANDB_MODE=disabled` for child processes.

To keep this check small, each rank uses two Drive instances, 128,000 replay
entries, a 16-step scenario, and resampling every 64 steps. It retains the full
network, 3,200 agent slots per instance, the production logical minibatch of
128,000, and the configured microbatch limit (currently 32,000, matching PPO).
It must perform learner updates, complete the selected benchmark, and write
final actor/learner checkpoints to count as successful. Compilation stays enabled.
The benchmark retains all 250 scenarios and 500 steps; smoke overrides affect
training collection only. It does not qualify the full replay, 20-worker layout
or full-scale environment throughput.

While it runs, another terminal can show GPU usage:

```bash
nvidia-smi -i 0,1 --query-gpu=index,memory.used,utilization.gpu --format=csv -l 1
```

Historical smoke reports are in
[reports/archived/smoke_runs/](../reports/archived/smoke_runs/).
The completed production sweep has
[all 16 trials completed](../runs/sac16_grid_20261012T032408Z/manifest.json).

## 3. Launch the full grid

Authenticate W&B once if necessary:

```bash
source .venv/bin/activate
wandb login
```

Choose idle GPU pairs, then run:

```bash
bash project/soft_actor_critic_tuning/scripts/launch_sweep.sh \
  --gpu-pairs '0,1;2,3' --max-concurrent 2
```

This queues all 16 trials, with two GPUs per trial and two concurrent trials.
Only rank zero logs each trial to **`pufferdrive_vanilla_sac_tuning`**, group
`vanilla_sac_grid`. Each trial has a unique W&B run name and output directory.
Model artifact uploads are enabled for these full runs; `no_model_upload=true`
disables artifacts alone. `--no-wandb` disables all external loggers and uploads.

To keep the launcher running after disconnecting the terminal:

```bash
mkdir -p project/soft_actor_critic_tuning/runs
nohup bash project/soft_actor_critic_tuning/scripts/launch_sweep.sh \
  --gpu-pairs '0,1;2,3' --max-concurrent 2 \
  > "project/soft_actor_critic_tuning/runs/launcher_$(date -u +%Y%m%dT%H%M%SZ).log" 2>&1 &
LAUNCHER_PID=$!
echo "$LAUNCHER_PID"
```

Save that PID if you need to stop this launcher later. A foreground run can be
stopped with Ctrl-C; a background run with `kill -TERM "$LAUNCHER_PID"` in the
same shell. The launcher terminates its owned trial processes and tracked vector workers.
An interrupted trial is marked cancelled; its final checkpoint is not guaranteed.

### Time and host-memory budgets

Each trial targets **three hours**, measured from its process launch, including
initialization, replay fill and benchmark evaluation. After a completed
collection/training/evaluation cycle, the monitor stops if the deadline has
passed or another cycle of the same duration would exceed it. This preserves
completed replay passes and leaves time for final artifacts; a trial can finish
earlier by roughly one cycle. Cycle durations can change, so this estimate is
not a guarantee. The parent allows up to 600 additional seconds before
forcefully terminating a trial and marking it `failed_timeout`.

The nominal training budget is **16 × 2 × 3 = 96 GPU-hours**. Two concurrent
trials take eight waves, approximately **24 hours**, plus shutdown overhead.
Four concurrent trials would take four waves, approximately **12 hours**, if
host RAM permits it.

For 923 observation features, the CPU replay alone uses about **56.5 GiB per
rank**. The launcher estimates replay plus 10% headroom before starting workers:

| Concurrent trials | GPUs | Estimated host RAM |
| --- | --- | --- |
| 1 | 2 | 124.2 GiB |
| 2, default | 4 | 248.4 GiB |
| 4 | 8 | 496.8 GiB |

This server had about 486 GiB available during preflight, so four concurrent
trials would fail the RAM guard. Keep two concurrent trials initially. The
estimate is a lower-bound planning check: simulator state, pinned staging,
Python, checkpoints, and other processes consume additional memory. GPU IDs
must be distinct, and the launcher does not reserve GPUs against other jobs.

## 4. Configuration and artifacts

The launch inputs are these partial configurations, merged in this order:

1. [SAC_tunable_config.yaml](../configs/round1/SAC_tunable_config.yaml): shared learner,
   vectorization, logging, and run settings.
2. [SAC_specific_config.yaml](../configs/round1/SAC_specific_config.yaml): replay,
   temperature, target critics, and the four grid axes.
3. [SAC_fixed_config.yaml](../configs/round1/SAC_fixed_config.yaml): fixed environment
   and network settings, which override overlapping fields.
4. [SAC_run_config.yaml](../configs/round1/SAC_run_config.yaml): launcher-only duration,
   GPU pairs, concurrency, and shutdown grace.

`SAC_all_configs.yaml` is a catalog of package defaults, not a launch input.
After merging, the launcher applies each trial's grid values and unique run
paths. Smoke mode then applies its small-test overrides. Read the generated
`configs/trial_XX.yaml` to see the actual input to the learner.

The production setup retains CARLA nightly's environment and backbone settings,
20 Drive instances × 3,200 agent slots per rank, and **8,192,000 replay entries
per rank**. As in PPO, `vec.num_workers=auto` resolves to 20 workers per rank,
receiving ten instances at a time. Two concurrent two-rank trials therefore
create 80 vector workers. SAC uses a discrete actor with the existing decoder into continuous Drive
controls. The actor and twin critics own independent backbones.

Training uses AdamW, actor/critic LR `0.0005`, gamma `0.999`, BF16 autocast,
CPU replay, and `torch.compile`. A logical minibatch has 128,000 transitions, accumulated
in four microbatches of 32,000, matching PPO's minibatch and microbatch sizes.
A full replay pass has 64 logical updates:
three epochs give 192 updates per cycle; ten give 640. Replacement ratios set
valid-transition quotas of 409,600 or 819,200 per rank after the initial fill.
LR annealing is disabled by the requested experiment choice.

`total_timesteps=10,000,000,000` matches PPO’s global collection ceiling. SAC
counts received agent rows across both ranks; it does not count replay passes
or optimizer updates toward this budget. For example, ten received instances
× 3,200 slots × two ranks add 64,000 global rows. Reusing those rows for ten
training passes does not add another tenfold amount to this counter. Training
stops at this ceiling or the three-hour time limit, whichever is reached first.
The previous one-trillion ceiling was a launcher choice to prioritize the timer,
not a SAC requirement.

### Training-time progression and final checkpoints

After **every collection/training cycle**, the actor runs `carla_fast`: all 250
baseline scenarios, 500 steps, `dt=0.1`, and `action_selection=mean`.
`evaluation_interval_epochs=1` counts outer cycles; it is independent of the
three or ten shuffled training passes inside a cycle. W&B records the benchmark
metrics as `eval_carla_fast/<metric>`, alongside ordinary training metrics.
Compare progression against `agent_steps` and elapsed time; different replay
ratios and pass counts produce different numbers of cycles in three hours.

`checkpoint_interval: null` disables periodic checkpoint writes. Normal
completion still writes `final_model.pt` and `trainer_state.pt`. Benchmark
reports are retained for each cycle; replay and live-environment state are
not included in learner checkpoints.

```text
runs/sac16_grid_<timestamp>/
  manifest.json                  # trial parameters, GPU allocation, status
  configs/trial_01.yaml           # resolved configuration before learner setup
  trial_01/
    console.log                  # torchrun and learner output
    metrics.jsonl                # rank-zero training metrics per logged cycle
    stop_reason.json             # deadline, budget_before_next_cycle, or nonfinite loss
    rank_0_runtime.json           # CUDA peaks, device, duration, error
    rank_1_runtime.json
    summary.json                 # final status, counters, artifact paths
    learner/
      config.yaml                # effective config saved by the existing learner
      final_model.pt             # actor for inference/evaluation
      trainer_state.pt           # learner state; replay/environment excluded
      eval/training/carla_fast/
        epoch_000001_step_<rows>/
          evaluation_summary.json # aggregate benchmark metrics and episode counts
          episode_metrics.csv     # individual completed episodes
          resolved_benchmark.yaml
```

`runs/` is ignored by Git. A successful trial needs zero exit status, both rank
runtime records, logged learner updates, final actor/trainer checkpoints, and
a complete finite benchmark summary for every selected benchmark at the final
epoch when evaluation is enabled. Its `summary.json` includes those benchmark
metrics and report paths.
Failures remain visible in its summary and manifest; the sweep exits nonzero if
any selected trial fails. Trials run independently; a failed trial does not stop
the remaining queue.

To retry selected combinations as fresh runs:

```bash
bash project/soft_actor_critic_tuning/scripts/launch_sweep.sh \
  --trial-ids '3,9' --gpu-pairs '0,1;2,3' --max-concurrent 2
```

Use a new directory for `--output-dir`; existing directories are refused.
There is no exact replay/environment continuation in this launcher.
