# Baseline after the August 24 upstream sync

## Purpose

This directory defines the first isolated baseline after the local
`develop-3.0` rebase onto upstream commit `8adcec26`. The reflog records that
rebase as completing on **August 24, 2026 at 00:52 EDT**, so the experiment is
named `baseline_run_sync_2026-08-24`.

The workflow has two stages:

1. Train a discrete policy in CARLA/Gigaflow for **10 billion global agent
   steps**.
2. Fine-tune a completed CARLA checkpoint in nuPlan SDC-only replay for **1
   billion global transitions**, sampling randomly with replacement from all
   169,715 training scenarios available on September 11, 2026.

It follows the third-run directory layout, but it deliberately restores the
second run's full 0.3-second CARLA training clock. The September 10 update adds
opt-in 0.3-second nuPlan fine-tuning; evaluation remains at 0.1 seconds.
See [nuPlan dt=0.3 workflow and validation](nuplan_dt03_workflow.md) and the
[step-by-step code-reading guide](nuplan_dt03_code_reading_guide.md).

## Timing contract

CARLA training pins these five coupled values:

| Setting | Value | Meaning |
|---|---:|---|
| `env.dt` | `0.3` | 3.33 policy transitions per simulated second |
| `env.scenario_length` | `2560` | 768 simulated seconds |
| `env.resample_frequency` | `256000` | second-run resampling cadence |
| `train.gamma` | `0.999` | per-transition reward discount |
| `train.gae_lambda` | `0.95` | per-transition GAE trace discount |

The 128-transition BPTT horizon therefore spans **38.4 simulated seconds**.
The global training budget remains 10 billion agent steps.

nuPlan fine-tuning enables `env.resample_replay_to_dt=true`: 201 logged states
at 0.1 seconds become 67 states at 0.3 seconds (indices 0, 3, ..., 198).
An episode has 66 transitions over **19.8 seconds**, with `gamma=0.999` and
`gae_lambda=0.95`. Its 1,000-transition rollout spans multiple reset episodes;
`rnn_name: null` keeps PPO feed-forward.

Every benchmark explicitly sets `env.dt=0.1` and
`env.resample_replay_to_dt=false`. nuPlan evaluation retains 200 transitions
and 20-second scenarios. Training and evaluation deliberately use different
action intervals; their transitions are not directly equivalent.

## Retained configuration and upstream behavior

The CARLA stage preserves 10 billion steps, BPTT 128, value bootstrapping,
uniform reward-coefficient sampling (`reward_log_sampling=false`),
`reset_accel_on_stop=false`, and disabled actor/critic head layer normalization.
The fine-tune uses a 1-billion-global-transition budget and the starting
workload documented in `nuplan_dt03_workflow.md`. GPU sizing is pending usable
hardware; starting values are not measured recommendations.

The synchronized code includes upstream fixes that change comparison semantics:

- red-light violations are one-step rear-bumper crossing events;
- vertical observations use the updated filtering and normalization behavior;
- spawned vehicles retain their selected wheelbase geometry;
- deterministic mean-action evaluation converts the expectation in physical
  action space.

Operational validation, action clipping, checkpoint retention, and replay
visualization fixes from the sync are retained as well. The new replay loader
option defaults to false; stepping, rewards, dynamics, and PPO remain unchanged.
Experiment launchers and profiling stay in this directory.

## Comparison rules

Compare checkpoints only when code revision, benchmark YAML, scenario set,
seed, controller matrix, and action-selection rule match. The 0.3-second CARLA
training stage is comparable to the second run's training clock, but results
are not a seed-only replication because the synchronized simulator and evaluator
contain the fixes listed above. The 0.1-second third-run CARLA training baseline
is a different timing experiment.

Old CARLA results produced with replay-controlled background vehicles are not
valid all-agent self-play comparisons. This workflow explicitly keeps policy
control for CARLA SDC and background vehicles; nuPlan single uses replay
backgrounds and SDC-only policy control; nuPlan multi uses policy vehicle
control with replay non-vehicles.

After any future sync that changes C or header files, rebuild before training:

```bash
source .venv/bin/activate
python setup.py build_ext --inplace --force
```

No rebuild is needed merely to add or edit this experiment workflow.

## Layout and commands

- `docs/`: experiment context, operating workflow, and code-reading map.
- `override_config/nightly_best.yaml`: 0.3-second CARLA training overrides.
- `override_config/nuplan_sdc_finetune.yaml`: opt-in 0.3-second nuPlan SDC fine-tune.
- `override_config/evaluation_benchmarks.yaml`: shared 0.1-second benchmarks.
- `train/`: two-GPU launchers, one seed per invocation.
- `eval/`: one-GPU standalone mean-action evaluation with 300-agent capacity.
- `render/`: one-GPU CSV-driven failure replay with observation capture.

All outputs live below `experiments/baseline_run_sync_2026-08-24`. CARLA runs
are named `baseline_run_sync_2026-08-24_TIMESTAMP_seedN`; fine-tunes are named
`baseline_run_sync_2026-08-24_nuplan_sdc_finetune_dt03_TIMESTAMP_seedN`. Custom
`RUN_NAME` values must retain the corresponding prefix and seed suffix so the
exactly-one-run-per-seed discovery guards remain reliable.

Run from the repository root, replacing seeds and GPU indices as needed:

```bash
# Inspect the CARLA launch without starting training.
CUDA_VISIBLE_DEVICES=0,1 DRY_RUN=1 \
  ./project/baseline_run_sync_2026-08-24/train/launch_2gpu.sh 0

# Train CARLA, then fine-tune its completed checkpoint on nuPlan.
CUDA_VISIBLE_DEVICES=0,1 \
  ./project/baseline_run_sync_2026-08-24/train/launch_2gpu.sh 0
CUDA_VISIBLE_DEVICES=0,1 \
  ./project/baseline_run_sync_2026-08-24/train/launch_nuplan_sdc_finetune_2gpu.sh 0

# Evaluate the CARLA and fine-tuned checkpoints.
CUDA_VISIBLE_DEVICES=0 \
  ./project/baseline_run_sync_2026-08-24/eval/evaluate_final_model.sh 0
CUDA_VISIBLE_DEVICES=0 \
  ./project/baseline_run_sync_2026-08-24/eval/evaluate_nuplan_sdc_finetune.sh 0

# Render at most 10 nuPlan-single offroad failures from each evaluation.
CUDA_VISIBLE_DEVICES=0 \
  ./project/baseline_run_sync_2026-08-24/render/render_eval_failures.sh \
  0 nuplan_single offroad 10
CUDA_VISIBLE_DEVICES=0 \
  ./project/baseline_run_sync_2026-08-24/render/render_nuplan_sdc_finetune_failures.sh \
  0 nuplan_single offroad 10
```

Seeds are restricted to `0`, `1`, and `2`. Training requires exactly two
distinct visible GPUs; evaluation and rendering require exactly one. The
fine-tune stages only the CARLA run's `final_model.pt` and `config.yaml`, never
its `trainer_state.pt`. It resumes only from the fine-tune run's own trainer
state. Run standalone evaluation before rendering because renderers consume the
saved `episode_metrics.csv`.
