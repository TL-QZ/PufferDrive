# Baseline PPO: 4B received simulator rows

Standalone configuration snapshot derived from
`project/baseline_run_sync_2026-08-24/override_config/nightly_best.yaml`,
its two-GPU launcher settings, and the current `pufferlib/config/puffer_drive.yaml`
defaults. Training uses the repository implementation, virtual environment, and
map resources, but does not load files from the original experiment project.

## Launch

From the repository root, inspect the command without starting training:

```bash
DRY_RUN=1 project/baseline_ppo_4b_carla/train/launch_2gpu.sh 0
```

Run seed 0 on a selected pair of GPUs:

```bash
CUDA_VISIBLE_DEVICES=0,1 project/baseline_ppo_4b_carla/train/launch_2gpu.sh 0
```

Seeds 0, 1, and 2 are supported. Each invocation creates a timestamped run name
under `experiments/baseline_ppo_4b_carla/`, with W&B project `pufferdrive` and
group `baseline_ppo_4b_carla`. GPU indices default to `0,1`.

## Budget and evaluation

`train.total_timesteps: 4000000000` is the aggregate received simulator agent-row
budget across both ranks, including masked rows. The existing PPO loader divides
this into 2B per rank. The training loop finishes complete collection/update
cycles, so the actual final count can exceed 4B. This is not an optimizer-step
count or a count of vector-environment calls.

`train.evaluation_interval_epochs: 2` follows the 4B SAC trial-98 setting in
`project/soft_actor_critic_target_entropy_scheduling/configs/round2_tes_sac/SAC_run_config.yaml`.
This matches collection-cycle frequency, not exact received-row spacing: PPO
fills rollouts whereas SAC refreshes a fraction of replay. Evaluation uses
`carla_fast`, mean actions, and a byte-identical local copy of the baseline
benchmark catalog. The existing trainer also evaluates at completion if needed.

The baseline training clock (`env.dt: 0.3`) and evaluation clock (`dt: 0.1`) are
preserved. PPO hyperparameters, rollout sizing, rewards, and environment settings
are unchanged. `override_config/ppo.yaml` explicitly snapshots all configuration
leaves so inherited values are reviewable locally. The launcher only supplies
seed, run identity, output directory, and the same baseline two-GPU sizing.

## Validation

Passed shell syntax, launcher dry run, full Hydra/schema composition, 4B-to-2B
per-rank budget conversion, resolved baseline config comparison, and benchmark
byte comparison. No training, CUDA execution, or runtime evaluation was launched.
