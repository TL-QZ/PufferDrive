# SAC sweep versus the CARLA nightly PPO baseline

**Historical audit:** the comparison below records the settings before the
user’s corrections. Subsequent configuration and smoke verification are in
[compiled evaluation validation](archived/smoke_runs/compiled_evaluation_2026-10-11.md).
Compilation, baseline workers, model learning rates, seed and the 10B global
collection ceiling have been restored. The user selected evaluation every
cycle, final-only checkpoints, and fixed learning rates for this experiment.


**Result: the grid is controlled correctly, but the fixed run settings do not
yet fully mirror PPO.** Environment, architecture settings, benchmark definitions,
and nominal minibatch geometry match. Several non-swept training/execution
settings were changed while preparing the launcher and need explicit correction
or an agreed exception before calling this a PPO-matched sweep.

The intended rule is: inherit the selected PPO baseline wherever the setting
has the same meaning; vary only the four grid parameters; retain necessary SAC
mechanisms and explicitly requested experiment controls.

## Reference and verification

The reference is the actual
[two-GPU baseline launcher](../../baseline_run_sync_2026-08-24/train/launch_2gpu.sh),
its [nightly overrides](../../baseline_run_sync_2026-08-24/override_config/nightly_best.yaml),
and inherited [Drive defaults](../../../pufferlib/config/puffer_drive.yaml).
The launcher supplies seed `0`, `1`, or `2`; this audit composed seed `0` as one
concrete baseline reference. It also supplies vector and minibatch overrides
that are absent from `nightly_best.yaml`, so comparing that YAML alone is insufficient.

The SAC reference is the production merge performed by
[run_sac16_grid.py](../scripts/run_sac16_grid.py): tunable, specific, then fixed
config, followed by grid values and unique run paths. The catalog
`SAC_all_configs.yaml` and reduced smoke settings are not production inputs.

PPO configuration was composed through its actual `load_config()` without
creating environments or loggers. A fresh SAC dry run generated all 16 resolved
configs. A field comparison verified that only `train.update_epochs`,
`train.replay_update_ratio`, `train.initial_alpha`, and `train.tau` vary between
trials, apart from `run_name` and `train.data_dir`. Every trial is fresh and has
the 32,000-row microbatch setting. No training, benchmark, or W&B session was
started for this audit.

## What matches

| Area | Verified settings |
| --- | --- |
| Environment | All 84 resolved fields match before SAC's required final-observation override: CARLA maps, Gigaflow, 3,200 slots/instance, jerk dynamics, `dt=0.3`, scenario length 2,560, resampling 256,000, rewards, randomization, observations, controllers and reset behavior. |
| Network architecture | All 21 `policy` fields match: encoder widths/activations, 1,024-wide three-layer backbones, 256-wide heads, normalization, discrete policy and independent backbones. Recurrence is off in both. |
| Benchmark definitions | All 14 `eval` fields match, including the baseline benchmark catalog and `action_selection=mean`. Both select `carla_fast` for scheduled evaluation. The shared catalog uses 250 scenarios, 500 steps, and `dt=0.1`; this evaluation/training clock difference is inherited from the baseline. |
| Shared learning settings | Actor LR `0.0005`, gamma `0.999`, AdamW, betas `(0.9,0.999)`, epsilon `1e-8`, weight decay `0.01`, gradient clipping `0.5`, BF16 AMP and nondeterministic backend settings match. |
| Collection and nominal update geometry | Two GPUs/trial, 20 Drive instances/rank, 3,200 slots/instance and ten instances/receive. PPO's automatic rollout size resolves to `20*3200*128=8,192,000` per rank, matching SAC replay capacity numerically. Both logical minibatches are 128,000/rank and microbatches 32,000/rank: four chunks and a nominal 256,000 transitions across two ranks per optimizer update. PPO advantage filtering means its actual contributing samples need not equal this nominal count. |

## Fixed differences that are not grid axes

These are the settings to correct or explicitly accept; they are not automatic
requirements of the SAC algorithm.

| Setting | Actual PPO baseline | Current SAC sweep | Interpretation / correction for literal parity |
| --- | --- | --- | --- |
| `train.evaluation_interval_epochs` | `100` | `null` | **Mismatch:** disables both periodic and automatic final benchmark evaluation. Restore `100` for the baseline cycle schedule. See timing limits below. |
| `train.checkpoint_interval` | `500` | `50` | **Mismatch:** both count outer collection/training cycles. Restore `500` for literal parity. Both also save on normal completion. |
| Critic learning rate | `0.0005` through PPO's model optimizer | `train.critic_learning_rate=0.0003` | **Mismatch:** PPO's AdamW includes actor and value-backbone parameters. SAC needs a separate twin-Q optimizer, but does not require a different LR. Use `0.0005` to inherit the baseline non-swept optimizer LR. |
| `train.compile` | `true` | `false` | **Mismatch:** an execution choice, not a SAC requirement. Restore `true` for parity and verify it with a short GPU test; existing GPU smoke evidence covers eager execution only. |
| `vec.num_workers` | `auto`, resolved to `20` per rank | `4` per rank | **Mismatch:** same number of simulated instances, different process layout. Restore `auto`/20 to match. With two concurrent two-rank trials this means 80 vector workers rather than 16, on a server with 48 physical CPU cores; scheduling contention must be understood before assuming identical throughput. |
| `train.seed` | Launcher selects `0`, `1`, or `2` | `42` for every trial | **Mismatch:** select one baseline seed, such as `0`, and hold it fixed across the 16 configurations. A multi-seed repeat is a separate experiment, not part of this grid. `vec.seed=42` already matches PPO. |
| `train.anneal_lr` | `true` | `false` | **Mismatch:** the launcher preparation disabled a shared learning mechanism. Literal parity restores cosine annealing, together with the baseline step budget. Do not merely enable it against the current enlarged ceiling. |
| `train.total_timesteps` | Configured `10,000,000,000` global; PPO's loader divides this into per-rank budgets under torchrun | `1,000,000,000,000` global ceiling | **Different run choice:** the oversized ceiling was introduced for the requested three-hour deadline. Literal config parity can retain the baseline 10B global budget while also imposing the requested three-hour cap. SAC already counts global received rows, so do not divide its setting again. Actual LR progress will still differ between algorithms and replay replacement ratios. |

Sources: [SAC shared settings](../configs/round1/SAC_tunable_config.yaml),
[SAC-specific settings](../configs/round1/SAC_specific_config.yaml),
[PPO optimizer and scheduler](../../../pufferlib/pufferl.py),
[SAC optimizers and scheduler](../../../pufferlib/pufferl_sac.py), and
[automatic worker resolution](../../../pufferlib/vector.py).

## Required or explicitly requested differences

| Difference | Reason |
| --- | --- |
| Four grid axes | `update_epochs=[3,10]`, `replay_update_ratio=[0.05,0.1]`, `initial_alpha=[0.1,0.2]`, `tau=[0.005,0.01]`: exactly 16 combinations. PPO's base epoch count is three; the ten-pass alternative is an explicit sweep choice. |
| SAC actor, twin Q networks, targets and temperature | Same architecture settings and action mapping, different required outputs and optimizer ownership. PPO has a scalar value estimator; SAC needs two action-value estimators, target critics, and learned alpha. Fixed alpha LR `0.0003` and automatic target entropy have no direct PPO setting to copy. |
| Replay retention and final observations | PPO replaces a rollout; SAC retains and refreshes complete valid transitions. Startup forces `env.capture_final_observations=true`, despite `false` in the fixed YAML, to bootstrap truncations correctly. The initial replay fill and replacement quotas do not have identical PPO collection semantics. |
| CPU replay | At 923 FP32 observation features and 8,192,000 entries, current/next observations plus scalar fields need about 56.5 GiB per rank. This exceeds a 48 GB A6000 before network memory. CPU storage is a necessary choice for this replay layout and capacity; it changes host/device transfer costs. |
| Experiment management | The requested three-hour trial cap, W&B project `pufferdrive_vanilla_sac_tuning`, distinct group, unique output paths, and forced fresh trials are deliberate. Two simultaneous trials are the current host-RAM-conscious default; the 16-trial nominal budget remains 96 GPU-hours. |

PPO-only GAE, clipping, value bootstrapping, advantage filtering and related
settings cannot be copied into the SAC loss as if they were shared parameters.
They are algorithm differences. PPO's numeric `train.batch_size` and SAC's
numeric replay capacity match here, but the tensors and refresh rules differ.

## Evaluation cadence and the three-hour deadline

Both trainers' `evaluation_interval_epochs` counts **outer collection/training
cycles**, not the replay passes selected by `train.update_epochs` and not
individual optimizer updates. SAC performs 64 logical updates per replay pass:
three passes produce 192 updates/cycle; ten produce 640.

The stopped production run
`runs/sac16_grid_20261012T012830Z/trial_01/metrics.jsonl` recorded cycle 1 at
732.8 seconds and cycle 5 at 3,429.8 seconds. The intervening cycles averaged
674.3 seconds, about 11.2 minutes, with three passes and two concurrent trials.
Thus a three-hour trial would finish only roughly 15–16 cycles at the observed
eager speed. **Restoring `100` does not provide an in-trial benchmark at this
speed.** It enables a final benchmark on normal SAC completion. Ten-pass trials
will have fewer cycles. Compiled performance has not been measured, so these
timings are observations of the stopped eager run, not predictions for a fixed
compiled profile.

Matching PPO's interval value and matching its evaluation frequency in elapsed
minutes are different requirements. A shorter/time-based cadence would be an
additional explicit experiment choice; it should not be silently substituted
while claiming all shared settings match PPO.

The soft deadline is checked after a full cycle, while the parent's hard deadline
is fixed at trial start + three hours + 600 seconds. Those 600 seconds cover
**cycle overshoot plus final evaluation, checkpoints, and shutdown**, rather than
starting after a cycle finishes. A long ten-pass cycle can therefore be killed
before final evaluation begins; final artifacts are not guaranteed in that case.
Existing `run_training_evaluation()`
also catches evaluation exceptions and returns an empty result, while the grid's
success check currently requires learner artifacts rather than benchmark
artifacts. Consequently a completed trial is not proof that its benchmark ran
successfully. Final benchmark duration and compiled execution need verification
before a strict benchmarked sweep is relaunched.

## Other fields and runtime details

Generic `sweep.*`, `max_runs=4`, profiling and autotune settings are inactive in
this grid launcher. Their differences from PPO's generic search defaults do not
change the 16 trials. The `sweep` epoch candidates `[3,5,15,30]` are not the
grid's `[3,10]` candidates. No hidden actor-LR sweep is applied.

The same numeric training seed does not imply identical cross-algorithm random
streams. In addition to different optimizer/action sampling operations, the
current SAC constructor derives reset seeds with `derive_rank_seeds`, whereas
PPO resets with `train_seed*world_size+rank`. Exact reset-stream identity would
require a separate implementation alignment; this is distinct from fixing the
current 42-versus-baseline-seed configuration mismatch.

This audit records the current state; no production config was changed during
the review. The baseline-matched claim should remain qualified until the fixed
differences above are corrected or explicitly accepted. Next: resolve that table
before generating a fresh launch directory.
