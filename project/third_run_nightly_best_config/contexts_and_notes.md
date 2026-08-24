# Third nightly-best baseline: purpose and upstream changes

## Experiment context

This directory is for the first baseline trained after synchronizing with
`upstream/3.0` through commit `8adcec26` (local planning snapshot: `126def4c`).
It follows the second-run structure:

1. Train the discrete policy in the continuous-action CARLA/Gigaflow simulator
   for **10 billion global agent steps**.
2. Fine-tune that checkpoint on **nuPlan single-agent replay** for **100 million
   global agent steps**.

This matches the second-run nuPlan fine-tuning budget.

## Required time-step alignment

The second run inherited `dt=0.3` for CARLA training, but its shared benchmark
configuration forced `dt=0.1` during evaluation. The same mismatch exists in the
original repository's nightly and Kesai launch scripts. Because the study
focuses on zero-shot CARLA-to-nuPlan transfer, the third run will use **`dt=0.1`
everywhere**: CARLA training, CARLA evaluation, nuPlan evaluation, and nuPlan
fine-tuning. This removes timestep as a domain-transfer confound and matches
nuPlan's 10 Hz recorded frames.

`dt` changes the physical time per policy action and directly scales jerk
integration (`delta_acceleration = jerk * dt`). To preserve the second run's
physical CARLA episode and resampling durations while training at `0.1`, use:

- `env.scenario_length=7680` (`768 s`)
- `env.resample_frequency=768000`
- `train.gamma=0.9996665555` (`0.999^(1/3)`)
- `train.gae_lambda=0.9830475725` (`0.95^(1/3)`)

The CARLA evaluation lengths remain 6,000 steps for the full 600 s benchmark
and 500 steps for the 50 s fast benchmark. The training budget remains 10
billion agent steps; at `dt=0.1`, this covers one-third of the simulated physical
time of the second run's 10-billion-step `dt=0.3` training, so the third run is a
new temporally aligned baseline rather than an exact reproduction.

The BPTT horizon remains 128 transitions to preserve the tested optimization
setup. Its physical horizon is therefore **12.8 s** (`128 x 0.1 s`), rather than
the second run's **38.4 s** (`128 x 0.3 s`). This is another intentional
consequence of choosing a shared 10 Hz clock; the transition budget and tensor
batch sizes remain unchanged, but their physical-time exposure does not.

## What changed since the second run

### Changes that affect training and evaluation

- **Traffic-light violations were corrected.** A violation is now a one-step
  event when the vehicle's rear bumper crosses a stop line while the light is
  red. The old code could repeatedly or incorrectly penalize a vehicle already
  inside an intersection. Red-light rewards and metrics therefore use different
  semantics from the second run.
- **Vertical observations changed.** The second run divided relative height by
  a fixed 4 m constant and omitted partners, road segments, and traffic controls
  more than 4 m above or below the ego. The new code keeps them when they pass
  the normal spatial-range checks and divides height by
  `env.obs_norm_z_m=10.0`. This fixes missing observations on steep maps such as
  CARLA Town03, but also changes the policy's input distribution.
- **Spawned vehicle steering geometry was fixed.** The simulator now keeps the
  selected vehicle wheelbase instead of always replacing it with `0.6 × vehicle
  length`. Steering and curvature can therefore differ for some vehicle types.

These are intentional simulator changes. A third-run score is therefore a new
baseline, not a seed-only replication of the second run.

### Evaluation-only change

- **Mean action selection was fixed and is now the default.** The policy emits
  probabilities over discrete actions, while the environment accepts a
  continuous jerk command. Because braking and acceleration have asymmetric
  physical ranges, the expectation must be computed in physical units. The old
  code averaged normalized actions first and could even produce the wrong sign.
  Training samples discrete actions and is unaffected; deterministic
  mean-action evaluation trajectories can change substantially.

The second-run final-evaluation scripts explicitly selected `mean`, so their
saved results used the old calculation. Do not compare those numbers directly
with third-run evaluations. Periodic second-run evaluation used `mode` and was
not affected by this particular fix.

### New switches whose defaults preserve second-run behavior

- `train.use_value_bootstrapping=true`: retains the existing truncation
  bootstrap heuristic. Setting it false defines a different PPO target.
- `env.reward_log_sampling=false`: retains the second run's **uniform** reward
  coefficient sampling. Log-uniform sampling is newly available but should not
  be enabled in this baseline without recording it as an ablation.
- `env.reset_accel_on_stop=false`: retains the previous jerk-dynamics behavior
  when velocity crosses zero. A reasonable follow-up is to test `true`, which
  sets both velocity and acceleration to zero at a stop and may prevent residual
  braking from turning into unintended reverse motion. Keep it `false` for this
  baseline so that the dynamics change is isolated as a separately named
  ablation rather than mixed into the upstream comparison.
- `policy.actor_head_layer_norm=false` and
  `policy.critic_head_layer_norm=false`: retain the old network architecture.
- `env.eval_perceived_size_margin_m=0.1`: exposes the old fixed evaluation
  margin as a setting without changing its default.

### Operational changes

- Evaluation now clips all continuous-environment actions, fails early on
  non-finite observations/actions, and reports action-space mismatches more
  clearly. Map-grid construction also validates invalid geometry.
- Standalone evaluation still defaults to `<run>/eval` and `final_eval_*` W&B
  keys; the output directory/prefix is now configurable.
- Checkpoint saving now keeps only the newest periodic checkpoint and newest
  best checkpoint. `final_model.pt` is not removed, but historical intermediate
  checkpoints must be copied elsewhere if they are needed.
- Replay visualization and upstream SLURM helper scripts received small fixes;
  these do not define the learning objective.

## Comparison rule

Compare models only when both are evaluated with the same code revision,
benchmark YAML, scenario set, seed, and action-selection rule. To measure model
quality under the updated simulator, re-evaluate both the second-run checkpoint
and third-run checkpoint with the updated evaluator. Keep the old reports as
results from the pre-sync simulator; they cannot be reproduced exactly through
configuration because the old height filter and old mean-action calculation no
longer have compatibility switches.

After every sync that changes `.c` or `.h` files, rebuild before launching:

```bash
source .venv/bin/activate
python setup.py build_ext --inplace --force
```

## Workflow layout and commands

- `override_config/nightly_best.yaml`: CARLA/Gigaflow baseline overrides.
- `override_config/nuplan_sdc_finetune.yaml`: nuPlan SDC-only fine-tuning
  overrides.
- `override_config/evaluation_benchmarks.yaml`: the shared CARLA and nuPlan
  benchmark definitions.
- `train/`: two-GPU launchers. Fine-tuning stages only `final_model.pt` and
  `config.yaml`; it never imports the CARLA run's optimizer/trainer state.
- `eval/`: standalone mean-action evaluation on `carla`, `nuplan_single`, and
  `nuplan_multi` with 300-agent capacity.
- `render/`: replay selected failures from the newest standalone evaluation
  CSV, with observation capture enabled.

All artifacts are written below
`experiments/third_run_nightly_best_config`. Run these commands from the
repository root, replacing the seed and GPU indices as needed:

```bash
# Inspect the CARLA command without launching it.
CUDA_VISIBLE_DEVICES=0,1 DRY_RUN=1 \
  ./project/third_run_nightly_best_config/train/launch_2gpu.sh 0

# Train CARLA, then fine-tune its completed seed on nuPlan.
CUDA_VISIBLE_DEVICES=0,1 \
  ./project/third_run_nightly_best_config/train/launch_2gpu.sh 0
CUDA_VISIBLE_DEVICES=0,1 \
  ./project/third_run_nightly_best_config/train/launch_nuplan_sdc_finetune_2gpu.sh 0

# Evaluate the CARLA checkpoint and the fine-tuned checkpoint.
CUDA_VISIBLE_DEVICES=0 \
  ./project/third_run_nightly_best_config/eval/evaluate_final_model.sh 0
CUDA_VISIBLE_DEVICES=0 \
  ./project/third_run_nightly_best_config/eval/evaluate_nuplan_sdc_finetune.sh 0

# Example: render at most 10 nuPlan-single offroad failures from each model.
CUDA_VISIBLE_DEVICES=0 \
  ./project/third_run_nightly_best_config/render/render_eval_failures.sh \
  0 nuplan_single offroad 10
CUDA_VISIBLE_DEVICES=0 \
  ./project/third_run_nightly_best_config/render/render_nuplan_sdc_finetune_failures.sh \
  0 nuplan_single offroad 10
```

Seeds accepted by these launchers are 0, 1, and 2. Training requires exactly
two visible GPUs; evaluation and rendering require exactly one. Run standalone
evaluation before failure rendering because the renderer consumes its saved
`episode_metrics.csv`.
