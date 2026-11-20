# SAC 4B GPU comparison

This isolated project derives TES trial 98 from
`project/soft_actor_critic_target_entropy_scheduling/configs/round2_tes_sac`
and PQL trial 98 from
`project/soft_actor_critic_multi_step_return_PQL/configs/PQL_official_round2/trial_98`.
The source configs are unchanged.

Both derivatives preserve the source 4B global received-row budget, 2-cycle
evaluation cadence, 2,048,000-row replay capacity, and evaluation catalog
contents. Catalog copies live beside each derivative config. Logical batches
remain TES 128,000 and PQL 64,000. Other source-specific settings, including
TES's 3,200 and PQL's 800 environment agent slots, are preserved.

The exact shared learner changes are `train.cpu_offload: false`,
`train.update_epochs: 1`, `train.replay_update_ratio: 0.3333333333333333`, and
`train.max_minibatch_size: 64000`. TES keeps trial-98 learning rates (actor and
critic `0.0005`, alpha `0.0003`). PQL changes its actor, critic, and alpha rates
from `0.0001`, `0.0001`, and `0.00007` to those TES values.

Trial 98 retains its original index in the 3x3x4x3 grid: gamma `0.999`, target
entropy `1.863680`, update epoch position 0, and replay-ratio position 1. The
selected epoch changes from 3 to 1 and selected ratio from 0.5 to 1/3. Both
`sweep` and `controlled_exp` axes reflect those new values. Derivative identity
changes are the local catalog paths, `train.data_dir` templates, and W&B groups:
`tes_round2_gpu_1ep` and `pql_official_round2_gpu_1ep_highlr`.

Run dry runs with:

```bash
bash project/sac_gpu_4b_comparison/launch_tes.sh --dry-run
bash project/sac_gpu_4b_comparison/launch_pql.sh --dry-run
```

Omit `--dry-run` to train. Each launcher accepts `--output-dir PATH` and
`--no-wandb`; default outputs go under the ignored local `runs/` directory.
W&B is online by default (`WANDB_MODE=online`). Launchers use `.venv` and call
the production `project/soft_actor_critic_tuning/scripts/launch_sweep.sh` with
trial 98, one concurrent trial, and GPU groups 0,1 (TES) or 2,3 (PQL). The
production dry-run configs/manifests provide recursive parity evidence; no
training is launched by a dry run.

## Optional evaluation cadence by received steps

The existing recipes keep their two-cycle evaluation schedule. SAC also supports
an opt-in interval measured in global received simulator agent rows:

```yaml
train:
  evaluation_interval_epochs: null
  evaluation_interval_agent_steps: 32768000
```

This matches the 4B PPO baseline's nominal evaluation spacing across both GPUs.
It counts all received slots, including masked slots, rather than valid replay
insertions or retained training rows. Evaluation happens after the first completed
collection/training cycle crossing a multiple of the interval, so it can overshoot
by one cycle. The next threshold remains aligned to the original global multiples;
crossing several thresholds in one cycle runs one evaluation, not catch-up repeats.
Final evaluation remains enabled and is not duplicated if the final cycle already
evaluated. On resume, the next threshold is the first multiple strictly above the
restored received-step count.

Only one interval may be non-null. With `evaluation_interval_agent_steps: null`
(the default), epoch-based behavior is unchanged. With both intervals null,
scheduled and final training evaluation remain disabled. The step-based option
is supported by the normal SAC/TES/PQL training entrypoint; other training loops
reject it rather than silently using an epoch schedule.

For a copied experiment bundle, put the two `train` values above in
`SAC_tunable_config.yaml` and remove the selected trial's epoch override from
`SAC_run_config.yaml`. For the single-trial recipes here, use:

```yaml
launcher:
  trial_evaluation_interval_epochs: {}
```

The launcher otherwise reapplies its old `98: 2` override. Keep all other launcher
keys. Inspect the resulting `configs/trial_98.yaml` with `--dry-run` before launch.
Evaluation-triggered checkpoints follow the new cadence; separately configured
checkpoint intervals are unchanged. This option does not change replay size,
collection quotas, training updates, filtering, or the step budget.
