# Round 2 enhance without critic-target entropy

Derived from `../round2_enhance/`. The only learner change is
`train.critic_entropy_in_target: false`: critic bootstrap targets omit the entropy
bonus. Actor entropy regularization, learned alpha, and adaptive TES remain enabled.
All other settings and benchmark contents are preserved.

Inherited enhanced learner settings:

- `train.minibatch_size: 64000`; `train.max_minibatch_size: 32000` (two microbatches per logical update).
- Actor and critic learning rates: `0.0003`; alpha learning rate: `0.0001`.
- Replay replacement: `0.25` after initial fill (512,000 transitions per rank per collection cycle).
- Adaptive TES copied from `round3_nstep2_tes_sac`: initial target `1.863680`, EMA `0.999`, mean/std thresholds `0.01/0.05`, decay factor `0.9`, cumulative qualifying count `200`, forced decay after 500M global received rows without a reduction. Alpha remains learned.

Trial `01` uses the 4B-row budget and evaluation every two collection cycles.
The singleton grid names the enhanced 25% replacement recipe independently of
its source trial ID.
Outputs and W&B group use `round2_enhance_no_critic_entropy`. Training has not been launched.

Preview trial `01`:

```bash
source .venv/bin/activate
bash project/soft_actor_critic_multi_step_return/scripts/launch_round2_enhance_no_critic_entropy.sh --trial 01 --dry-run --no-wandb
```

To train, remove `--dry-run --no-wandb`; use `--gpus ID,ID` to select two GPUs
(default: `0,1`).

## Goal spacing: one third of the baseline

Training uses `min_goal_spacing: 6.666666666666667` and
`max_goal_spacing: 66.66666666666667` meters (previously 20–200 m).
The private `evaluation_benchmarks.yaml` uses 10 m for both bounds
(previously 30 m), including scheduled `carla_fast` evaluation.
Round 1 and the shared benchmark catalog are unchanged. Goal radius, goal count,
observation normalization, rewards, and simulation timestep are unchanged.

For standalone SAC evaluation, explicitly pass this override alongside the
checkpoint and benchmark selection; checkpoint loading alone does not restore
the benchmark catalog path:

```text
eval.benchmark_config=project/soft_actor_critic_multi_step_return/configs/round2_enhance_no_critic_entropy/evaluation_benchmarks.yaml
```

The shared tuning `evaluate_carla.py` selects the baseline catalog, not this
private catalog. `nuplan_multi` also requires `eval.num_agents=300` or greater;
the inherited default of 128 remains unchanged.
