# Round 2: four-step SAC returns

This isolated setup inherits the selected Round 7 candidates used by Round 1: trials 97 and 98. Trial 97 uses replay refresh `0.1`, a 2B global received-row budget, and evaluation every 10 collection cycles. Trial 98 uses refresh `0.5`, a 4B budget, and evaluation every 2 cycles. The inherited two-GPU and learner settings are otherwise unchanged; goal spacing is reduced as described below.

`train.n_steps: 4` sums up to four consecutive valid rewards. At `env.dt=0.3` seconds, the horizon is up to 1.2 simulated seconds; episode and invalid-mask boundaries can shorten it. Outputs use the `round2_nstep4` directory and W&B group.

Launch either trial:

```bash
source .venv/bin/activate
bash project/soft_actor_critic_multi_step_return/scripts/launch_round2.sh --trial 97
bash project/soft_actor_critic_multi_step_return/scripts/launch_round2.sh --trial 98
```

Preview configuration and command construction without starting training:

```bash
bash project/soft_actor_critic_multi_step_return/scripts/launch_round2.sh --trial 97 --dry-run --no-wandb --output-dir /tmp/sac_round2_nstep4_trial97
bash project/soft_actor_critic_multi_step_return/scripts/launch_round2.sh --trial 98 --dry-run --no-wandb --output-dir /tmp/sac_round2_nstep4_trial98
```

No training has been started for this round.

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
eval.benchmark_config=project/soft_actor_critic_multi_step_return/configs/round2_nstep4/evaluation_benchmarks.yaml
```

The shared tuning `evaluate_carla.py` selects the baseline catalog, not this
private catalog. `nuplan_multi` also requires `eval.num_agents=300` or greater;
the inherited default of 128 remains unchanged.
