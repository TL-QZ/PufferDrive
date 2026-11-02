# Round 3: two-step SAC with adaptive target entropy

Based on the reduced-goal-distance TES-SAC Round 3 configuration in
`project/soft_actor_critic_target_entropy_scheduling/configs/round3_tes_sac`.
Adds `train.n_steps: 2` and isolated experiment paths/group. All other YAML
settings and trial mappings are inherited.

Two consecutive valid rewards contribute to each replay target, with a maximum
horizon of 0.6 simulation seconds at `env.dt=0.3`. Trajectory boundaries can
shorten the return. Use fresh replay with the standard SAC collector.

Training goal spacing remains 6.666666666666667–66.66666666666667 meters.
The private evaluation catalog retains 10-meter spacing for both bounds,
including scheduled `carla_fast` evaluation.

Adaptive target entropy starts at 1.863680 nats, with EMA decay 0.999,
mean/std thresholds 0.01/0.05, decay factor 0.9, and 200 qualifying updates.
Forced decay occurs after 500 million global received agent rows without a
reduction. Alpha remains learned.

| Trial | Replay refresh | Global received-row budget | Evaluation interval |
|---|---|---|---|
| 97 | 0.1 | 2 billion | 10 collection cycles |
| 98 | 0.5 | 4 billion | 2 collection cycles |

Both trials retain gamma 0.999, three update epochs, 2,048,000 replay rows per
rank, and two GPUs. W&B group and run folder are `round3_nstep2_tes_sac`.

Preview either trial (replace 97 with 98 for the second candidate):

```bash
source .venv/bin/activate
bash project/soft_actor_critic_multi_step_return/scripts/launch_round3.sh --trial 97 --dry-run --no-wandb
```

To train, remove `--dry-run --no-wandb`; use `--gpus ID,ID` to change GPU IDs.
No training was launched during preparation. Dry runs validate configuration
and command construction, not live GPU training or policy quality.

Standalone evaluation must explicitly select the private catalog:

```text
eval.benchmark_config=project/soft_actor_critic_multi_step_return/configs/round3_nstep2_tes_sac/evaluation_benchmarks.yaml
```

Checkpoint loading does not restore this catalog path. For `nuplan_multi`, also
set `eval.num_agents=300` or greater (the inherited default remains 128).
