# Round 4.1: two-step SAC with lower initial learning rates

Round 4.1 copies Round 4's five YAML inputs and selected trials. It changes the
initial actor, critic, and alpha learning rates; cosine annealing stays enabled,
so the table shows each optimizer's starting rate.

| Learning rate | Round 4 | Round 4.1 |
|---|---:|---:|
| Actor (`train.learning_rate`) | 0.0005 | 0.0001 |
| Critic (`train.critic_learning_rate`) | 0.0005 | 0.0001 |
| Alpha (`train.alpha_learning_rate`) | 0.0003 | 0.00007 |

Inherited settings include `train.n_steps: 2`, `initial_alpha: 0.1`,
`train.anneal_lr: true`, `train.minibatch_size: 64000`, and
`train.max_minibatch_size: 32000`. Adaptive target entropy retains its target
(1.863680), EMA (0.999), thresholds (0.01/0.05), decay (0.9), required count
(200), and forced-decay horizon (500M global rows). TES and collapse diagnostics
remain enabled. Trials 97/98 retain their 2B/4B received-row budgets and
10/2-cycle evaluation intervals. The new private evaluation catalog retains
10-meter goal spacing.

Preview both trials with distinct temporary output directories:

```bash
source .venv/bin/activate
bash project/soft_actor_critic_multi_step_return/scripts/launch_round4_1.sh --trial 97 --dry-run --no-wandb --output-dir /tmp/round4_1_nstep2_tes_sac_trial97
bash project/soft_actor_critic_multi_step_return/scripts/launch_round4_1.sh --trial 98 --dry-run --no-wandb --output-dir /tmp/round4_1_nstep2_tes_sac_trial98
```

To train, remove `--dry-run --no-wandb` and omit `--output-dir` to use the
launcher’s timestamped Round 4.1 output path. Outputs use W&B group
`round4_1_nstep2_tes_sac` and the Round 4.1 learner data directory.

Standalone evaluation should set
`eval.benchmark_config=project/soft_actor_critic_multi_step_return/configs/round4_1_nstep2_tes_sac/evaluation_benchmarks.yaml`.
