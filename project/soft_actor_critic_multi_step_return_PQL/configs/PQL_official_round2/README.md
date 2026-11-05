# PQL official Round 2

This round repeats the PQL official Round 1 pair with the original, wider goal spacing restored. Relative to Round 1, the only training/evaluation setting changed is goal spacing; the PQL horizon remains 128 because the experiment is not increasing the multi-step return horizon.

Training uses `env.min_goal_spacing: 20.0` and `env.max_goal_spacing: 200.0` meters. The shared evaluation catalog uses 30.0 meters for both minimum and maximum goal spacing. All other learner, replay, entropy-schedule, budget, and evaluation-cadence settings are copied from Round 1.

The two trial codes retain the Round 1 replay replacement comparison:

| Code name | Generated trial ID | `train.replay_update_ratio` |
|---|---:|---:|
| `PQL_official_round2_98` | 98 | 0.50 |
| `PQL_official_round2_01` | 1 | 0.25 |

Both runs use 800 agents per environment, 20 environments, a 2,048,000-transition replay capacity per rank, PQL horizon 128 and lambda 0.95, a 4B global received-row budget, and evaluation every two collection cycles. Actor and critic learning rates are `1e-4`; alpha learning rate is `7e-5`. Target-entropy EMA standard deviation must be at most `0.025`, with 2,000 qualifying learner updates before an adaptive reduction. Learning-rate annealing and the remaining Round 4.1 settings are shared.

The Round 4.1 evaluation catalog is shared by both configs apart from restoring the baseline 30 m goal spacing. Trial 98 preserves its original position in the 108-combination grid, selecting gamma `0.999`, target entropy `1.863680`, three update epochs, and 50% replacement. Trial 01 uses singleton axes for the same common values and 25% replacement. The different grid metadata preserves the requested generated IDs; it does not change the learner recipe beyond the replacement rate.

Preview either run without training:

```bash
bash project/soft_actor_critic_multi_step_return_PQL/scripts/launch_PQL_official_round2.sh --trial 98 --dry-run
bash project/soft_actor_critic_multi_step_return_PQL/scripts/launch_PQL_official_round2.sh --trial 01 --dry-run
```

Validation: launcher syntax and dry runs for both trials passed. Generated
Round 2 configs match their Round 1 counterparts exactly except for training
goal spacing and isolated run/group/output/catalog paths. The shared evaluation
catalog differs only in goal spacing. No training was launched.
