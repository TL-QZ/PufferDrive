# PQL official Round 1

This paired comparison uses the selected Round 4.1 PQL trial-98 recipe for both runs. The intended training difference is replay replacement rate:

| Code name | Generated trial ID | `train.replay_update_ratio` |
|---|---:|---:|
| `PQL_official_round1_98` | 98 | 0.50 |
| `PQL_official_round1_01` | 1 | 0.25 |

Both runs use 800 agents per environment, 20 environments, a 2,048,000-transition replay capacity per rank, PQL horizon 128 and lambda 0.95, a 4B global received-row budget, and evaluation every two collection cycles. Actor and critic learning rates are `1e-4`; alpha learning rate is `7e-5`. Target-entropy EMA standard deviation must be at most `0.025`, with 2,000 qualifying learner updates before an adaptive reduction. Learning-rate annealing and the remaining Round 4.1 settings are shared.

The Round 4.1 evaluation catalog is shared by both configs. Trial 98 preserves its original position in the 108-combination grid, selecting gamma `0.999`, target entropy `1.863680`, three update epochs, and 50% replacement. Trial 01 uses singleton axes for the same common values and 25% replacement. The different grid metadata preserves the requested generated IDs; it does not change the learner recipe beyond the replacement rate.

Preview either run without training:

```bash
bash project/soft_actor_critic_multi_step_return_PQL/scripts/launch_PQL_official_round1.sh --trial 98 --dry-run
bash project/soft_actor_critic_multi_step_return_PQL/scripts/launch_PQL_official_round1.sh --trial 01 --dry-run
```
